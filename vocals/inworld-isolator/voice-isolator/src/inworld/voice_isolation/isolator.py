# Copyright 2026 Inworld AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""VoiceIsolator, the public inference API, in offline or streaming mode."""

from collections import abc
import contextlib
import dataclasses
import functools
import json

import numpy as np
import torch

from inworld.voice_isolation import audio as audio_lib
from inworld.voice_isolation import model as modeling
from inworld.voice_isolation import registry
from inworld.voice_isolation import streaming

MODES = ("offline", "streaming")
PRECISIONS = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
# "auto": the training precision, bfloat16 autocast, on a CUDA device; float32
# elsewhere, where bfloat16 autocast is slow or unsupported
DEFAULT_PRECISION = "auto"
STREAM_PIECE_SECONDS = 0.5
STREAM_BATCH = 16
STREAM_WINDOW_FRAMES = 600
STREAM_CROSSFADE_FRAMES = 200

# the first PyTorch whose MPS backend runs both released models and matches the
# CPU: 2.2 lacks the FFT ops, 2.3 streams wrong output, 2.4 and 2.5 fail
# offline (unsupported dtype, missing unfold_backward)
MPS_MIN_TORCH = "2.6"


@functools.cache
def _mps_fft_matches_cpu() -> bool:
    """Whether an FFT round trip on MPS runs and agrees with the CPU.

    Catches the macOS releases whose Metal backend lacks the FFT.
    """
    x = torch.linspace(-1.0, 1.0, 64)
    try:
        y = torch.fft.irfft(torch.fft.rfft(x.to("mps")), n=64).cpu()
    except (RuntimeError, NotImplementedError):
        return False
    return bool(torch.allclose(y, x, atol=1e-5))


def default_device() -> torch.device:
    """Returns the device a VoiceIsolator uses when none is given.

    CUDA when available; else Apple's Metal backend (MPS) when available on
    PyTorch ``MPS_MIN_TORCH`` or later and its FFT works; else the CPU. An
    explicit ``device="mps"`` is honoured on any version.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (torch.backends.mps.is_available() and torch.torch_version.TorchVersion(
            torch.__version__) >= MPS_MIN_TORCH and _mps_fft_matches_cpu()):
        return torch.device("mps")
    return torch.device("cpu")


def _windows(window_frames: int | None, crossfade_frames: int | None,
             default: tuple) -> tuple:
    """Resolves the stream's windowing: None takes the default, 0 turns it off.

    Returns ``(window, crossfade)`` in frames, or ``(None, None)`` for the
    plain stream whose attention looks back over its whole cache.
    """
    window = default[0] if window_frames is None else int(window_frames)
    if not window:
        return None, None
    crossfade = default[1] if crossfade_frames is None else int(
        crossfade_frames)
    if not 0 < crossfade < window:
        raise ValueError(
            "crossfade_frames must be positive and shorter than window_frames.")
    return window, crossfade


_STATE_PREFIXES = ("_forward_module.", "module.", "model.", "isolator.")
_TRAINING_ONLY_PREFIXES = (
    "discriminator.",
    "disc.",
    "ema.",
    "optimizer.",
    "scheduler.",
    "enhance_decoder.",
    # the auxiliary decision head, trained beside the output and never served
    "decision_head.",
)


def _strip_prefixes(state: dict) -> dict:
    for p in _STATE_PREFIXES:
        if any(k.startswith(p) for k in state):
            state = {
                (k[len(p):] if k.startswith(p) else k): v
                for k, v in state.items()
            }
    return state


def _architecture(raw: dict, path: str) -> dict:
    """Finds the architecture dictionary inside a checkpoint's config."""
    config = raw.get("config")
    if isinstance(config, dict) and "modeling" in config:
        config = config["modeling"].get("compact_family")
    if not isinstance(config, dict) or "width" not in config:
        raise ValueError(
            f"{path}: no embedded architecture config; refusing to guess the"
            " architecture.")
    return config


def read_checkpoint(path: str) -> tuple[dict, dict]:
    """The architecture dictionary and the weights of a checkpoint.

    Supported: a ``.safetensors`` file whose metadata carries the config JSON
    (the release format), or a training checkpoint ``.pt``
    holding plain tensors and the embedded architecture config. The weights
    come with training wrapper prefixes stripped and training-side extras
    still among them; ``build_model`` sorts those out.
    """
    if path.endswith(".safetensors"):
        # pylint: disable-next=import-outside-toplevel
        import safetensors.torch

        with safetensors.safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
        if "config" not in meta:
            raise ValueError(
                f"{path}: safetensors metadata lacks the 'config' entry.")
        return json.loads(meta["config"]), _strip_prefixes(
            safetensors.torch.load_file(path))
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    # torch raises several types here
    except Exception as e:  # pylint: disable=broad-exception-caught
        raise ValueError(
            f"{path}: not loadable as plain tensors ({type(e).__name__}); "
            "checkpoints are never unpickled with code execution enabled; "
            "export the weights to safetensors instead.") from e
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: unsupported checkpoint object {type(raw)}.")
    state = raw.get("model") or raw.get("state_dict") or raw
    return _architecture(raw, path), _strip_prefixes(state)


def build_model(
        architecture: dict,
        state: dict,
        path: str,
        device: str | torch.device = "cpu") -> modeling.VoiceIsolatorNetwork:
    """The isolator of ``architecture`` carrying ``state``, in eval mode.

    ``path`` names the checkpoint in error messages. Every learned weight must
    be present; only training-side extras may be spare.
    """
    config = modeling.VoiceIsolatorConfig.from_dict(architecture)
    model = modeling.VoiceIsolatorNetwork(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    _check_state_keys(path, missing, unexpected)
    return model.to(device).eval()


def load_model(
        path: str,
        device: str | torch.device = "cpu") -> modeling.VoiceIsolatorNetwork:
    """Builds the isolator's network from a checkpoint, with its architecture.

    The architecture is always taken from the checkpoint (see
    ``read_checkpoint``), never from arguments.
    """
    architecture, state = read_checkpoint(path)
    return build_model(architecture, state, path, device)


def _check_state_keys(path: str, missing: list[str],
                      unexpected: list[str]) -> None:
    """Requires every learned weight; only training-side extras may be spare."""
    spare = [k for k in unexpected if not k.startswith(_TRAINING_ONLY_PREFIXES)]
    if missing or spare:
        raise RuntimeError(
            f"{path}: checkpoint does not match the architecture: "
            f"{len(missing)} missing key(s) {missing[:4]}, "
            f"{len(spare)} unexpected key(s) {spare[:4]}.")


class VoiceIsolator:
    """Isolates voices at 48 kHz, offline for best quality or streaming.

    Example:
        from inworld import voice_isolation
        isolator = voice_isolation.VoiceIsolator("inworld/voice-isolator")
        voice = isolator.isolate(wav, sample_rate)

    The constructor takes the Hub repository id (``mode`` picks its
    checkpoint), a registry name, an ``hf://`` spec, a local checkpoint path
    or a loaded network; ``load`` is the same call with the arguments
    documented.

    ``mode="streaming"`` runs a causal checkpoint such as
    ``voice-isolator-v1-realtime``
    (or a dual-mode one, the same weights run causally) with a trained
    look-ahead of ``lookahead_frames`` frames: ``isolate`` then runs the
    stream itself over the clip, exactly what a live session gives, and
    ``stream()`` returns such a session for audio that arrives in pieces.
    Offline, ``offline_passes`` runs the model again over its own output, in
    the same chunks and precision: the checkpoint's own count unless one is
    passed, one for the released files.
    """

    def __init__(
        self,
        model: modeling.VoiceIsolatorNetwork | str = registry.REPO_ID,
        stage2: None = None,
        device: str | torch.device | None = None,
        precision: str = DEFAULT_PRECISION,
        mode: str = "offline",
        lookahead_frames: int = 0,
        block_frames: int = 4,
        cache_frames: int | None = None,
        window_frames: int | None = None,
        crossfade_frames: int | None = None,
        offline_passes: int | None = None,
    ) -> None:
        # The digest-bound benchmark helpers still pass stage2=None.
        if stage2 is not None:
            raise ValueError("Stage-2 restoration is not supported; use"
                             " isolate() with a released model.")
        if precision != "auto" and precision not in PRECISIONS:
            raise ValueError(
                f"precision must be 'auto' or one of {sorted(PRECISIONS)},"
                f" got {precision!r}")
        self.device = torch.device(device) if device else default_device()
        if isinstance(model, str):
            model = load_model(registry.resolve(model, mode), self.device)
        if precision == "auto":
            precision = "bf16" if self.device.type == "cuda" else "fp32"
        self.precision = precision
        self.model = model.to(self.device).eval()
        self.offline_passes = _passes(offline_passes, self.model.config, mode)
        if block_frames < 1:
            raise ValueError("block_frames must be at least 1.")
        self.block_frames = block_frames
        if cache_frames is not None and cache_frames < 1:
            raise ValueError("cache_frames must be at least 1.")
        self.cache_frames = cache_frames
        self.window_frames, self.crossfade_frames = _windows(
            window_frames, crossfade_frames,
            (STREAM_WINDOW_FRAMES, STREAM_CROSSFADE_FRAMES))
        self._mode = None
        self._lookahead = 0
        self.lookahead_frames = lookahead_frames
        self.mode = mode

    @classmethod
    def load(
        cls,
        model: str = registry.REPO_ID,
        stage2: None = None,
        device: str | torch.device | None = None,
        precision: str = DEFAULT_PRECISION,
        mode: str = "offline",
        lookahead_frames: int = 0,
        block_frames: int = 4,
        cache_frames: int | None = None,
        window_frames: int | None = None,
        crossfade_frames: int | None = None,
        offline_passes: int | None = None,
    ) -> "VoiceIsolator":
        """Loads a released model, an ``hf://`` spec or a local checkpoint.

        The Hub repository id, ``inworld/voice-isolator``, loads the
        checkpoint for ``mode``; a registry name loads one directly.
        An ``hf://<org>/<repo>/<file>[@<revision>]`` spec names any file on the
        Hub; local
        checkpoints may be a ``.safetensors`` file or a training ``.pt``.
        ``mode`` is "offline" (every frame sees its whole chunk) or
        "streaming" (causal, with ``lookahead_frames`` frames of look-ahead,
        one of the checkpoint's trained positions, processed ``block_frames``
        frames at a time, the time attention remembering ``cache_frames``
        frames, by default the checkpoint's attention window, else its
        training chunk). In streaming
        mode the time attention never looks back across a window of
        ``window_frames`` frames: a second lane starts ``crossfade_frames``
        before each boundary and the outputs blend over the crossfade
        (``streaming.Stream`` has the semantics), which keeps a long session
        from drifting. None takes the defaults (600 and 200 frames: 6 s
        windows with a 2 s crossfade); 0 turns the
        windows off, the attention then looking back over its whole cache.
        ``precision``
        "auto" (the default) runs the
        network under bfloat16 autocast on a CUDA device, the precision it
        was trained in, and in float32 elsewhere; "fp32", "fp16" or "bf16"
        force one. The STFT, the iSTFT and the selective scan stay in
        float32 in every case. ``offline_passes`` runs the offline model
        again over its own output: None takes the checkpoint's own count (one
        for the released files); each pass costs one offline render, so two
        double the time. Above one it applies to offline mode only.
        """
        return cls(model,
                   stage2,
                   device,
                   precision=precision,
                   mode=mode,
                   lookahead_frames=lookahead_frames,
                   block_frames=block_frames,
                   cache_frames=cache_frames,
                   window_frames=window_frames,
                   crossfade_frames=crossfade_frames,
                   offline_passes=offline_passes)

    @property
    def sample_rate(self) -> int:
        """The sample rate expected by the model, in Hz."""
        return self.model.config.sample_rate

    @property
    def frame_seconds(self) -> float:
        """The duration of one model frame, the unit of the look-ahead."""
        return self.model.config.hop_length / self.sample_rate

    @property
    def lookahead_choices(self) -> tuple[int, ...]:
        """The look-ahead positions, in frames, the checkpoint trained at."""
        config = self.model.config
        if config.lookahead_frames_choices:
            return config.lookahead_frames_choices
        return (config.lookahead_frames,) if config.causal else ()

    @property
    def mode(self) -> str:
        """The active inference mode: offline or streaming."""
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
        """Selects an inference mode supported by the loaded checkpoint."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}.")
        config = self.model.config
        if mode == "offline":
            if config.dual_mode:
                self.model.set_offline()
            elif config.causal:
                raise ValueError("This checkpoint only streams; load it with"
                                 " mode='streaming'.")
        else:
            if not self.lookahead_choices:
                raise ValueError(
                    "This checkpoint has no streaming mode: it was trained"
                    " offline only.")
            self.model.set_lookahead(self._lookahead)
        self._mode = mode

    @property
    def lookahead_frames(self) -> int:
        """The look-ahead of the streaming mode, in frames."""
        return self._lookahead

    @lookahead_frames.setter
    def lookahead_frames(self, frames: int) -> None:
        """Selects a trained look-ahead position for streaming inference."""
        if self.lookahead_choices and frames not in self.lookahead_choices:
            raise ValueError(
                f"lookahead_frames must be one of {self.lookahead_choices}"
                f" ({self.frame_seconds * 1000:g} ms each), got {frames}.")
        self._lookahead = frames
        if self._mode == "streaming":
            self.model.set_lookahead(frames)

    def latency_samples(self, block_frames: int | None = None) -> int:
        """The streaming mode's algorithmic latency in samples.

        The analysis window plus the look-ahead frames. ``block_frames`` is
        accepted for compatibility and changes nothing: a frame is computed
        as soon as its window is complete, so the block sets throughput, not
        latency.
        """
        del block_frames
        config = self.model.config
        return config.n_fft + config.hop_length * self._lookahead

    def _autocast(self):
        if PRECISIONS[self.precision] is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type,
                              dtype=PRECISIONS[self.precision])

    def _stream_render(self, inp: torch.Tensor, lengths=None) -> torch.Tensor:
        """The stream over ``inp`` [batch, samples] in half-second pieces."""
        session = self.stream(lengths=lengths)  # the session's own defaults
        piece = int(STREAM_PIECE_SECONDS * self.sample_rate)
        pieces = [
            session.push(inp[:, i:i + piece])
            for i in range(0, inp.shape[1], piece)
        ]
        pieces.append(session.flush())
        return torch.cat(pieces, dim=1).to(inp.device)

    def _prepare(self, audio, sample_rate) -> "_Prepared":
        """Makes a clip mono, at the model rate and on the device."""
        x = audio_lib.to_mono(audio)
        if not x.numel():
            raise ValueError("The audio is empty: there is nothing to isolate.")
        if not torch.isfinite(x).all():
            raise ValueError("The audio contains NaN or infinite samples.")
        a = audio_lib.resample(x, sample_rate, self.sample_rate).to(self.device)
        return _Prepared(a, x.shape[-1], isinstance(audio, np.ndarray))

    def _finish(self, y, p: "_Prepared", sample_rate, output_sample_rate):
        y = audio_lib.fit_length(y.float(), p.a.shape[-1])
        out_sr = output_sample_rate or sample_rate
        y = audio_lib.resample(y.cpu(), self.sample_rate, out_sr)
        if out_sr == sample_rate:
            y = audio_lib.fit_length(y, p.n_in)
        return y.numpy() if p.as_numpy else y

    def _separate(self, inp, chunk_kwargs):
        """Runs the model over the prepared audio.

        In streaming mode the stream itself runs over the clip; the chunks
        of the offline render do not apply.
        """
        if self._mode == "streaming":
            batched = inp.ndim == 2
            out = self._stream_render(inp if batched else inp[None],
                                      chunk_kwargs.get("lengths"))
            return out if batched else out[0]
        windows = {k: v for k, v in chunk_kwargs.items() if k != "lengths"}
        with torch.inference_mode(), self._autocast():
            out = self.model.enhance(inp, **windows)
            # Each further pass reads the previous pass's output, in the
            # same chunks and precision.
            for _ in range(self.offline_passes - 1):
                out = self.model.enhance(out, **windows)
            return out

    def _run(self, audio, sample_rate, output_sample_rate, chunk_kwargs):
        p = self._prepare(audio, sample_rate)
        y = self._separate(p.a, chunk_kwargs)
        return self._finish(y, p, sample_rate, output_sample_rate)

    def isolate_many(
        self,
        clips: abc.Sequence[np.ndarray | torch.Tensor],
        sample_rate: int,
        output_sample_rate: int | None = None,
        chunk_seconds: float | None = None,
        overlap_seconds: float | None = None,
    ) -> list[np.ndarray | torch.Tensor]:
        """Isolates several clips, one pass per group of equal lengths.

        Offline nothing is padded, so each result matches ``isolate`` up to the
        rounding of batched kernels. In streaming mode clips of similar length
        stream in lockstep, padded with silence at their ends, which a causal
        stream cannot see before it arrives; batching there multiplies
        throughput. Returns the outputs in input order. Every prepared clip
        sits on the device at once, so bound the list by the device's memory.
        """
        kw = _chunk_kwargs(chunk_seconds, overlap_seconds,
                           self.model.config.hop_length)
        prepared = [self._prepare(c, sample_rate) for c in clips]
        out = [None] * len(clips)
        if self._mode == "streaming":
            order = sorted(range(len(prepared)),
                           key=lambda i: -prepared[i].a.shape[-1])
            batches = [
                order[k:k + STREAM_BATCH]
                for k in range(0, len(order), STREAM_BATCH)
            ]
        else:
            groups: dict[int, list[int]] = {}
            for i, p in enumerate(prepared):
                groups.setdefault(p.a.shape[-1], []).append(i)
            batches = list(groups.values())
        for idx in batches:
            longest = max(prepared[i].a.shape[-1] for i in idx)
            inp = torch.stack(
                [audio_lib.fit_length(prepared[i].a, longest) for i in idx])
            lengths = [prepared[i].a.shape[-1] for i in idx]
            y = self._separate(inp, dict(kw, lengths=lengths))
            for j, i in enumerate(idx):
                out[i] = self._finish(y[j], prepared[i], sample_rate,
                                      output_sample_rate)
        return out

    def isolate(
        self,
        audio: np.ndarray | torch.Tensor,
        sample_rate: int,
        output_sample_rate: int | None = None,
        chunk_seconds: float | None = None,
        overlap_seconds: float | None = None,
    ) -> np.ndarray | torch.Tensor:
        """Isolates the voice, at the input rate unless overridden.

        Offline, long audio runs in overlapping chunks (``chunk_seconds``,
        ``overlap_seconds``); in streaming mode the stream itself runs over the
        clip and the chunks do not apply.
        """
        return self._run(
            audio,
            sample_rate,
            output_sample_rate,
            _chunk_kwargs(chunk_seconds, overlap_seconds,
                          self.model.config.hop_length),
        )

    def stream(self,
               block_frames: int | None = None,
               cache_frames: int | None = None,
               lengths: list[int] | None = None,
               window_frames: int | None = None,
               crossfade_frames: int | None = None) -> streaming.Stream:
        """Starts a streaming session (``mode="streaming"`` only).

        Push mono float samples at ``sample_rate`` in pieces of any size and
        receive the output samples that became final; ``flush`` returns the
        rest. Up to ``block_frames`` frames are computed together, a
        throughput knob: a frame is computed as soon as its analysis window
        is complete, so the block adds no latency. ``cache_frames`` bounds the
        time attention's past (the session's default, else the checkpoint's
        attention window, else its training chunk); a checkpoint with an
        attention window never looks further back than that, and for any
        other a longer memory than the model was trained with is outside its
        training distribution. ``window_frames`` and
        ``crossfade_frames`` bound the attention to windows with a crossfaded
        second lane: None keeps the session's, 0 turns the windows off. A
        batch of clips padded to one length passes ``lengths``, each clip's
        own.
        """
        if self._mode != "streaming":
            raise RuntimeError(
                "stream() needs mode='streaming'; set the isolator's mode"
                " first.")
        window, crossfade = _windows(
            window_frames, crossfade_frames,
            (self.window_frames or 0, self.crossfade_frames or
             STREAM_CROSSFADE_FRAMES))
        return streaming.Stream(self.model,
                                block_frames=block_frames or self.block_frames,
                                cache_frames=cache_frames or self.cache_frames,
                                autocast_dtype=PRECISIONS[self.precision],
                                lengths=lengths,
                                window_frames=window,
                                crossfade_frames=crossfade)


@dataclasses.dataclass
class _Prepared:
    """A clip ready for the model: audio on the device, length and type."""

    a: torch.Tensor
    n_in: int
    as_numpy: bool


def _passes(requested, config, mode: str) -> int:
    """Returns the offline passes: the checkpoint's own count unless one is
    requested, a requested count checked against the checkpoint and the
    mode."""
    if requested is None:
        return config.offline_passes
    if (isinstance(requested, bool) or not isinstance(requested, int) or
            requested < 1):
        raise ValueError("offline_passes must be a whole number of passes, at"
                         f" least 1, got {requested!r}.")
    if requested > 1 and (config.causal or mode == "streaming"):
        raise ValueError(
            "offline_passes above 1 applies to the offline mode of a"
            " two-sided or dual-mode checkpoint.")
    return requested


def _chunk_kwargs(chunk_seconds, overlap_seconds, min_samples: int) -> dict:
    """Returns the window settings to pass on, checked before any audio."""
    kw = {}
    if chunk_seconds is not None:
        kw["chunk_seconds"] = chunk_seconds
    if overlap_seconds is not None:
        kw["overlap_seconds"] = overlap_seconds
    modeling.window_samples(
        kw.get("chunk_seconds", 1.0),
        kw.get("overlap_seconds", 0.0),
        audio_lib.SR,
        min_samples,
    )
    return {k: float(v) for k, v in kw.items()}
