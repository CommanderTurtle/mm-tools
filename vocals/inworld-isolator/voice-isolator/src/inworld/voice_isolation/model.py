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
"""The isolator: a band-split dual-path network over the 48 kHz STFT.

Each frame's spectrum is cut into bands, every band becomes one token, and
blocks of time attention (rotary positions, exclusive self attention) and
dual-path Mamba (along time, then across bands) refine the tokens before
per-band heads predict a bounded complex mask and a bounded residual per
bin. A checkpoint is either two-sided (offline: every frame sees its whole
chunk; a dual-mode one also runs causally with the same weights) or causal
only (streaming: attention and the time scan are causal and the decoder's
output for a frame is read a trained number of frames later, the
look-ahead). A causal checkpoint may bound its time attention to a window of
frames (``attention_window_frames``); the stream keeps the same bound.
"""

import dataclasses
import math

import torch

from inworld.voice_isolation import scan

TRANSFORMER = "transformer"
MAMBA = "mamba"
BLOCK_TYPES = (TRANSFORMER, MAMBA)
EXCLUSIVE_AXES = ("both", "time", "band")
DEFAULT_BAND_PLAN = ((100, 1000), (250, 4000), (500, 8000), (1000, 16000),
                     (2000, 24000))
TRAINING_ONLY_FIELDS = frozenset({
    "activation_checkpointing",
    "checkpoint_every",
    "attention_backend",
    "offline_mode_prob",
    "dual_distill",
    "hint_sha256",
    "hint_bf16",
    "grid_width",
    "grid_hidden",
    "grid_kernel",
    "grid_stride",
    "grid_heads",
    "grid_qk_channels",
    "grid_output",
    "grid_normalize_input",
    "decision_head",
})

REQUIRED_VALUES = {
    "representation": "bands",
    "decoder_type": "complex_mask_residual",
    "hint_checkpoint": None,
}


@dataclasses.dataclass(frozen=True)
class VoiceIsolatorConfig:
    """Architecture of a checkpoint; always read from the checkpoint."""

    width: int = 320
    block_types: tuple[str, ...] = (TRANSFORMER, MAMBA, TRANSFORMER, MAMBA,
                                    TRANSFORMER, MAMBA, TRANSFORMER)
    num_heads: int = 8
    n_fft: int = 2048
    hop_length: int = 480
    sample_rate: int = 48000
    band_plan: tuple[tuple[int, int], ...] = DEFAULT_BAND_PLAN
    correction_bound: float = 0.5
    chunk_seconds: float = 12.0
    overlap_seconds: float = 2.0
    exclusive_self_attention: bool = True
    exclusive_axes: str = "time"
    mamba_state: int = 16
    mamba_conv: int = 4
    mamba_expand: int = 2
    causal: bool = False
    lookahead_frames: int = 0
    lookahead_frames_choices: tuple[int, ...] = ()
    dual_mode: bool = False
    mode_film: bool = False
    attention_window_frames: int = 0
    offline_passes: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_types", tuple(self.block_types))
        object.__setattr__(self, "band_plan",
                           tuple(tuple(pair) for pair in self.band_plan))
        object.__setattr__(self, "lookahead_frames_choices",
                           tuple(self.lookahead_frames_choices))
        if self.width < 1 or self.num_heads < 1 or self.width % (
                2 * self.num_heads):
            raise ValueError(
                "width must be a positive multiple of twice num_heads.")
        if not self.block_types or any(
                kind not in BLOCK_TYPES for kind in self.block_types):
            raise ValueError(f"block_types must be drawn from {BLOCK_TYPES}.")
        if self.n_fft < 4 or self.n_fft % 2 or not (0 < self.hop_length <=
                                                    self.n_fft // 2):
            raise ValueError(
                "n_fft must be even and hop_length within (0, n_fft / 2].")
        if self.sample_rate < 1 or not self.band_plan or any(
                width < 1 or upto < 1 for width, upto in self.band_plan):
            raise ValueError("band_plan needs positive widths and limits.")
        if not 0 < self.correction_bound < 1:
            raise ValueError("correction_bound must lie in (0, 1).")
        if (self.chunk_seconds * self.sample_rate < self.n_fft or
                not 0 < self.overlap_seconds <= self.chunk_seconds / 2):
            raise ValueError(
                "chunk_seconds must cover one window and overlap_seconds lie"
                " in (0, chunk_seconds / 2].")
        if self.exclusive_axes not in EXCLUSIVE_AXES:
            raise ValueError(f"exclusive_axes must be one of {EXCLUSIVE_AXES}.")
        if min(self.mamba_state, self.mamba_conv, self.mamba_expand) < 1:
            raise ValueError("Mamba dimensions must be positive.")
        choices = self.lookahead_frames_choices
        if list(choices) != sorted(set(choices)) or (choices and
                                                     choices[0] < 0):
            raise ValueError(
                "lookahead_frames_choices must be distinct non-negative frame"
                " counts in ascending order.")
        if self.lookahead_frames < 0:
            raise ValueError("lookahead_frames cannot be negative.")
        if choices and self.lookahead_frames not in choices:
            raise ValueError(
                "lookahead_frames must be one of lookahead_frames_choices.")
        if (self.lookahead_frames > 0 or
                choices) and not (self.causal or self.dual_mode):
            raise ValueError("A look-ahead needs a causal or dual-mode model.")
        if self.dual_mode and (self.causal or not choices):
            raise ValueError(
                "dual_mode builds a two-sided model switched per forward:"
                " leave causal off and list the streaming positions.")
        if self.mode_film and not choices:
            raise ValueError("mode_film needs lookahead_frames_choices.")
        if self.attention_window_frames < 0 or (self.attention_window_frames and
                                                not self.causal):
            raise ValueError(
                "attention_window_frames needs a causal model and cannot be"
                " negative.")
        if (isinstance(self.offline_passes, bool) or
                not isinstance(self.offline_passes, int) or
                self.offline_passes < 1):
            raise ValueError(
                "offline_passes must be a whole number of passes, at least 1.")
        if self.offline_passes > 1 and self.causal:
            raise ValueError(
                "A causal model streams in one pass: offline_passes above 1"
                " needs a two-sided or dual-mode model.")

    @classmethod
    def from_dict(cls, raw: dict) -> "VoiceIsolatorConfig":
        """Builds the config from a checkpoint's architecture dictionary.

        Training-only fields are ignored; a value this package does not
        implement is refused rather than guessed.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        for name, value in REQUIRED_VALUES.items():
            if raw.get(name, value) != value:
                raise ValueError(
                    f"This package implements {name}={value!r}, not"
                    f" {raw[name]!r}.")
        unknown = sorted(
            name for name, value in raw.items()
            if name not in known and name not in TRAINING_ONLY_FIELDS and
            name not in REQUIRED_VALUES and value not in (0, False, None, "",
                                                          [], {}))
        if unknown:
            raise ValueError(
                "Checkpoint config uses fields this package does not"
                f" implement: {unknown}. Upgrade inworld-voice-isolation.")
        return cls(**{name: raw[name] for name in known if name in raw})

    def to_dict(self) -> dict:
        """The config as plain JSON-ready values."""
        return dataclasses.asdict(self)


def band_edges(config: VoiceIsolatorConfig) -> list[tuple[int, int]]:
    """Bin ranges [start, end) of the bands, covering every bin once."""
    n_bins = config.n_fft // 2 + 1
    hz_per_bin = config.sample_rate / config.n_fft
    edges: list[tuple[int, int]] = []
    start = 0
    for width_hz, upto_hz in config.band_plan:
        width = max(1, round(width_hz / hz_per_bin))
        upto = min(n_bins, round(upto_hz / hz_per_bin))
        while start < upto:
            end = min(start + width, upto)
            edges.append((start, end))
            start = end
    if start < n_bins:
        edges.append((start, n_bins))
    return edges


def rope_tables(frames: int, head_dim: int, device,
                dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine and sine tables [frames, head_dim / 2] of rotary positions."""
    half = head_dim // 2
    inv = 10000.0**(-torch.arange(0, half, device=device, dtype=torch.float32) /
                    half)
    pos = torch.arange(frames, device=device, dtype=torch.float32)
    angles = torch.outer(pos, inv)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> torch.Tensor:
    """Rotates ``x`` [batch, heads, frames, head_dim] by the tables."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos[None, None], sin[None, None]
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)


def exclusive(output: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Removes each token's component along its own value, in float32."""
    o32, v32 = output.float(), value.float()
    coefficient = (o32 * v32).sum(-1, keepdim=True) / (v32 * v32).sum(
        -1, keepdim=True).clamp_min(1e-6)
    return (o32 - coefficient * v32).to(output.dtype)


def bounded_correction(raw: torch.Tensor, reference: torch.Tensor,
                       bound: float) -> torch.Tensor:
    """Bounds each complex correction's magnitude relative to its bin."""
    components = torch.tanh(raw.float())
    norm = torch.linalg.vector_norm(components, dim=-1,
                                    keepdim=True).clamp(min=1)
    return components / norm * (bound * reference.abs()[..., None])


def window_mask(frames: int, window: int, device) -> torch.Tensor:
    """[frames, frames] causal mask that also drops keys ``window`` or more
    frames behind the query: each frame sees itself and the ``window - 1``
    frames before it."""
    positions = torch.arange(frames, device=device)
    lag = positions[:, None] - positions[None, :]
    return (lag >= 0) & (lag < window)


class _TimeAttention(torch.nn.Module):
    """Self attention along time with rotary positions.

    ``window`` > 0 (causal only) bounds every frame's past to that many
    frames, itself included, the attention window the checkpoint was trained
    with; streaming keeps the same bound over its cache.
    """

    def __init__(self,
                 width: int,
                 num_heads: int,
                 exclusive_attention: bool,
                 window: int = 0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.exclusive = exclusive_attention
        self.window = window
        self.qkv = torch.nn.Linear(width, 3 * width)
        self.proj = torch.nn.Linear(width, width)

    def forward(self, x: torch.Tensor, causal: bool) -> torch.Tensor:
        batch, frames, width = x.shape
        qkv = self.qkv(x).view(batch, frames, 3, self.num_heads,
                               self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        cos, sin = rope_tables(frames, self.head_dim, x.device, query.dtype)
        query, key = apply_rope(query, cos, sin), apply_rope(key, cos, sin)
        if causal and 0 < self.window < frames:
            output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=window_mask(frames, self.window, x.device))
        else:
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=causal)
        if self.exclusive:
            output = exclusive(output, value)
        return self.proj(output.transpose(1, 2).reshape(batch, frames, width))


class _TimeBlock(torch.nn.Module):
    """A pre-norm transformer layer along time."""

    def __init__(self, config: VoiceIsolatorConfig, exclusive_attention: bool):
        super().__init__()
        width = config.width
        self.norm1 = torch.nn.LayerNorm(width)
        self.attn = _TimeAttention(width, config.num_heads, exclusive_attention,
                                   config.attention_window_frames)
        self.norm2 = torch.nn.LayerNorm(width)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, 4 * width),
            torch.nn.GELU(),
            torch.nn.Identity(),
            torch.nn.Linear(4 * width, width),
        )

    def forward(self, x: torch.Tensor, causal: bool) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal)
        return x + self.ffn(self.norm2(x))


class _BandTransformer(torch.nn.Module):
    """A pre-norm transformer layer across the bands of one frame."""

    def __init__(self, config: VoiceIsolatorConfig, exclusive_attention: bool):
        super().__init__()
        width = config.width
        self.exclusive = exclusive_attention
        self.self_attn = torch.nn.MultiheadAttention(width,
                                                     config.num_heads,
                                                     batch_first=True)
        self.linear1 = torch.nn.Linear(width, 4 * width)
        self.linear2 = torch.nn.Linear(4 * width, width)
        self.norm1 = torch.nn.LayerNorm(width)
        self.norm2 = torch.nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, bands, width = x.shape
        heads = self.self_attn.num_heads
        qkv = torch.nn.functional.linear(self.norm1(x),
                                         self.self_attn.in_proj_weight,
                                         self.self_attn.in_proj_bias)
        qkv = qkv.view(batch, bands, 3, heads,
                       width // heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value)
        if self.exclusive:
            output = exclusive(output, value)
        x = x + self.self_attn.out_proj(
            output.transpose(1, 2).reshape(batch, bands, width))
        return x + self.linear2(
            torch.nn.functional.gelu(self.linear1(self.norm2(x))))


class DualPathAttention(torch.nn.Module):
    """Attention along time within each band, then across bands."""

    def __init__(self, config: VoiceIsolatorConfig):
        super().__init__()
        on_time = config.exclusive_self_attention and config.exclusive_axes in (
            "both", "time")
        on_band = config.exclusive_self_attention and config.exclusive_axes in (
            "both", "band")
        self.time_transformer = _TimeBlock(config, on_time)
        self.band_transformer = _BandTransformer(config, on_band)

    def forward(self, x: torch.Tensor, causal: bool) -> torch.Tensor:
        batch, bands, frames, width = x.shape
        x = self.time_transformer(x.reshape(batch * bands, frames, width),
                                  causal)
        x = x.reshape(batch, bands, frames, width).permute(0, 2, 1, 3)
        x = self.band_transformer(x.reshape(batch * frames, bands, width))
        return x.reshape(batch, frames, bands, width).permute(0, 2, 1, 3)


class Mamba(torch.nn.Module):
    """One selective state-space layer, with a carried state for streams."""

    def __init__(self, width: int, state: int, convolution: int, expand: int):
        super().__init__()
        inner = expand * width
        self.d_state = state
        self.dt_rank = math.ceil(width / 16)
        self.in_proj = torch.nn.Linear(width, 2 * inner, bias=False)
        self.conv1d = torch.nn.Conv1d(inner,
                                      inner,
                                      convolution,
                                      groups=inner,
                                      padding=convolution - 1)
        self.x_proj = torch.nn.Linear(inner,
                                      self.dt_rank + 2 * state,
                                      bias=False)
        self.dt_proj = torch.nn.Linear(self.dt_rank, inner)
        self.A_log = torch.nn.Parameter(
            torch.log(torch.arange(1, state + 1,
                                   dtype=torch.float32)).repeat(inner, 1))
        self.D = torch.nn.Parameter(torch.ones(inner))
        self.out_proj = torch.nn.Linear(inner, width, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        frames = x.shape[1]
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        history = u.transpose(1, 2)
        if state is not None:
            history = torch.cat((state[0].to(history.dtype), history), dim=-1)
        convolved = torch.nn.functional.conv1d(
            history,
            self.conv1d.weight.to(history.dtype),
            self.conv1d.bias.to(history.dtype),
            padding=0 if state is not None else self.conv1d.padding[0],
            groups=self.conv1d.groups,
        )[..., :frames]
        needed = self.conv1d.kernel_size[0] - 1
        padded = torch.nn.functional.pad(history, (needed, 0))
        tail = padded[..., padded.shape[-1] - needed:]
        u = torch.nn.functional.silu(convolved).transpose(1, 2)
        dt, b, c = torch.split(self.x_proj(u),
                               [self.dt_rank, self.d_state, self.d_state],
                               dim=-1)
        delta = torch.nn.functional.softplus(
            torch.nn.functional.linear(dt, self.dt_proj.weight).float() +
            self.dt_proj.bias.float())
        y, ssm = scan.selective_scan(u, delta, -torch.exp(self.A_log.float()),
                                     b, c, self.D, z,
                                     None if state is None else state[1])
        return self.out_proj(y.to(xz.dtype)), (tail, ssm)


class BidirectionalMamba(torch.nn.Module):
    """Forward and backward scans projected together; forward only if causal."""

    def __init__(self, width: int, state: int, convolution: int, expand: int,
                 causal: bool):
        super().__init__()
        self.causal = causal
        self.forward_blocks = Mamba(width, state, convolution, expand)
        self.backward_blocks = None if causal else Mamba(
            width, state, convolution, expand)
        self.output_proj = torch.nn.Linear(width if causal else 2 * width,
                                           width)
        self.norm = torch.nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        forward, _ = self.forward_blocks(x)
        if self.causal:
            return self.norm(self.output_proj(forward + x))
        if causal:
            return self.norm(
                self.output_proj(
                    torch.cat((forward + x, torch.zeros_like(x)), dim=-1)))
        backward, _ = self.backward_blocks(x.flip(1))
        return self.norm(
            self.output_proj(
                torch.cat((forward + x, backward.flip(1) + x), dim=-1)))


class DualPathMamba(torch.nn.Module):
    """Mamba along time within each band, then across bands."""

    def __init__(self, config: VoiceIsolatorConfig):
        super().__init__()
        arguments = (config.width, config.mamba_state, config.mamba_conv,
                     config.mamba_expand)
        self.time_mamba = BidirectionalMamba(*arguments, causal=config.causal)
        self.freq_mamba = BidirectionalMamba(*arguments, causal=False)

    def forward(self, x: torch.Tensor, causal: bool) -> torch.Tensor:
        batch, bands, frames, width = x.shape
        time = x.reshape(batch * bands, frames, width)
        time = time + self.time_mamba(time, causal)
        frequency = time.reshape(batch, bands, frames,
                                 width).transpose(1, 2).reshape(
                                     batch * frames, bands, width)
        frequency = frequency + self.freq_mamba(frequency)
        return frequency.reshape(batch, frames, bands,
                                 width).transpose(1, 2).contiguous()


class _Decoder(torch.nn.Module):
    """Per-band heads: a complex mask and a residual for every bin."""

    def __init__(self, width: int, edges: list[tuple[int, int]]):
        super().__init__()
        self.edges = edges
        self.condition = torch.nn.Linear(2, 2 * width)
        self.register_buffer("controls",
                             torch.tensor([1.0, 0.0]),
                             persistent=False)
        torch.nn.init.zeros_(self.condition.weight)
        torch.nn.init.zeros_(self.condition.bias)
        self.heads = torch.nn.ModuleList()
        for start, end in edges:
            head = torch.nn.Sequential(
                torch.nn.LayerNorm(width),
                torch.nn.Linear(width, width),
                torch.nn.GELU(),
                torch.nn.Linear(width, 4 * (end - start)),
            )
            torch.nn.init.zeros_(head[-1].weight)
            torch.nn.init.zeros_(head[-1].bias)
            self.heads.append(head)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, _, frames, _ = features.shape
        scale, shift = self.condition(self.controls).chunk(2, dim=-1)
        features = features * (1 + scale) + shift
        return torch.cat([
            head(features[:, index]).float().reshape(batch, frames, end - start,
                                                     4)
            for index, ((start, end),
                        head) in enumerate(zip(self.edges, self.heads))
        ],
                         dim=2).permute(0, 2, 1, 3)


def read_ahead(raw: torch.Tensor, frames: int) -> torch.Tensor:
    """The decoder's output for frame t taken from frame t + ``frames``.

    The last frame serves every frame beyond it.
    """
    total = raw.shape[2]
    index = (torch.arange(total, device=raw.device) + frames).clamp(max=total -
                                                                    1)
    return raw[:, :, index]


def window_samples(
    chunk_seconds: float,
    overlap_seconds: float,
    sample_rate: int,
    min_samples: int = 1,
) -> tuple[int, int]:
    """Returns the window and its crossfade in samples.

    The crossfade is capped at half the window.

    Raises:
        ValueError: A length is not a finite number, the window is shorter than
            ``min_samples``, or the crossfade is negative. Such settings would
            drop or blank audio.
    """
    lengths = {}
    for name, value in (
        ("chunk_seconds", chunk_seconds),
        ("overlap_seconds", overlap_seconds),
    ):
        try:
            lengths[name] = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} must be a number of seconds, not {value!r}") from None
        if not math.isfinite(lengths[name]):
            raise ValueError(f"{name} must be finite, not {value!r}")
    chunk = int(lengths["chunk_seconds"] * sample_rate)
    if chunk < min_samples:
        raise ValueError(f"chunk_seconds={chunk_seconds!r} is shorter than "
                         f"{1000 * min_samples / sample_rate:g} ms")
    if lengths["overlap_seconds"] < 0:
        raise ValueError(
            f"overlap_seconds must not be negative, not {overlap_seconds!r}")
    return chunk, min(int(lengths["overlap_seconds"] * sample_rate), chunk // 2)


class VoiceIsolatorNetwork(torch.nn.Module):
    """The isolator's network over one chunk of audio, in either of its modes.

    ``forward`` takes a batch of equally long mono clips at the model rate
    and returns the isolated voice; ``enhance`` renders clips of any length
    in overlapping chunks. A dual-mode checkpoint starts offline;
    ``set_lookahead`` moves it to one of its trained streaming positions and
    ``set_offline`` back.
    """

    def __init__(self, config: VoiceIsolatorConfig) -> None:
        super().__init__()
        self.config = config
        self.edges = band_edges(config)
        self.encoders = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.LayerNorm(2 * (end - start)),
                torch.nn.Linear(2 * (end - start), config.width))
            for start, end in self.edges
        ])
        self.band_embedding = torch.nn.Parameter(
            torch.randn(len(self.edges), config.width) * 0.02)
        choices = config.lookahead_frames_choices
        if choices:
            positions = len(choices) + (1 if config.dual_mode else 0)
            self.lookahead_embedding = torch.nn.Embedding(
                positions, config.width)
            torch.nn.init.zeros_(self.lookahead_embedding.weight)
            self.register_buffer("lookahead_choices",
                                 torch.tensor(choices),
                                 persistent=False)
            default = (len(choices) if config.dual_mode else choices.index(
                config.lookahead_frames))
            self.register_buffer("lookahead_index",
                                 torch.tensor(default),
                                 persistent=False)
            if config.mode_film:
                self.mode_film = torch.nn.Embedding(
                    positions, 2 * len(config.block_types) * config.width)
                torch.nn.init.zeros_(self.mode_film.weight)
        self.blocks = torch.nn.ModuleList([
            DualPathAttention(config)
            if kind == TRANSFORMER else DualPathMamba(config)
            for kind in config.block_types
        ])
        self.denoise_decoder = _Decoder(config.width, self.edges)
        self.register_buffer("window",
                             torch.hann_window(config.n_fft),
                             persistent=False)

    @property
    def num_params(self) -> int:
        """The number of learned parameters in the network."""
        return sum(p.numel() for p in self.parameters())

    @property
    def offline(self) -> bool:
        """Whether every frame sees its whole chunk now."""
        if self.config.dual_mode:
            return int(self.lookahead_index) == len(
                self.config.lookahead_frames_choices)
        return not self.config.causal

    @property
    def lookahead_frames(self) -> int:
        """The look-ahead in force, zero offline."""
        if not self.config.lookahead_frames_choices:
            return self.config.lookahead_frames
        if self.offline:
            return 0
        return self.config.lookahead_frames_choices[int(self.lookahead_index)]

    def set_lookahead(self, frames: int) -> int:
        """Moves to the trained streaming position with this look-ahead."""
        choices = self.config.lookahead_frames_choices
        if not choices:
            if frames != self.config.lookahead_frames or not self.config.causal:
                raise ValueError(
                    "This checkpoint has no streaming positions to choose from."
                )
            return frames
        if frames not in choices:
            raise ValueError(
                f"Look-ahead {frames} is not among the trained positions"
                f" {choices}.")
        self.lookahead_index.fill_(choices.index(frames))
        return frames

    def set_offline(self) -> None:
        """Returns a dual-mode checkpoint to its two-sided mode."""
        if not self.config.dual_mode:
            raise ValueError(
                "Only a dual-mode checkpoint switches between offline and"
                " streaming.")
        self.lookahead_index.fill_(len(self.config.lookahead_frames_choices))

    def default_window(self) -> tuple[float, float]:
        """The render window and crossfade in seconds: the checkpoint's own."""
        return (self.config.chunk_seconds, self.config.overlap_seconds)

    def stft(self, audio: torch.Tensor) -> torch.Tensor:
        """Complex spectra [batch, bins, frames] of audio [batch, samples]."""
        with torch.autocast(device_type=audio.device.type, enabled=False):
            return torch.stft(
                audio.float(),
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                window=self.window.float(),
                center=True,
                pad_mode="reflect" if audio.shape[-1] > self.config.n_fft //
                2 else "constant",
                return_complex=True,
            )

    def istft(self, spectrum: torch.Tensor, length: int) -> torch.Tensor:
        """Audio [batch, length] of complex spectra [batch, bins, frames]."""
        with torch.autocast(device_type=spectrum.device.type, enabled=False):
            return torch.istft(
                spectrum,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                window=self.window.float(),
                center=True,
                length=length,
            )

    def encode(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Band tokens [batch, bands, frames, width] of the spectra."""
        components = torch.view_as_real(spectrum).permute(0, 2, 1, 3)
        features = torch.stack([
            encoder(components[:, :, start:end].flatten(2))
            for (start, end), encoder in zip(self.edges, self.encoders)
        ],
                               dim=1)
        features = features + self.band_embedding[None, :, None]
        if self.config.lookahead_frames_choices:
            features = features + self.lookahead_embedding(
                self.lookahead_index)[None, None, None]
        return features

    def film(self) -> torch.Tensor | None:
        """Per-block scale and shift [blocks, 2, width] of the position."""
        if not self.config.mode_film:
            return None
        return self.mode_film(self.lookahead_index).view(
            len(self.blocks), 2, self.config.width)

    def features(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Backbone features [batch, bands, frames, width] of the spectra."""
        causal = not self.offline
        features = self.encode(spectrum)
        film = self.film()
        for index, block in enumerate(self.blocks):
            features = block(features, causal)
            if film is not None:
                features = features * (1 + film[index, 0]) + film[index, 1]
        return features

    def correction(self, raw: torch.Tensor,
                   spectrum: torch.Tensor) -> torch.Tensor:
        """The complex change of the spectrum the decoder's output encodes."""
        mask_delta = torch.view_as_complex(
            (2 * torch.tanh(raw[..., :2])).contiguous())
        residual = bounded_correction(raw[..., 2:], spectrum,
                                      self.config.correction_bound)
        return spectrum * mask_delta + torch.view_as_complex(
            residual.contiguous())

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Isolates floating-point mono audio shaped [batch, samples]."""
        if (audio.ndim != 2 or not audio.shape[-1] or
                not audio.is_floating_point()):
            raise ValueError(
                "Expected nonempty floating-point mono audio [batch, samples].")
        spectrum = self.stft(audio)
        raw = self.denoise_decoder(self.features(spectrum))
        if not self.offline and self.lookahead_frames:
            raw = read_ahead(raw, self.lookahead_frames)
        return audio + self.istft(self.correction(raw, spectrum),
                                  audio.shape[-1])

    @torch.inference_mode()
    def enhance(
        self,
        audio: torch.Tensor,
        chunk_seconds: float | None = None,
        overlap_seconds: float | None = None,
    ) -> torch.Tensor:
        """Isolates ``[samples]`` or ``[batch, samples]`` of any length.

        Windows of ``chunk_seconds`` overlap by ``overlap_seconds`` with a
        linear crossfade; a clip no longer than one window is a single pass.
        Every window of a longer clip is a full one: the last ends at the
        clip's end, reaching back over the window before it, and fades in
        where that window fades out, so the tail keeps a whole window of
        context.
        """
        default_chunk, default_overlap = self.default_window()
        chunk, overlap = window_samples(
            default_chunk if chunk_seconds is None else chunk_seconds,
            default_overlap if overlap_seconds is None else overlap_seconds,
            self.config.sample_rate,
            self.config.hop_length,
        )
        squeeze = audio.ndim == 1
        if squeeze:
            audio = audio[None]
        n = audio.shape[-1]
        if n <= chunk:
            out = self(audio)
        else:
            out = audio.new_zeros(audio.shape[0], n, dtype=torch.float32)
            weight = audio.new_zeros(n, dtype=torch.float32)
            starts = list(range(0, n - chunk, chunk - overlap)) + [n - chunk]
            for index, start in enumerate(starts):
                end = start + chunk
                w = torch.ones(chunk, device=audio.device)
                if index > 0:
                    fade_in = starts[index - 1] + chunk - overlap - start
                    w[:fade_in] = 0.0
                    if overlap > 0:
                        w[fade_in:fade_in + overlap] = torch.linspace(
                            0.0, 1.0, overlap, device=audio.device)
                if overlap > 0 and end < n:
                    w[-overlap:] = torch.linspace(1.0,
                                                  0.0,
                                                  overlap,
                                                  device=audio.device)
                out[:, start:end] += self(audio[:, start:end]) * w
                weight[start:end] += w
            out = out / weight.clamp(min=1e-6)
        out = out.float()
        return out[0] if squeeze else out
