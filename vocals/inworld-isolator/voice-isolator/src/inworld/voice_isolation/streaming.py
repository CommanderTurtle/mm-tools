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
"""Streaming inference: the isolated voice leaves as the audio arrives.

Samples are pushed in pieces of any size. Every block of new frames runs
once with the state carried (time attention over a sliding cache of past
keys, the time Mamba's recurrence, the STFT's overlap-add), and output
samples are released as soon as every frame that touches them has been
synthesized. The state is never reset within a session: only ``flush`` ends
it. Several clips stream in lockstep as a batch, each with its own state:
padded with silence to one length and given their own lengths, each ends
exactly as it would alone, since nothing looks ahead of the look-ahead. The
latency is the STFT's window plus the look-ahead frames. Partial blocks run
immediately; the maximum block size controls throughput, not latency.

With ``window_frames`` and ``crossfade_frames`` set, the time attention never
looks back across a window boundary: a second lane starts before each
boundary with the Mamba states cloned and the attention empty, the two
decoder outputs blend over the crossfade, and the new lane carries on. The
recurrence is kept throughout; only the attention's past is bounded.
"""

import contextlib
import weakref

import numpy as np
import torch

from inworld.voice_isolation import model as modeling

# Every stream of a model reads the same packed band weights (about 100 MB on
# the device for the released realtime model), so they are packed once per
# model rather than once per session.
_PACKED_BANDS = weakref.WeakKeyDictionary()
# A lane's attention buffers start with this many slots (frames) and double.
_FIRST_SLOTS = 64


class _AttentionState:
    """One lane's time attention past for one attention block.

    Keys and values sit in buffers [rows, heads, capacity, head_dim] written
    in place, and a key is rotated once, when it is written, at its slot: a
    slot's index is its rotary position. ``end`` counts the slots in use.
    """

    def __init__(self):
        self.keys = None
        self.values = None
        self.end = 0


class _AttentionCache:
    """What every attention block of a stream shares: the past each block
    may see (``frames``), the buffers' capacity, the rotary tables of every
    slot and the masks by block and past size.

    A lane that never outgrows ``frames`` plus a block (a windowed lane
    within its window and crossfade) keeps every key in the slot it was
    written to, so its rotations are those of positions counted from the
    lane's first frame, as the cache's own positions were. Beyond that, the
    buffer is compacted when full: the last ``frames`` keys move to the
    front and are rotated back by the distance they moved.
    """

    def __init__(self, frames: int, capacity: int, head_dim: int, window: int,
                 device) -> None:
        self.frames, self.capacity, self.window = frames, capacity, window
        self.cos, self.sin = modeling.rope_tables(capacity, head_dim, device,
                                                  torch.float32)
        self.masks = {}

    def tables(self, start: int, stop: int,
               dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotary tables of slots [start, stop) in ``dtype``."""
        return (self.cos[start:stop].to(dtype), self.sin[start:stop].to(dtype))

    def mask(self, frames: int, past: int) -> torch.Tensor:
        """Which of ``past`` earlier and ``frames`` new keys each new frame
        sees: itself and what precedes it, within the trained window."""
        key = (frames, past)
        if key not in self.masks:
            device = self.cos.device
            positions = torch.arange(past + frames, device=device)
            queries = (past + torch.arange(frames, device=device))[:, None]
            mask = positions[None, :] <= queries
            if self.window:
                # The training bound: a frame sees itself and window - 1
                # before it, whatever the block holds and the cache keeps.
                mask &= positions[None, :] > queries - self.window
            self.masks[key] = mask
        return self.masks[key]


class _MambaState:

    def __init__(self):
        self.carried = None


def _clone_states(states: list) -> list:
    """Lane B's start: the Mamba states cloned (the recurrence does not
    drift), the attention caches empty."""
    out = []
    for state in states:
        if isinstance(state, _MambaState):
            fresh = _MambaState()
            if state.carried is not None:
                fresh.carried = tuple(part.clone() for part in state.carried)
            out.append(fresh)
        else:
            out.append(_AttentionState())
    return out


def _compact(state: _AttentionState, cache: _AttentionCache) -> None:
    """Moves the last ``cache.frames`` keys and values to the front, the
    keys rotated back by the distance they moved."""
    keep = min(state.end, cache.frames)
    shift = state.end - keep
    if not shift:
        return
    cos, sin = cache.tables(shift, shift + 1, state.keys.dtype)
    kept = state.keys[:, :, shift:state.end].clone()
    state.keys[:, :, :keep] = modeling.apply_rope(kept, cos, -sin)
    state.values[:, :, :keep] = state.values[:, :, shift:state.end].clone()
    state.end = keep


def _reserve(state: _AttentionState, slots: int, capacity: int,
             key: torch.Tensor, value: torch.Tensor) -> None:
    """Makes a lane's buffers hold at least ``slots`` slots.

    Buffers start at ``_FIRST_SLOTS`` and double, up to ``capacity``, as the
    lane's past grows, so a short stream, or a batch of them, holds only what
    it uses. Used slots keep their index, which is their rotary position.
    """
    held = 0 if state.keys is None else state.keys.shape[2]
    if slots <= held:
        return
    rows, _, heads, size = key.shape
    shape = (rows, heads, min(capacity, max(slots, 2 * held,
                                            _FIRST_SLOTS)), size)
    keys, values = key.new_empty(shape), value.new_empty(shape)
    if state.end:
        keys[:, :, :state.end] = state.keys[:, :, :state.end]
        values[:, :, :state.end] = state.values[:, :, :state.end]
    state.keys, state.values = keys, values


def _attend_lane(state: _AttentionState, cache: _AttentionCache,
                 query: torch.Tensor, key: torch.Tensor,
                 value: torch.Tensor) -> torch.Tensor:
    """One lane's attention for its new frames [rows, frames, heads, dim]:
    writes the new keys and values, returns [rows, heads, frames, dim]."""
    frames = key.shape[1]
    if state.end + frames > cache.capacity:
        _compact(state, cache)
    _reserve(state, state.end + frames, cache.capacity, key, value)
    first = state.end
    cos, sin = cache.tables(first, first + frames, query.dtype)
    state.keys[:, :, first:first + frames] = modeling.apply_rope(
        key.transpose(1, 2), cos, sin)
    state.values[:, :, first:first + frames] = value.transpose(1, 2)
    state.end = first + frames
    past = min(first, cache.frames)
    return torch.nn.functional.scaled_dot_product_attention(
        modeling.apply_rope(query.transpose(1, 2), cos, sin),
        state.keys[:, :, first - past:state.end],
        state.values[:, :, first - past:state.end],
        attn_mask=cache.mask(frames, past))


def _attend(block: modeling.DualPathAttention, features: torch.Tensor,
            states: list, cache: _AttentionCache) -> torch.Tensor:
    """One attention block over the new frames [batch, bands, frames, width].

    ``features`` stacks one batch per lane in ``states``. Each new frame sees
    the last ``cache.frames`` keys of its own lane before the block and the
    block's keys up to itself; a checkpoint's attention window bounds each
    query exactly as in training. Rotary positions are relative, so a key
    rotated at its slot needs no rotating again. Only the attention itself
    runs lane by lane; every layer around it runs once for all of them.
    """
    time_block = block.time_transformer
    attn = time_block.attn
    batch, bands, frames, width = features.shape
    rows = features.reshape(batch * bands, frames, width)
    qkv = attn.qkv(time_block.norm1(rows)).view(batch * bands, frames, 3,
                                                attn.num_heads, attn.head_dim)
    query, key, value = qkv.unbind(2)
    if len(states) == 1:
        output = _attend_lane(states[0], cache, query, key, value)
    else:
        output = torch.cat([
            _attend_lane(state, cache, *parts) for state, parts in zip(
                states, zip(
                    *(t.chunk(len(states)) for t in (query, key, value))))
        ])
    if attn.exclusive:
        output = modeling.exclusive(output, value.transpose(1, 2))
    rows = rows + attn.proj(
        output.transpose(1, 2).reshape(batch * bands, frames, width))
    rows = rows + time_block.ffn(time_block.norm2(rows))
    banded = block.band_transformer(
        rows.view(batch, bands, frames,
                  width).permute(0, 2, 1, 3).reshape(batch * frames, bands,
                                                     width))
    return banded.view(batch, frames, bands, width).permute(0, 2, 1, 3)


class _PackedBands:
    """The per-band encoders and decoder heads as a few batched operations.

    The model runs one small layer per band, 46 of them, for each of the
    encoder and the decoder; in a stream's block of a few frames each is a
    separate GPU dispatch. Here the bands' weights are stacked, zero-padded
    to the widest band, so each step is one batched operation over every
    band, and the valid outputs are gathered back in the bands' order. The
    results equal the modules' to float32 rounding.
    """

    def __init__(self, model: modeling.VoiceIsolatorNetwork) -> None:
        edges = model.edges
        widths = [end - start for start, end in edges]
        inputs, outputs = 2 * max(widths), 4 * max(widths)
        bins = edges[-1][1]
        device = model.window.device
        self.bands = len(edges)
        gather = torch.full((self.bands, inputs), 2 * bins, dtype=torch.long)
        valid = torch.zeros(self.bands, inputs)
        keep = []
        for band, (start, end) in enumerate(edges):
            gather[band, :2 * (end - start)] = torch.arange(2 * start, 2 * end)
            valid[band, :2 * (end - start)] = 1.0
            keep.append(band * outputs + torch.arange(4 * (end - start)))
        self.gather = gather.to(device)
        self.valid = valid.to(device)
        self.counts = torch.tensor([2.0 * w for w in widths],
                                   device=device)[:, None]
        self.keep = torch.cat(keep).to(device)

        def stack(tensors, size):
            padded = torch.zeros(self.bands,
                                 size,
                                 *tensors[0].shape[1:],
                                 device=device)
            for band, tensor in enumerate(tensors):
                padded[band, :tensor.shape[0]] = tensor.detach()
            return padded

        encoders = [(encoder[0], encoder[1]) for encoder in model.encoders]
        self.encoder_eps = encoders[0][0].eps
        self.encoder_norm_weight = stack([n.weight for n, _ in encoders],
                                         inputs)
        self.encoder_norm_bias = stack([n.bias for n, _ in encoders], inputs)
        self.encoder_weight = stack([l.weight.t() for _, l in encoders], inputs)
        self.encoder_bias = torch.stack([l.bias.detach() for _, l in encoders
                                        ])[:, None]
        heads = model.denoise_decoder.heads
        self.decoder = model.denoise_decoder
        self.head_norm = heads[0][0]
        self.head_norm_weight = torch.stack(
            [h[0].weight.detach() for h in heads])[:, None]
        self.head_norm_bias = torch.stack([h[0].bias.detach() for h in heads
                                          ])[:, None]
        self.head_weight = torch.stack(
            [h[1].weight.detach().t() for h in heads])
        self.head_bias = torch.stack([h[1].bias.detach() for h in heads])[:,
                                                                          None]
        self.out_weight = torch.stack([
            torch.nn.functional.pad(h[3].weight.detach().t(),
                                    (0, outputs - h[3].weight.shape[0]))
            for h in heads
        ])
        self.out_bias = torch.stack([
            torch.nn.functional.pad(h[3].bias.detach(),
                                    (0, outputs - h[3].bias.shape[0]))
            for h in heads
        ])[:, None]

    def encode(self, model: modeling.VoiceIsolatorNetwork,
               spectrum: torch.Tensor) -> torch.Tensor:
        """``model.encode``: band tokens [batch, bands, frames, width]."""
        batch, bins, frames = spectrum.shape
        components = torch.view_as_real(spectrum).permute(0, 2, 1, 3).reshape(
            batch * frames, 2 * bins)
        components = torch.nn.functional.pad(components, (0, 1))
        x = components[:, self.gather].transpose(0, 1)
        mean = x.sum(-1, keepdim=True) / self.counts[:, None]
        centered = (x - mean) * self.valid[:, None]
        variance = (centered * centered).sum(
            -1, keepdim=True) / self.counts[:, None]
        x = centered * torch.rsqrt(variance + self.encoder_eps)
        x = x * self.encoder_norm_weight[:, None] + self.encoder_norm_bias[:,
                                                                           None]
        x = torch.baddbmm(self.encoder_bias, x, self.encoder_weight)
        features = x.view(self.bands, batch, frames, -1).transpose(0, 1)
        features = features + model.band_embedding[None, :, None]
        if model.config.lookahead_frames_choices:
            features = features + model.lookahead_embedding(
                model.lookahead_index)[None, None, None]
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """``model.denoise_decoder``: [batch, bins, frames, 4] in float32."""
        batch, bands, frames, width = features.shape
        scale, shift = self.decoder.condition(self.decoder.controls).chunk(
            2, dim=-1)
        x = (features * (1 + scale) + shift).transpose(0, 1).reshape(
            bands, batch * frames, width)
        x = torch.nn.functional.layer_norm(x, (width,), None, None,
                                           self.head_norm.eps)
        x = x * self.head_norm_weight + self.head_norm_bias
        x = torch.nn.functional.gelu(
            torch.baddbmm(self.head_bias, x, self.head_weight))
        x = torch.baddbmm(self.out_bias, x, self.out_weight)
        x = x.transpose(0, 1).reshape(batch * frames, -1)[:, self.keep]
        return x.float().view(batch, frames, -1, 4).permute(0, 2, 1, 3)


def _packed_bands(model: modeling.VoiceIsolatorNetwork) -> _PackedBands:
    """The model's band layers packed once and shared by all its streams.

    The packed copy is rebuilt when a band weight is replaced, moves to
    another device or dtype, or changes in place (``load_state_dict`` copies
    into the existing tensors, which only their version counter records).
    """
    key = []
    for module in (*model.encoders, *model.denoise_decoder.heads):
        for tensor in module.parameters():
            if tensor.is_inference():
                # Weights made under torch.inference_mode() keep no version
                # counter, so a change in place could go unseen: pack them
                # for this stream alone.
                return _PackedBands(model)
            # pylint: disable-next=protected-access
            version = tensor._version
            key.append(
                (tensor.data_ptr(), version, tensor.device, tensor.dtype))
    key = tuple(key)
    cached = _PACKED_BANDS.get(model)
    if cached is None or cached[0] != key:
        cached = (key, _PackedBands(model))
        _PACKED_BANDS[model] = cached
    return cached[1]


def _time_mamba(mamba: modeling.BidirectionalMamba, inputs: torch.Tensor,
                states: list) -> torch.Tensor:
    """The time Mamba at its streaming position: the forward scan carried,
    each lane's state for its own rows of ``inputs``."""
    if len(states) == 1:
        carried = states[0].carried
    elif states[0].carried is None:
        carried = None
    else:
        carried = tuple(
            torch.cat(parts)
            for parts in zip(*(state.carried for state in states)))
    forward, carried = mamba.forward_blocks(inputs, carried)
    if len(states) == 1:
        states[0].carried = carried
    else:
        for state, parts in zip(
                states, zip(*(part.chunk(len(states)) for part in carried))):
            state.carried = parts
    if mamba.causal:
        return mamba.norm(mamba.output_proj(forward + inputs))
    return mamba.norm(
        mamba.output_proj(
            torch.cat((forward + inputs, torch.zeros_like(inputs)), dim=-1)))


def _scan(block: modeling.DualPathMamba, features: torch.Tensor,
          states: list) -> torch.Tensor:
    """One dual-path Mamba block over the new frames of every lane's batch."""
    batch, bands, frames, width = features.shape
    rows = features.reshape(batch * bands, frames, width)
    time = rows + _time_mamba(block.time_mamba, rows, states)
    frequency = time.view(batch, bands, frames, width).permute(0, 2, 1, 3)
    frequency = frequency.reshape(batch * frames, bands, width)
    frequency = frequency + block.freq_mamba(frequency)
    return frequency.view(batch, frames, bands, width).permute(0, 2, 1, 3)


class Stream:
    """Runs a model at its streaming position over audio in pieces.

    ``push`` takes new mono samples at the model's rate, ``[samples]`` or a
    batch ``[clips, samples]`` of clips streamed in lockstep, and returns the
    output samples that became final; ``flush`` ends the stream and returns
    the rest. Up to ``block_frames`` frames are computed together, a
    throughput knob: a frame is computed as soon as its analysis window is
    complete, so the block adds no latency; ``cache_frames`` bounds the time
    attention's past (by
    default the checkpoint's attention window, else its training chunk). The
    cache is trimmed after each block and a block ends where a push ends, so
    once a session's past exceeds the cache a frame's context, and with it the
    plain stream's output, varies slightly with how the audio was pushed;
    within the cache it does not. A checkpoint trained with an attention
    window keeps that exact bound in the stream (each frame sees itself and
    the ``window - 1`` frames before it), so with a cache of at least
    ``window - 1`` frames its output equals the whole-clip forward and does
    not depend on the pushes at all. A batch of clips padded with silence to
    one length passes ``lengths``, each clip's own, so every clip ends as it
    would alone. A NumPy array in gives a NumPy array out; a tensor gives a CPU
    tensor, of the rank that was pushed. The band encoders and decoder heads
    are packed when the stream starts (shared with the model's other
    streams): weights loaded into the model during a session apply from the
    next one.

    Windowed attention (``window_frames`` and ``crossfade_frames``, both None
    by default: the plain stream, bit-exact). The time attention never looks
    back across a window boundary, which removes the drift of the cached past
    on long inputs while the recurrent state is kept. The semantics:

    1. Lanes and blocks. Lane A carries the output. Frame positions count
       from the session's first analysis frame; the first crossfade starts at
       frame ``window_frames - crossfade_frames`` and the first boundary is
       frame ``window_frames``. Blocks are cut where a crossfade starts and
       where a window ends, so no block crosses either frame and, with a
       cache that never trims (point 4), the output does not depend on how
       the audio was pushed. Lane B starts with the
       block that begins at the crossfade's first frame: its Mamba states are
       clones of lane A's after the frame before, its attention caches start
       empty.
    2. What is blended. The raw decoder output per frame (complex mask delta
       and bounded residual), linearly: lane B's weight is
       ``(f - crossfade_start + 1) / crossfade_frames`` for the frames of the
       crossfade and lane A's the rest. The correction (the mask's tanh and
       the residual's bound) is applied to the blend, so this is a blend of
       the two lanes' decoder outputs, close to but not exactly a blend of
       their output spectra; synthesis then overlap-adds the corrected frames
       with the single envelope.
    3. The swap. Frame ``k * window_frames - 1`` is the last frame of window
       ``k`` (lane B's weight is 1 there); after it, lane B becomes lane A and
       the old lane is dropped, so every frame from the boundary on is the new
       lane's; ``boundaries`` records ``k * window_frames``. The next
       crossfade starts ``window_frames - crossfade_frames`` frames into the
       new window. An input that ends inside a crossfade flushes the blended
       frames computed so far; no swap is forced.
    4. ``cache_frames`` bounds the attention cache within each lane; a lane's
       context never exceeds ``window_frames + crossfade_frames`` frames, so
       a cache at or above that is a no-op and a smaller one bounds further,
       with the plain stream's dependence on the pushes.
    """

    def __init__(
        self,
        model: modeling.VoiceIsolatorNetwork,
        block_frames: int = 4,
        cache_frames: int | None = None,
        autocast_dtype: torch.dtype | None = None,
        lengths: list[int] | None = None,
        window_frames: int | None = None,
        crossfade_frames: int | None = None,
    ) -> None:
        if model.offline:
            raise ValueError(
                "Streaming needs the model at a streaming position: call"
                " set_lookahead first (or load with mode='streaming').")
        if block_frames < 1:
            raise ValueError("Process at least one frame per block.")
        config = model.config
        self.model = model
        self.block_frames = block_frames
        self.cache_frames = (cache_frames or config.attention_window_frames or
                             max(
                                 1,
                                 round(config.chunk_seconds *
                                       config.sample_rate / config.hop_length)))
        self.lookahead = model.lookahead_frames
        self.n_fft, self.hop = config.n_fft, config.hop_length
        self.sample_rate = config.sample_rate
        self.autocast_dtype = autocast_dtype
        self.states = [
            _AttentionState()
            if isinstance(block, modeling.DualPathAttention) else _MambaState()
            for block in model.blocks
        ]
        if (window_frames is None) != (crossfade_frames is None):
            raise ValueError("window_frames and crossfade_frames go together.")
        if window_frames is not None and not (0 < crossfade_frames <
                                              window_frames):
            raise ValueError("crossfade_frames must be positive and shorter"
                             " than window_frames.")
        self.window_frames, self.crossfade_frames = (window_frames,
                                                     crossfade_frames)
        self.states_b = None
        self._spare = [None] * len(self.states)
        self.boundaries = []
        device = model.window.device
        # A windowed lane lasts its window and the crossfade before it; one
        # that never outgrows the cache plus a block needs no compacting.
        lane = (None if window_frames is None else window_frames +
                crossfade_frames)
        if lane is not None and lane <= self.cache_frames + block_frames:
            capacity = lane
        else:
            capacity = self.cache_frames + max(self.cache_frames // 2,
                                               block_frames)
        self._bands = _packed_bands(model)
        self._attention = _AttentionCache(self.cache_frames, capacity,
                                          config.width // config.num_heads,
                                          config.attention_window_frames,
                                          device)
        self._rank = None
        self._audio = torch.zeros(1, 0, device=device)
        self._audio_base = 0
        self._frames = 0
        self._emitted = 0
        self._raw = []
        self._spectra = []
        self._pending_base = 0
        self._ola = torch.zeros(1, self.n_fft, device=device)
        self._envelope = torch.zeros(1, self.n_fft, device=device)
        self._ola_base = 0
        self._released = 0
        self._closed = False
        self._limits = (None if lengths is None else
                        [int(n) // self.hop + 1 for n in lengths])

    @property
    def latency_samples(self) -> int:
        """Input samples that must follow an output sample before its release.

        The analysis window of the last frame that touches it and the
        look-ahead frames, to within one hop of frame rounding; the block
        adds nothing, since a frame is computed as soon as its window is
        complete.
        """
        return self.n_fft + self.hop * self.lookahead

    @property
    def latency_seconds(self) -> float:
        """The stream's algorithmic latency in seconds."""
        return self.latency_samples / self.sample_rate

    def _context(self):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self._audio.device.type,
                              dtype=self.autocast_dtype)

    def _analyze(self, first: int, count: int) -> torch.Tensor:
        """Spectra [count, clips, bins] of frames from ``first``, together:
        each frame's window of the audio, zeros where there is none."""
        start = first * self.hop - self.n_fft // 2
        stop = (first + count - 1) * self.hop - self.n_fft // 2 + self.n_fft
        total = self._audio_base + self._audio.shape[1]
        lo, hi = max(start, self._audio_base), min(stop, total)
        if hi > lo:
            audio = self._audio[:, lo - self._audio_base:hi - self._audio_base]
        else:
            lo = hi = start
            audio = self._audio[:, :0]
        if lo > start or hi < stop:
            audio = torch.nn.functional.pad(audio, (lo - start, stop - hi))
        segments = audio.unfold(1, self.n_fft, self.hop)
        return torch.fft.rfft(segments * self.model.window.float()).transpose(
            0, 1)

    def _run_block(self, spectra: torch.Tensor) -> torch.Tensor:
        """The backbone and decoder over spectra [frames, batch, bins]."""
        model = self.model
        with self._context():
            features = self._bands.encode(model, spectra.permute(1, 2, 0))
            film = model.film()
            if self.window_frames is None:
                raw = self._bands.decode(
                    self._backbone(features, [self.states], film))
            else:
                raw = self._windowed(features, film, spectra.shape[0])
        return raw.permute(2, 0, 1, 3)

    def _backbone(self, features: torch.Tensor, lanes: list,
                  film: torch.Tensor | None) -> torch.Tensor:
        """The blocks over the new frames for each lane's states, the lanes
        stacked into one batch: [lanes * batch, bands, frames, width]."""
        if len(lanes) > 1:
            features = features.repeat(len(lanes), 1, 1, 1)
        for index, block in enumerate(self.model.blocks):
            states = [lane[index] for lane in lanes]
            if isinstance(block, modeling.DualPathAttention):
                features = _attend(block, features, states, self._attention)
            else:
                features = _scan(block, features, states)
            if film is not None:
                features = features * (1 + film[index, 0]) + film[index, 1]
        return features

    def _next_cut(self) -> int:
        """The first frame after the current one that a block may not cross
        into: the next crossfade start or window boundary (none without
        windows)."""
        if self.window_frames is None:
            return self._frames + self.block_frames
        start = self._frames - self._frames % self.window_frames
        crossfade_start = start + self.window_frames - self.crossfade_frames
        if self._frames < crossfade_start:
            return crossfade_start
        return start + self.window_frames

    def _windowed(self, features: torch.Tensor, film: torch.Tensor | None,
                  frames: int) -> torch.Tensor:
        """Two lanes across a window boundary (the class docstring's
        semantics): lane B starts at the crossfade's first frame with the
        Mamba states cloned and the attention empty, the decoder outputs blend
        linearly over the crossfade, and lane B carries on from the boundary.
        Blocks never cross a crossfade start or a boundary (``_next_cut``)."""
        first = self._frames % self.window_frames
        crossfade_start = self.window_frames - self.crossfade_frames
        if first + frames > self.window_frames or (first < crossfade_start <
                                                   first + frames):
            raise RuntimeError("a block crossed a crossfade start or a window"
                               " boundary")
        decoder = self._bands.decode
        if first >= crossfade_start and self.states_b is None:
            # cloned before lane A consumes this block: lane B starts from the
            # frame before the crossfade
            self.states_b = _clone_states(self.states)
            # the last lane's attention buffers, rather than new ones: a lane
            # reads only the slots it has written
            for state, spare in zip(self.states_b, self._spare):
                if spare is not None:
                    state.keys, state.values = spare
            self._spare = [None] * len(self.states)
        if first < crossfade_start:
            return decoder(self._backbone(features, [self.states], film))
        # both lanes in one pass: the same features, each lane's own states
        raw, raw_b = decoder(
            self._backbone(features, [self.states, self.states_b],
                           film)).chunk(2)
        offsets = torch.arange(first, first + frames, device=features.device)
        weight = ((offsets - crossfade_start + 1).float() /
                  self.crossfade_frames).clamp(0.0, 1.0)
        weight = weight.view(1, 1, -1, 1).to(raw.dtype)
        raw = raw * (1 - weight) + raw_b * weight
        if first + frames == self.window_frames:
            self.boundaries.append(self._frames + frames)
            self._spare = [
                (state.keys,
                 state.values) if isinstance(state, _AttentionState) and
                state.keys is not None else None for state in self.states
            ]
            self.states, self.states_b = self.states_b, None
        return raw

    def _synthesize(self) -> None:
        """Synthesizes the next frame from the decoder's output read
        ``lookahead`` frames later; the last computed frame serves every frame
        beyond it, and a clip's own last frame once the clip has ended."""
        index = self._emitted
        rows = []
        for clip in range(self._audio.shape[0]):
            last = (self._frames if self._limits is None else min(
                self._frames, self._limits[clip])) - 1
            source = min(index +
                         self.lookahead, last) if index <= last else index
            rows.append(self._raw[source - self._pending_base][clip])
        raw = torch.stack(rows)
        spectrum = self._spectra[index - self._pending_base]
        delta = self.model.correction(raw, spectrum)
        frame = torch.fft.irfft(delta, n=self.n_fft) * self.model.window.float()
        active = torch.ones(self._audio.shape[0], 1, device=frame.device)
        if self._limits is not None:
            active = torch.tensor(
                [[float(index < limit)] for limit in self._limits],
                device=frame.device)
        start = index * self.hop - self._ola_base
        needed = start + self.n_fft
        if needed > self._ola.shape[1]:
            grow = needed - self._ola.shape[1]
            self._ola = torch.cat(
                (self._ola, self._ola.new_zeros(self._ola.shape[0], grow)),
                dim=1)
            self._envelope = torch.cat(
                (self._envelope,
                 self._envelope.new_zeros(self._envelope.shape[0], grow)),
                dim=1)
        self._ola[:, start:needed] += frame * active
        self._envelope[:,
                       start:needed] += (self.model.window.float()**2 * active)
        self._emitted += 1

    def _advance(self, limit: int) -> None:
        """Computes every frame below ``limit`` in blocks, then synthesizes."""
        while self._frames < limit:
            count = min(self.block_frames, limit - self._frames,
                        self._next_cut() - self._frames)
            spectra = self._analyze(self._frames, count)
            raw = self._run_block(spectra)
            self._raw.extend(raw.unbind(0))
            self._spectra.extend(spectra.unbind(0))
            self._frames += count
        if self._limits is None:
            self._synthesize_ready()
        while self._emitted + self.lookahead < self._frames:
            self._synthesize()
        self._trim()

    def _synthesize_ready(self) -> None:
        """Synthesizes, together, every frame whose look-ahead frame has been
        computed, as ``_synthesize`` would one by one: the correction and the
        inverse transforms run once, the overlap-add frame by frame in order.
        """
        count = self._frames - self.lookahead - self._emitted
        if count <= 0:
            return
        first = self._emitted - self._pending_base
        raw = torch.stack(self._raw[first + self.lookahead:first +
                                    self.lookahead + count])
        spectrum = torch.stack(self._spectra[first:first + count])
        window = self.model.window.float()
        frames = torch.fft.irfft(self.model.correction(raw, spectrum),
                                 n=self.n_fft) * window
        start = self._emitted * self.hop - self._ola_base
        needed = start + (count - 1) * self.hop + self.n_fft
        if needed > self._ola.shape[1]:
            grow = needed - self._ola.shape[1]
            self._ola = torch.nn.functional.pad(self._ola, (0, grow))
            self._envelope = torch.nn.functional.pad(self._envelope, (0, grow))
        squared = window**2
        for k in range(count):
            at = start + k * self.hop
            self._ola[:, at:at + self.n_fft] += frames[k]
            self._envelope[:, at:at + self.n_fft] += squared
        self._emitted += count

    def _trim(self) -> None:
        """Drops what no frame still needs."""
        drop = self._emitted - self._pending_base
        self._raw, self._spectra = self._raw[drop:], self._spectra[drop:]
        self._pending_base = self._emitted
        keep_from = self._released - self._ola_base
        if keep_from > 0:
            self._ola = self._ola[:, keep_from:]
            self._envelope = self._envelope[:, keep_from:]
            self._ola_base = self._released
        needed_from = min(self._released,
                          max(0, self._frames * self.hop - self.n_fft // 2))
        drop_audio = needed_from - self._audio_base
        if drop_audio > 0:
            self._audio = self._audio[:, drop_audio:]
            self._audio_base = needed_from

    def _release(self, upto: int) -> torch.Tensor:
        """Output samples [released, upto): input plus synthesized change."""
        total = self._audio_base + self._audio.shape[1]
        start, stop = self._released, min(upto, total)
        if stop <= start:
            return self._audio.new_zeros(self._audio.shape[0], 0)
        offset = self.n_fft // 2 - self._ola_base
        envelope = self._envelope[:, start + offset:stop + offset]
        delta = self._ola[:, start + offset:stop +
                          offset] / envelope.clamp(min=1e-11)
        self._released = stop
        return self._audio[:, start - self._audio_base:stop -
                           self._audio_base] + delta

    def _accept(self, audio) -> tuple[torch.Tensor, bool]:
        as_numpy = isinstance(audio, np.ndarray)
        samples = torch.as_tensor(audio)
        if samples.ndim not in (1, 2) or not samples.is_floating_point():
            raise ValueError(
                "Push mono float samples [samples] or a batch [clips, samples]."
            )
        rank = samples.ndim
        if self._rank is None:
            self._rank = rank
            clips = samples.shape[0] if rank == 2 else 1
            if self._limits is not None and len(self._limits) != clips:
                raise ValueError("lengths names every clip of the batch.")
            if rank == 2:
                self._audio = self._audio.new_zeros(clips, 0)
                self._ola = self._ola.new_zeros(clips, self.n_fft)
                self._envelope = self._envelope.new_zeros(clips, self.n_fft)
        if rank != self._rank or (rank == 2 and
                                  samples.shape[0] != self._audio.shape[0]):
            raise ValueError(
                "Every push of a session has the same rank and batch.")
        samples = samples.detach().to(self._audio.device, torch.float32)
        return samples if rank == 2 else samples[None], as_numpy

    def _deliver(self, released: torch.Tensor, as_numpy: bool):
        released = released.detach().cpu()
        if self._rank != 2:
            released = released[0]
        return released.numpy() if as_numpy else released

    @torch.inference_mode()
    def push(self,
             audio: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Feeds new samples; returns the output samples that became final."""
        if self._closed:
            raise RuntimeError("The stream was flushed.")
        samples, as_numpy = self._accept(audio)
        self._audio = torch.cat((self._audio, samples), dim=1)
        total = self._audio_base + self._audio.shape[1]
        computable = (total - self.n_fft // 2) // self.hop + 1
        self._advance(max(0, computable))
        released = self._release(self._emitted * self.hop - self.n_fft // 2)
        self._trim()
        return self._deliver(released, as_numpy)

    @torch.inference_mode()
    def flush(self, as_numpy: bool = False) -> np.ndarray | torch.Tensor:
        """Ends the stream and returns every remaining output sample."""
        if self._closed:
            return self._deliver(self._audio.new_zeros(self._audio.shape[0], 0),
                                 as_numpy)
        self._closed = True
        total = self._audio_base + self._audio.shape[1]
        self._advance(total // self.hop + 1)
        while self._emitted < self._frames:
            self._synthesize()
        return self._deliver(self._release(total), as_numpy)
