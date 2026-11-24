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
"""The selective state-space scan behind every Mamba block, in plain torch.

The recurrence ``h_t = exp(delta_t a) h_{t-1} + delta_t b_t u_t`` and its
readout ``y_t = c_t . h_t + d u_t`` run frame by frame in float32, the
per-frame factors prepared a block of frames at a time. The same function
serves whole clips and a stream's few new frames, which pass the state they
carry. When the ``mamba_ssm`` kernels are installed and the audio is on a
GPU, a whole clip's scan runs through them instead; the result is the same to
float32 rounding. On Apple's GPU (``mps``) every scan runs as one Metal
kernel compiled on first use: each thread keeps one channel's state in
registers, where the plain recurrence would write every frame's state to
memory and spend several GPU operations per frame.
"""

import torch

BLOCK = 64
METAL_STATES = 16
_KERNEL: dict = {}
_METAL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void selective_scan(
    device float* y [[buffer(0)]],
    device float* last [[buffer(1)]],
    device const float* u [[buffer(2)]],
    device const float* delta [[buffer(3)]],
    device const float* a [[buffer(4)]],
    device const float* b [[buffer(5)]],
    device const float* c [[buffer(6)]],
    device const float* d [[buffer(7)]],
    device const float* z [[buffer(8)]],
    device const float* carried [[buffer(9)]],
    constant uint& batch [[buffer(10)]],
    constant uint& frames [[buffer(11)]],
    constant uint& channels [[buffer(12)]],
    constant uint& states [[buffer(13)]],
    constant uint& gated [[buffer(14)]],
    constant uint& carrying [[buffer(15)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= batch * channels) {
    return;
  }
  const uint row = index / channels;
  const uint channel = index % channels;
  float h[16];
  float rate[16];
  for (uint n = 0; n < states; ++n) {
    rate[n] = a[channel * states + n];
    h[n] = carrying ? carried[index * states + n] : 0.0f;
  }
  const float skip = d[channel];
  for (uint t = 0; t < frames; ++t) {
    const uint at = (row * frames + t) * channels + channel;
    const uint bt = (row * frames + t) * states;
    const float step = delta[at];
    const float x = u[at];
    const float drive = step * x;
    float out = 0.0f;
    for (uint n = 0; n < states; ++n) {
      h[n] = precise::exp(step * rate[n]) * h[n] + drive * b[bt + n];
      out += h[n] * c[bt + n];
    }
    out += x * skip;
    if (gated) {
      const float g = z[at];
      out *= precise::divide(g, 1.0f + precise::exp(-g));
    }
    y[at] = out;
  }
  for (uint n = 0; n < states; ++n) {
    last[index * states + n] = h[n];
  }
}
"""


def kernel():
    """The fused selective scan of ``mamba_ssm`` when installed, else None."""
    if "scan" not in _KERNEL:
        try:
            # pylint: disable-next=import-outside-toplevel
            from mamba_ssm.ops import selective_scan_interface
            _KERNEL["scan"] = selective_scan_interface.selective_scan_fn
        except ImportError:
            _KERNEL["scan"] = None
    return _KERNEL["scan"]


def metal_kernel():
    """The scan compiled for Apple's GPU, or None where it cannot be."""
    if "metal" not in _KERNEL:
        try:
            _KERNEL["metal"] = torch.mps.compile_shader(_METAL_SOURCE)
        except (AttributeError, RuntimeError):
            _KERNEL["metal"] = None
    return _KERNEL["metal"]


def _metal(u, delta, a, b, c, d, z, state, library):
    batch, frames, channels = u.shape
    states = a.shape[-1]
    y = u.new_empty(batch, frames, channels)
    last = u.new_empty(batch, channels, states)
    if batch * channels:
        u, delta, b, c = (t.contiguous() for t in (u, delta, b, c))
        library.selective_scan(
            y,
            last,
            u,
            delta,
            a.contiguous(),
            b,
            c,
            d.contiguous(),
            u if z is None else z.float().contiguous(),
            u if state is None else state.float().contiguous(),
            batch,
            frames,
            channels,
            states,
            int(z is not None),
            int(state is not None),
            threads=batch * channels)
    return y, last


def _channels_first(t: torch.Tensor) -> torch.Tensor:
    return t.transpose(1, 2).contiguous()


def _fused(u, delta, a, b, c, d, z, fused):
    y, last = fused(_channels_first(u),
                    _channels_first(delta),
                    a,
                    _channels_first(b),
                    _channels_first(c),
                    d,
                    z=None if z is None else _channels_first(z.float()),
                    delta_bias=None,
                    delta_softplus=False,
                    return_last_state=True)
    return y.transpose(1, 2), last


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    z: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs the scan over ``u`` and returns its output and final state.

    Args:
        u: Inputs [batch, frames, channels].
        delta: Step sizes [batch, frames, channels], already positive.
        a: State decay rates [channels, state], negative.
        b: Input projections [batch, frames, state].
        c: Output projections [batch, frames, state].
        d: Skip gains [channels].
        z: Optional gate [batch, frames, channels]; the output is multiplied
            by its SiLU.
        state: The state [batch, channels, state] carried in, zeros when
            None.

    Returns:
        The output [batch, frames, channels] in float32 and the state after
        the last frame.
    """
    u, delta, b, c = (t.float() for t in (u, delta, b, c))
    a, d = a.float(), d.float()
    fused = kernel() if state is None and u.is_cuda else None
    if fused is not None:
        return _fused(u, delta, a, b, c, d, z, fused)
    if u.device.type == "mps" and a.shape[-1] <= METAL_STATES:
        library = metal_kernel()
        if library is not None:
            return _metal(u, delta, a, b, c, d, z, state, library)
    batch, frames, channels = u.shape
    h = (u.new_zeros(batch, channels, a.shape[-1])
         if state is None else state.float())
    outputs = []
    for start in range(0, frames, BLOCK):
        stop = min(start + BLOCK, frames)
        step = delta[:, start:stop]
        decay = torch.exp(step[..., None] * a)
        driven = (step * u[:, start:stop])[..., None] * b[:, start:stop,
                                                          None, :]
        readout = c[:, start:stop, None, :]
        for t in range(stop - start):
            h = decay[:, t] * h + driven[:, t]
            outputs.append((h * readout[:, t]).sum(-1))
    y = torch.stack(outputs, 1) + u * d
    if z is not None:
        y = y * torch.nn.functional.silu(z.float())
    return y, h


def selective_scan_reference(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    z: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The scan one frame at a time: what ``selective_scan`` must match."""
    u, delta, b, c = (t.float() for t in (u, delta, b, c))
    a, d = a.float(), d.float()
    batch, frames, channels = u.shape
    h = (u.new_zeros(batch, channels, a.shape[-1])
         if state is None else state.float())
    ys = []
    for t in range(frames):
        h = (torch.exp(delta[:, t, :, None] * a) * h +
             delta[:, t, :, None] * b[:, t, None, :] * u[:, t, :, None])
        ys.append((h * c[:, t, None, :]).sum(-1) + d * u[:, t])
    y = torch.stack(ys, 1)
    if z is not None:
        y = y * torch.nn.functional.silu(z.float())
    return y, h
