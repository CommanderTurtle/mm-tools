#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Streaming pixel decoding of LingBot chunks (Sec. 3.2 of the paper).

The pixel decoder maps the DiT's pre-head
hidden states of a chunk's final denoising step to RGB. ``StreamingDecoder`` evaluates only
the four new latent frames: each of the three causal ConvNeXt blocks keeps its previous two
input frames (zeros at the start of a video, which equals the causal zero padding). A chunk
yields 16 RGB frames; the caller drops the first three frames of a video once.

``CapturedDecoder`` compiles and graph-captures one step (1 GPU). ``SpatialDecoder`` splits
the 29 token rows into bands, one per rank, with a three-row halo covering the receptive
field of the three blocks, and gathers the RGB bands on rank 0 (8 GPUs).
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .conditioning import CHUNK
from .dit import GRID_H, GRID_W

__all__ = ["StreamingDecoder", "CapturedDecoder", "SpatialDecoder"]

COMPILE_OPTIONS = {"max_autotune": True, "coordinate_descent_tuning": True,
                   "emulate_precision_casts": True, "triton.cudagraphs": False}


class StreamingDecoder(torch.nn.Module):
    """Four-latent causal decoding with per-block input history (``rows`` token rows)."""

    def __init__(self, decoder, rows=GRID_H, width=GRID_W):
        super().__init__()
        self.decoder = decoder
        head = self.decoder
        self.rows, self.width, self.dim = rows, width, head.dim
        self.in_dim = head.dim
        self.tokens = CHUNK * rows * width
        self.blocks = list(head.conv_low) + list(head.conv_high)
        weight = next(head.parameters())
        for i in range(len(self.blocks)):
            self.register_buffer(f"history_{i}", torch.zeros(1, head.dim, 2, rows, width, device=weight.device,
                                                             dtype=weight.dtype), persistent=False)

    @property
    def history(self):
        return [getattr(self, f"history_{i}") for i in range(len(self.blocks))]

    def reset(self):
        for value in self.history:
            value.zero_()

    def forward(self, features, modulation):
        """``features [1, 4 * rows * 52, in_dim]``, ``modulation [1, 4, 1, in_dim]`` -> ``[1, 16, 3, 16 rows, 832]``."""
        head = self.decoder
        dtype = head.modulation.dtype
        features, modulation = features.to(dtype), modulation.to(dtype)
        shift, scale = (head.modulation.unsqueeze(1) + modulation).chunk(2, dim=2)
        value = head.norm(features).unflatten(1, (CHUNK, self.rows * self.width)) * (1 + scale) + shift
        current = value.reshape(1, CHUNK, self.rows, self.width, self.dim).permute(0, 4, 1, 2, 3).contiguous()
        current = head.entry_pw(head.entry_norm(current))
        for block, history in zip(self.blocks, self.history):
            inputs = torch.cat((history, current), dim=2)
            hidden = F.conv3d(F.pad(inputs, (1, 1, 1, 1, 0, 0)), block.dw.weight, block.dw.bias,
                              stride=block.dw.stride, dilation=block.dw.dilation, groups=block.dw.groups)
            hidden = block.pw2(block.act(block.pw1(block.norm(hidden))))
            history.copy_(inputs[:, :, -2:])
            current = current + hidden
        tokens = current.permute(0, 2, 3, 4, 1).contiguous().view(1, CHUNK, self.rows * self.width, self.dim)
        pixels = head.fc_out(tokens).reshape(1, CHUNK, self.rows, self.width, 4, 16, 16, 3)
        return pixels.permute(0, 1, 4, 7, 2, 5, 3, 6).reshape(1, 4 * CHUNK, 3, 16 * self.rows, 16 * self.width)


class CapturedDecoder:
    """Compiled, graph-captured ``StreamingDecoder`` with FP32 input staging (1 GPU)."""

    def __init__(self, decoder, *, compile=True, graphs=True, warmup=4):
        self.runtime = runtime = StreamingDecoder(decoder)
        device = next(decoder.parameters()).device
        self.features = torch.zeros(1, runtime.tokens, runtime.in_dim, device=device)
        self.modulation = torch.zeros(1, CHUNK, 1, runtime.in_dim, device=device)
        self.call = (torch.compile(runtime.forward, fullgraph=True, dynamic=False, options=dict(COMPILE_OPTIONS))
                     if compile else runtime.forward)
        self.graph = None
        with torch.inference_mode():
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    self.call(self.features, self.modulation)
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize(device)
            if graphs:
                runtime.reset()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=stream):
                    self.output = self.call(self.features, self.modulation)
                torch.cuda.current_stream(device).wait_stream(stream)
                torch.cuda.synchronize(device)
            runtime.reset()

    def reset(self):
        self.runtime.reset()

    def __call__(self, features, modulation):
        self.features.copy_(features)
        self.modulation.copy_(modulation)
        if self.graph is None:
            return self.call(self.features, self.modulation)
        self.graph.replay()
        return self.output

    def close(self):
        torch.cuda.synchronize(self.features.device)
        self.graph = None


class SpatialDecoder:
    """Row-band context-parallel decoding; returns RGB on rank 0 (a flag tensor elsewhere).

    ``features`` are the full (all-gathered) chunk features on every rank.
    """

    def __init__(self, decoder, *, compile=True, graphs=True, warmup=4):
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        device = next(decoder.parameters()).device
        self.row_start, self.row_end = self.rank * GRID_H // self.world, (self.rank + 1) * GRID_H // self.world
        self.halo_start, self.halo_end = max(0, self.row_start - 3), min(GRID_H, self.row_end + 3)
        self.core_rows, self.max_rows = self.row_end - self.row_start, math.ceil(GRID_H / self.world)
        self.runtime = StreamingDecoder(decoder, rows=self.halo_end - self.halo_start)
        self.dim = self.runtime.in_dim
        self.full_features = torch.zeros(CHUNK * GRID_H * GRID_W, self.dim, device=device)
        self.tile = torch.zeros(1, self.runtime.tokens, self.dim, device=device)
        self.modulation = torch.zeros(1, CHUNK, 1, self.dim, device=device)
        dtype = next(decoder.parameters()).dtype
        self.shard = torch.zeros(4 * CHUNK, 3, 16 * self.max_rows, 16 * GRID_W, device=device, dtype=dtype)
        self.gathered = torch.zeros(self.world * 4 * CHUNK, 3, 16 * self.max_rows, 16 * GRID_W, device=device,
                                    dtype=dtype)
        self.ready = torch.zeros(1, device=device, dtype=torch.int32)
        self.call = (torch.compile(self.runtime.forward, fullgraph=True, dynamic=False, options=dict(COMPILE_OPTIONS))
                     if compile else self.runtime.forward)
        self.graph = None
        with torch.inference_mode():
            for _ in range(warmup):  # compile every rank's tile shape before entering collectives
                self.call(self.tile, self.modulation)
            dist.barrier()
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    self._step()
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize(device)
            self.runtime.reset()
            if graphs:
                dist.barrier()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=stream):
                    self.output = self._step()
                torch.cuda.current_stream(device).wait_stream(stream)
                torch.cuda.synchronize(device)
                self.runtime.reset()
            dist.barrier()

    def _step(self):
        tile = self.full_features.view(1, CHUNK, GRID_H, GRID_W, self.dim)[:, :, self.halo_start:self.halo_end]
        self.tile.copy_(tile.reshape(1, self.runtime.tokens, self.dim))
        pixels = self.call(self.tile, self.modulation)
        crop = (self.row_start - self.halo_start) * 16
        self.shard[:, :, :self.core_rows * 16].copy_(pixels[0, :, :, crop:crop + self.core_rows * 16])
        dist.all_gather_into_tensor(self.gathered, self.shard)
        if self.rank == 0:
            bands = self.gathered.view(self.world, 4 * CHUNK, 3, 16 * self.max_rows, 16 * GRID_W)
            rows = [(r + 1) * GRID_H // self.world - r * GRID_H // self.world for r in range(self.world)]
            output = torch.cat([bands[r, :, :, :rows[r] * 16] for r in range(self.world)], dim=2).unsqueeze(0)
        else:
            output = self.ready
        dist.broadcast(self.ready, src=0)  # keeps every rank in step with rank 0's assembly of the frames
        return output

    def reset(self):
        self.runtime.reset()

    def __call__(self, features, modulation):
        self.full_features.copy_(features.reshape(-1, self.dim))
        self.modulation.copy_(modulation)
        if self.graph is None:
            return self._step()
        self.graph.replay()
        return self.output

    def close(self):
        torch.cuda.synchronize(self.full_features.device)
        self.graph = None
