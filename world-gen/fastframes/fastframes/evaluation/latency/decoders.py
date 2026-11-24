#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Pixel-rendering latency: the ConvNeXt pixel decoder and the Wan VAE (Sec. 3.2, Fig. 3).

All decoders render one latent frame into four RGB frames (480x832 at 480p).

* ``StreamingPixelDecoder`` runs the released ``PixelDecoder`` one frame at a time. Each
  causal 3x3x3 depthwise convolution keeps its previous two input frames in a persistent
  buffer (instead of the zero padding of the clip-level forward); everything else is the
  release arithmetic (FP32 master weights under BF16 autocast for FastFrames-S, BF16 for
  FastFrames-L including its 5120->1536 bridge). Frozen BF16 copies of the convolution and
  linear weights avoid per-call casts. The call is captured in a CUDA graph; its output
  is bitwise identical to the clip-level decoder (checked in every ``profile_decoder`` run).
  With ``world > 1`` the pre-head features (sharded by the context-parallel DiT) are
  all-gathered, every rank renders a band of patch rows plus a three-row halo (three
  radius-one spatial convolutions), and the RGB bands are gathered on rank 0.
* Wan VAE: ``VideoVAE.decode_to_pixel(z, use_cache=True)`` on one latent frame (``--chunk 3``:
  three latent frames, the Self-Forcing / CausVid unit) with the causal feature cache warm
  (the streaming decoder of the autoregressive baselines).
"""
from __future__ import annotations

import math
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from dit import Geometry
from models import load_decoder
from timing import capture_graph, compare

from fastframes.checkpoints import load_state_dict
from fastframes.models.pixel_decoder import BridgedPixelDecoder, align_frames

__all__ = ["StreamingPixelDecoder", "profile_decoder", "profile_vae"]


def _autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


class StreamingPixelDecoder(torch.nn.Module):
    """Frame-by-frame ``PixelDecoder`` over ``rows`` patch rows of the ``(h, w)`` token grid."""

    def __init__(self, decoder, grid, rows=None):
        super().__init__()
        self.bridge = decoder.bridge if isinstance(decoder, BridgedPixelDecoder) else None
        self.core = decoder.pixel_head if self.bridge is not None else decoder
        self.height = grid[0] if rows is None else rows
        self.width = grid[1]
        self.tokens = self.height * self.width
        core = self.core
        self.blocks = list(core.conv_low) + list(core.conv_high)
        convs = [core.entry_pw] + [c for b in self.blocks for c in (b.dw, b.pw1, b.pw2)]
        self.conv_w = [c.weight.detach().to(torch.bfloat16) for c in convs]
        self.conv_b = [c.bias.detach().to(torch.bfloat16) for c in convs]
        self.fc_w = core.fc_out.weight.detach().to(torch.bfloat16)
        self.fc_b = core.fc_out.bias.detach().to(torch.bfloat16)
        device = core.fc_out.weight.device
        self.history = [torch.zeros(1, core.dim, 2, self.height, self.width, device=device, dtype=torch.bfloat16)
                        for _ in self.blocks]

    def reset(self):
        for h in self.history:
            h.zero_()

    def _conv(self, index, layer, x, padding):
        return F.conv3d(x, self.conv_w[index], self.conv_b[index], stride=layer.stride, padding=padding,
                        dilation=layer.dilation, groups=layer.groups)

    def forward(self, features, modulation):
        """features ``[1, rows*w, in_dim]``, modulation ``[1, 1, 1, in_dim]`` -> ``[1, 4, 3, 16*rows, 16*w]``."""
        core = self.core
        if self.bridge is not None:
            features, modulation = self.bridge(features), self.bridge(modulation)
        shift, scale = (core.modulation.unsqueeze(1) + modulation).chunk(2, dim=2)
        x = core.norm(features).unsqueeze(1) * (1 + scale) + shift
        x = x.reshape(1, 1, self.height, self.width, core.dim).permute(0, 4, 1, 2, 3).contiguous()
        x = self._conv(0, core.entry_pw, core.entry_norm(x), 0)
        for i, (block, history) in enumerate(zip(self.blocks, self.history)):
            inputs = torch.cat((history, x), dim=2)
            pad = block.dw._pad
            y = self._conv(1 + 3 * i, block.dw, F.pad(inputs, (pad[0], pad[1], pad[2], pad[3], 0, 0)), 0)
            y = self._conv(2 + 3 * i, block.pw1, block.norm(y), 0)
            y = self._conv(3 + 3 * i, block.pw2, block.act(y), 0)
            history.copy_(inputs[:, :, -2:])
            x = x + y
        tokens = x.permute(0, 2, 3, 4, 1).contiguous().view(1, 1, self.tokens, core.dim)  # GEMM layout of the release
        out = F.linear(tokens, self.fc_w, self.fc_b)
        p_t, p_h, p_w = core.patch_size
        out = out.reshape(1, 1, self.height, self.width, p_t, p_h, p_w, core.out_dim)
        return out.permute(0, 1, 4, 7, 2, 5, 3, 6).reshape(1, p_t, core.out_dim, self.height * p_h, self.width * p_w)


class DecoderStep:
    """Static inputs + CUDA graph around one per-frame decoder call (optionally row-parallel)."""

    def __init__(self, decoder, geometry: Geometry, in_dim, *, rank=0, world=1, cuda_graph=True):
        self.world, self.rank, self.decoder = world, rank, decoder
        h, w = geometry.grid
        self.h, self.w = h, w
        device = next(decoder.parameters()).device
        tokens = h * w
        self.local_features = torch.zeros(1, tokens // world, in_dim, device=device, dtype=torch.bfloat16)
        self.modulation = torch.zeros(1, 1, 1, in_dim, device=device, dtype=torch.bfloat16)
        if world > 1:
            self.full = torch.zeros(1, tokens, in_dim, device=device, dtype=torch.bfloat16)
            self.row0, self.row1 = rank * h // world, (rank + 1) * h // world
            halo = len(decoder.pixel_head.conv_low) + len(decoder.pixel_head.conv_high) \
                if isinstance(decoder, BridgedPixelDecoder) else len(decoder.conv_low) + len(decoder.conv_high)
            self.halo0, self.halo1 = max(0, self.row0 - halo), min(h, self.row1 + halo)
            self.band = math.ceil(h / world)
            self.runtime = StreamingPixelDecoder(decoder, (h, w), rows=self.halo1 - self.halo0)
            self.tile = torch.zeros(1, (self.halo1 - self.halo0) * w, in_dim, device=device, dtype=torch.bfloat16)
            self.rgb_shard = torch.zeros(4, 3, self.band * 16, w * 16, device=device, dtype=torch.bfloat16)
            self.rgb_all = torch.zeros(world * 4, 3, self.band * 16, w * 16, device=device, dtype=torch.bfloat16)
            self.ready = torch.zeros(1, device=device)
        else:
            self.runtime = StreamingPixelDecoder(decoder, (h, w))
        self.graph = None
        if cuda_graph:
            with _autocast():
                self.graph, self.output = capture_graph(self._step, before_capture=self._reset)
            self._reset()

    def _reset(self):
        self.runtime.reset()
        if self.world > 1:
            dist.barrier()

    def _step(self):
        if self.world == 1:
            return self.runtime(self.local_features, self.modulation)
        dist.all_gather_into_tensor(self.full.view(-1), self.local_features.view(-1))
        self.tile.copy_(self.full[:, self.halo0 * self.w:self.halo1 * self.w])
        rgb = self.runtime(self.tile, self.modulation)
        start, rows = (self.row0 - self.halo0) * 16, (self.row1 - self.row0) * 16
        self.rgb_shard[:, :, :rows].copy_(rgb[0, :, :, start:start + rows])
        dist.all_gather_into_tensor(self.rgb_all, self.rgb_shard)
        if self.rank == 0:
            bands = self.rgb_all.view(self.world, 4, 3, self.band * 16, self.w * 16)
            out = torch.cat([bands[r, :, :, :((r + 1) * self.h // self.world - r * self.h // self.world) * 16]
                             for r in range(self.world)], dim=2).unsqueeze(0)
        else:
            out = self.ready
        dist.broadcast(self.ready, src=0)
        return out

    def __call__(self, features=None, modulation=None):
        if features is not None:
            self.local_features.copy_(features)
            self.modulation.copy_(modulation)
        if self.graph is not None:
            self.graph.replay()
            return self.output
        with _autocast():
            return self._step()


def profile_decoder(args, ctx, record):
    torch.backends.cudnn.benchmark = True
    decoder = load_decoder(args.model, args.decoder, ctx.device)
    geometry = Geometry.from_resolution(args.resolution, args.frames)
    in_dim = decoder.bridge.in_features if isinstance(decoder, BridgedPixelDecoder) else decoder.dim
    params = sum(p.numel() for p in decoder.parameters())
    g = torch.Generator(device=ctx.device).manual_seed(args.seed)
    frames = geometry.frames
    feats = torch.randn(1, frames * geometry.tokens, in_dim, generator=g, device=ctx.device, dtype=torch.bfloat16)
    mods = 0.2 * torch.randn(1, frames, 1, in_dim, generator=g, device=ctx.device, dtype=torch.bfloat16)
    step = DecoderStep(decoder, geometry, in_dim, rank=ctx.rank, world=ctx.world, cuda_graph=not args.no_cuda_graph)

    # Streaming (graph) output over a clip vs the release clip-level decoder (FastFrames.decode).
    shard = geometry.tokens // ctx.world
    frames_out = []
    step._reset()
    for f in range(frames):
        x = feats[:, f * geometry.tokens:(f + 1) * geometry.tokens][:, ctx.rank * shard:(ctx.rank + 1) * shard]
        frames_out.append(step(x, mods[:, f:f + 1]).clone())
    torch.cuda.synchronize()
    check = None
    if ctx.is_main:
        stream = torch.cat(frames_out, dim=1)
        grid = torch.tensor([[frames, *geometry.grid]], device=ctx.device)
        with torch.no_grad(), _autocast():
            dense = decoder(feats, mods, grid)
        check = {"streaming_vs_clip_decoder_last81": compare(align_frames(stream.float(), 4 * frames - 3),
                                                             align_frames(dense.float(), 4 * frames - 3))}
    step._reset()
    record["settings"] = dict(kind="convnext", parameters=params,
                              weights=args.decoder or "random", in_dim=in_dim, cuda_graph=step.graph is not None,
                              decoder_dtype=str(next(decoder.parameters()).dtype), row_parallel=ctx.world > 1,
                              cudnn_benchmark=True)
    record["check"] = check
    x0 = feats[:, :geometry.tokens][:, ctx.rank * shard:(ctx.rank + 1) * shard].contiguous()
    step.local_features.copy_(x0)
    step.modulation.copy_(mods[:, :1])
    return step


# ---------------------------------------------------------------------------------- VAE
def profile_vae(args, ctx):
    """Median CUDA-event time of streaming VAE decode calls for latent positions 2..F-1."""
    from fastframes.wan.vae import VideoVAE
    if ctx.world != 1:
        raise ValueError("the VAE decoder runs on one GPU")
    vae = VideoVAE(args.vae).to(ctx.device, dtype=torch.bfloat16) if args.vae else _random_vae(ctx.device)
    decoder_params = sum(p.numel() for p in vae.model.decoder.parameters()) + \
        sum(p.numel() for p in vae.model.conv2.parameters())
    geometry = Geometry.from_resolution(args.resolution, args.frames)
    if args.latents:
        latents = _load_latents(args.latents).to(ctx.device, torch.bfloat16)
    else:
        g = torch.Generator(device=ctx.device).manual_seed(args.seed)
        latents = torch.randn(1, geometry.frames, 16, geometry.height, geometry.width, generator=g,
                              device=ctx.device, dtype=torch.bfloat16)

    @torch.compile
    def fn(z):
        return vae.decode_to_pixel(z, use_cache=True)

    chunk = args.chunk
    starts = list(range(0, latents.shape[1] - chunk + 1, chunk))

    def stream(record_events):
        vae.model.clear_cache()
        events = []
        for f in starts:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            out = fn(latents[:, f:f + chunk])
            b.record()
            events.append((a, b, out.shape[1]))
        torch.cuda.synchronize()
        return [(a.elapsed_time(b), n) for a, b, n in events] if record_events else None

    started = time.perf_counter()
    with torch.no_grad():
        dense = vae.decode_to_pixel(latents)  # compile/cudnn warm dense decode (not timed)
        for _ in range(2):
            stream(False)
        prepare = time.perf_counter() - started
        runs = [stream(True) for _ in range(args.streams_measured)]
    # Steady state: calls starting at latent position >= 2 (the first two positions fill the causal cache).
    steady_calls = [i for i, f in enumerate(starts) if f >= 2]
    steady = [run[i][0] for run in runs for i in steady_calls]
    first = [run[0][0] for run in runs]
    second = [run[1][0] for run in runs]
    return {"latency_ms": statistics.median(steady),
            "settings": dict(compile="torch.compile", dtype="bfloat16",
                             latents_per_call=chunk, decoder_parameters=decoder_params,
                             weights=args.vae or "random", latents=args.latents or "random N(0, 1)",
                             frames_per_call=runs[0][steady_calls[0]][1], dense_frames=int(dense.shape[1])),
            "prepare_seconds": prepare,
            "timing": {"statistic": f"median CUDA-event time of {chunk}-latent calls starting at latent "
                                    f"position >= 2, over {args.streams_measured} streams (2 untimed warm-up streams)",
                       "steady": {"count": len(steady), "median_ms": statistics.median(steady),
                                  "mean_ms": statistics.mean(steady), "min_ms": min(steady), "max_ms": max(steady)},
                       "first_call_median_ms": statistics.median(first),
                       "second_call_median_ms": statistics.median(second),
                       "samples_ms": [[t for t, _ in run] for run in runs]}}


def _random_vae(device):
    from fastframes.wan.vae import VideoVAE, WanVAE_
    vae = VideoVAE.__new__(VideoVAE)
    torch.nn.Module.__init__(vae)
    vae.mean = torch.tensor(VideoVAE.MEAN)
    vae.std = torch.tensor(VideoVAE.STD)
    vae.model = WanVAE_(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                        temperal_downsample=[False, True, True], dropout=0.0).eval().requires_grad_(False)
    return vae.to(device, dtype=torch.bfloat16)


def _load_latents(path):
    if str(path).endswith(".safetensors"):
        state = load_state_dict(path)
        return next(iter(state.values()))
    return torch.load(path, map_location="cpu", weights_only=True)
