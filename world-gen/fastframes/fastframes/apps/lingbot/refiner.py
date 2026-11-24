#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""KV refinement of a LingBot chunk (Sec. 3.3 of the paper).

After the chunk's final denoising step the DiT has written the chunk's *noisy* keys and
values into the rolling cache. The ``KVRefinementNetwork`` maps them, together with the
clean latent, to refined keys and values that replace the cache slot; this removes the
upstream clean re-evaluation of the chunk at ``t = 0`` (only the first chunk keeps it).

LingBot's refinement networks operate on canonical keys (RMS-normalized, before RoPE) and
raw values, i.e. ``DiTResult.key`` / ``DiTResult.value``. The stem input is the DiT's own
36-channel patch projection of ``[x0, image condition]``; the stem attends within each
latent frame. Repaired keys are rotated with the chunk's absolute RoPE before they are
committed. All arithmetic is BF16 without autocast. The per-layer heads run as batched GEMMs,
split into groups of ``heads_per_stream`` layers on parallel CUDA streams.

Context parallel (8 GPUs): ranks ``2f`` and ``2f + 1`` evaluate the stem of latent frame ``f``;
each rank refines its 754 query tokens for all layers; the results are all-gathered.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .conditioning import CHUNK, LATENT_HEIGHT, LATENT_WIDTH
from .dit import FRAME_TOKENS, READ_FRAMES, TOKENS, rope_real

__all__ = ["KVRefiner"]


class _Stem(torch.nn.Module):
    """``phi_v`` of the chunk (or of this rank's half frame): patch projection -> stem -> ``phi_proj``."""

    def __init__(self, net, patch, timestep, world, rank):
        super().__init__()
        self.net, self.patch, self.world, self.rank = net, patch, world, rank
        self.frames = CHUNK if world == 1 else 1
        device = next(net.parameters()).device
        self.register_buffer("embedding", net.embed_timestep(timestep, self.frames, device, torch.bfloat16))
        self.phi_proj = net.heads_grouped.phi_proj[str(net.heads_grouped.gemms[0].d_v)]

    def forward(self, x, y):
        if self.world > 1:
            frame = self.rank // 2
            x, y = x[:, frame:frame + 1], y[:, frame:frame + 1]
        # Conv3d(36 -> dim, kernel (1, 2, 2)) as a channel-major GEMM plus BF16 bias.
        value = torch.cat((x, y), 0).to(torch.bfloat16)
        columns = value.reshape(36, self.frames, LATENT_HEIGHT // 2, 2, LATENT_WIDTH // 2, 2)
        columns = columns.permute(0, 3, 5, 1, 2, 4).reshape(144, -1)
        stem = (self.patch.weight.flatten(1) @ columns) + self.patch.bias[:, None]
        stem = stem.T.reshape(self.frames, FRAME_TOKENS, -1)
        phi = self.net.compute_phi(stem, self.embedding).reshape(1, self.frames * FRAME_TOKENS, -1)
        if self.world > 1:
            start = self.rank % 2 * (FRAME_TOKENS // 2)
            phi = phi[:, start:start + FRAME_TOKENS // 2]
        return self.phi_proj(phi[0])


class _Heads(torch.nn.Module):
    """Refinement heads of layers ``[lo, hi)`` with their timestep modulation folded in."""

    names = ("proj_k_w", "proj_k_b", "proj_v_w", "proj_v_b", "fuse_w", "fuse_b",
             "m0_w", "m0_b", "m2_w", "m2_b", "dK_w", "dK_b", "dV_w", "dV_b")

    def __init__(self, net, embedding, lo, hi):
        super().__init__()
        group = net.heads_grouped.gemms[0]
        self.count, self.heads, self.head_dim = hi - lo, net.num_heads, net.head_dim
        for name in self.names:
            self.register_buffer(name, getattr(group, name)[lo:hi].detach())
        e = F.silu(embedding).reshape(1, 1, net.d_a).expand(net.num_blocks, 1, net.d_a).contiguous()
        shift, scale = torch.baddbmm(group.mod_b, e, group.mod_w).chunk(2, -1)
        self.register_buffer("shift", shift[lo:hi].detach())
        self.register_buffer("scale", scale[lo:hi].detach())
        self.register_buffer("vscale", group.v_logscale[lo:hi].exp().detach())

    def forward(self, k, v, phi):
        pk = torch.baddbmm(self.proj_k_b, k, self.proj_k_w)
        pv = torch.baddbmm(self.proj_v_b, v, self.proj_v_w)
        h = torch.baddbmm(self.fuse_b, torch.cat((pk, pv, phi[None].expand(self.count, -1, -1)), -1), self.fuse_w)
        h = F.layer_norm(h, (h.shape[-1],)) * (1 + self.scale) + self.shift
        h = torch.baddbmm(self.m0_b, h, self.m0_w)
        h = torch.baddbmm(self.m2_b, F.gelu(h, approximate="tanh"), self.m2_w)
        kr = k + torch.baddbmm(self.dK_b, h, self.dK_w)
        shaped = kr.reshape(self.count, -1, self.heads, self.head_dim)
        ref = k.reshape(self.count, -1, self.heads, self.head_dim)
        kr = (shaped * (ref.norm(dim=-1, keepdim=True) / (shaped.norm(dim=-1, keepdim=True) + 1e-6))).reshape_as(k)
        vr = (v + torch.baddbmm(self.dV_b, h, self.dV_w)) * self.vscale
        return kr, vr


class KVRefiner:
    """Refine the chunk's cache slot in place: ``refiner(x0, y, rope, read_frames)``.

    ``key`` / ``value`` are the DiT's captured canonical K/V ``[layers, 1, 6032, heads, head_dim]``;
    ``cache_k`` / ``cache_v`` the rolling cache ``[layers, 1, 18 * 1508, heads, head_dim]``.
    """

    def __init__(self, net, patch, cache_k, cache_v, key, value, timestep, *, world=1, rank=0,
                 heads_per_stream=None, compile=True, graphs=True):
        import torch._inductor.config as inductor_config
        inductor_config.emulate_precision_casts = True
        if world not in (1, 8):
            raise ValueError("KV refinement supports 1 or 8 GPUs")
        layers, dim = net.num_blocks, net.dim
        heads_per_stream = heads_per_stream or layers
        if layers % heads_per_stream:
            raise ValueError("heads_per_stream must divide the number of layers")
        self.world, self.rank, self.layers, self.dim = world, rank, layers, dim
        self.cache_k, self.cache_v = cache_k, cache_v
        self.k, self.v = key.view(layers, TOKENS, dim), value.view(layers, TOKENS, dim)
        self.use_graphs, self.graphs = graphs, {}
        device = self.k.device
        self.x = torch.empty(16, CHUNK, LATENT_HEIGHT, LATENT_WIDTH, device=device)
        self.y = torch.empty(20, CHUNK, LATENT_HEIGHT, LATENT_WIDTH, device=device)
        self.rope = torch.empty(TOKENS, 1, net.head_dim // 2, 2, device=device)
        self.count = heads_per_stream
        with torch.autocast("cuda", enabled=False):
            self.stem_module = _Stem(net, patch, timestep, world, rank)
            embedding = net.embed_timestep(timestep, 1, device, torch.bfloat16)
            self.head_modules = [_Heads(net, embedding, lo, lo + self.count) for lo in range(0, layers, self.count)]

        def compiled(fn):
            if not compile:
                return fn
            return torch.compile(fn, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")
        self.stem = compiled(self.stem_module)
        self.heads = [compiled(h) for h in self.head_modules]
        self.rotate = compiled(rope_real)
        self.commit = compiled(self._commit)
        self.streams = [torch.cuda.Stream(device=device) for _ in self.heads] if len(self.heads) > 1 else []
        self.local_tokens = TOKENS // world
        self.local = torch.empty(2, layers, self.local_tokens, dim, device=device, dtype=torch.bfloat16)
        self.gathered = (torch.empty(world, 2, layers, self.local_tokens, dim, device=device, dtype=torch.bfloat16)
                         if world > 1 else None)
        self.pool = torch.cuda.graph_pool_handle()

    def _commit(self, key, value, read_frames: int):
        end = read_frames * FRAME_TOKENS
        shape = (self.layers, 1, TOKENS, self.cache_k.shape[-2], self.cache_k.shape[-1])
        self.cache_k[:, :, end - TOKENS:end].copy_(key.reshape(shape))
        self.cache_v[:, :, end - TOKENS:end].copy_(value.reshape(shape))

    def _refine(self, head, first, last, phi, lo, hi):
        kr, vr = head(self.k[first:last, lo:hi], self.v[first:last, lo:hi], phi)
        heads = self.cache_k.shape[-2]
        physical = self.rotate(kr.reshape(last - first, self.local_tokens, heads, -1), self.rope[lo:hi])
        self.local[0, first:last].copy_(physical.reshape(last - first, self.local_tokens, self.dim))
        self.local[1, first:last].copy_(vr)

    def _body(self, read_frames):
        with torch.autocast("cuda", enabled=False):
            phi = self.stem(self.x, self.y)
            lo = self.rank * self.local_tokens
            hi = lo + self.local_tokens
            if self.streams:
                caller = torch.cuda.current_stream(self.k.device)
                for group, (head, stream) in enumerate(zip(self.heads, self.streams)):
                    stream.wait_stream(caller)
                    with torch.cuda.stream(stream):
                        self._refine(head, group * self.count, (group + 1) * self.count, phi, lo, hi)
                for stream in self.streams:
                    caller.wait_stream(stream)
            else:
                self._refine(self.heads[0], 0, self.layers, phi, lo, hi)
            if self.world > 1:
                dist.all_gather_into_tensor(self.gathered.view(-1, self.local_tokens, self.dim),
                                            self.local.view(-1, self.local_tokens, self.dim))
                key = self.gathered[:, 0].permute(1, 0, 2, 3).reshape(self.layers, TOKENS, self.dim)
                value = self.gathered[:, 1].permute(1, 0, 2, 3).reshape(self.layers, TOKENS, self.dim)
            else:
                key, value = self.local.unbind(0)
            self.commit(key, value, read_frames)

    @torch.no_grad()
    def __call__(self, x0, y, rope, read_frames):
        if read_frames not in READ_FRAMES:
            raise ValueError(f"unsupported cache length {read_frames}")
        self.x.copy_(x0)
        self.y.copy_(y)
        self.rope.copy_(rope)
        if not self.use_graphs:
            return self._body(read_frames)
        if read_frames not in self.graphs:
            side = torch.cuda.Stream(device=self.k.device)
            side.wait_stream(torch.cuda.current_stream(self.k.device))
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._body(read_frames)
            torch.cuda.current_stream(self.k.device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                self._body(read_frames)
            self.graphs[read_frames] = graph
        self.graphs[read_frames].replay()

    def close(self):
        torch.cuda.synchronize(self.k.device)
        self.graphs.clear()
