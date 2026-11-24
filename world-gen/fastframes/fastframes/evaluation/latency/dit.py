#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""DiT evaluation for latency profiling (Sec. 3.1 and 3.4 of the paper).

``FrameDiT`` evaluates one generation unit (``chunk`` latent frames; 1 for FastFrames
and Causal Forcing++, 3 for Self-Forcing and CausVid) of a Wan2.1
``fastframes.wan.causal_model.CausalWanModel`` against a persistent, full KV cache,
in one of two modes:

* ``baseline`` -- the reference per-frame computation of ``CausalWanModel.forward_frame``
  (Conv3d patch embedding, separate Q/K/V projections, complex-valued RoPE, cuDNN SDPA),
  with the timestep conditioning evaluated inside every call and the text K/V cached,
  as in the public autoregressive Wan2.1 implementations. Compiled with ``torch.compile``
  (default mode). This is the "BF16 + cuDNN attention + torch.compile" baseline.
* ``optimized`` -- the systems-level optimizations of Sec. 3.4:
  persistent KV buffers written in place; precomputed RoPE tables, text K/V and
  timestep modulation; one fused QKV GEMM per layer; split-KV attention with a
  log-sum-exp merge; per-block ``torch.compile`` and a CUDA graph over the whole
  forward. With ``world > 1`` it is context parallel: each rank owns a contiguous
  slice of the unit's query tokens, the live K/V of every layer are all-gathered
  through NVLink symmetric memory, and every rank keeps the full KV history.

Both modes use the parameters of the ``fastframes`` model, so the profiled computation
is the released model (``check_dit.py`` measures the numerical agreement).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from timing import capture_graph  # also puts the repository root on sys.path
from torch.nn.attention import SDPBackend, sdpa_kernel

from fastframes.wan.attention import attention
from fastframes.wan.causal_model import CausalWanModel, allocate_kv_cache
from fastframes.wan.model import causal_rope_apply

__all__ = ["Geometry", "RESOLUTIONS", "FrameDiT", "split_kv_attention", "rope_table"]

# Latent (height, width) of the officially supported Wan2.1 resolutions.
RESOLUTIONS = {"480p": (60, 104), "720p": (90, 160)}


@dataclass(frozen=True)
class Geometry:
    """Latent frames of ``height x width``; the full KV cache holds ``frames`` frames."""

    height: int = 60
    width: int = 104
    frames: int = 21

    @classmethod
    def from_resolution(cls, resolution, frames=21):
        return cls(*RESOLUTIONS[resolution], frames)

    @property
    def grid(self):
        return self.height // 2, self.width // 2

    @property
    def tokens(self):
        """Tokens per latent frame (1560 at 480p, 3600 at 720p)."""
        return self.grid[0] * self.grid[1]

    @property
    def kv_len(self):
        return self.frames * self.tokens


def split_kv_attention(q, k, v, split, scale):
    """Exact attention of ``q [1, H, Lq, D]`` over ``k, v [1, Lk, H, D]`` (cuDNN).

    With ``split > 1`` the KV sequence is cut into ``split`` equal segments that are
    processed as one batched cuDNN call; the partial outputs are merged with their
    log-sum-exp weights in FP32. This exposes more parallel work when ``Lq`` (one latent
    frame) is small relative to the cache.
    """
    if split == 1:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return F.scaled_dot_product_attention(q, k.transpose(1, 2), v.transpose(1, 2))
    _, length, heads, dim = k.shape
    if length % split:
        raise ValueError(f"KV length {length} is not divisible by split={split}")
    k = k.view(split, length // split, heads, dim).transpose(1, 2)
    v = v.view(split, length // split, heads, dim).transpose(1, 2)
    out, lse, *_ = torch.ops.aten._scaled_dot_product_cudnn_attention(
        q.expand(split, -1, -1, -1).contiguous(), k, v, None, True, 0.0, False, False, scale=scale)
    if lse.ndim == 3:
        lse = lse.unsqueeze(-1)
    weight = torch.exp(lse - lse.amax(dim=0, keepdim=True))
    return ((out.float() * weight).sum(0, keepdim=True) / weight.sum(0, keepdim=True)).to(q.dtype)


def _rope(x, table):
    """RoPE as FP32 real rotations; ``table`` is ``[L, 1, D/2, 2]`` (cos, sin)."""
    pairs = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    re, im = pairs[..., 0], pairs[..., 1]
    cos, sin = table[..., 0], table[..., 1]
    return torch.stack((re * cos - im * sin, re * sin + im * cos), dim=-1).reshape_as(x).to(x.dtype)


def rope_table(model, start_frame, num_frames, grid):
    """FP32 ``(cos, sin)`` table ``[F*h*w, 1, D/2, 2]`` for latent frames ``start_frame + [0, F)``."""
    h, w = grid
    freqs = model.freqs.to(model.patch_embedding.weight.device)
    half = freqs.shape[1]
    t, y, x = freqs.split([half - 2 * (half // 3), half // 3, half // 3], dim=1)
    f = num_frames
    table = torch.cat((
        t[start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
        y[:h].view(1, h, 1, -1).expand(f, h, w, -1),
        x[:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ), dim=-1).reshape(f * h * w, 1, half)
    return torch.view_as_real(table).float().contiguous()


def _native_block(block, x, e0, grid_sizes, freqs, start_frame, text, cache, live):
    """``CausalWanAttentionBlock.forward_frame`` (reference path) for any number of frames."""
    mod = (block.modulation.unsqueeze(1) + e0).chunk(6, dim=2)
    h = (block.norm1(x).unsqueeze(1) * (1 + mod[1]) + mod[0]).flatten(1, 2)
    sa = block.self_attn
    q, k, v = sa.qkv(h)
    q = causal_rope_apply(q, grid_sizes, freqs, start_frame=start_frame).type_as(v)
    k = causal_rope_apply(k, grid_sizes, freqs, start_frame=start_frame).type_as(v)
    cache["k"][:, live].copy_(k)
    cache["v"][:, live].copy_(v)
    y = sa.o(attention(q, cache["k"][:, :live.stop], cache["v"][:, :live.stop]).flatten(2))
    x = x + (y.unsqueeze(1) * mod[2]).flatten(1, 2)
    x = x + block.cross_attn(block.norm3(x), None, cache=text)
    y = block.ffn((block.norm2(x).unsqueeze(1) * (1 + mod[4]) + mod[3]).flatten(1, 2))
    return x + (y.unsqueeze(1) * mod[5]).flatten(1, 2)


def _optimized_block(block, x, e0, key_cache, value_cache, text_k, text_v, qkv_w, qkv_b, rope,
                     live, gather, split, scale, group_name):
    """One transformer block on this rank's query tokens (optimized path).

    ``gather`` is ``None`` on one GPU, otherwise the ``(shard, full)`` symmetric-memory
    buffers through which the unit's live K/V are all-gathered across ranks.
    """
    sa, ca = block.self_attn, block.cross_attn
    b, tokens, _ = x.shape
    shape = (b, tokens, sa.num_heads, sa.head_dim)
    mod = (block.modulation.unsqueeze(1) + e0).chunk(6, dim=2)
    h = (block.norm1(x).unsqueeze(1) * (1 + mod[1]) + mod[0]).flatten(1, 2)
    q, k, v = F.linear(h, qkv_w, qkv_b).chunk(3, dim=-1)
    q = _rope(sa.norm_q(q).view(shape), rope)
    k = _rope(sa.norm_k(k).view(shape), rope)
    v = v.view(shape)
    if gather is None:
        key_cache[:, live].copy_(k)
        value_cache[:, live].copy_(v)
    else:
        shard, full = gather
        shard.copy_(torch.cat((k, v), dim=-1).view(tokens, sa.num_heads, 2 * sa.head_dim))
        torch.ops.symm_mem.multimem_all_gather_out(shard, group_name, full)
        live_k, live_v = full.split(sa.head_dim, dim=-1)
        key_cache[:, live].copy_(live_k.unsqueeze(0))
        value_cache[:, live].copy_(live_v.unsqueeze(0))
    y = split_kv_attention(q.transpose(1, 2), key_cache[:, :live.stop], value_cache[:, :live.stop], split, scale)
    y = sa.o(y.transpose(1, 2).reshape(b, tokens, -1))
    x = x + (y.unsqueeze(1) * mod[2]).flatten(1, 2)
    cq = ca.norm_q(ca.q(block.norm3(x))).view(shape)
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        y = F.scaled_dot_product_attention(cq.transpose(1, 2), text_k.transpose(1, 2), text_v.transpose(1, 2))
    x = x + ca.o(y.transpose(1, 2).reshape(b, tokens, -1))
    y = block.ffn((block.norm2(x).unsqueeze(1) * (1 + mod[4]) + mod[3]).flatten(1, 2))
    return x + (y.unsqueeze(1) * mod[5]).flatten(1, 2)


class FrameDiT:
    """One DiT evaluation of the last ``chunk`` latent frames against a full KV cache.

    ``set_inputs`` fills the static latent / timestep / prompt buffers, ``prepare`` compiles
    (and, for ``optimized``, captures a CUDA graph) and ``__call__`` runs one evaluation,
    returning the flow prediction ``[1, chunk, 16, H, W]`` (complete on every rank).
    ``self.features`` holds the pre-head hidden states of this rank's tokens.
    """

    def __init__(self, model: CausalWanModel, geometry: Geometry, *, mode="optimized", chunk=1,
                 split_kv=1, rank=0, world=1, group_name=None, compile_mode="max-autotune-no-cudagraphs",
                 cuda_graph=True):
        if mode not in ("baseline", "optimized"):
            raise ValueError(mode)
        if world > 1 and mode != "optimized":
            raise ValueError("context parallelism is implemented for the optimized path only")
        self.model, self.geometry, self.mode, self.chunk = model, geometry, mode, chunk
        self.split_kv, self.rank, self.world, self.group_name = split_kv, rank, world, group_name
        self.compile_mode, self.cuda_graph = compile_mode, cuda_graph
        self.device = model.patch_embedding.weight.device
        self.start_frame = geometry.frames - chunk
        tokens = chunk * geometry.tokens
        if tokens % world:
            raise ValueError(f"{tokens} query tokens are not divisible by world={world}")
        self.tokens, self.local = tokens, tokens // world
        self.token_slice = slice(rank * self.local, (rank + 1) * self.local)
        self.live = slice(self.start_frame * geometry.tokens, geometry.kv_len)
        heads, head_dim = model.num_heads, model.dim // model.num_heads
        self.scale = head_dim ** -0.5
        self.kv_cache = allocate_kv_cache(model, geometry.frames, geometry.tokens, device=self.device)
        self.latent = torch.zeros(1, chunk, 16, geometry.height, geometry.width, device=self.device,
                                  dtype=torch.bfloat16)
        self.timestep = torch.zeros(1, 1, device=self.device)
        self.prompt = torch.zeros(1, model.text_len, model.config.text_dim, device=self.device,
                                  dtype=torch.bfloat16)
        self.grid_sizes = torch.tensor([[chunk, *geometry.grid]], dtype=torch.long)
        self._fn, self.graph, self.features = None, None, None
        if mode == "optimized":
            self.patch_w = model.patch_embedding.weight.detach().reshape(model.dim, -1).contiguous()
            self.patch_b = model.patch_embedding.bias
            self.qkv = [(torch.cat([b.self_attn.q.weight, b.self_attn.k.weight, b.self_attn.v.weight]).contiguous(),
                         torch.cat([b.self_attn.q.bias, b.self_attn.k.bias, b.self_attn.v.bias]).contiguous())
                        for b in model.blocks]
            self.rope = rope_table(model, self.start_frame, chunk, geometry.grid)[self.token_slice].contiguous()
            self.gather, self.out_gather = [None] * model.num_layers, None
            if world > 1:
                import torch.distributed._symmetric_memory as symm_mem

                def buffer(*shape):
                    t = symm_mem.empty(*shape, dtype=torch.bfloat16, device=self.device)
                    symm_mem.rendezvous(t, group_name)
                    return t
                self.gather = [(buffer(self.local, heads, 2 * head_dim), buffer(tokens, heads, 2 * head_dim))
                               for _ in model.blocks]
                self.out_gather = (buffer(self.local, 64), buffer(tokens, 64))

    # ------------------------------------------------------------------ inputs
    @torch.no_grad()
    def set_inputs(self, latent=None, timestep=833.0, prompt=None, seed=0):
        """Copy inputs into the static buffers (random when omitted); precompute fixed conditioning."""
        g = torch.Generator(device=self.device).manual_seed(seed)

        def rand(t):
            return torch.randn(t.shape, generator=g, device=self.device, dtype=t.dtype)

        self.latent.copy_(rand(self.latent) if latent is None else latent)
        self.prompt.copy_(rand(self.prompt) if prompt is None else prompt)
        self.timestep.fill_(float(timestep))
        # Text K/V are projected once per prompt in both modes (as in the reference implementations).
        self.text = self.model.text_cache(self.prompt)
        if self.mode == "optimized":
            self.e0, self.e_head = self.model.time_conditioning(self.timestep, torch.bfloat16)
            self.text_kv = [(c["k"].contiguous(), c["v"].contiguous()) for c in self.text]

    @torch.no_grad()
    def fill_cache(self, seed=1):
        """Fill the KV history with random BF16 values (the cost does not depend on them)."""
        g = torch.Generator(device=self.device).manual_seed(seed)
        for c in self.kv_cache:
            for t in (c["k"], c["v"]):
                t.copy_(torch.randn(t.shape, generator=g, device=self.device, dtype=t.dtype))

    # ------------------------------------------------------------------ baseline
    def _baseline_forward(self):
        m = self.model
        if m.freqs.device != self.device:
            m.freqs = m.freqs.to(self.device)
        e0, e_head = m.time_conditioning(self.timestep, torch.bfloat16)
        x = m.patch_embedding(self.latent.permute(0, 2, 1, 3, 4)).flatten(2).transpose(1, 2)
        for i, block in enumerate(m.blocks):
            x = _native_block(block, x, e0, self.grid_sizes, m.freqs, self.start_frame, self.text[i],
                              self.kv_cache[i], self.live)
        self.features = x
        return torch.stack(m.unpatchify(m.head(x, e_head), self.grid_sizes)).permute(0, 2, 1, 3, 4)

    # ------------------------------------------------------------------ optimized
    def _patch(self):
        h, w = self.geometry.grid
        x = self.latent.reshape(1, self.chunk, 16, h, 2, w, 2).permute(0, 1, 3, 5, 2, 4, 6)
        x = x.reshape(1, self.tokens, 64)
        return F.linear(x[:, self.token_slice], self.patch_w, self.patch_b)

    def _head(self, x):
        h, w = self.geometry.grid
        tokens = self.model.head(x, self.e_head).reshape(self.local, 64)
        if self.out_gather is not None:
            shard, full = self.out_gather
            shard.copy_(tokens)
            torch.ops.symm_mem.multimem_all_gather_out(shard, self.group_name, full)
            tokens = full
        # Wan unpatchify of the (1, 2, 2) patches into [1, chunk, 16, H, W].
        return tokens.view(1, self.chunk, h, w, 2, 2, 16).permute(0, 1, 6, 2, 4, 3, 5).reshape(
            1, self.chunk, 16, 2 * h, 2 * w)

    def _optimized_forward(self, block_fn=_optimized_block, patch_fn=None, head_fn=None):
        x = (patch_fn or self._patch)()
        for i, block in enumerate(self.model.blocks):
            cache = self.kv_cache[i]
            x = block_fn(block, x, self.e0, cache["k"], cache["v"], *self.text_kv[i], *self.qkv[i], self.rope,
                         self.live, self.gather[i], self.split_kv, self.scale, self.group_name)
        self.features = x
        return (head_fn or self._head)(x)

    # ------------------------------------------------------------------ execution
    @torch.no_grad()
    def eager(self):
        """Uncompiled, ungraphed forward of the selected mode (numerical reference)."""
        return self._baseline_forward() if self.mode == "baseline" else self._optimized_forward()

    @torch.no_grad()
    def prepare(self, compile_warmup=4, barrier=lambda: None):
        """Compile the forward (and for ``optimized`` capture it into a CUDA graph)."""
        import torch._dynamo
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
        mode = None if self.compile_mode == "default" else self.compile_mode
        if self.mode == "baseline":
            fn = torch.compile(self._baseline_forward, mode=mode) if self.compile_mode else self._baseline_forward
        elif self.compile_mode:
            block = torch.compile(_optimized_block, dynamic=False, mode=mode)
            patch = torch.compile(self._patch, dynamic=False, mode=mode)
            head = torch.compile(self._head, dynamic=False, mode=mode)
            def fn():
                return self._optimized_forward(block, patch, head)
        else:
            fn = self._optimized_forward
        for _ in range(compile_warmup):
            fn()
            torch.cuda.synchronize()
            barrier()
        self._fn = fn
        if self.mode == "optimized" and self.cuda_graph:
            self.graph, self.output = capture_graph(fn, before_capture=barrier)
            barrier()

    @torch.no_grad()
    def __call__(self):
        if self.graph is not None:
            self.graph.replay()
            return self.output
        return self._fn()
