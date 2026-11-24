# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modifications Copyright (C) 2026 Apple Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see ACKNOWLEDGEMENTS.
# Modified for FastFrames (shared layers and the bidirectional Wan2.1 backbone); the frame-offset
# RoPE ``causal_rope_apply`` is adapted from Self-Forcing (Apache-2.0).
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .attention import attention, cross_attention

__all__ = [
    "WanConfig",
    "WAN_CONFIGS",
    "WanModel",
    "WanRMSNorm",
    "WanLayerNorm",
    "WanCrossAttention",
    "sinusoidal_embedding_1d",
    "rope_params",
    "rope_apply",
    "causal_rope_apply",
]


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    return torch.polar(torch.ones_like(freqs), freqs)


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """3-D RoPE on ``[B, L, H, D]`` tokens whose first frame is ``start_frame``."""
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        output.append(torch.cat([x_i, x[i, seq_len:]]))
    return torch.stack(output).type_as(x)


def rope_apply(x, grid_sizes, freqs):
    return causal_rope_apply(x, grid_sizes, freqs, start_frame=0)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def qkv(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    def forward(self, x, grid_sizes, freqs):
        q, k, v = self.qkv(x)
        x = attention(rope_apply(q, grid_sizes, freqs), rope_apply(k, grid_sizes, freqs), v)
        return self.o(x.flatten(2))


class WanCrossAttention(WanSelfAttention):
    """Text cross-attention. ``cache`` optionally holds the projected text K/V."""

    def text_kv(self, context):
        b, n, d = context.size(0), self.num_heads, self.head_dim
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        return k, v

    def forward(self, x, context, cache=None):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k, v = (cache["k"], cache["v"]) if cache is not None else self.text_kv(context)
        return self.o(cross_attention(q, k, v).flatten(2))


class WanAttentionBlock(nn.Module):

    def __init__(self, dim, ffn_dim, num_heads, qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, grid_sizes, freqs, context):
        e = (self.modulation + e).chunk(6, dim=1)
        y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], grid_sizes, freqs)
        x = x + y * e[2]
        x = x + self.cross_attn(self.norm3(x), context)
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        return x + y * e[5]


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + e[1]) + e[0])


def unpatchify(x, grid_sizes, patch_size, out_dim):
    """``[B, L, prod(patch) * C]`` tokens -> list of ``[C, F, H, W]`` videos."""
    out = []
    for u, v in zip(x, grid_sizes.tolist()):
        u = u[:math.prod(v)].view(*v, *patch_size, out_dim)
        u = torch.einsum("fhwpqrc->cfphqwr", u)
        out.append(u.reshape(out_dim, *[i * j for i, j in zip(v, patch_size)]))
    return out


def init_wan_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    nn.init.xavier_uniform_(model.patch_embedding.weight.flatten(1))
    for m in model.text_embedding.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
    for m in model.time_embedding.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
    nn.init.zeros_(model.head.head.weight)


@dataclass(frozen=True)
class WanConfig:
    dim: int
    ffn_dim: int
    num_heads: int
    num_layers: int
    patch_size: tuple = (1, 2, 2)
    text_len: int = 512
    in_dim: int = 16
    freq_dim: int = 256
    text_dim: int = 4096
    out_dim: int = 16
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6


WAN_CONFIGS = {
    "Wan2.1-T2V-1.3B": WanConfig(dim=1536, ffn_dim=8960, num_heads=12, num_layers=30),
    "Wan2.1-T2V-14B": WanConfig(dim=5120, ffn_dim=13824, num_heads=40, num_layers=40),
}


class WanBackbone(nn.Module):
    """Embeddings and hyper-parameters shared by the bidirectional and causal models."""

    block_class = None
    head_class = Head

    def __init__(self, config: WanConfig):
        super().__init__()
        self.config = config
        dim = config.dim
        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.dim = dim
        self.freq_dim = config.freq_dim
        self.out_dim = config.out_dim
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.patch_embedding = nn.Conv3d(
            config.in_dim, dim, kernel_size=config.patch_size, stride=config.patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(config.text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(config.freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            self.block_class(dim, config.ffn_dim, config.num_heads, config.qk_norm,
                             config.cross_attn_norm, config.eps)
            for _ in range(config.num_layers)
        ])
        self.head = self.head_class(dim, config.out_dim, config.patch_size, config.eps)
        self.freqs = self.rope_table()
        init_wan_weights(self)
        self.gradient_checkpointing = False

    def rope_table(self):
        """Complex128 RoPE table; deliberately not a buffer so ``.to(dtype)`` keeps it."""
        d = self.dim // self.num_heads
        with torch.device("cpu"):
            return torch.cat([
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ], dim=1)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def embed_text(self, prompt_embeds):
        """Zero-pad T5 embeddings to ``text_len`` tokens and project them."""
        return self.text_embedding(torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in prompt_embeds
        ]))

    def unpatchify(self, x, grid_sizes):
        return unpatchify(x, grid_sizes, self.patch_size, self.out_dim)

    def _run_block(self, block, *args, **kwargs):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(block, *args, use_reentrant=False, **kwargs)
        return block(*args, **kwargs)


class WanModel(WanBackbone):
    """Bidirectional Wan2.1 T2V backbone (teacher / score networks during training)."""

    block_class = WanAttentionBlock

    def forward(self, x, t, context, seq_len):
        """x: ``[B, C, F, H, W]``; t: ``[B]``; context: T5 embeddings ``[B, L, 4096]``."""
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        assert max(u.size(1) for u in x) <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        context = self.embed_text(context)
        for block in self.blocks:
            x = self._run_block(block, x, e0, grid_sizes, self.freqs, context)
        x = self.head(x, e)
        return torch.stack(self.unpatchify(x, grid_sizes))
