#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Pixel decoders (Sec. 3.2 of the paper).

The decoder reads the DiT's pre-head hidden states of the frame's final
denoising step (plus the head's time modulation) and emits RGB directly,
replacing the VAE decoder. ``PixelDecoder`` is the causal ConvNeXt decoder of
FastFrames. It maps ``[B, F*H*W, dim]`` tokens on the ``(F, 30, 52)`` DiT grid to
``[B, 4F, 3, 480, 832]`` pixels in ``[-1, 1]``; callers drop the first three
frames (``align_frames``) to obtain the 81 output frames.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ..wan.model import WanLayerNorm

__all__ = ["PixelDecoder", "BridgedPixelDecoder", "align_frames"]


def unpatchify_tokens(tokens, grid_sizes, patch_size, out_dim):
    """``[B, F, H*W, prod(patch)*C]`` -> ``[B, F*p_t, C, H*p_h, W*p_w]`` (Wan layout)."""
    p_t, p_h, p_w = patch_size
    out = []
    for item, (f, h, w) in zip(tokens, grid_sizes.tolist()):
        u = torch.einsum("fhwpqrc->cfphqwr", item.reshape(f, h, w, p_t, p_h, p_w, out_dim))
        out.append(u.reshape(out_dim, f * p_t, h * p_h, w * p_w))
    return torch.stack(out).permute(0, 2, 1, 3, 4).contiguous()


def align_frames(pixels: torch.Tensor, num_frames: int = 81) -> torch.Tensor:
    """Drop leading frames so that ``pixels`` has ``num_frames`` frames (84 -> 81)."""
    if pixels.shape[1] < num_frames:
        raise ValueError(f"decoder produced {pixels.shape[1]} < {num_frames} frames")
    return pixels[:, pixels.shape[1] - num_frames:]


class _CausalConv3d(nn.Conv3d):
    """Conv3d with left-only temporal padding and symmetric spatial padding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pad_t, pad_h, pad_w = self.padding
        self._pad = (pad_w, pad_w, pad_h, pad_h, 2 * pad_t, 0)
        self.padding = (0, 0, 0)

    def forward(self, x):
        return super().forward(F.pad(x, self._pad))


class _ChannelLayerNorm3d(nn.Module):
    """LayerNorm over the channels of a ``[B, C, T, H, W]`` tensor."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, x):
        x = x.movedim(1, -1)
        x = F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self.eps)
        return x.movedim(-1, 1).contiguous()


class _CausalResDWBlock3d(nn.Module):
    """Causal depthwise conv -> LN -> pointwise MLP, zero-initialised residual."""

    def __init__(self, dim: int, kernel: Tuple[int, int, int] = (3, 3, 3), ffn_mult: int = 4):
        super().__init__()
        self.dw = _CausalConv3d(dim, dim, kernel_size=kernel, padding=tuple(k // 2 for k in kernel), groups=dim)
        self.norm = _ChannelLayerNorm3d(dim)
        self.pw1 = nn.Conv3d(dim, dim * ffn_mult, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv3d(dim * ffn_mult, dim, kernel_size=1)
        nn.init.zeros_(self.pw2.weight)
        nn.init.zeros_(self.pw2.bias)

    def forward(self, x):
        return x + self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))


class PixelDecoder(nn.Module):
    """AdaLN -> channel LN + 1x1 mixer -> causal ConvNeXt blocks -> linear unpatchify."""

    def __init__(self, dim: int = 1536, pixel_patch_size: Tuple[int, int, int] = (4, 16, 16),
                 out_dim: int = 3, n_low_blocks: int = 2, n_high_blocks: int = 1,
                 ffn_mult: int = 4, eps: float = 1e-6):
        super().__init__()
        self.dim, self.out_dim = int(dim), int(out_dim)
        self.patch_size = tuple(pixel_patch_size)
        self.gradient_checkpointing = False
        self.norm = WanLayerNorm(dim, eps)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        self.entry_norm = _ChannelLayerNorm3d(dim)
        self.entry_pw = nn.Conv3d(dim, dim, kernel_size=1)
        self.conv_low = nn.Sequential(*[_CausalResDWBlock3d(dim, ffn_mult=ffn_mult) for _ in range(n_low_blocks)])
        self.conv_high = nn.Sequential(*[_CausalResDWBlock3d(dim, ffn_mult=ffn_mult) for _ in range(n_high_blocks)])
        self.fc_out = nn.Linear(dim, math.prod(self.patch_size) * self.out_dim)
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def _run_stack(self, stack, value):
        if self.gradient_checkpointing and torch.is_grad_enabled():
            for block in stack:
                value = torch.utils.checkpoint.checkpoint(block, value, use_reentrant=False)
            return value
        return stack(value)

    def forward(self, x, e_mod, grid_sizes):
        """x: ``[B, F*L, dim]``; e_mod: ``[B, F, 1, dim]``; grid_sizes: ``[B, 3]`` token grid."""
        num_frames, frame_tokens = e_mod.shape[1], x.shape[1] // e_mod.shape[1]
        shift, scale = (self.modulation.unsqueeze(1) + e_mod).chunk(2, dim=2)
        x = self.norm(x).unflatten(1, (num_frames, frame_tokens)) * (1 + scale) + shift
        f, h, w = grid_sizes[0].tolist()
        value = x.reshape(x.shape[0], f, h, w, self.dim).permute(0, 4, 1, 2, 3).contiguous()
        value = self.entry_pw(self.entry_norm(value))
        value = self._run_stack(self.conv_low, value)
        value = self._run_stack(self.conv_high, value)
        b, c, f, h, w = value.shape
        tokens = self.fc_out(value.permute(0, 2, 3, 4, 1).contiguous().view(b, f, h * w, c))
        grid = torch.tensor([[f, h, w]] * b, dtype=torch.long, device=tokens.device)
        return unpatchify_tokens(tokens, grid, self.patch_size, self.out_dim)


class BridgedPixelDecoder(nn.Module):
    """A bias-free projection of hidden states and modulation into a narrower decoder.

    FF-L uses it to decode Wan2.1-14B features (5120) with the 1536-wide decoder.
    """

    def __init__(self, in_dim: int = 5120, dim: int = 1536, bridge_seed: int = 20260821):
        super().__init__()
        self.bridge = nn.Linear(in_dim, dim, bias=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(bridge_seed))
            nn.init.normal_(self.bridge.weight, std=in_dim**-0.5)
        self.pixel_head = PixelDecoder(dim)

    def forward(self, x, e_mod, grid_sizes):
        return self.pixel_head(self.bridge(x), self.bridge(e_mod), grid_sizes)
