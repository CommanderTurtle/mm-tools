#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""KV refinement network (Sec. 3.3 of the paper).

After the final denoising step of a frame, the DiT has written *noisy* keys and
values into the cache slot of that frame. Instead of a second, clean DiT pass
over the frame, this network maps (noisy K/V, clean-latent features) to refined
K/V and overwrites the slot:

  * a shared per-token stem ``phi = E(patch_embed(x0))`` (one small transformer
    over the frame's tokens);
  * one light head per DiT layer:
    ``K' = renorm(K + dK)`` (post-RoPE, per-head norm preserved) and
    ``V' = (V + dV) * exp(s_V)``.

All residual branches are zero-initialised, so an untrained network is the
identity (the no-refinement baseline). The heads of all layers are evaluated as
batched ``baddbmm`` GEMMs, which keeps the network launch-efficient.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

__all__ = ["KVRefinementNetwork"]


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    half = dim // 2
    position = position.type(torch.float64)
    div = torch.arange(half, dtype=torch.float64, device=position.device).div(half)
    sinusoid = torch.outer(position, torch.pow(10000.0, -div))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).float()


def _sdpa(q, k, v):
    """``[B, H, L, D]`` attention; cuDNN for half precision on CUDA, math kernel otherwise."""
    if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v)
    return F.scaled_dot_product_attention(q, k, v)


class TimestepEmbedder(nn.Module):

    def __init__(self, d_a: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, d_a), nn.SiLU(), nn.Linear(d_a, d_a))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_embedding_1d(self.freq_dim, t.flatten().float())
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class Modulation(nn.Module):
    """adaLN: ``e [B, d]`` -> ``n`` vectors ``[B, 1, d]``; zero-initialised."""

    def __init__(self, d_a: int, n: int):
        super().__init__()
        self.n = n
        self.lin = nn.Linear(d_a, n * d_a)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, e: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.lin(F.silu(e)).unsqueeze(1).chunk(self.n, dim=-1)


class StemBlock(nn.Module):
    """Pre-norm spatial transformer block with adaLN and zero-initialised gates."""

    def __init__(self, d_a: int, n_heads: int = 8, mlp_mult: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_a // n_heads
        self.norm1 = nn.LayerNorm(d_a, elementwise_affine=False)
        self.qkv = nn.Linear(d_a, 3 * d_a)
        self.proj = nn.Linear(d_a, d_a)
        self.norm2 = nn.LayerNorm(d_a, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_a, d_a * mlp_mult), nn.GELU(approximate="tanh"), nn.Linear(d_a * mlp_mult, d_a))
        self.mod = Modulation(d_a, 6)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        sh1, sc1, g1, sh2, sc2, g2 = self.mod(e)
        h = self.norm1(x) * (1 + sc1) + sh1
        q, k, v = self.qkv(h).reshape(b, s, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        x = x + g1 * self.proj(_sdpa(q, k, v).transpose(1, 2).reshape(b, s, -1))
        h = self.norm2(x) * (1 + sc2) + sh2
        return x + g2 * self.mlp(h)


class GroupGEMM(nn.Module):
    """Refinement heads for a group of layers sharing one width ``(d_v, d_k)``.

    Weights of all members are stacked on a leading dimension; modulation and the
    V log-scale remain per layer.
    """

    def __init__(self, num_members: int, dim: int, d_a: int, d_v: int, d_k: int,
                 num_heads: int, head_dim: int, mlp_mult: int = 2):
        super().__init__()
        self.dim, self.d_a, self.d_v, self.d_k = dim, d_a, d_v, d_k
        self.num_heads, self.head_dim = num_heads, head_dim
        n = int(num_members)

        def weight(i, o, zero=False):
            return nn.Parameter(torch.zeros(n, i, o) if zero else torch.randn(n, i, o) * (i ** -0.5))

        def bias(o):
            return nn.Parameter(torch.zeros(n, 1, o))

        self.proj_k_w, self.proj_k_b = weight(dim, d_k), bias(d_k)
        self.proj_v_w, self.proj_v_b = weight(dim, d_v), bias(d_v)
        self.fuse_w, self.fuse_b = weight(d_k + 2 * d_v, d_v), bias(d_v)
        self.m0_w, self.m0_b = weight(d_v, d_v * mlp_mult), bias(d_v * mlp_mult)
        self.m2_w, self.m2_b = weight(d_v * mlp_mult, d_v), bias(d_v)
        self.mod_w, self.mod_b = weight(d_a, 2 * d_v, zero=True), bias(2 * d_v)
        self.dK_w, self.dK_b = weight(d_v, dim, zero=True), bias(dim)
        self.dV_w, self.dV_b = weight(d_v, dim, zero=True), bias(dim)
        self.v_logscale = nn.Parameter(torch.zeros(n, 1, dim))

    def _renorm_per_head(self, x, ref):
        b, s, _ = x.shape
        xh = x.reshape(b, s, self.num_heads, self.head_dim)
        rh = ref.reshape(b, s, self.num_heads, self.head_dim)
        xh = xh * (rh.norm(dim=-1, keepdim=True) / (xh.norm(dim=-1, keepdim=True) + 1e-6))
        return xh.reshape(b, s, -1)

    def forward(self, nK, nV, phi_v, e, layers=slice(None)):
        """nK, nV: ``[M, S, dim]``; phi_v: ``[M, S, d_v]``; e: ``[*, d_a]``.

        ``layers`` selects the ``M`` members the inputs belong to (default: all), e.g. one
        rank's share of the layers when the refinement is split across GPUs.
        """
        b = nK.shape[0]
        pk = torch.baddbmm(self.proj_k_b[layers], nK, self.proj_k_w[layers])
        pv = torch.baddbmm(self.proj_v_b[layers], nV, self.proj_v_w[layers])
        h = torch.baddbmm(self.fuse_b[layers], torch.cat([pk, pv, phi_v], dim=-1), self.fuse_w[layers])
        e1 = F.silu(e).reshape(1, 1, self.d_a).expand(b, 1, self.d_a).contiguous()
        shift, scale = torch.baddbmm(self.mod_b[layers], e1, self.mod_w[layers]).chunk(2, dim=-1)
        h = F.layer_norm(h, (self.d_v,)) * (1 + scale) + shift
        h = torch.baddbmm(self.m0_b[layers], h, self.m0_w[layers])
        h = torch.baddbmm(self.m2_b[layers], F.gelu(h, approximate="tanh"), self.m2_w[layers])
        k = self._renorm_per_head(nK + torch.baddbmm(self.dK_b[layers], h, self.dK_w[layers]), ref=nK)
        v = (nV + torch.baddbmm(self.dV_b[layers], h, self.dV_w[layers])) * torch.exp(self.v_logscale[layers])
        return k, v


class GroupedKVRepairHeads(nn.Module):
    """The refinement heads of all layers, evaluated together as one stacked ``GroupGEMM``.

    ``gemms`` and ``phi_proj`` keep the layout of the released state dicts
    (``heads_grouped.gemms.0.*``, ``heads_grouped.phi_proj.<d_a>.*``).
    """

    def __init__(self, num_blocks: int, dim: int, d_a: int, num_heads: int, head_dim: int, mlp_mult: int = 2):
        super().__init__()
        self.num_blocks = num_blocks
        self.gemms = nn.ModuleList([GroupGEMM(num_blocks, dim, d_a, d_a, d_a, num_heads, head_dim, mlp_mult)])
        self.phi_proj = nn.ModuleDict({str(d_a): nn.Linear(d_a, d_a, bias=False)})

    def forward(self, nK_all, nV_all, phi, e):
        """nK_all, nV_all: ``[L, S, dim]``; phi: ``[1, S, d_a]`` or ``[S, d_a]``."""
        phi2 = phi[0] if phi.dim() == 3 else phi
        gemm = self.gemms[0]
        phi_v = self.phi_proj[str(gemm.d_v)](phi2).unsqueeze(0).expand(self.num_blocks, -1, -1)
        return gemm(nK_all, nV_all, phi_v, e)


class KVRefinementNetwork(nn.Module):
    """Shared clean-latent stem plus one refinement head per DiT layer.

    Per frame::

        e = net.embed_timestep(t_ref, 1, device, dtype)
        phi = net.compute_phi(patch_embed(x0), e)
        K, V = net.repair_all(K_noisy, V_noisy, phi, e)   # [layers, tokens, dim]
    """

    def __init__(self, dim: int = 1536, num_heads: int = 12, head_dim: int = 128, num_blocks: int = 30,
                 d_a: int = 768, n_stem: int = 4, stem_n_heads: int = 8,
                 stem_mlp_mult: int = 4, head_mlp_mult: int = 2, freq_dim: int = 256):
        super().__init__()
        assert dim == num_heads * head_dim, (dim, num_heads, head_dim)
        self.dim, self.num_heads, self.head_dim = dim, num_heads, head_dim
        self.num_blocks, self.d_a, self.n_stem = num_blocks, d_a, n_stem
        self.t_embed = TimestepEmbedder(d_a, freq_dim)
        self.stem_in = nn.Linear(dim, d_a)
        self.stem_blocks = nn.ModuleList([StemBlock(d_a, stem_n_heads, stem_mlp_mult) for _ in range(n_stem)])
        self.stem_norm = nn.LayerNorm(d_a, elementwise_affine=True)
        self.heads_grouped = GroupedKVRepairHeads(num_blocks, dim, d_a, num_heads, head_dim, head_mlp_mult)

    def embed_timestep(self, timestep, batch_size: int, device, dtype) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([float(timestep)], device=device)
        timestep = timestep.to(device).float().reshape(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(batch_size)
        return self.t_embed(timestep).to(dtype)

    def compute_phi(self, stem_tokens: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """stem_tokens: ``[B, S, dim]`` patch-embedded clean latent -> ``[B, S, d_a]``."""
        x = self.stem_in(stem_tokens)
        for block in self.stem_blocks:
            x = block(x, e)
        return self.stem_norm(x)

    def repair_all(self, nK_all, nV_all, phi, e):
        return self.heads_grouped(nK_all, nV_all, phi, e)

    @property
    def repaired_block_indices(self) -> List[int]:
        return list(range(self.num_blocks))
