"""The image head: a bidirectional transformer over the decision tokens' final hidden states."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


def sinusoid(t, dim):
    """[B] -> [B, dim]: sines and cosines of t at frequencies 1 ... 1/10000"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


class TimeMLP(nn.Module):
    """a scalar per image -> a vector of the backbone's width, added to every decision token's input embedding"""

    def __init__(self, hidden, dim=config.TIME_DIM):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, t):
        return self.net(sinusoid(t, self.dim))


class CondEmbed(nn.Module):
    """the masked share t of the random state -> the image head's condition vector [B, dim] (the other inputs are constants)"""

    def __init__(self, dim=config.COND_DIM, freq=config.TIME_DIM):
        super().__init__()
        self.freq = freq
        self.net = nn.Sequential(nn.Linear(5 * freq, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        zero, one = torch.zeros_like(t), torch.ones_like(t)
        return self.net(torch.cat([sinusoid(p, self.freq) for p in (t, zero, zero, zero, one)], 1))


class HeadBlock(nn.Module):
    """self-attention over all code positions + MLP, each modulated by the condition (adaLN-Zero)"""

    def __init__(self, width, heads, cdim, mlp_ratio=4):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.n2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, mlp_ratio * width), nn.GELU(approximate="tanh"), nn.Linear(mlp_ratio * width, width))
        self.ada = nn.Linear(cdim, 6 * width)

    def forward(self, x, c):
        b, n, w = x.shape
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(F.silu(c)).unsqueeze(1).chunk(6, dim=-1)
        h = self.n1(x) * (1 + sc1) + sh1
        q, k, v = self.qkv(h).view(b, n, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)                      # bidirectional: every position sees every other
        x = x + g1 * self.proj(a.transpose(1, 2).reshape(b, n, w))
        h = self.n2(x) * (1 + sc2) + sh2
        return x + g2 * self.mlp(h)


class ImageHead(nn.Module):
    """decision tokens' final hidden states [B, N, in_dim] + the random state [B, N, code_dim] + c
    -> (projection of u to the code space [B, N, code_dim], the features u [B, N, width])"""

    def __init__(self, in_dim, code_dim=config.CODE_DIM, n_tok=config.NUM_CODES, width=config.HEAD_WIDTH, depth=config.HEAD_DEPTH,
                 heads=config.HEAD_HEADS, cdim=config.COND_DIM):
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.zin = nn.Linear(code_dim, width)
        self.pos = nn.Parameter(torch.zeros(n_tok, width))
        self.blocks = nn.ModuleList([HeadBlock(width, heads, cdim) for _ in range(depth)])
        self.nf = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.ada_f = nn.Linear(cdim, 2 * width)
        self.out = nn.Linear(width, code_dim)

    def forward(self, hq, state, c):
        x = self.inp(hq) + self.zin(state) + self.pos
        for blk in self.blocks:
            x = blk(x, c)
        sh, sc = self.ada_f(F.silu(c)).unsqueeze(1).chunk(2, dim=-1)
        u = self.nf(x) * (1 + sc) + sh
        return self.out(u), u
