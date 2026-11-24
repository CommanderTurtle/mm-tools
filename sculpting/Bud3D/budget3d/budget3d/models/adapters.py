"""The modules Budget3D adds to Idol's denoiser.

Four additions, each gated so that it contributes exactly nothing at initialisation:

  ScaleMixer3D    a cascade that pools 2 Sobol points into a 16^3 cell and 8 cells into an
                  8^3 cube, so a merged token has an embedding without re-running anything
                  upstream.  The pooled path is a plain mean; the learned path ends in a
                  zero-initialised Linear.
  MipNeRFIPE3D    an integrated positional encoding of the cube a coarse token covers:
                  the sinusoids of its centre, damped by the cube's variance, so a token
                  that owns more volume gets a smoother position.  Ends zero-initialised.
  LatentRefine3D  the inverse of the merge: one coarse token's single 16-channel prediction
                  is turned back into one prediction per Sobol point it owns, by a small
                  AdaLN transformer over the token's own children.  Ends zero-initialised.
  bp_feat_proj    a training-only head that predicts the teacher's per-point trunk feature
                  from the refiner's hidden state, for the feature-distillation term.

Together they are ~8 M parameters on a 370 M backbone.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero(linear):
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


def _trunc(shape, std=0.02):
    return nn.Parameter(nn.init.trunc_normal_(torch.empty(*shape), std=std))


def fourier3d(pos, octaves=4):
    """pos [..., 3] -> [..., 6 * octaves] sin/cos features (24 dims at 4 octaves)."""
    w = (2.0 ** torch.arange(octaves, device=pos.device, dtype=pos.dtype)) * math.pi
    ang = pos.unsqueeze(-1) * w
    return torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (h * self.weight).type_as(x)


# ---------------------------------------------------------------------------------------
# (a) multi-scale embedding
# ---------------------------------------------------------------------------------------
class ScaleMixer3D(nn.Module):
    """n_sub child token features -> one parent token feature.

    At initialisation the learned branch is zero, so the parent is exactly the mean of its
    children's pretrained embeddings.
    """

    def __init__(self, dim, n_sub, bott=256):
        super().__init__()
        self.n_sub = n_sub
        self.pos = _trunc((n_sub, dim))
        self.down = nn.Linear(dim, bott)
        self.spatial_w = nn.Parameter(torch.eye(n_sub).expand(bott, n_sub, n_sub).contiguous()
                                      + 0.02 * torch.randn(bott, n_sub, n_sub))
        self.spatial_b = nn.Parameter(torch.zeros(n_sub, bott))
        self.up = _zero(nn.Linear(n_sub * bott, dim))

    def forward(self, sub):
        """sub [..., n_sub, D] -> [..., D]."""
        h = F.gelu(self.down(sub + self.pos))
        h = torch.einsum("...sc,cts->...tc", h, self.spatial_w) + self.spatial_b
        return sub.mean(-2) + self.up(F.gelu(h).flatten(-2))


# ---------------------------------------------------------------------------------------
# (c) position and scale
# ---------------------------------------------------------------------------------------
class MipNeRFIPE3D(nn.Module):
    """Integrated positional encoding of an axis-aligned cube, in Idol's [0, 1]^3 frame.

    The cube centre is mapped to mu = (2c - 1) * pi, and each octave is damped by the
    cube's variance.  The second pair of feature groups uses expm1 of the *excess* variance
    over the s=2 cell, so an s=2 token -- the finest merged scale -- contributes nothing
    there and the encoding grows smoothly with the token's volume.
    """

    def __init__(self, dim, bott=256, octaves=8, base_scale=2, ipe_scale=1.0):
        super().__init__()
        self.octaves, self.ipe_scale = octaves, ipe_scale
        self.base_var = ipe_scale * (2 * math.pi / 32 * base_scale) ** 2 / 12
        self.register_buffer("omega", 2.0 ** torch.arange(octaves, dtype=torch.float32), persistent=False)
        self.mlp = nn.Sequential(nn.Linear(12 * octaves, bott), nn.SiLU(), _zero(nn.Linear(bott, dim)))

    def forward(self, centres, scale):
        """centres [..., 3] in [0, 1]^3, scale in {2, 4} -> [..., dim]."""
        mu = (2 * centres.float() - 1) * math.pi
        var_s = self.ipe_scale * (2 * math.pi * scale / 32) ** 2 / 12
        w = self.omega
        ang = mu.unsqueeze(-1) * w
        damp = torch.exp(-0.5 * w ** 2 * var_s)
        excess = torch.expm1(-0.5 * w ** 2 * max(var_s - self.base_var, 0.0))
        feats = torch.cat([ang.sin() * damp, ang.cos() * damp,
                           ang.sin() * excess, ang.cos() * excess], dim=-1).flatten(-2)
        return self.mlp(feats)


# ---------------------------------------------------------------------------------------
# (b) token-local refinement
# ---------------------------------------------------------------------------------------
class RefineBlock(nn.Module):
    """AdaLN-Zero self-attention over one parent token's children, conditioned on that token.

    Both residual branches are gated by the zero-initialised ``ada`` projection, and their
    output projections are zero too, so the block is the identity at initialisation.
    """

    def __init__(self, width, heads, cond_dim, mlp_ratio=4.0):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = _zero(nn.Linear(width, width))
        inner = int(width * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(width, inner), nn.GELU(approximate="tanh"),
                                 _zero(nn.Linear(inner, width)))
        self.ada = _zero(nn.Linear(cond_dim, 6 * width))

    def forward(self, x, cond):
        """x [M, n, W], cond [M, C] -> [M, n, W]."""
        m, n, w = x.shape
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(F.silu(cond)).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1) + shift1
        q, k, v = self.qkv(h).reshape(m, n, 3, self.heads, self.head_dim).unbind(2)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = x + gate1 * self.proj(a.transpose(1, 2).reshape(m, n, w))
        h = self.norm2(x) * (1 + scale2) + shift2
        return x + gate2 * self.mlp(h)


class RefineStage(nn.Module):
    """One parent latent -> ``n_ratio`` child latents (a cube's cells, or a cell's points).

    The child prediction starts as a nearest-repeat of the parent -- what the released
    model would have produced across the whole merged region -- and the transformer adds a
    correction through a zero-initialised output projection.  Its inputs are the repeated
    parent latent, a zero-initialised U-Net skip from the noisy state z_t of each child,
    the child's position relative to the parent cube in [-1, 1]^3, a learned child slot,
    and, in the second stage of the s=4 cascade, the hidden state carried over from the
    first stage.
    """

    def __init__(self, n_ratio, width=128, depth=2, heads=4, cond_dim=1024, lat_ch=16, zt_skip=True):
        super().__init__()
        self.n_ratio, self.zt_skip = n_ratio, zt_skip
        self.bf16_blocks = True
        self.embed = nn.Linear(lat_ch, width)
        self.embed_zt = _zero(nn.Linear(lat_ch, width))
        self.pos_proj = nn.Linear(24, width)
        self.slots = _trunc((n_ratio, width))
        self.blocks = nn.ModuleList([RefineBlock(width, heads, cond_dim) for _ in range(depth)])
        self.out_norm = RMSNorm(width)
        self.out = _zero(nn.Linear(width, lat_ch))

    def forward(self, lat_parent, cond, rel_pos, z_stage=None, carry=None):
        """lat_parent [M, lat_ch], cond [M, C], rel_pos [n_ratio, 3],
        z_stage [M, n_ratio, lat_ch], carry [M, W] -> (child latents, hidden)."""
        rep = lat_parent.unsqueeze(1).expand(-1, self.n_ratio, -1)
        h = self.embed(rep) + self.slots + self.pos_proj(fourier3d(rel_pos))
        if self.zt_skip and z_stage is not None:
            h = h + self.embed_zt(z_stage)
        if carry is not None:
            h = h + carry.unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=self.bf16_blocks and torch.cuda.is_available()):
            for blk in self.blocks:
                h = blk(h, cond)
        h = h.float()
        return rep + self.out(self.out_norm(h)), h


def wiener_hf_gain(t, rho=0.25):
    """The analytical high-frequency gain w_hf(t) = t / max(t^2 + rho^2 (1 - t)^2, 1e-6).

    A merged token predicts one velocity for the whole cube, which cannot carry the detail
    that is already visible in z_t itself.  Adding this multiple of the within-cube
    residual of z_t restores it with no parameters: the gain is the Wiener-optimal weight
    for a signal whose high band is pure noise at t = 1 and pure data at t = 0.
    """
    t = torch.as_tensor(t, dtype=torch.float32)
    return t / torch.clamp(t ** 2 + rho ** 2 * (1 - t) ** 2, min=1e-6)
