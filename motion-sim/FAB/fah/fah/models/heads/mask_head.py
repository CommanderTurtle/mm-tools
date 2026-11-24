"""Hand instance-mask head on the shared hand latent: the decoder's final image tokens are upsampled 4x
and fused with a shallow stride-4 RGB stem; the hand's final pose token is projected to a query and
dot-produced with the per-pixel embedding -> mask logits of *this crop's* hand at 1/4 crop resolution.
Channels: 0 = hand (wrist to fingertips; SAM-box / mesh convention), 1 = hand + visible arm (VISOR convention)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight, self.bias, self.eps = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c)), eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class HandMaskHead(nn.Module):
    def __init__(self, img_dim=1280, tok_dim=1024, dim=64, stem=32, channels=2):
        super().__init__()
        self.dim, self.channels = dim, channels
        self.img_proj = nn.Sequential(nn.Conv2d(img_dim, 256, 1), LayerNorm2d(256), nn.GELU())
        self.up = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, 2), LayerNorm2d(128), nn.GELU(),
                                nn.ConvTranspose2d(128, dim, 2, 2), nn.GELU())
        self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, 2, 1), nn.GELU(), nn.Conv2d(stem, stem, 3, 2, 1), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(dim + stem, dim, 3, padding=1), LayerNorm2d(dim), nn.GELU(),
                                  nn.Conv2d(dim, dim, 1))
        self.query = nn.Sequential(nn.Linear(tok_dim, tok_dim), nn.GELU(), nn.Linear(tok_dim, dim * channels))

    def forward(self, img_tokens, grid, pose_token, img):
        """img_tokens (B, h*w, C) final decoder image tokens; grid (h, w); pose_token (B, C_tok);
        img (B, 3, S, S) normalised crop. Returns logits (B, channels, 4h, 4w)."""
        B = img_tokens.shape[0]
        f = img_tokens.transpose(1, 2).reshape(B, -1, *grid).float()
        f = self.up(self.img_proj(f))
        s = self.stem(img.float())
        if s.shape[-2:] != f.shape[-2:]:
            s = F.interpolate(s, size=f.shape[-2:], mode="bilinear", align_corners=False)
        f = self.fuse(torch.cat([f, s], 1))
        q = self.query(pose_token.float()).view(B, self.channels, self.dim)
        return torch.einsum("bkc,bchw->bkhw", q, f) / math.sqrt(self.dim)


class HandVerifyHead(nn.Module):
    """Verification logits from the final pose token: [hand present in this crop, crop is correctly mirrored
    (i.e. the claimed handedness is right)]. Used to re-score low-confidence detections."""

    def __init__(self, tok_dim=1024, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(tok_dim, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, pose_token):
        return self.mlp(pose_token.float())
