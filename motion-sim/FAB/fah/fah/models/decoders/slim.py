"""CPU-friendly replacement for the 6-layer, 1024-wide promptable hand decoder.

Q learned queries (conditioned on the CLIFF camera condition) cross-attend to the ray-conditioned image
tokens in a small transformer; query 0 is projected to the decoder's 1024-d pose-token space and fed to the
unchanged MHR hand head / camera head (and the mask / verify heads), so every output keeps its format.
Trained by distillation from the full decoder + the same GT losses (train/finetune.py --slim_decoder).
"""
import math

import torch
import torch.nn as nn


def sincos_2d(h, w, dim, device):
    y, x = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                          torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    omega = 1.0 / (10000 ** (torch.arange(dim // 4, device=device, dtype=torch.float32) / (dim // 4)))
    out = [torch.sin(x.flatten()[:, None] * omega), torch.cos(x.flatten()[:, None] * omega),
           torch.sin(y.flatten()[:, None] * omega), torch.cos(y.flatten()[:, None] * omega)]
    return torch.cat(out, 1)[None]  # (1, h*w, dim)


class SlimHandDecoder(nn.Module):
    def __init__(self, img_dim=1280, dim=384, layers=2, heads=6, queries=4, tok_dim=1024, cond_dim=3, kp_queries=0):
        super().__init__()
        self.dim = dim
        # optional per-joint queries (spatially grounded like the full decoder's keypoint tokens); they feed
        # the pose query through self-attention and get a 2D keypoint auxiliary output
        self.kp = nn.Embedding(kp_queries, dim) if kp_queries else None
        self.kp_head = nn.Linear(dim, 2) if kp_queries else None
        queries = max(queries, 3)
        self.img_proj = nn.Sequential(nn.LayerNorm(img_dim), nn.Linear(img_dim, dim))
        self.queries = nn.Embedding(queries, dim)
        self.cond = nn.Linear(cond_dim, dim)
        layer = nn.TransformerDecoderLayer(dim, heads, dim * 4, dropout=0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(dim)
        self.to_token = nn.Linear(dim, tok_dim)
        self.last_image_embedding = self.last_tokens = None

    def forward(self, image_embeddings, condition_info):
        """image_embeddings (B, C, h, w) ray-conditioned; condition_info (B, 3). Returns (B, tok_dim) pose token."""
        B, C, h, w = image_embeddings.shape
        mem = self.img_proj(image_embeddings.flatten(2).transpose(1, 2).float())
        mem = mem + sincos_2d(h, w, self.dim, mem.device)
        q = self.queries.weight[None].expand(B, -1, -1)
        if self.kp is not None:
            q = torch.cat([q, self.kp.weight[None].expand(B, -1, -1)], 1)
        q = q + self.cond(condition_info.float())[:, None]
        out = self.norm(self.decoder(q, mem))
        nq = self.queries.num_embeddings
        self.kp_uv = self.kp_head(out[:, nq:]).sigmoid() if self.kp is not None else None  # crop-normalised 2D
        tok = self.to_token(out[:, :nq])
        # for the mask / verify heads (image tokens at the slim width, final pose token)
        self.last_image_embedding, self.last_tokens = mem, tok
        return tok[:, 0]
