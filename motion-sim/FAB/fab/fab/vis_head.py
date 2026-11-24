"""Per-keypoint visibility head: for each of the 70 MHR keypoints and the 14
evaluation joints, 3-class logits -- visible / occluded (by the body itself, another person or an object) / outside
the crop -- from the dense head's own features at the point (heatmap-pooled image features + its query, kept through
dense_head.keep_feats, as the contact head), its 2D and root-relative 3D, the decoder's pose token, and (attn_dim > 0)
a cross-attention of the point to the image tokens around it (who / what is in front). Labels: visibility
sidecars (train_fab.py --vis_labels vis; format in docs/training.md); trained on a frozen model (--vis_head --vis_only) or jointly.
"""

import torch
import torch.nn as nn

N_VIS_POINTS = 84  # MHR70 + J14, the dense head's first 84 points
VIS_CLASSES = ("visible", "occluded", "out")


class VisHead(nn.Module):
    def __init__(self, tok_dim, feat_dim, n_points=N_VIS_POINTS, hidden=256, attn_dim=128, img_dim=1280, human=False):
        """human: a second classifier on the same features for human-annotation semantics (COCO / WholeBody / Human-Art
        v-flags: visible = v 2, occluded = v 1, out = a labelled point outside the crop); forward then returns
        (rendered logits, human logits)."""
        super().__init__()
        self.register_buffer("points", torch.arange(n_points, dtype=torch.long))
        self.emb = nn.Parameter(torch.randn(n_points, 32) * 0.02)  # which point (the classes' priors differ per point)
        self.pt = nn.Sequential(nn.LayerNorm(feat_dim + 5 + 32), nn.Linear(feat_dim + 5 + 32, hidden), nn.GELU())
        self.tok = nn.Sequential(nn.LayerNorm(tok_dim), nn.Linear(tok_dim, hidden), nn.GELU())
        self.attn_dim = attn_dim
        if attn_dim:
            self.mem = nn.Linear(img_dim, attn_dim)
            self.mem_pos = nn.Linear(2, attn_dim)
            self.q = nn.Linear(hidden, attn_dim)
            self.q_pos = nn.Linear(2, attn_dim)
            self.attn = nn.MultiheadAttention(attn_dim, 4, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden * 2 + attn_dim, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.mlp_h = nn.Sequential(nn.Linear(hidden * 2 + attn_dim, hidden), nn.GELU(), nn.Linear(hidden, 3)) \
            if human else None

    def forward(self, token, feats, xyz, uv, img_tokens=None):
        """token (B, C_tok), feats (B, P, F) dense features of the points, xyz (B, P, 3) root-relative (m), uv (B, P, 2)
        crop units, img_tokens (B, C, h, w) -> logits (B, P, 3)."""
        B, P = feats.shape[:2]
        x = torch.cat([feats, xyz * 5.0, uv * 2.0 - 1.0, self.emb[None].expand(B, -1, -1)], -1)
        f = self.pt(x)
        parts = [f, self.tok(token)[:, None].expand(-1, P, -1)]
        if self.attn_dim:
            _, C, h, w = img_tokens.shape
            gy, gx = torch.meshgrid((torch.arange(h, device=f.device) + 0.5) / h,
                                    (torch.arange(w, device=f.device) + 0.5) / w, indexing="ij")
            mem = self.mem(img_tokens.float().flatten(2).transpose(1, 2)) + self.mem_pos(
                torch.stack([gx, gy], -1).view(1, -1, 2).float())
            a, _ = self.attn(self.q(f) + self.q_pos(uv), mem, mem, need_weights=False)
            parts.append(a)
        h = torch.cat(parts, -1)
        if self.mlp_h is not None:
            return self.mlp(h), self.mlp_h(h)
        return self.mlp(h)


def has_vis_head(weights):
    return any(k.startswith("vis_head.") for k in weights)


def vis_head_args(weights):
    """add_vis_head(**...) of a saved head."""
    return {"attn_dim": int(weights["vis_head.mem.weight"].shape[0]) if "vis_head.mem.weight" in weights else 0,
            "human": "vis_head.mlp_h.0.weight" in weights}
