"""Explicit contact head (FAB-L / FAB-S): per-crop contact probabilities of the left / right
heel and toe, hands and knees (fab/contact_regions.CONTACT_NAMES) from the decoder's pose token and the dense head's
features at the foot / knee / wrist keypoints (the heatmap-pooled image features + query, their 2D and root-relative 3D),
plus the body's camera translation. Trained on a frozen model (train_fab.py --contact_head --contact_only), so the pose
outputs are unchanged; `contact_feat` (the last hidden layer) feeds the video contact model.
"""

import torch
import torch.nn as nn

# MHR70: ankles, big / small toes, heels (13-20), knees (11, 12), wrists (62 left, 41 right)
CONTACT_POINTS = [13, 14, 15, 16, 17, 18, 19, 20, 11, 12, 62, 41]
N_CONTACT = 8


class ContactHead(nn.Module):
    """attn_dim > 0: each keypoint's feature also cross-attends to the image tokens (positions: the keypoint's 2D, each
    token's grid cell), for ground cues around the feet (floor, shadows) beyond the heatmap-pooled features."""

    def __init__(self, tok_dim, feat_dim, n_out=N_CONTACT, points=CONTACT_POINTS, hidden=512, attn_dim=0, img_dim=1280):
        super().__init__()
        self.register_buffer("points", torch.as_tensor(points, dtype=torch.long))
        self.pt = nn.Sequential(nn.LayerNorm(feat_dim + 5), nn.Linear(feat_dim + 5, 128), nn.GELU())
        self.tok = nn.Sequential(nn.LayerNorm(tok_dim), nn.Linear(tok_dim, 256), nn.GELU())
        self.attn_dim = attn_dim
        if attn_dim:
            self.mem = nn.Linear(img_dim, attn_dim)
            self.mem_pos = nn.Linear(2, attn_dim)
            self.q = nn.Linear(128, attn_dim)
            self.q_pos = nn.Linear(2, attn_dim)
            self.attn = nn.MultiheadAttention(attn_dim, 4, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(len(points) * (128 + attn_dim) + 256 + 3, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.out = nn.Linear(hidden, n_out)

    def forward(self, token, feats, xyz, uv, cam_t, img_tokens=None):
        """token (B, C_tok), feats (B, P, F) dense features of `points`, xyz (B, P, 3) root-relative m, uv (B, P, 2) crop
        units, cam_t (B, 3) m, img_tokens (B, C, h, w) (attn_dim > 0) -> logits (B, n_out), hidden (B, hidden)."""
        x = torch.cat([feats, xyz * 5.0, uv * 2.0 - 1.0], -1)
        z = cam_t[:, 2:3].clamp_min(0.3)
        c = torch.cat([cam_t[:, :2] / z, torch.log(z)], -1)
        f = self.pt(x)
        parts = [f.flatten(1)]
        if self.attn_dim:
            B, C, h, w = img_tokens.shape
            gy, gx = torch.meshgrid((torch.arange(h, device=f.device) + 0.5) / h,
                                    (torch.arange(w, device=f.device) + 0.5) / w, indexing="ij")
            mem = self.mem(img_tokens.float().flatten(2).transpose(1, 2)) + self.mem_pos(
                torch.stack([gx, gy], -1).view(1, -1, 2).float())
            a, _ = self.attn(self.q(f) + self.q_pos(uv), mem, mem, need_weights=False)
            parts.append(a.flatten(1))
        h = self.mlp(torch.cat(parts + [self.tok(token), c], -1))
        return self.out(h), h


def has_contact_head(weights):
    return any(k.startswith("contact_head.") for k in weights)


def contact_attn_dim(weights):
    """Attention width of a saved contact head (0: none)."""
    w = weights.get("contact_head.q.weight")
    return int(w.shape[0]) if w is not None else 0


def contact_head_args(weights):
    """add_contact_head kwargs of a saved head: attention width, outputs (8 static, or 8 static + 8 any contact)."""
    w = weights.get("contact_head.out.weight")
    return {"attn_dim": contact_attn_dim(weights), "n_out": int(w.shape[0]) if w is not None else N_CONTACT}
