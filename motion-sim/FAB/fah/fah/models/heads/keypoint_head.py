"""Direct keypoint head on the shared hand latent, independent of the MHR / MANO parameters (whose
fingertips often miss the image evidence). Per point: a heatmap over the 4x-upsampled final decoder image
tokens (+ a stride-4 RGB stem, as in the mask head), queried by the final pose token plus a learned point
embedding; soft-argmax -> crop-normalised 2D; the features pooled under the heatmap -> root-relative 3D (m,
camera axes). Points = 21 MANO-convention joints, then K dense surface points (MHR hand-mesh vertices
`surface_idx`, indices into the hand sub-mesh)."""
import math

import numpy as np

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from .mask_head import LayerNorm2d


class HandKeypointHead(nn.Module):
    def __init__(self, img_dim=1280, tok_dim=1024, dim=96, stem=32, n_joints=21, surface_idx=(), pool=False, dense=(0, 0), stride=4,
                 depth=False, vis=False, dense_vidx=None, parts=0, part_hr=False, part_depth=2, contact=0, contact_dists=(10.0,),
                 contact_dim=128, contact_tokmem=False, contact_geo=True):
        super().__init__()
        # dense=(n_bone, n_vert): MHR-free layout -- joints, then points along the 20 bones, then MANO-topology vertices
        self.register_buffer("dense", torch.as_tensor(list(dense), dtype=torch.long))
        # which template vertices the n_vert surface points are (all 778 by default; a subset -> full mesh via the
        # spectral basis at inference)
        if dense_vidx is not None:
            self.register_buffer("dense_vidx", torch.as_tensor(list(dense_vidx), dtype=torch.long))
        # decoder-free use (no pose token): the token is pooled from the image tokens + the CLIFF camera condition
        self.pool = nn.Sequential(nn.LayerNorm(img_dim + 3), nn.Linear(img_dim + 3, tok_dim), nn.GELU()) if pool else None
        # decoder-free root depth: log(z * crop size / crop focal), i.e. CLIFF-style normalised by the crop scale
        self.depth = nn.Sequential(nn.LayerNorm(tok_dim), nn.Linear(tok_dim, 256), nn.GELU(), nn.Linear(256, 1)) if depth else None
        self.dim, self.n_joints = dim, n_joints
        self.register_buffer("surface_idx", torch.as_tensor(list(surface_idx), dtype=torch.long))
        n = n_joints + len(surface_idx) + int(sum(dense))
        self.img_proj = nn.Sequential(nn.Conv2d(img_dim, 256, 1), LayerNorm2d(256), nn.GELU())
        if stride == 4:  # heatmaps at 4x the token grid
            self.up = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, 2), LayerNorm2d(128), nn.GELU(),
                                    nn.ConvTranspose2d(128, dim, 2, 2), nn.GELU())
            self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, 2, 1), nn.GELU(), nn.Conv2d(stem, stem, 3, 2, 1), nn.GELU())
        else:  # stride 8, light CPU variant: heatmaps at 2x the token grid
            self.up = nn.Sequential(nn.ConvTranspose2d(256, dim, 2, 2), nn.GELU())
            self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, 2, 1), nn.GELU(), nn.Conv2d(stem, stem, 3, 2, 1), nn.GELU(),
                                      nn.Conv2d(stem, stem, 3, 2, 1), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(dim + stem, dim, 3, padding=1), LayerNorm2d(dim), nn.GELU(),
                                  nn.Conv2d(dim, dim, 3, padding=1), LayerNorm2d(dim), nn.GELU(), nn.Conv2d(dim, dim, 1))
        self.point = nn.Parameter(torch.randn(n, dim) * 0.02)
        self.query = nn.Sequential(nn.Linear(tok_dim, 512), nn.GELU(), nn.Linear(512, dim))
        self.qmlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.log_temp = nn.Parameter(torch.tensor(1.0))
        self.xyz = nn.Sequential(nn.Linear(2 * dim + tok_dim, 256), nn.GELU(), nn.Linear(256, 3))
        # per-point visibility logit (self- / object-occluded points: their 2D comes from the 3D, not the heatmap)
        self.vis = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, 64), nn.GELU(), nn.Linear(64, 1)) if vis else None
        # part segmentation (0 background, 1 palm, 2-16 finger segments: train.part_masks): one query per class on the same
        # feature map as the point heatmaps, so parts, joints and dense points share one representation
        self.part_emb = nn.Parameter(torch.randn(parts, dim) * 0.02) if parts else None
        # its own query path (sharing the point queries' MLPs let the part loss move every joint / point query)
        self.part_tok = nn.Linear(tok_dim, dim) if parts else None
        self.part_q = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)) if parts else None
        # per-pixel classes need a smooth map: the heatmap features carry the transposed-conv lattice (every 2nd / 4th
        # pixel), which point heatmaps average out but a pixel classifier copies -> 2x2-pooled features + two 3x3 convs,
        # logits upsampled back to the heatmap resolution
        blocks = []
        for _ in range(part_depth - 1):  # part_depth 3x3 convs (more context for e.g. curled fingers seen from the back)
            blocks += [nn.Conv2d(dim, dim, 3, padding=1), LayerNorm2d(dim), nn.GELU()]
        self.part_conv = nn.Sequential(*blocks, nn.Conv2d(dim, dim, 3, padding=1)) if parts else None
        # part_hr: image-guided refinement to half the crop resolution (sharper part / hand boundaries): the logits upsampled
        # 2x, refined from a light stride-2 image stem (residual; zero-initialised = plain upsampling at the start)
        self.part_stem2 = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.GELU(), nn.Conv2d(16, 16, 3, padding=1), nn.GELU()) if parts and part_hr else None
        self.part_refine = nn.Sequential(nn.Conv2d(parts + 16, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, parts, 3, padding=1)) if parts and part_hr else None
        if self.part_refine is not None:
            nn.init.zeros_(self.part_refine[-1].weight)
            nn.init.zeros_(self.part_refine[-1].bias)
        # hand-object contact: a logit per dense surface point + one per hand. Point tokens = learned per-point embedding +
        # the pose token + the detached point geometry (pooled image features under the point's heatmap, query, root-relative
        # 3D, visibility); then `contact` layers of self-attention over the points and cross-attention to the 4x-pooled
        # feature map (+ 2D sin-cos positions). No other output changes. contact_dists: contact at several tolerances (mm;
        # ordinal outputs per point and hand, e.g. 5 / 10 / 20 / 35); the 10 mm output stays the main one (last_contact), all of
        # them in last_contact_k
        self.n_vert, self.contact_on = int(dense[1]), bool(contact)
        # contact_geo configuration: True = the point geometry above; False = image tokens only (no per-point geometry input);
        # "external" = the caller builds the per-point inputs (contact_from), e.g. from a fitted MANO mesh
        self.contact_geo = contact_geo if contact_geo == "external" else bool(contact_geo)
        self.contact_main = list(contact_dists).index(10.0) if 10.0 in contact_dists else 0
        if contact:
            cd, nk = contact_dim, len(contact_dists)
            if nk > 1:  # saved with the weights (the loader reads the tolerances); single 10 mm heads stay as trained
                self.register_buffer("contact_dists", torch.as_tensor(list(contact_dists), dtype=torch.float32))
            # per-tolerance logit offset fitted on validation data (the class-balanced loss shifts probabilities up): with it the
            # F1-optimal vertex threshold is 0.5 (fah.api.CONTACT_THRESH); zero for heads saved without it
            self.register_buffer("contact_bias", torch.zeros(nk))
            self.contact_in = nn.Sequential(nn.LayerNorm(2 * dim + 4), nn.Linear(2 * dim + 4, cd), nn.GELU(), nn.Linear(cd, cd)) \
                if contact_geo else None
            self.contact_tok = nn.Linear(tok_dim, cd)
            self.contact_img = nn.Sequential(nn.Conv2d(dim, cd, 1), LayerNorm2d(cd))
            self.contact_pos = nn.Parameter(torch.randn(self.n_vert, cd) * 0.02)
            self.contact_mix = nn.ModuleList(nn.TransformerDecoderLayer(cd, max(4, cd // 32), 2 * cd, dropout=0.0, batch_first=True,
                                                                        norm_first=True) for _ in range(contact))
            self.contact_out = nn.Sequential(nn.LayerNorm(cd), nn.Linear(cd, nk))
            self.contact_hand = nn.Sequential(nn.LayerNorm(2 * cd), nn.Linear(2 * cd, 64), nn.GELU(), nn.Linear(64, nk))
            # contact_tokmem: the encoder tokens (img_dim, the token grid) also join the cross-attention memory (read-only)
            self.contact_tokmem = nn.Sequential(nn.LayerNorm(img_dim), nn.Linear(img_dim, cd)) if contact_tokmem else None
            # contact_branch (attach_contact_branch): trainable copies of the encoder's last blocks on its frozen hidden state
            self.contact_cd, self.contact_branch, self._br_in = cd, None, None

    def forward(self, img_tokens, grid, pose_token, img, cond=None, return_conf=False):
        """img_tokens (B, h*w, C) final decoder image tokens; grid (h, w); pose_token (B, C_tok);
        img (B, 3, S, S) crop; pose_token None -> pooled (needs cond (B, 3)). Returns uv (B, N, 2) in [0, 1] crop units, xyz (B, N, 3) root-relative (m),
        logits (B, N, 4h, 4w). return_conf also exposes last_conf, reusing the
        probabilities already needed for the coordinates and pooled features."""
        B = img_tokens.shape[0]
        f = img_tokens.transpose(1, 2).reshape(B, -1, *grid).float()
        f = self.up(self.img_proj(f))
        s = self.stem(img.float())
        if s.shape[-2:] != f.shape[-2:]:
            s = F.interpolate(s, size=f.shape[-2:], mode="bilinear", align_corners=False)
        f = self.fuse(torch.cat([f, s], 1))
        H, W = f.shape[-2:]
        if pose_token is None:
            pose_token = self.pool(torch.cat([img_tokens.float().mean(1), cond.float()], -1))
        pose = pose_token.float()
        q = self.qmlp(self.query(pose)[:, None] + self.point[None])  # (B, N, dim)
        logits = torch.einsum("bnc,bchw->bnhw", q, f) * (self.log_temp.exp() / math.sqrt(self.dim))
        self.last_part_logits = None
        if self.part_emb is not None and getattr(self, "parts_on", True):  # (B, n_parts, H, W) logits at the heatmap resolution
            qp = self.part_q(self.part_tok(pose)[:, None] + self.part_emb[None])
            g = getattr(self, "part_grad", 1.0)  # training: < 1 scales the part losses' gradient into the shared features
            fs = f if g >= 1 else f * g + f.detach() * (1 - g)
            fp = self.part_conv(F.avg_pool2d(fs, 2))
            pl = torch.einsum("bkc,bchw->bkhw", qp, fp) / math.sqrt(self.dim)
            pl = F.interpolate(pl, size=f.shape[-2:], mode="bilinear", align_corners=False)
            if self.part_refine is not None:
                s2 = self.part_stem2(img.float())
                up = F.interpolate(pl, size=s2.shape[-2:], mode="bilinear", align_corners=False)
                pl = up + self.part_refine(torch.cat([up, s2], 1))
            self.last_part_logits = pl
        p = logits.flatten(2).softmax(-1).view_as(logits)
        if return_conf:
            self.last_conf = p.flatten(2).amax(-1)
        xs = (torch.arange(W, device=f.device, dtype=f.dtype) + 0.5) / W
        ys = (torch.arange(H, device=f.device, dtype=f.dtype) + 0.5) / H
        pu = p
        r = getattr(self, "local_r", 0)
        if r:  # inference: soft-argmax in a window around the peak (no averaging between modes of a multi-modal map)
            idx = logits.flatten(2).argmax(-1)
            iy, ix = (idx // W)[..., None, None], (idx % W)[..., None, None]
            gy, gx = torch.arange(H, device=f.device)[:, None], torch.arange(W, device=f.device)[None, :]
            win = ((gy - iy).abs() <= r) & ((gx - ix).abs() <= r)
            pu = logits.masked_fill(~win, float("-inf")).flatten(2).softmax(-1).view_as(logits)
            # local_split: keep the global uv as output, the local one aside (estimator uses it for the surface 2D only)
            self.last_uv_local = torch.stack([(pu.sum(2) * xs).sum(-1), (pu.sum(3) * ys).sum(-1)], -1)
            if getattr(self, "local_split", False):
                pu = p
        uv = torch.stack([(pu.sum(2) * xs).sum(-1), (pu.sum(3) * ys).sum(-1)], -1)
        pooled = torch.einsum("bnhw,bchw->bnc", p, f)
        self.last_pooled = pooled  # per-point features under the heatmaps (MANO head input)
        self.last_vis = self.vis(torch.cat([pooled, q], -1))[..., 0] if self.vis is not None else None
        # = self.xyz(cat[pooled, q, pose]) with the pose part of the first layer computed once per hand
        w, d = self.xyz[0].weight, 2 * self.dim
        h = F.linear(torch.cat([pooled, q], -1), w[:, :d]) + F.linear(pose, w[:, d:], self.xyz[0].bias)[:, None]
        xyz = self.xyz[2](self.xyz[1](h))
        if self.contact_on and self.contact_geo != "external":  # (B, n_vert) per-surface-point logits, (B,) hand logit
            a = xyz.shape[1] - self.n_vert
            vis = self.last_vis[:, a:, None] if self.last_vis is not None else xyz.new_zeros(B, self.n_vert, 1)
            geo = torch.cat([pooled[:, a:], q[:, a:], xyz[:, a:] * 10, vis], -1) if self.contact_geo else None
            self.contact_from(geo, pose, f, img_tokens.transpose(1, 2).reshape(B, -1, *grid) if self.contact_tokmem is not None else None)
        elif self.contact_on:  # contact_geo "external": the caller builds the point inputs and calls contact_from
            self.last_f, self.last_q, self.last_pose = f, q, pose
        return uv, xyz, logits

    def contact_from(self, geo, pose, f, tok=None):
        """Contact logits from the per-point inputs geo (B, n_vert, 2 dim + 4; None: image tokens only), the pose token, the heatmap
        feature map f and (contact_tokmem) the encoder tokens tok (B, img_dim, h, w) -> self.last_contact (B, n_vert),
        self.last_contact_hand (B,)."""
        c = self.contact_tok(pose.detach())[:, None] + self.contact_pos[None]
        if geo is not None:
            c = c + self.contact_in(geo.detach())
        fm = self.contact_img(F.avg_pool2d(f.detach(), 4))
        mem = (fm + sincos_2d(fm.shape[1], *fm.shape[-2:], fm.device)).flatten(2).transpose(1, 2)
        if tok is not None:
            tm = self.contact_tokmem(tok.detach().float().flatten(2).transpose(1, 2))
            tm = tm + sincos_2d(tm.shape[-1], *tok.shape[-2:], tm.device).flatten(1).transpose(0, 1)[None]
            mem = torch.cat([mem, tm], 1)
        if getattr(self, "contact_branch", None) is not None:  # + the branch's tokens (its own copies of the encoder's last blocks)
            h, kw = self._br_in  # the encoder's hidden state at the branch's first block (forward pre-hook / TensorRT output)
            self._br_in = None
            br = self.contact_branch
            with torch.autocast(h.device.type, dtype=torch.bfloat16, enabled=h.is_cuda):
                h = h.detach()
                for blk in br["blocks"]:
                    if getattr(self, "contact_branch_ckpt", False) and torch.is_grad_enabled():  # activation checkpointing (memory only)
                        h = torch.utils.checkpoint.checkpoint(blk, h, use_reentrant=False, **kw)
                    else:
                        h = blk(h, **kw)
                h = br["norm"](h)[:, self.contact_branch_prefix:]
            bm = br["proj"](h.float())
            g = int(round(bm.shape[1] ** 0.5))
            bm = bm + sincos_2d(bm.shape[-1], g, g, bm.device).flatten(1).transpose(0, 1)[None]
            mem = torch.cat([mem, bm], 1)
        for blk in self.contact_mix:
            c = blk(c, mem)
        self.last_contact_k = self.contact_out(c) - self.contact_bias  # (B, n_vert, K)
        self.last_contact_hand_k = self.contact_hand(torch.cat([c.mean(1), c.amax(1)], -1))  # (B, K)
        # the main tolerance as fresh dense tensors (packed into one host copy, also for one hand; no host-to-device copy, so
        # CUDA graphs can capture it)
        self.last_contact = self.last_contact_k[..., self.contact_main].clone(memory_format=torch.contiguous_format)
        self.last_contact_hand = self.last_contact_hand_k[:, self.contact_main].clone(memory_format=torch.contiguous_format)
        return self.last_contact, self.last_contact_hand


def attach_contact_branch(encoder, head, k):
    """Contact branch on a StudentEncoder: copies of its last k ViT blocks + final norm and a projection, stored in the head
    (keypoint_head.contact_branch.*, trained with the contact head), run on the frozen encoder's hidden state before those
    blocks: a forward pre-hook reads it on the PyTorch path, the pose TensorRT engine gives it as a second output
    (hidden<block>, scripts/build_pose_engine.py --hidden_block). The encoder's own forward and every pose output are
    unchanged. Also keeps the blocks' keyword arguments (rotary embedding of the fixed token grid) for the TensorRT path."""
    import copy

    vit = encoder.vit
    L = len(vit.blocks)
    dev = next(vit.parameters()).device
    head.contact_branch = nn.ModuleDict(dict(blocks=copy.deepcopy(vit.blocks[L - k:]), norm=copy.deepcopy(vit.norm),
                                             proj=nn.Sequential(nn.LayerNorm(vit.embed_dim), nn.Linear(vit.embed_dim, head.contact_cd)))).to(dev)
    head.contact_branch_prefix, head.contact_branch_block = encoder.num_prefix, L - k

    def grab(mod, args, kwargs):
        head._br_in = (args[0], kwargs)

    hook = vit.blocks[L - k].register_forward_pre_hook(grab, with_kwargs=True)
    with torch.no_grad():  # the blocks' kwargs for one input (input-independent at the fixed grid)
        p = next(encoder.parameters())
        encoder(torch.zeros(1, 3, encoder.input_size, encoder.input_size, device=dev, dtype=p.dtype))
    head.contact_branch_kw, head._br_in = head._br_in[1], None
    return hook


def sincos_2d(c, h, w, device):
    """(c, h, w) fixed 2D sin-cos positions of a feature map (x in the first c / 2 channels, y in the rest)."""
    k = torch.arange(c // 4, device=device, dtype=torch.float32)
    fr = 1.0 / (100.0 ** (k / max(c // 4 - 1, 1)))
    ys, xs = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h, (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    ex = (xs[None] * fr[:, None] * 2 * math.pi)[:, None, :].expand(-1, h, -1)
    ey = (ys[None] * fr[:, None] * 2 * math.pi)[:, :, None].expand(-1, -1, w)
    return torch.cat([ex.sin(), ex.cos(), ey.sin(), ey.cos()], 0)


def dense_subset(rest, n, mode="fps", joints=None):
    """n of the template's vertices (rest (V, 3)): 'fps' farthest-point sampling (uniform over the surface);
    'fingers' FPS with distances on the fingers counted 1.6x (denser where the outline / articulation is),
    needs the rest joints (21, OpenPose order)."""
    rest = np.asarray(rest, np.float64)
    w = np.ones(len(rest))
    if mode == "fingers":
        jt = np.asarray(joints, np.float64)
        palm = jt[[0, 1, 5, 9, 13, 17]]
        d_palm = np.linalg.norm(rest[:, None] - palm[None], axis=-1).min(1)
        d_fing = np.linalg.norm(rest[:, None] - np.delete(jt, [0, 1, 5, 9, 13, 17], 0)[None], axis=-1).min(1)
        w = np.where(d_fing < d_palm, 1.6, 1.0)
    idx = [int(np.argmin(rest[:, 1]))]
    d = np.linalg.norm(rest - rest[idx[0]], axis=1) * w
    for _ in range(n - 1):
        i = int(d.argmax())
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(rest - rest[i], axis=1) * w)
    return np.sort(np.asarray(idx))


def farthest_points(verts, k, start=0):
    """Greedy farthest-point sampling of k vertex indices (verts (V, 3))."""
    verts = torch.as_tensor(verts, dtype=torch.float32)
    idx = [start]
    d = (verts - verts[start]).norm(dim=-1)
    for _ in range(k - 1):
        i = int(d.argmax())
        idx.append(i)
        d = torch.minimum(d, (verts - verts[i]).norm(dim=-1))
    return idx


BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def bone_points(j, per_bone=2):
    """(..., 21, C) joints -> (..., 20 * per_bone, C) points evenly inside every bone (parent -> child)."""
    a = j[..., [p for p, _ in BONES], :]
    b = j[..., [c for _, c in BONES], :]
    ts = [(k + 1) / (per_bone + 1) for k in range(per_bone)]
    return torch.cat([a + t * (b - a) for t in ts], -2)
