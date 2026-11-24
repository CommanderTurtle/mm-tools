"""Direct dense keypoints: per-point heatmaps over the 4x-upsampled
image tokens (+ a stride-4 RGB stem), queried by the decoder's pose token plus a learned point embedding;
soft-argmax -> 2D (crop units in [0, 1]); the features pooled under each heatmap -> root-relative 3D (m, camera axes).

Points decoder (all optional): 2.5D points (each point's 3D on its own heatmap 2D ray
at a regressed depth, p25d), a wider / deeper point transformer (xyz_dim, xyz_layers) whose memory can also hold the
decoder's tokens (xyz_mem).

Points (in this order): the 70 MHR keypoints (2D labels, teacher 3D), the 14 H36M-regressor joints of the SMPL
evaluation skeleton, and `surface_idx` SMPL template vertices (farthest-point sampled). Each output is a modality
of its own: 2D from the heatmaps, skeleton from the joint points, surface from the vertex points; none is derived
from the parametric mesh (weak agreement only, through the shared pose token and optional reprojection terms).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_MHR, N_J14 = 70, 14
MASK_KP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]  # "+kp" prompts: the 17 COCO body keypoints in MHR70


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight, self.bias, self.eps = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c)), eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return self.weight[:, None, None] * (x - u) / torch.sqrt(s + self.eps) + self.bias[:, None, None]


class BodyDenseHead(nn.Module):
    def __init__(self, surface_idx, img_dim=1280, tok_dim=1024, dim=96, stem=32, up=4, residual=False, parts=0,
                 xyz_layers=0, part_depth=0, hr=0, mask_depth=0, mask_query="box", xyz_dim=256, xyz_mem="img",
                 p25d=0.0, p25d_border=0.02, p25d_vis=False, ray_in=False, uv_off=0.0):
        super().__init__()
        self.register_buffer("surface_idx", torch.as_tensor(list(surface_idx), dtype=torch.long))
        # residual: the 3D output is a zero-initialised offset on the parametric mesh's corresponding points (the
        # caller adds the detached anchors: MHR keypoints, mesh-regressed joints, mapped mesh vertices)
        self.register_buffer("residual", torch.tensor(int(residual)))
        n = N_MHR + N_J14 + len(surface_idx)
        self.dim = dim
        self.img_proj = nn.Sequential(nn.Conv2d(img_dim, 256, 1), LayerNorm2d(256), nn.GELU())
        if up == 4:  # heatmaps at 4x the token grid (stride 4 at 512 px), stride-4 pixel stem
            self.up = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, 2), LayerNorm2d(128), nn.GELU(),
                                    nn.ConvTranspose2d(128, dim, 2, 2), LayerNorm2d(dim), nn.GELU())
            self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, 2, 1), nn.GELU(), nn.Conv2d(stem, stem, 3, 2, 1), nn.GELU())
        else:  # 2x (stride 8)
            self.up = nn.Sequential(nn.ConvTranspose2d(256, dim, 2, 2), LayerNorm2d(dim), nn.GELU())
            self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, 2, 1), nn.GELU(), nn.Conv2d(stem, stem, 3, 2, 1), nn.GELU(),
                                      nn.Conv2d(stem, stem, 3, 2, 1), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(dim + stem, dim, 3, padding=1), LayerNorm2d(dim), nn.GELU(),
                                  nn.Conv2d(dim, dim, 1))
        self.point = nn.Parameter(torch.randn(n, dim) * 0.02)
        self.query = nn.Sequential(nn.Linear(tok_dim, 512), nn.GELU(), nn.Linear(512, dim))
        self.qmlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.log_temp = nn.Parameter(torch.tensor(1.0))
        self.xyz = nn.Sequential(nn.Linear(2 * dim + tok_dim, xyz_dim), nn.GELU(), nn.Linear(xyz_dim, 3))
        # xyz_layers > 0: the points' 3D features go through transformer layers before the xyz MLP -- self-attention
        # among all points (a body's root-relative layout needs global reasoning that each point's local feature
        # lacks) and cross-attention to the image tokens; positions: each point's own 2D, each token's grid cell
        self.pt = None
        if xyz_layers:
            layer = nn.TransformerDecoderLayer(xyz_dim, 8, 4 * xyz_dim, dropout=0.0, activation="gelu", batch_first=True,
                                               norm_first=True)
            self.pt = nn.TransformerDecoder(layer, xyz_layers)
            self.pt_pos = nn.Linear(2, xyz_dim)
            self.pt_mem_pos = nn.Linear(2, xyz_dim)
        # points decoder, off by default. xyz_dim: width of the xyz MLP and the point
        # transformer (default 256); xyz_mem "img+dec": the transformer's memory also holds the decoder's final tokens
        # (pose token + the 70 keypoint tokens), through a zero-initialised projection
        self.pt_mem = self.pt_dec = self.pt_dec_pos = None
        if xyz_layers and xyz_dim != 256:
            self.pt_mem = nn.Linear(256, xyz_dim)
        if xyz_layers and "dec" in xyz_mem:
            self.pt_dec = nn.Linear(tok_dim, xyz_dim)
            nn.init.zeros_(self.pt_dec.weight)
            nn.init.zeros_(self.pt_dec.bias)
            self.pt_dec_pos = nn.Parameter(torch.randn(1 + N_MHR, xyz_dim) * 0.02)
        # p25d > 0: 2.5D points (MeTRAbs / NLF). The regressed xyz is the points' metric root-relative 3D P (kept as
        # self.last_xyz3d); the output is each point back-projected along its own heatmap 2D ray through the crop camera
        # at depth P_z + t_z (t: closed-form translation of P onto the rays) minus t, mixed with P at weight 1 - p25d;
        # points within p25d_border of the crop edge (truncated: their heatmaps pin to it) keep P. The 2D is detached
        # (the 3D losses move the depths, not the heatmaps) unless p25d_uvgrad
        self.p25d_mix, self.p25d_border, self.p25d_uvgrad = float(p25d), float(p25d_border), False
        # training only (train_fab --dense_25d_uvgrad_vis / --dense_25d_uvgrad_scale): (gate (B, K, 1) for the first K
        # points or None, share for the other points) of the 3D losses' gradient that reaches each point's heatmap 2D
        # through the lift; the forward value is unchanged
        self.uv_grad_gate = None
        if p25d:
            self.register_buffer("p25d", torch.tensor([float(p25d), float(p25d_border)]))
        # p25d_vis: the lift's weight per point x the visibility head's P(visible) (FABDecoder, after fab/
        # vis_head.py), so occluded / out-of-frame keypoints keep the regressed 3D; saved as the p25d_vis buffer
        self.p25d_vis_gate = bool(p25d_vis)
        if p25d_vis:
            self.register_buffer("p25d_vis", torch.ones(1))
        # ray_in: the 3D features also get each point's unit viewing ray (from its detached 2D through the crop camera;
        # zero-initialised), so the depth / elevation along the ray is predicted knowing where the ray points -- the image
        # tokens the head reads are taken before the decoder's ray conditioning and carry no camera
        self.pt_ray = None
        if ray_in:
            self.pt_ray = nn.Linear(3, xyz_dim)
            nn.init.zeros_(self.pt_ray.weight)
            nn.init.zeros_(self.pt_ray.bias)
        # uv_off > 0: amodal 2D for truncated bodies. The soft-argmax 2D is a mean of cell centres, so a point outside the
        # crop pins to its edge; a zero-initialised offset from the point's pooled feature and query, weighted 1 at the
        # edge -> 0 at uv_off (crop units) from it, lets those points leave the crop; interior points are unchanged
        self.uv_off = None
        if uv_off:
            self.uv_off = nn.Linear(2 * dim, 2)
            nn.init.zeros_(self.uv_off.weight)
            nn.init.zeros_(self.uv_off.bias)
            self.register_buffer("uv_off_band", torch.tensor([float(uv_off)]))
        if residual:
            nn.init.zeros_(self.xyz[2].weight)
            nn.init.zeros_(self.xyz[2].bias)
        # part segmentation (0 background + PART_NAMES): one query per class on the feature map of the point heatmaps
        # so parts, joints and surface points share one representation
        self.part_emb = nn.Parameter(torch.randn(parts, dim) * 0.02) if parts else None
        # + a small conv head on the same map (zero-initialised: starts as the query logits); trainable alone on a
        # frozen model (train_fab.py --parts_only) so the part output can be added without moving the other outputs
        self.part_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), LayerNorm2d(dim), nn.GELU(),
                                       nn.Conv2d(dim, parts, 1)) if parts else None
        if parts:
            nn.init.zeros_(self.part_conv[-1].weight)
            nn.init.zeros_(self.part_conv[-1].bias)
        # part_depth > 0: a deeper part head instead -- its own query path (sharing the point
        # queries' MLPs lets the part loss move every point query) and part_depth 3x3 convs on the 2x2-pooled map (the
        # transposed-conv lattice of the heatmap features, which a per-pixel classifier copies), logits upsampled back
        self.part_tok = self.part_q = self.part_dconv = None
        if parts and part_depth:
            self.part_conv = None
            self.part_tok = nn.Linear(tok_dim, dim)
            self.part_q = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            blocks = []
            for _ in range(part_depth - 1):
                blocks += [nn.Conv2d(dim, dim, 3, padding=1), LayerNorm2d(dim), nn.GELU()]
            self.part_dconv = nn.Sequential(*blocks, nn.Conv2d(dim, dim, 3, padding=1))
        # inference only (> 0): 2D from the soft-argmax inside this many heatmap cells around each map's peak (a
        # multi-modal map's global expectation lands between the modes); the pooled 3D features keep the full map
        self.local_window = 0
        self.keep_logits = False  # inference diagnostics: keep the point logits as self.last_logits
        self.keep_feats = None  # point indices: keep their pooled features + queries as self.last_feats (contact head)
        # hr > 0: the first hr points (the 70 MHR keypoints) take their 2D from a 2x finer map (stride 2 at 512 px): the
        # heatmap features bilinearly upsampled + a zero-initialised refinement from a stride-2 pixel stem, so a warm
        # start begins at (nearly) its old 2D; the pooled 3D features keep the base map
        self.hr = hr
        if hr:
            self.register_buffer("hr_points", torch.tensor(int(hr)))
            self.hr_stem = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.GELU(), nn.Conv2d(16, 16, 3, padding=1), nn.GELU())
            self.hr_refine = nn.Sequential(nn.Conv2d(dim + 16, dim, 3, padding=1), LayerNorm2d(dim), nn.GELU(),
                                           nn.Conv2d(dim, dim, 3, padding=1))
            nn.init.zeros_(self.hr_refine[-1].weight)
            nn.init.zeros_(self.hr_refine[-1].bias)
        self.tok_dim = tok_dim
        self.mask_tok = self.mask_qvec = self.mask_q = self.mask_dconv = self.mask_stem = self.mask_refine = None
        self.mask_down, self.mask_pts, self.mask_extent = None, False, False
        self.mask_full = self.mask_corners = self.mask_kp = False
        if mask_depth:
            self.add_mask_head(mask_depth, mask_query)

    def add_mask_head(self, depth=4, query="box"):
        """Person instance mask (the visible person with clothing and hair, trained on human mask labels; train_fab.py
        --mask_depth / --mask_only): a query on `depth` 3x3 convs over the heatmap feature map (1/4 crop, + the crop's
        x / y coordinates), dot product -> logits, bilinearly upsampled to 1/2 crop and refined from a stride-2 pixel stem
        (zero-initialised residual) -> (B, 1, S/2, S/2). query "box": one learned query, so the mask is the person the
        crop (= the detector box) is centred on, from the image features alone -- independent of the decoder's pose
        token, so it can be fed back as that person's prompt when people overlap; "token": the person's pose token.
        "+pts" (e.g. "box+pts"): one more input channel, the model's own projected dense points (14 joints + surface,
        detached) splatted on the map: a person-specific prior for crowds. "+extent": one more channel, 1 inside the
        prompt box (forward mask_box, crop units): the square crop drops the box's aspect, which tells overlapping
        people apart. Reads the shared map only: nothing else reads it."""
        dim = self.dim
        base, self.mask_pts = query.split("+")[0], "+pts" in query
        self.mask_extent = "+extent" in query
        if self.mask_extent:  # saved with the head: marks the extent channel for load_mask_head / dense_mask_query
            self.register_buffer("mask_extent_on", torch.ones(1))
        if base == "box":
            self.mask_qvec = nn.Parameter(torch.randn(dim) * 0.02)
        else:
            self.mask_tok = nn.Linear(self.tok_dim, dim)
        self.mask_q = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        # "+kp": SAM-style point prompts from the model's own predicted 2D (never GT): one channel per COCO body keypoint
        # of the target (the dense head's MHR keypoints, detached), a Gaussian of 1.5 map cells
        self.mask_kp = "+kp" in query
        if self.mask_kp:
            self.register_buffer("mask_kp_idx", torch.tensor(MASK_KP, dtype=torch.long))
        blocks, c = [], dim + 2 + int(self.mask_pts) + int(self.mask_extent) + (len(MASK_KP) if self.mask_kp else 0)
        # "+pool": the convs at 1/8 crop after a learned 2 x 2 stride-2 downsample (4x cheaper; the refinement keeps
        # the boundaries)
        self.mask_down = nn.Conv2d(c, c, 2, 2) if "+pool" in query else None
        for _ in range(depth - 1):
            blocks += [nn.Conv2d(c, dim, 3, padding=1), LayerNorm2d(dim), nn.GELU()]
            c = dim
        self.mask_dconv = nn.Sequential(*blocks, nn.Conv2d(c, dim, 3, padding=1))
        self.mask_stem = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.GELU(), nn.Conv2d(16, 16, 3, padding=1), nn.GELU())
        self.mask_refine = nn.Sequential(nn.Conv2d(17, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 3, padding=1))
        nn.init.zeros_(self.mask_refine[-1].weight)
        nn.init.zeros_(self.mask_refine[-1].bias)
        # "+full": a coarse-to-fine decoder to the full crop: refined at 1/4 crop with the dense head's stride-4 RGB stem
        # features, at 1/2 (mask_stem), and at the crop's own resolution (a stride-1 stem) -> (B, 1, S, S); each step a
        # zero-initialised residual
        self.mask_full = "+full" in query
        if self.mask_full:
            sc = self.stem[-2].out_channels
            self.mask_up4 = nn.Sequential(nn.Conv2d(1 + sc, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 3, padding=1))
            self.mask_stem1 = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.GELU())
            self.mask_refine1 = nn.Sequential(nn.Conv2d(9, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 3, padding=1))
            for m in (self.mask_up4, self.mask_refine1):
                nn.init.zeros_(m[-1].weight)
                nn.init.zeros_(m[-1].bias)
        # "+corners": SAM-style box-corner prompt tokens: the query = the learned query + a Fourier embedding of the
        # box corners, then 2 cross-attention blocks over the conv features (+ their x / y), so the query can pick the
        # person the box was drawn for (needs mask_box)
        self.mask_corners = "+corners" in query
        if self.mask_corners:
            self.mask_box_emb = nn.Sequential(nn.Linear(4 * 2 * 8, dim), nn.GELU(), nn.Linear(dim, dim))
            self.mask_pos = nn.Conv2d(2, dim, 1)
            self.mask_attn = nn.ModuleList([nn.MultiheadAttention(dim, 4, batch_first=True) for _ in range(2)])
            self.mask_ln = nn.ModuleList([nn.LayerNorm(dim) for _ in range(4)])
            self.mask_mlp = nn.ModuleList([nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
                                           for _ in range(2)])
        return self

    def mask_corner_query(self, G, mask_box):
        """+corners: G (B, dim, h, w) mask conv features, mask_box (B, 4) crop units -> the query (B, dim)."""
        B, C, h, w = G.shape
        fr = (2.0 ** torch.arange(8, device=G.device, dtype=G.dtype)) * math.pi
        ang = mask_box.to(G.dtype)[..., None] * fr  # (B, 4, 8)
        pe = torch.cat([ang.sin(), ang.cos()], -1).flatten(1)
        q = (self.mask_qvec.to(G.dtype)[None] + self.mask_box_emb(pe))[:, None]  # (B, 1, dim)
        cy, cx = torch.meshgrid(torch.linspace(0, 1, h, device=G.device, dtype=G.dtype),
                                torch.linspace(0, 1, w, device=G.device, dtype=G.dtype), indexing="ij")
        kv = (G + self.mask_pos(torch.stack([cx, cy])[None])).flatten(2).transpose(1, 2)  # (B, hw, dim)
        for a, (l1, l2), mlp in zip(self.mask_attn, zip(self.mask_ln[0::2], self.mask_ln[1::2]), self.mask_mlp):
            q = q + a(l1(q), kv, kv, need_weights=False)[0]
            q = q + mlp(l2(q))
        return self.mask_q(q[:, 0])

    def load_mask_head(self, weights):
        """Attach a separately trained person-mask head ({"dense_head.mask_*": tensor}); False when there is none."""
        w = {k.removeprefix("dense_head."): v for k, v in weights.items() if k.startswith("dense_head.mask_")}
        depth = sum(1 for k, v in w.items() if k.startswith("mask_dconv.") and k.endswith(".weight") and v.ndim == 4)
        if not depth:
            return False
        ext = "mask_extent_on" in w
        nkp = len(w["mask_kp_idx"]) if "mask_kp_idx" in w else 0
        pts = w["mask_dconv.0.weight"].shape[1] == self.dim + 3 + int(ext) + nkp
        self.add_mask_head(depth, ("box" if "mask_qvec" in w else "token") + ("+pts" if pts else "")
                           + ("+extent" if ext else "") + ("+pool" if "mask_down.weight" in w else "")
                           + ("+full" if "mask_up4.0.weight" in w else "") + ("+corners" if "mask_pos.weight" in w else "")
                           + ("+kp" if nkp else ""))
        missing, unexpected = self.load_state_dict(w, strict=False)
        bad = [k for k in missing if k.startswith("mask_")] + list(unexpected)
        if bad:
            raise ValueError(f"mask head load: {bad[:6]}")
        return True

    def load_part_head(self, weights):
        """Attach a separately trained part head (e.g. trained with --parts_only on this frozen model): weights =
        {"dense_head.part_*": tensor} of the original or the d6-style head. Returns False when there is none."""
        w = {k.removeprefix("dense_head."): v for k, v in weights.items() if k.startswith("dense_head.part_")}
        if "part_emb" not in w:
            return False
        parts, dim = w["part_emb"].shape
        depth = sum(1 for k, v in w.items() if k.startswith("part_dconv.") and k.endswith(".weight") and v.ndim == 4)
        self.part_emb = nn.Parameter(torch.zeros(parts, dim))
        if depth:
            self.part_conv = None
            self.part_tok = nn.Linear(w["part_tok.weight"].shape[1], dim)
            self.part_q = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
            blocks = []
            for _ in range(depth - 1):
                blocks += [nn.Conv2d(dim, dim, 3, padding=1), LayerNorm2d(dim), nn.GELU()]
            self.part_dconv = nn.Sequential(*blocks, nn.Conv2d(dim, dim, 3, padding=1))
        else:
            self.part_tok = self.part_q = self.part_dconv = None
            self.part_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), LayerNorm2d(dim), nn.GELU(),
                                           nn.Conv2d(dim, parts, 1))
        missing, unexpected = self.load_state_dict(w, strict=False)
        bad = [k for k in missing if k.startswith("part_")] + list(unexpected)
        if bad:
            raise ValueError(f"part head load: {bad[:6]}")
        return True

    def forward(self, img_tokens, pose_token, img, point_count=None, cam_int=None, affine=None, dec_tokens=None,
                mask_box=None):
        """img_tokens (B, C, h, w), pose_token (B, C_tok), img (B, 3, S, S) crop in [0, 1] -> uv (B, N, 2) in [0, 1]
        crop units, xyz (B, N, 3) root-relative (m), part logits (B, P, 4h, 4w) or None. p25d: cam_int (B, 3, 3) and
        affine (B, 2, 3) (full image -> crop px); xyz_mem "img+dec": dec_tokens (B, 1 + 70, C_tok)."""
        g = self.img_proj(img_tokens.float())
        f = self.up(g)
        s = self.stem(img.float())
        if s.shape[-2:] != f.shape[-2:]:
            s = F.interpolate(s, size=f.shape[-2:], mode="bilinear", align_corners=False)
        f = self.fuse(torch.cat([f, s], 1))
        H, W = f.shape[-2:]
        pose = pose_token.float()
        # Point queries do not interact: callers needing only the MHR keypoints can omit
        # the training-only surface queries without changing the retained predictions.
        if point_count is not None and not 1 <= point_count <= len(self.point):
            raise ValueError("point_count must select a nonempty prefix of the saved points")
        if point_count is not None and self.pt is not None:
            raise ValueError("points interact through the xyz transformer: all points are needed")
        points = self.point if point_count is None else self.point[:point_count]
        q = self.qmlp(self.query(pose)[:, None] + points[None])
        logits = torch.einsum("bnc,bchw->bnhw", q, f) * (self.log_temp.exp() / math.sqrt(self.dim))
        p = logits.flatten(2).softmax(-1).view_as(logits)
        xs = (torch.arange(W, device=f.device, dtype=f.dtype) + 0.5) / W
        ys = (torch.arange(H, device=f.device, dtype=f.dtype) + 0.5) / H
        uv = torch.stack([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], -1)
        if self.local_window > 0 and not self.training:
            uv = local_soft_argmax(logits, self.local_window)
        if self.keep_logits:
            self.last_logits = logits
        if self.hr:
            k = min(self.hr, len(points))
            fu = F.interpolate(f, scale_factor=2, mode="bilinear", align_corners=False)
            sh = self.hr_stem(img.float())
            if sh.shape[-2:] != fu.shape[-2:]:
                sh = F.interpolate(sh, size=fu.shape[-2:], mode="bilinear", align_corners=False)
            fh = fu + self.hr_refine(torch.cat([fu, sh], 1))
            lh = torch.einsum("bnc,bchw->bnhw", q[:, :k], fh) * (self.log_temp.exp() / math.sqrt(self.dim))
            Hh, Wh = fh.shape[-2:]
            ph = lh.flatten(2).softmax(-1).view_as(lh)
            xh = (torch.arange(Wh, device=f.device, dtype=ph.dtype) + 0.5) / Wh
            yh = (torch.arange(Hh, device=f.device, dtype=ph.dtype) + 0.5) / Hh
            uvh = torch.stack([(ph.sum(2) * xh).sum(-1), (ph.sum(3) * yh).sum(-1)], -1)
            if self.local_window > 0 and not self.training:
                uvh = local_soft_argmax(lh, 2 * self.local_window)
            if self.keep_logits:
                self.last_logits_hr = lh
            uv = torch.cat([uvh.to(uv.dtype), uv[:, k:]], 1)
        pooled = torch.einsum("bnhw,bchw->bnc", p, f)
        if self.uv_off is not None:  # edge band only (amodal 2D of truncated points); rays / lift / outputs use it
            band = (1 - torch.minimum(uv, 1 - uv).amin(-1, keepdim=True) / self.uv_off_band).clamp(0, 1)
            uv = uv + band * self.uv_off(torch.cat([pooled, q], -1)).float()
        if self.keep_feats is not None:
            self.last_feats = torch.cat([pooled, q], -1)[:, self.keep_feats]
        w, d = self.xyz[0].weight, 2 * self.dim
        h = F.linear(torch.cat([pooled, q], -1), w[:, :d]) + F.linear(pose, w[:, d:], self.xyz[0].bias)[:, None]
        if self.pt_ray is not None:  # each point's viewing ray (its own 2D through the crop camera), camera axes
            h = h + self.pt_ray(self.rays(uv, cam_int, affine, img.shape[-1])).to(h.dtype)
        if self.pt is not None:
            hg, wg = g.shape[-2:]
            cy, cx = torch.meshgrid((torch.arange(hg, device=g.device) + 0.5) / hg,
                                    (torch.arange(wg, device=g.device) + 0.5) / wg, indexing="ij")
            mem = g.flatten(2).transpose(1, 2)
            if self.pt_mem is not None:
                mem = self.pt_mem(mem)
            mem = mem + self.pt_mem_pos(torch.stack([cx, cy], -1).view(1, -1, 2).to(g.dtype))
            if self.pt_dec is not None:  # + the decoder's tokens
                mem = torch.cat([mem, (self.pt_dec(dec_tokens.float()) + self.pt_dec_pos).to(mem.dtype)], 1)
            h = self.pt(h + self.pt_pos(uv), mem)
        xyz = self.xyz[2](self.xyz[1](h))
        if self.p25d_mix > 0:
            self.last_xyz3d = xyz
            xyz = self.lift25d(uv, xyz, cam_int, affine, img.shape[-1])
        parts = None
        if self.part_emb is not None and self.part_dconv is not None:
            qp = self.part_q(self.part_tok(pose)[:, None] + self.part_emb[None])
            fp = self.part_dconv(F.avg_pool2d(f, 2))
            parts = F.interpolate(torch.einsum("bnc,bchw->bnhw", qp, fp) / math.sqrt(self.dim), size=(H, W),
                                  mode="bilinear", align_corners=False)
        elif self.part_emb is not None:
            qp = self.qmlp(self.query(pose)[:, None] + self.part_emb[None])
            parts = torch.einsum("bnc,bchw->bnhw", qp, f) / math.sqrt(self.dim) + self.part_conv(f)
        if self.mask_dconv is not None:  # person mask: a 4th output (only when the mask head exists)
            if self.mask_qvec is not None:  # box-conditioned: the same query for every crop
                qm = self.mask_q(self.mask_qvec.to(f.dtype))[None].expand(f.shape[0], -1)
            else:
                qm = self.mask_q(self.mask_tok(pose))
            cy, cx = torch.meshgrid(torch.linspace(-1, 1, H, device=f.device, dtype=f.dtype),
                                    torch.linspace(-1, 1, W, device=f.device, dtype=f.dtype), indexing="ij")
            fm = [f, torch.stack([cx, cy])[None].expand(f.shape[0], -1, -1, -1)]
            if self.mask_pts:  # the model's own projected points (joints + surface), splatted and dilated 2 cells
                pu = uv[:, N_MHR:].detach()
                ix = (pu[..., 0] * W).long().clamp(0, W - 1) + (pu[..., 1] * H).long().clamp(0, H - 1) * W
                occ = torch.zeros(f.shape[0], H * W, device=f.device, dtype=f.dtype).scatter_(1, ix, 1.0)
                fm.append(F.max_pool2d(occ.view(-1, 1, H, W), 5, 1, 2))
            if self.mask_extent:  # the prompt box (B, 4) x0, y0, x1, y1 in crop units: 1 inside (cell centres)
                if mask_box is None:  # no box given (e.g. 3D evals): the padded square's centre 0.8 (padding 1.25)
                    mask_box = f.new_tensor([0.1, 0.1, 0.9, 0.9]).expand(f.shape[0], -1)
                bx = mask_box.to(f.dtype)
                gx, gy = (cx + 1) / 2, (cy + 1) / 2
                fm.append(((gx[None] >= bx[:, 0, None, None]) & (gx[None] <= bx[:, 2, None, None])
                           & (gy[None] >= bx[:, 1, None, None]) & (gy[None] <= bx[:, 3, None, None])).to(f.dtype)[:, None])
            if self.mask_kp:  # point prompts at the target's own predicted keypoints (cell centres, as uv)
                ku = uv[:, self.mask_kp_idx].detach().to(f.dtype)  # (B, K, 2) crop units
                d2 = (xs.view(1, 1, 1, W) - ku[..., 0, None, None]) ** 2 + (ys.view(1, 1, H, 1) - ku[..., 1, None, None]) ** 2
                fm.append(torch.exp(-d2 * (W * W / (2 * 1.5 ** 2))))
            fm = torch.cat(fm, 1)
            if self.mask_down is not None:
                fm = self.mask_down(fm)
            G = self.mask_dconv(fm)
            if self.mask_corners:
                if mask_box is None:
                    mask_box = f.new_tensor([0.1, 0.1, 0.9, 0.9]).expand(f.shape[0], -1)
                qm = self.mask_corner_query(G, mask_box)
            ml = torch.einsum("bc,bchw->bhw", qm, G)[:, None] / math.sqrt(self.dim)
            if self.mask_full:  # 1/4 crop with the stride-4 stem features first
                ml = F.interpolate(ml, size=s.shape[-2:], mode="bilinear", align_corners=False)
                ml = ml + self.mask_up4(torch.cat([ml, s.to(ml.dtype)], 1))
            sm = self.mask_stem(img.float())
            ml = F.interpolate(ml, size=sm.shape[-2:], mode="bilinear", align_corners=False)
            ml = ml + self.mask_refine(torch.cat([ml, sm.to(ml.dtype)], 1))
            if self.mask_full:  # the crop's own resolution
                s1 = self.mask_stem1(img.float())
                ml = F.interpolate(ml, size=s1.shape[-2:], mode="bilinear", align_corners=False)
                ml = ml + self.mask_refine1(torch.cat([ml, s1.to(ml.dtype)], 1))
            return uv, xyz, parts, ml
        return uv, xyz, parts

    @staticmethod
    def rays(uv, K, A, size):
        """uv (B, N, 2) crop units (detached), K (B, 3, 3), A (B, 2, 3) full image -> crop px -> (B, N, 3) unit rays."""
        q, K, A = uv.detach().float(), K.float(), A.float()
        px, py = q[..., 0] * size - A[:, None, 0, 2], q[..., 1] * size - A[:, None, 1, 2]
        a, b, c, d = A[:, None, 0, 0], A[:, None, 0, 1], A[:, None, 1, 0], A[:, None, 1, 1]
        det = a * d - b * c
        u, v = (d * px - b * py) / det, (a * py - c * px) / det
        r = torch.stack([(u - K[:, None, 0, 2]) / K[:, None, 0, 0], (v - K[:, None, 1, 2]) / K[:, None, 1, 1],
                         torch.ones_like(u)], -1)
        return r / r.norm(dim=-1, keepdim=True)

    def lift25d(self, uv, P, K, A, size):
        """2.5D points (p25d): uv (B, N, 2) crop units, P (B, N, 3) root-relative 3D (m), K (B, 3, 3), A (B, 2, 3) ->
        (B, N, 3) in P's frame. t = argmin sum_i |P_i + t - (P_iz + t_z) r_i|^2 over the rays r_i of uv (full perspective,
        algebraic error; t_x, t_y eliminated in closed form, then a scalar solve for t_z), points off the crop edge
        only; each point -> (P_iz + t_z) r_i - t. Elementwise ops only (CUDA graphs, ONNX)."""
        g = self.uv_grad_gate  # (per-keypoint gate (B, K, 1) or None, value for the other points)
        if g is None:
            q = (uv if self.p25d_uvgrad else uv.detach()).float()
        else:  # value = uv exactly (uv - uv.detach() is 0); gradient into uv x the gate
            gk, rest = g
            gg = torch.full_like(uv[..., :1], float(rest), dtype=torch.float32)
            if gk is not None:
                gg = torch.cat([gk.float(), gg[:, gk.shape[1]:]], 1)
            q = uv.detach().float() + gg * (uv.float() - uv.detach().float())
        P, K, A = P.float(), K.float(), A.float()
        px, py = q[..., 0] * size - A[:, None, 0, 2], q[..., 1] * size - A[:, None, 1, 2]
        a, b, c, d = A[:, None, 0, 0], A[:, None, 0, 1], A[:, None, 1, 0], A[:, None, 1, 1]
        det = a * d - b * c
        u, v = (d * px - b * py) / det, (a * py - c * px) / det  # full-image px
        x, y = (u - K[:, None, 0, 2]) / K[:, None, 0, 0], (v - K[:, None, 1, 2]) / K[:, None, 1, 1]
        e = self.p25d_border
        inside = ((q > e) & (q < 1 - e)).all(-1)
        w = inside.float()
        w = torch.where(w.sum(1, keepdim=True) >= 3, w, torch.ones_like(w))  # (nearly) all at the edge: every point
        w = w / w.sum(1, keepdim=True)
        ex, ey = (w * x).sum(1, keepdim=True), (w * y).sum(1, keepdim=True)
        mx, my = (w * (x * P[..., 2] - P[..., 0])).sum(1, keepdim=True), (w * (y * P[..., 2] - P[..., 1])).sum(1, keepdim=True)
        ax, ay = P[..., 0] - x * P[..., 2] + mx, P[..., 1] - y * P[..., 2] + my  # residuals ax + cx t_z, ay + cy t_z
        cx, cy = ex - x, ey - y
        tz = -(w * (ax * cx + ay * cy)).sum(1, keepdim=True) / (w * (cx * cx + cy * cy)).sum(1, keepdim=True).clamp_min(1e-10)
        tz = tz.clamp_min(0.3)
        z = P[..., 2] + tz
        self.last_abs_z = z  # each point's absolute camera depth (m): train_fab --w dense_tz
        lift = torch.stack([x * z - (mx + ex * tz), y * z - (my + ey * tz), P[..., 2]], -1)
        return torch.where(inside[..., None], self.p25d_mix * lift + (1 - self.p25d_mix) * P, P)


def dense_mask_depth(weights):
    """3x3 convs of a saved person-mask head (0: none)."""
    return sum(1 for k, v in weights.items() if k.startswith("dense_head.mask_dconv.") and k.endswith(".weight")
               and v.ndim == 4)


def dense_mask_query(weights):
    """Query of a saved person-mask head: "box" (one learned query) or "token" (the pose token), + "+pts" when it reads
    the projected dense points."""
    q = "box" if "dense_head.mask_qvec" in weights else "token"
    w0 = weights.get("dense_head.mask_dconv.0.weight")
    ext = "dense_head.mask_extent_on" in weights
    nkp = len(weights["dense_head.mask_kp_idx"]) if "dense_head.mask_kp_idx" in weights else 0
    q += "+pts" if w0 is not None and w0.shape[1] == weights["dense_head.point"].shape[1] + 3 + int(ext) + nkp else ""
    return (q + ("+extent" if ext else "") + ("+pool" if "dense_head.mask_down.weight" in weights else "")
            + ("+full" if "dense_head.mask_up4.0.weight" in weights else "")
            + ("+corners" if "dense_head.mask_pos.weight" in weights else "") + ("+kp" if nkp else ""))


def dense_xyz_layers(weights):
    """Number of xyz transformer layers of a saved dense head (0 for the per-point MLP)."""
    ids = {k.split(".")[3] for k in weights if k.startswith("dense_head.pt.layers.")}
    return len(ids)


def dense_head_kwargs(weights):
    """Points-decoder options of a saved dense head for add_dense_head(**...): xyz_dim, xyz_mem, p25d / p25d_border,
    p25d_vis ({} for older heads)."""
    kw = {}
    w0 = weights.get("dense_head.xyz.0.weight")
    if w0 is not None and w0.shape[0] != 256:
        kw["xyz_dim"] = int(w0.shape[0])
    if "dense_head.pt_dec.weight" in weights:
        kw["xyz_mem"] = "img+dec"
    if "dense_head.p25d" in weights:
        kw["p25d"], kw["p25d_border"] = (round(float(x), 6) for x in weights["dense_head.p25d"])
    if "dense_head.p25d_vis" in weights:
        kw["p25d_vis"] = True
    if "dense_head.pt_ray.weight" in weights:
        kw["ray_in"] = True
    if "dense_head.uv_off.weight" in weights:
        kw["uv_off"] = round(float(weights["dense_head.uv_off_band"]), 6)
    return kw


def local_soft_argmax(logits, r):
    """logits (B, N, H, W) -> uv (B, N, 2) in [0, 1]: the soft-argmax within r cells (Chebyshev) of each map's peak."""
    H, W = logits.shape[-2:]
    peak = logits.flatten(2).argmax(-1)
    py, px = (peak // W)[..., None, None], (peak % W)[..., None, None]
    iy = torch.arange(H, device=logits.device)[:, None]
    ix = torch.arange(W, device=logits.device)[None]
    near = ((iy - py).abs() <= r) & ((ix - px).abs() <= r)
    p = logits.masked_fill(~near, float("-inf")).flatten(2).softmax(-1).view_as(logits)
    xs = (torch.arange(W, device=logits.device, dtype=p.dtype) + 0.5) / W
    ys = (torch.arange(H, device=logits.device, dtype=p.dtype) + 0.5) / H
    return torch.stack([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], -1)


def dense_hr(weights):
    """Points with a 2x-resolution heatmap in a saved dense head (0: none)."""
    return int(weights["dense_head.hr_points"]) if "dense_head.hr_points" in weights else 0


def dense_part_depth(weights):
    """3x3 convs of a saved d6-style part head (0: the original query + part_conv head, or none)."""
    return sum(1 for k, v in weights.items() if k.startswith("dense_head.part_dconv.") and k.endswith(".weight")
               and v.ndim == 4)


def mirror_map(verts):
    """Left-right vertex correspondence of a (near-)symmetric template (x = 0 plane): nearest neighbour of each
    vertex's mirror image; returns (map, mean / max mirror distance in m)."""
    verts = np.asarray(verts, np.float64)
    m = verts * np.array([-1.0, 1.0, 1.0])
    d2 = ((m[:, None] - verts[None]) ** 2).sum(-1)
    idx = d2.argmin(1)
    dist = np.sqrt(d2[np.arange(len(verts)), idx])
    return idx, float(dist.mean()), float(dist.max())


def symmetric_subset(verts, k):
    """~k template vertices: farthest-point samples closed under the mirror map; returns (sorted indices,
    position of each index's mirror in that list)."""
    mirror, _, _ = mirror_map(verts)
    idx = set(farthest_points(verts, max(k // 2, 1)).tolist())
    while True:
        more = {int(mirror[i]) for i in idx} - idx
        if not more:
            break
        idx |= more
    idx = np.sort(np.asarray(sorted(idx)))
    pos = {int(v): i for i, v in enumerate(idx)}
    return idx, np.asarray([pos[int(mirror[v])] for v in idx])


def farthest_points(verts, k, start=None):
    """Greedy farthest-point sampling of k vertex indices of verts (V, 3) (sorted)."""
    verts = np.asarray(verts, np.float64)
    idx = [int(np.argmin(verts[:, 1])) if start is None else start]
    d = np.linalg.norm(verts - verts[idx[0]], axis=1)
    for _ in range(k - 1):
        i = int(d.argmax())
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(verts - verts[i], axis=1))
    return np.sort(np.asarray(idx))
