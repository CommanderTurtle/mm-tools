"""FAB-L / FAB-S: SAM 3D Body's promptable decoder and heads on a small student backbone.

The student (a DINOv3 ViT at the teacher's 512 px input, 16 px patches -> the same 32 x 32 token grid) is
projected to the teacher's embedding width and replaces the 840M backbone; everything after the backbone (ray
conditioning, prompt / keypoint tokens, iterative decoder, MHR and camera heads) is SAM 3D Body's own, initialised
from its checkpoint. Trained with token-level feature distillation against the frozen teacher backbone plus the
FAB output / ground-truth losses (train/train_fab.py --arch fabd).
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn

from .model import BACKBONES


class StudentBackbone(nn.Module):
    """timm DINOv3 ViT -> (B, out_dim, H/16, W/16), the teacher backbone's output format."""

    def __init__(self, backbone: str = "dinov3_vitb16", size: int = 512, out_dim: int = 1280, pretrained: bool = True):
        super().__init__()
        import timm

        timm_name, dim = BACKBONES[backbone]
        self.vit = timm.create_model(timm_name, pretrained=pretrained, img_size=size, num_classes=0,
                                     dynamic_img_size=False)
        self.num_prefix = self.vit.num_prefix_tokens
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim))

    def forward(self, x, extra_embed=None):
        t = self.proj(self.vit.forward_features(x)[:, self.num_prefix:])
        b, n, c = t.shape
        h = int(round(n ** 0.5))
        return t.transpose(1, 2).reshape(b, c, h, n // h)


class FABDecoder(nn.Module):
    """forward(img (B, 3, 512, 512) in [0, 1], bbox_center, bbox_scale, ori_size, cam_int, affine) -> SAM 3D Body's
    `mhr` output dict (+ unconditioned "backbone_embeddings" for the feature loss)."""

    def __init__(self, sam3db, backbone: str = "dinov3_vitb16", size: int = 512, pretrained: bool = True,
                 dec_layers: int = 0):
        super().__init__()
        self.body = sam3db
        if dec_layers:  # SAM 3D Body's 3-layer estimate is within 0.2 mm of its 6-layer one (3DPW PA-MPJPE)
            self.body.decoder.layers = self.body.decoder.layers[:dec_layers]
        out_dim = self._teacher_dim(sam3db)
        self.body.backbone = StudentBackbone(backbone, size, out_dim, pretrained)
        self.body.backbone_dtype = torch.float32  # the student runs under autocast
        self.size = size
        self.g1_head = None
        self.kp2d_head = None
        self.dense_head = None
        self.contact_head = None
        self.vis_head = None
        self.dense_point_count = None  # inference-only prefix; None retains every training query
        self.map_outputs = False  # inference: + part_map / mask_map (B, 256, 256) uint8 (fab/seg_maps.py)

    def add_dense_head(self, surface_idx, residual=False, parts=False, xyz_layers=0, part_depth=0, hr=0, mask_depth=0,
                       mask_query="box", xyz_dim=256, xyz_mem="img", p25d=0.0, p25d_border=0.02, p25d_vis=False,
                       ray_in=False, uv_off=0.0):
        """Direct dense keypoints (fab/dense_head.py): 70 MHR keypoints, 14 evaluation joints and the SMPL
        template vertices `surface_idx`, each with heatmap 2D and regressed root-relative 3D; parts: + body part
        segmentation (fab/body_parts.py; part_depth > 0: the d6-style head); xyz_layers: transformer layers
        before the 3D regression; hr: the first hr points' 2D from a 2x finer heatmap; xyz_dim / xyz_mem / p25d: the
        points decoder (saved heads: dense_head.dense_head_kwargs)."""
        from .body_parts import N_PARTS
        from .dense_head import BodyDenseHead

        self.dense_head = BodyDenseHead(surface_idx, img_dim=self._teacher_dim(self.body),
                                        tok_dim=self.body.head_pose.proj.embed_dims, residual=residual,
                                        parts=N_PARTS if parts else 0, xyz_layers=xyz_layers, part_depth=part_depth,
                                        hr=hr, mask_depth=mask_depth, mask_query=mask_query, xyz_dim=xyz_dim,
                                        xyz_mem=xyz_mem, p25d=p25d, p25d_border=p25d_border, p25d_vis=p25d_vis,
                                        ray_in=ray_in, uv_off=uv_off)
        return self.dense_head

    def add_kp2d_head(self):
        """Independent 2D keypoints: a zero-initialised residual (crop pixels) on the mesh's projected keypoints from
        the decoder's final keypoint tokens, supervised by 2D labels; the mesh keeps its own losses (weak agreement)."""
        dim = self.body.head_pose.proj.embed_dims
        self.kp2d_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 2))
        nn.init.zeros_(self.kp2d_head[-1].weight)
        nn.init.zeros_(self.kp2d_head[-1].bias)
        return self.kp2d_head

    def add_g1_head(self, hidden: int = 512, joints_input: bool = False):
        """In-model retargeting: pose token -> Unitree G1 joint angles (29, URDF order) + root rotation (6D, camera
        frame), trained on G1 retargeting targets (the g1_body sidecars, docs/training.md). joints_input: also feed the (detached, pelvis-relative) 3D MHR
        keypoints of the mesh path, so the head can learn the deterministic skeleton -> robot mapping."""
        dim = self.body.head_pose.proj.embed_dims + (70 * 3 if joints_input else 0)
        self.g1_joints_input = joints_input
        self.g1_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                     nn.GELU(), nn.Linear(hidden, 29 + 6))
        nn.init.zeros_(self.g1_head[-1].weight)
        with torch.no_grad():
            self.g1_head[-1].bias.copy_(torch.tensor([0.0] * 29 + [1.0, 0, 0, 0, 1.0, 0]))
        return self.g1_head

    def add_contact_head(self, attn_dim=0, n_out=8):
        """Explicit contact probabilities (fab/contact_head.py) from the pose token and the dense head's features
        at the foot / knee / wrist keypoints; needs the dense head."""
        from .contact_head import ContactHead

        self.contact_head = ContactHead(self.body.head_pose.proj.embed_dims, 2 * self.dense_head.dim, attn_dim=attn_dim,
                                        img_dim=self._teacher_dim(self.body), n_out=n_out)
        self.contact_residual = bool(int(self.dense_head.residual))  # host flag: no device read in the forward (graphs)
        return self.contact_head

    def add_vis_head(self, attn_dim=128, human=False):
        """Per-keypoint visibility (fab/vis_head.py): visible / occluded / outside the crop for the 70 MHR
        keypoints and the 14 joints, from the dense head's features at those points; needs the dense head.
        human: + the human-annotation classifier (then vis_logits = human, vis_logits_rendered = rendered labels)."""
        from .vis_head import VisHead

        self.vis_head = VisHead(self.body.head_pose.proj.embed_dims, 2 * self.dense_head.dim, attn_dim=attn_dim,
                                img_dim=self._teacher_dim(self.body), human=human)
        self.vis_residual = bool(int(self.dense_head.residual))
        return self.vis_head

    @staticmethod
    def _teacher_dim(sam3db):
        enc = getattr(sam3db.backbone, "encoder", None)
        return int(getattr(enc, "embed_dim", 1280))

    def forward(self, img, bbox_center, bbox_scale, ori_size, cam_int, affine=None, return_vertices=True, mask=None,
                mask_score=None, mask_box=None):
        """mask (B, 1, S, S) / mask_score (B,): optional person-mask prompt in crop coordinates (default: none).
        mask_box (B, 4): the prompt box x0, y0, x1, y1 in crop units, read only by a "+extent" person-mask head."""
        b = img.shape[0]
        dev = img.device
        batch = {
            "img": img[:, None], "bbox_center": bbox_center[:, None], "bbox_scale": bbox_scale[:, None],
            "ori_img_size": ori_size[:, None].float(), "cam_int": cam_int, "affine_trans": affine[:, None],
            "person_valid": torch.ones(b, 1, device=dev),
            "mask": torch.zeros(b, 1, 1, self.size, self.size, device=dev) if mask is None else mask[:, None].float(),
            "mask_score": torch.zeros(b, 1, device=dev) if mask_score is None else mask_score.reshape(b, 1).float(),
            "img_size": torch.full((b, 1, 2), float(self.size), device=dev),
        }
        self.body._initialize_batch(batch)
        self.body.final_vertices = return_vertices
        out = self.body.forward_step(batch, decoder_type="body")
        mhr = dict(out["mhr"])
        mhr["image_embeddings"] = out["image_embeddings"]
        mhr["backbone_embeddings"] = out["backbone_embeddings"]
        mhr["token"] = mhr.get("pose_token")
        if self.kp2d_head is not None and mhr.get("kp_tokens") is not None and affine is not None:
            d_crop = self.kp2d_head(mhr["kp_tokens"].float()).float() * self.size  # (B, 70, 2) crop pixels
            lin = affine[:, :, :2].float()  # full image -> crop (similarity, possibly rolled)
            d_full = torch.linalg.solve(lin[:, None].expand(-1, d_crop.shape[1], -1, -1), d_crop[..., None])[..., 0]
            mhr["pred_keypoints_2d_ind"] = mhr["pred_keypoints_2d"][..., :2].float() + d_full
        if self.dense_head is not None and mhr["token"] is not None:
            if self.contact_head is not None:
                self.dense_head.keep_feats = self.contact_head.points
            if self.vis_head is not None:  # the visibility points (0-83) include the contact head's (MHR keypoints)
                self.dense_head.keep_feats = self.vis_head.points
            kw = {}
            if self.dense_head.p25d_mix > 0 or self.dense_head.pt_ray is not None:  # 2.5D lift / viewing rays: the crop camera
                kw.update(cam_int=cam_int, affine=affine)
            if self.dense_head.pt_dec is not None:  # point transformer memory + the decoder's tokens
                kw["dec_tokens"] = torch.cat([mhr["token"].reshape(b, 1, -1), mhr["kp_tokens"]], 1)
            if (self.dense_head.mask_extent or self.dense_head.mask_corners) and self.dense_head.mask_dconv is not None:
                kw["mask_box"] = mask_box
            uv, xyz, parts, *mask = self.dense_head(out["image_embeddings"], mhr["token"].reshape(b, -1), img,
                                                    point_count=self.dense_point_count, **kw)
            mhr["dense_uv"], mhr["dense_xyz"] = uv, xyz
            if self.dense_head.p25d_mix > 0:
                mhr["dense_xyz3d"] = self.dense_head.last_xyz3d  # the regressed 3D before the 2.5D lift
                mhr["dense_abs_z"] = self.dense_head.last_abs_z  # the lift's absolute point depths (m)
            if parts is not None:
                mhr["part_logits"] = parts
            if mask:
                mhr["mask_logits"] = mask[0]  # (B, 1, S/2, S/2) person mask
            if self.map_outputs and (parts is not None or mask):  # inference: the crop maps inside the (graphed) forward
                from .seg_maps import crop_maps_gpu

                pm, mm = crop_maps_gpu(parts, mask[0] if mask else None)
                if pm is not None:
                    mhr["part_map"] = pm
                if mm is not None:
                    mhr["mask_map"] = mm
            if self.contact_head is not None:
                ix = self.contact_head.points
                pts = xyz[:, ix].float()
                if self.contact_residual:  # offsets on the MHR keypoints (the residual head's anchors)
                    pts = pts + mhr["pred_keypoints_3d"][:, ix].float().detach()
                cf = self.dense_head.last_feats if self.vis_head is None else self.dense_head.last_feats[:, ix]
                logits, hid = self.contact_head(mhr["token"].reshape(b, -1).float(), cf.float(),
                                                pts, uv[:, ix].float(), mhr["pred_cam_t"].float(),
                                                out["image_embeddings"] if self.contact_head.attn_dim else None)
                mhr["contact_logits"], mhr["contact_feat"], mhr["contact_pts"] = logits, hid, pts
            if self.vis_head is not None:
                vp = xyz[:, :84].float()
                if self.vis_residual:  # MHR keypoint offsets on the mesh's keypoints (the joints' offsets ~ 0)
                    vp = torch.cat([vp[:, :70] + mhr["pred_keypoints_3d"].float().detach(), vp[:, 70:]], 1)
                vp = vp - vp[:, 72:74].mean(1, keepdim=True)  # root-relative at the J14 hips
                mhr["vis_logits"] = self.vis_head(mhr["token"].reshape(b, -1).float(), self.dense_head.last_feats.float(),
                                                  vp, uv[:, :84].float(),
                                                  out["image_embeddings"] if self.vis_head.attn_dim else None)
                if isinstance(mhr["vis_logits"], tuple):  # human-semantics head: the primary output (deployed /
                    # benchmarked as vis_logits); the rendered-label classifier stays as vis_logits_rendered
                    mhr["vis_logits_rendered"], mhr["vis_logits"] = mhr["vis_logits"]
            if self.vis_head is not None and self.dense_head.p25d_mix > 0 and self.dense_head.p25d_vis_gate:
                # vis-gated 2.5D lift: each keypoint's lift weighted by P(visible) (detached), so occluded / out-of-frame
                # keypoints keep the regressed 3D; points without a visibility output (the surface) keep the lift
                P = self.dense_head.last_xyz3d.float()
                pv = mhr["vis_logits"].float().softmax(-1)[..., 0].detach()
                g = torch.cat([pv, torch.ones_like(P[:, pv.shape[1]:, 0])], 1)[..., None]
                mhr["dense_xyz"] = P + g * (xyz.float() - P)
        if self.g1_head is not None and mhr["token"] is not None:
            x = mhr["token"].float().reshape(b, -1)
            if getattr(self, "g1_joints_input", False):
                kp = mhr["pred_keypoints_3d"].float().detach()
                x = torch.cat([x, (kp - kp[:, [9, 10]].mean(1, keepdim=True)).reshape(b, -1) * 5.0], -1)
            g = self.g1_head(x)
            mhr["g1_q"], mhr["g1_rot6d"] = g[:, :29], g[:, 29:]
        return mhr

    @property
    def head_pose(self):
        return self.body.head_pose

    @property
    def head_camera(self):
        return self.body.head_camera


def build_fab_decoder(teacher: str | Path, backbone: str = "dinov3_vitb16", size: int = 512,
                      device: str = "cuda", pretrained: bool = True, dec_layers: int = 0, teacher_weights: bool = False):
    """-> (student FABDecoder, frozen teacher backbone), built from SAM 3D Body's config and the MHR model in the folder
    `teacher` (model_config.yaml, assets/mhr_model.pt). Inference loads every network weight from the FAB checkpoint, so
    SAM 3D Body's network weights are not needed; teacher_weights=True also loads <teacher>/model.ckpt (training)."""
    from fab import load_sam_3d_body

    teacher = Path(teacher)
    teacher = teacher.parent if teacher.suffix == ".ckpt" else teacher
    sam3db, _ = load_sam_3d_body(checkpoint_path=str(teacher / "model.ckpt") if teacher_weights else "", device=device,
                                 mhr_path=str(teacher / "assets" / "mhr_model.pt"), config_dir=str(teacher))
    teacher_backbone = copy.deepcopy(sam3db.backbone).eval()
    for q in teacher_backbone.parameters():
        q.requires_grad_(False)
    model = FABDecoder(sam3db, backbone, size, pretrained, dec_layers).to(device)
    return model, teacher_backbone
