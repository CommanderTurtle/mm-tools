"""FAB's decoder-free image model: a small DINOv3 ViT + token pooling + SAM 3D Body's MHR and camera heads.

Decoder-free: the person crop goes through one ViT; a learned query, conditioned on the crop's position and
size in the camera (CLIFF-style), attends to the patch tokens and becomes the pose token that SAM 3D Body's
MHR head (rig parameters -> mesh, 70 keypoints) and perspective camera head read. The heads keep SAM 3D Body's
architecture and output format (so the evaluator and the estimator API are shared) and start from its weights;
the rig runs in eager PyTorch (MHRHead.enable_fast), so gradients flow through it.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .models.heads import build_head
from .utils.config import get_config

BACKBONES = {
    # timm name, embed dim
    "dinov3_vits16": ("vit_small_patch16_dinov3.lvd1689m", 384),
    "dinov3_vits16plus": ("vit_small_plus_patch16_dinov3.lvd1689m", 384),
    "dinov3_vitb16": ("vit_base_patch16_dinov3.lvd1689m", 768),
    "dinov3_vitl16": ("vit_large_patch16_dinov3.lvd1689m", 1024),
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FABImage(nn.Module):
    def __init__(self, cfg, backbone: str = "dinov3_vitb16", size: int = 384, pool_layers: int = 2,
                 pretrained: bool = True, iters: int = 1):
        super().__init__()
        import timm

        timm_name, dim = BACKBONES[backbone]
        self.backbone_name, self.size = backbone, size
        # fixed input size: position embeddings / RoPE precomputed (CUDA-graph and export friendly)
        self.vit = timm.create_model(timm_name, pretrained=pretrained, img_size=size, num_classes=0,
                                     dynamic_img_size=False)
        self.num_prefix = self.vit.num_prefix_tokens
        self.cond = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerDecoderLayer(dim, nhead=max(dim // 64, 1), dim_feedforward=4 * dim, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.pool = nn.TransformerDecoder(layer, num_layers=pool_layers)
        tok_dim = cfg.MODEL.DECODER.DIM
        self.to_token = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, tok_dim))
        self.head_pose = build_head(cfg, cfg.MODEL.PERSON_HEAD.POSE_TYPE)
        self.head_camera = build_head(cfg, cfg.MODEL.PERSON_HEAD.CAMERA_TYPE)
        self.init_pose = nn.Embedding(1, self.head_pose.npose)
        self.init_camera = nn.Embedding(1, self.head_camera.ncam)
        # iterative refinement (as SAM 3D Body's decoder): each extra pass feeds the current estimate back into the
        # query, cross-attends to the patch tokens again and lets the head predict a correction of the estimate
        self.iters = iters
        if iters > 1:
            self.feedback = nn.Linear(self.head_pose.npose + self.head_camera.ncam, dim)
            nn.init.zeros_(self.feedback.weight)
            nn.init.zeros_(self.feedback.bias)
            self.refine = nn.ModuleList([nn.TransformerDecoder(layer, num_layers=1) for _ in range(iters - 1)])
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), False)

    @staticmethod
    def camera_condition(bbox_center, bbox_scale, ori_size, cam_int):
        """CLIFF condition: box centre offset from the image centre and box size, both over the focal length."""
        f = cam_int[:, 0, 0]
        cx = (bbox_center[:, 0] - ori_size[:, 0] / 2) / f
        cy = (bbox_center[:, 1] - ori_size[:, 1] / 2) / f
        return torch.stack([cx, cy, bbox_scale[:, 0] / f], dim=-1)

    def forward(self, img, bbox_center, bbox_scale, ori_size, cam_int, return_vertices=True):
        """img (B, 3, S, S) RGB in [0, 1] (the teacher's crop, resized); geometry in original-image pixels.
        Returns SAM 3D Body's `mhr` output dict (pred_keypoints_3d / _2d in original pixels, pred_vertices,
        pred_cam_t, pose / shape parameters, ...)."""
        batch = img.shape[0]
        tokens = self.vit.forward_features((img - self.mean) / self.std)
        cond = self.camera_condition(bbox_center, bbox_scale, ori_size, cam_int)
        query = self.query.expand(batch, -1, -1) + self.cond(cond.to(tokens.dtype))[:, None]
        h = self.pool(query, tokens)
        token = self.to_token(h[:, 0]).float()

        last = self.iters == 1
        out = self.head_pose(token, self.init_pose.weight.expand(batch, -1), return_vertices=return_vertices and last)
        pred_cam = self.head_camera(token, self.init_camera.weight.expand(batch, -1))
        aux = []
        for k in range(self.iters - 1):
            aux.append(out["pred_keypoints_3d"])
            fb = self.feedback(torch.cat([out["pred_raw"], pred_cam], -1).to(h.dtype))
            h = self.refine[k](h + fb[:, None], tokens)
            token = self.to_token(h[:, 0]).float()
            last = k == self.iters - 2
            out = self.head_pose(token, out["pred_raw"], return_vertices=return_vertices and last)
            pred_cam = self.head_camera(token, pred_cam)
        cam = self.head_camera.perspective_projection(
            out["pred_keypoints_3d"], pred_cam, bbox_center, bbox_scale[:, 0], ori_size, cam_int
        )
        out.update(cam)
        out["pred_cam"] = pred_cam
        out["token"] = token
        if aux:
            out["aux_keypoints_3d"] = aux  # intermediate estimates (deep supervision)
        return out

    def load_teacher_heads(self, ckpt_path: str | Path) -> list[str]:
        """Initialise the MHR / camera heads and their init embeddings from SAM 3D Body."""
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        prefixes = ("head_pose.", "head_camera.", "init_pose.", "init_camera.")
        sub = {k: v for k, v in state.items() if k.startswith(prefixes) and not k.startswith("head_pose.mhr.")}
        missing, unexpected = self.load_state_dict(sub, strict=False)
        return [k for k in missing if k.startswith(prefixes) and not k.startswith("head_pose.mhr.")] + list(unexpected)


def load_fab_image(cfg_path: str | Path, mhr_path: str | Path, ckpt: str | Path | None = None,
                   device: str = "cuda", **kwargs) -> FABImage:
    cfg = get_config(str(cfg_path))
    cfg.defrost()
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = str(mhr_path)
    cfg.freeze()
    model = FABImage(cfg, pretrained=ckpt is None, **kwargs)
    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        weights = dict(state.get("model", state))
        weights.update(state.get("ema") or {})  # EMA float weights over the raw state (integer buffers kept)
        model.load_state_dict(weights, strict=False)
    model = model.to(device).eval()
    model.head_pose.enable_fast()
    return model
