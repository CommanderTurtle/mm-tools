"""Adapter: official WiLoR (https://github.com/rolpotamias/WiLoR) -> benchmark prediction format.

Follows WiLoR's demo.py (YOLO detector, conf 0.3, ViTDetDataset with rescale_factor 2.0). The crop
camera is converted to the full image with the TRUE intrinsics (WiLoR's demo assumes
focal = 5000/256 * max(W, H) and a centered principal point); PA-metrics are unaffected.
"""
import os
import sys

import numpy as np
import torch


class WiLoRMethod:
    name = "wilor"

    def __init__(self, repo=None, device="cuda", det_conf=0.3, det_iou=0.5, rescale_factor=2.0,
                 fast=False, batch_size=16, ckpt=None, cfg_path=None, **_):
        repo = repo or os.path.join(os.environ.get("THIRDPARTY", "thirdparty"), "WiLoR")  # checkout: docs/evaluation.md (baselines)
        sys.path.insert(0, repo)
        cwd = os.getcwd()
        os.chdir(repo)  # WiLoR resolves ./mano_data and ./pretrained_models relative to CWD
        try:
            from ultralytics import YOLO
            from wilor.datasets.vitdet_dataset import ViTDetDataset
            from wilor.models import load_wilor

            model, cfg = load_wilor(checkpoint_path=ckpt or "./pretrained_models/wilor_final.ckpt",
                                    cfg_path=cfg_path or "./pretrained_models/model_config.yaml")
            detector = YOLO("./pretrained_models/detector.pt")
        finally:
            os.chdir(cwd)
        if fast:
            torch.set_float32_matmul_precision("high")
            model = model.half()
            model.backbone = torch.compile(model.backbone)
            model.backbone.skip_blocks = True
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.detector = detector.to(self.device)
        self.cfg, self.ViTDetDataset = cfg, ViTDetDataset
        self.det_conf, self.det_iou = det_conf, det_iou  # same detector + NMS IoU as the fah adapter
        self.rescale_factor, self.fast, self.batch_size = rescale_factor, fast, batch_size
        self.name = "wilor-fast" if fast else "wilor"

    @torch.no_grad()
    def __call__(self, image_bgr, K):
        det = self.detector(image_bgr, conf=self.det_conf, iou=self.det_iou, verbose=False)[0]
        if len(det.boxes) == 0:
            return []
        boxes = det.boxes.xyxy.cpu().numpy()
        right = det.boxes.cls.cpu().numpy()
        scores = det.boxes.conf.cpu().numpy()
        ds = self.ViTDetDataset(self.cfg, image_bgr, boxes, right, rescale_factor=self.rescale_factor, fp16=self.fast)
        loader = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
        if K is None:  # unknown camera: WiLoR's own default (focal 5000/256 * max side, centered)
            h, w = image_bgr.shape[:2]
            f = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * max(h, w)
            K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32)
        f, cx0, cy0 = float(K[0, 0]), float(K[0, 2]), float(K[1, 2])
        out_hands = []
        for batch in loader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = self.model(batch)
            mult = 2 * batch["right"] - 1
            cam = out["pred_cam"].float().clone()
            cam[:, 1] = mult * cam[:, 1]
            bc, bsz = batch["box_center"].float(), batch["box_size"].float()
            bs = bsz * cam[:, 0] + 1e-9
            cam_t = torch.stack([cam[:, 1] + 2 * (bc[:, 0] - cx0) / bs,
                                 cam[:, 2] + 2 * (bc[:, 1] - cy0) / bs,
                                 2 * f / bs], dim=-1).cpu().numpy()
            verts = out["pred_vertices"].float().cpu().numpy()
            joints = out["pred_keypoints_3d"].float().cpu().numpy()
            # WiLoR's own 2D (weak-perspective in the crop, [-0.5, 0.5] crop units), as in its demo
            kp2 = out["pred_keypoints_2d"].float().cpu().numpy()
            bc_np, bsz_np = bc.cpu().numpy(), bsz.cpu().numpy()
            m = mult.cpu().numpy()
            for n in range(verts.shape[0]):
                v, j = verts[n].copy(), joints[n].copy()
                v[:, 0] *= m[n]
                j[:, 0] *= m[n]
                j3, v3 = j + cam_t[n], v + cam_t[n]
                uv = j3[:, :2] / np.clip(j3[:, 2:], 1e-6, None) * f + np.array([cx0, cy0])
                k2 = kp2[n].copy()
                k2[:, 0] *= m[n]
                uv_crop = k2 * bsz_np[n] + bc_np[n]
                idx = len(out_hands)
                out_hands.append(dict(is_right=bool(right[idx] > 0.5), score=float(scores[idx]), bbox=boxes[idx],
                                      joints3d=j3, verts3d=v3, joints2d=uv, joints2d_crop=uv_crop))
        return out_hands
