"""Adapter: official HACO (Jung & Lee, "Learning Dense Hand Contact Estimation from Imbalanced Data", NeurIPS 2025;
https://github.com/dqj5182/HACO_RELEASE, CC BY-NC 4.0; checkpoint haco_neurips_hamer_checkpoint.ckpt from the HF repo
dqj5182/haco-checkpoints) -> per-hand dense contact on MANO's 778 vertices.

The contact baseline of the evaluation (docs/evaluation.md): HACO's ViT-H (HaMeR backbone) on one hand crop, as its test-time
generate_patch_image_contact (the box warped axis-aligned to 256 x 256, aspect not preserved, black border; left hands
mirrored first), sigmoid of contact_out; the hand score = the max vertex probability, the flag = score >= thr. Crops:
  hand2x  the demo's crop: the hand box x 2 about its centre (ho_big_bbox_expand_ratio 2.0) -- HOT3D / DexYCB with
          ground-truth boxes (the contact table) and videos with detector boxes;
  mask13  HACO's MOW evaluation crop: mask2bbox of the hand + object mask with expansion 1.3 (each side out by 0.65 x the
          size, clipped to the image) -- MOW test (mask_box below);
  box13   the hand box x 1.3 about its centre (a variant, not used in the tables).
jpeg_quality: the HOT3D / DexYCB crops of the evaluation went through a JPEG round trip at quality 95 (the default here; it
moves single probabilities by up to ~0.2, vertex decisions at 0.56 agree on ~99.5 %); the MOW crops did not
(jpeg_quality=None reproduces those predictions exactly).
Threshold: 0.56 = HACO's threshold picked like ours (pooled vertex F1 on the pools' validation sequences); MOW is reported at
HACO's own 0.52. Boxes: given (ground truth: predict()), or WiLoR's YOLO hand detector with the WiLoR / HaMeR adapters'
conf 0.3 / NMS IoU 0.5 (__call__), so every baseline on a video shares one detection.

Run in HACO's own environment (its README); the checkout under $THIRDPARTY/HACO (release_checkpoint/, data/base_data).
"""
import os
import sys

import cv2
import numpy as np


def haco_patch(img, box_xywh, is_right, out=256):
    """HACO generate_patch_image_contact at test time (no scale / rotation): the box (x, y, w, h) -> out x out, axis-aligned
    (aspect not preserved), black border; left hands mirrored first."""
    x, y, w, h = box_xywh
    if not is_right:
        img, x = img[:, ::-1], img.shape[1] - (x + w)
    cx, cy = x + 0.5 * w, y + 0.5 * h
    src = np.float32([[cx, cy], [cx, cy + 0.5 * h], [cx + 0.5 * w, cy]])
    dst = np.float32([[out / 2, out / 2], [out / 2, out], [out, out / 2]])
    return cv2.warpAffine(np.ascontiguousarray(img), cv2.getAffineTransform(src, dst), (out, out), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def crop_box(box_xyxy, rule, width, height):
    """A hand box (x0, y0, x1, y1) -> HACO's crop window (x, y, w, h) by rule hand2x | box13 | mask13."""
    x0, y0, x1, y1 = (float(v) for v in box_xyxy)
    w, h = x1 - x0, y1 - y0
    if rule == "hand2x":
        return x0 + w / 2 - w, y0 + h / 2 - h, 2 * w, 2 * h
    if rule == "box13":
        return x0 + w / 2 - 0.65 * w, y0 + h / 2 - 0.65 * h, 1.3 * w, 1.3 * h
    if rule == "mask13":
        lo = np.maximum(np.array([x0, y0]) - 0.65 * np.array([w, h]), 0)
        hi = np.minimum(np.array([x1, y1]) + 0.65 * np.array([w, h]), [width - 1, height - 1])
        return lo[0], lo[1], hi[0] - lo[0] + 1, hi[1] - lo[1] + 1
    raise ValueError(rule)


def mask_box(mask, expand=1.3):
    """HACO's MOW crop: mask2bbox of a binary hand + object mask with expansion `expand` -> (x, y, w, h) (integer pixels)."""
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1
    x0, y0 = max(0, int(x0 - w * expand / 2)), max(0, int(y0 - h * expand / 2))
    x1, y1 = min(mask.shape[1] - 1, int(x1 + w * expand / 2)), min(mask.shape[0] - 1, int(y1 + h * expand / 2))
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


class HACOMethod:
    name = "haco"

    def __init__(self, repo=None, wilor_repo=None, device="cuda", crop="hand2x", thr=0.56, det_conf=0.3, det_iou=0.5,
                 batch_size=32, ckpt=None, jpeg_quality=95, **_):
        import torch
        from torchvision.transforms import Normalize

        root = os.environ.get("THIRDPARTY", "thirdparty")
        repo = os.path.abspath(repo or os.path.join(root, "HACO"))  # checkout: docs/evaluation.md (baselines)
        self.wilor_repo = os.path.abspath(wilor_repo or os.path.join(root, "WiLoR"))
        sys.path.insert(0, repo)
        cwd = os.getcwd()
        os.chdir(repo)  # HACO resolves its config, MANO data and checkpoint relative to CWD
        try:
            from lib.core.config import cfg, update_config
            from lib.models.model import HACO

            update_config(backbone_type="hamer", exp_dir=os.path.join(os.environ.get("TMPDIR", "/tmp"), "haco_eval"))
            self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
            model = HACO().to(self.device).eval()
            model.load_state_dict(torch.load(ckpt or "release_checkpoint/haco_neurips_hamer_checkpoint.ckpt", map_location=self.device)["state_dict"])
        finally:
            os.chdir(cwd)
        self.model, self.norm = model, Normalize(mean=cfg.MODEL.img_mean, std=cfg.MODEL.img_std)
        self.crop, self.thr, self.det_conf, self.det_iou, self.batch_size = crop, thr, det_conf, det_iou, batch_size
        self.jpeg_quality = jpeg_quality
        self.detector = None
        self.name = f"haco-{crop}"

    def contact(self, patches_bgr):
        """(n, 256, 256, 3) BGR uint8 patches -> (n, 778) contact probabilities."""
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(patches_bgr), self.batch_size):
                x = torch.stack([self.norm(torch.from_numpy(np.ascontiguousarray(p[:, :, ::-1]).transpose(2, 0, 1).astype(np.float32) / 255.0)).float()
                                 for p in patches_bgr[i:i + self.batch_size]]).to(self.device)
                out.append(self.model({"input": {"image": x}}, mode="test")["contact_out"].sigmoid().float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 778), np.float32)

    def predict(self, image_bgr, boxes_xyxy, is_right, scores=None, windows=None):
        """Given hand boxes (xyxy, image pixels) and sides -> one prediction per box. windows: explicit crop windows (x, y, w, h)
        instead of the crop rule (e.g. mask_box for MOW)."""
        H, W = image_bgr.shape[:2]
        win = windows if windows is not None else [crop_box(b, self.crop, W, H) for b in boxes_xyxy]
        patches = [haco_patch(image_bgr, w, bool(r)) for w, r in zip(win, is_right)]
        if self.jpeg_quality:  # the evaluation's crops were JPEG files (cv2.imwrite quality 95, read back)
            patches = [cv2.imdecode(cv2.imencode(".jpg", q, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])[1], cv2.IMREAD_COLOR) for q in patches]
        p = self.contact(patches)
        scores = scores if scores is not None else [1.0] * len(p)
        return [dict(bbox=np.asarray(b, np.float32), is_right=bool(r), score=float(s), contact=q.astype(np.float32), contact_score=float(q.max()),
                     contact_flag=bool(q.max() >= self.thr)) for b, r, s, q in zip(boxes_xyxy, is_right, scores, p)]

    def __call__(self, image_bgr, K=None):
        """WiLoR's hand detector (the WiLoR / HaMeR adapters' settings) -> HACO on each detected hand."""
        if self.detector is None:
            from ultralytics import YOLO

            self.detector = YOLO(os.path.join(self.wilor_repo, "pretrained_models", "detector.pt")).to(self.device)
        det = self.detector(image_bgr, conf=self.det_conf, iou=self.det_iou, verbose=False)[0]
        if len(det.boxes) == 0:
            return []
        return self.predict(image_bgr, det.boxes.xyxy.cpu().numpy(), det.boxes.cls.cpu().numpy() > 0.5, det.boxes.conf.cpu().numpy())
