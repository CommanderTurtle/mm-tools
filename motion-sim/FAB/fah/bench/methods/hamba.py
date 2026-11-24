"""Adapter: official Hamba (https://github.com/humansensinglab/Hamba, NeurIPS 2024; ViT-H + graph-guided bi-scanning
Mamba decoder, CC-BY-NC 4.0) -> benchmark prediction format.

Hamba's demo finds hands with Detectron2 ViTDet person boxes + ViTPose whole-body keypoints (as HaMeR), so it gets
WiLoR's YOLO hand detector with the same conf / NMS as the WiLoR / HaMeR adapters: the comparison isolates the pose
model. Crops follow Hamba's demo (rescale_factor 2.0): Hamba's ViTDetDataset is HaMeR's (anti-alias blur, 256x256
patch, left hands flipped) plus a MediaPipe pass on the crop whose 2D keypoints the released checkpoint does not use
(USE_INIT_REGRESSION: the graph-guided joint tokens are grid-sampled at its own initial-regressor 2D joints) and that
only drops crops where MediaPipe finds no hand, so HaMeR's ViTDetDataset is used. Same output conventions as HaMeR, so
the crop -> full-image camera with the TRUE intrinsics is the WiLoR adapter's.
"""
import os
import sys

import torch

from bench.methods.wilor import WiLoRMethod


class HambaMethod(WiLoRMethod):
    name = "hamba"

    def __init__(self, repo=None, hamer_repo=None, wilor_repo=None, device="cuda", det_conf=0.3, det_iou=0.5,
                 rescale_factor=2.0, batch_size=16, ckpt=None, **_):
        root = os.environ.get("THIRDPARTY", "thirdparty")
        repo = repo or os.path.join(root, "Hamba")  # checkout: docs/evaluation.md (baselines)
        hamer_repo = hamer_repo or os.path.join(root, "hamer")
        wilor_repo = wilor_repo or os.path.join(root, "WiLoR")
        sys.path[:0] = [repo, hamer_repo]
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # Hamba's (unused) MeshRenderer defaults to osmesa: fails with PyOpenGL 3.1.0
        cwd = os.getcwd()
        os.chdir(repo)  # Hamba resolves ./_DATA (MANO) relative to CWD
        try:
            from ultralytics import YOLO
            from hamer.datasets.vitdet_dataset import ViTDetDataset
            from hamba.models import load_hamba

            model, cfg = load_hamba(ckpt or "./ckpts/hamba/checkpoints/hamba.ckpt")
            detector = YOLO(os.path.join(wilor_repo, "pretrained_models", "detector.pt"))
        finally:
            os.chdir(cwd)
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.detector = detector.to(self.device)
        self.cfg, self.ViTDetDataset = cfg, ViTDetDataset
        self.det_conf, self.det_iou = det_conf, det_iou
        self.rescale_factor, self.fast, self.batch_size = rescale_factor, False, batch_size
        self.name = "hamba"
