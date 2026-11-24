"""Adapter: official HaMeR (https://github.com/geopavlakos/hamer) -> benchmark prediction format.

HaMeR has no hand detector of its own (its demo runs ViTPose whole-body keypoints on Detectron2 person
boxes), so it gets WiLoR's YOLO hand detector with the same conf / NMS as the WiLoR adapter: the comparison
isolates the pose model. Crops (ViTDetDataset, rescale_factor 2.0) and the crop -> full-image camera with
the TRUE intrinsics follow the WiLoR adapter (WiLoR is a HaMeR fork with the same output conventions).
"""
import os
import sys

import torch

from bench.methods.wilor import WiLoRMethod


class HaMeRMethod(WiLoRMethod):
    name = "hamer"

    def __init__(self, repo=None, wilor_repo=None, device="cuda", det_conf=0.3, det_iou=0.5, rescale_factor=2.0,
                 batch_size=16, ckpt=None, **_):
        root = os.environ.get("THIRDPARTY", "thirdparty")
        repo = repo or os.path.join(root, "hamer")  # checkout: docs/evaluation.md (baselines)
        wilor_repo = wilor_repo or os.path.join(root, "WiLoR")
        sys.path.insert(0, repo)
        cwd = os.getcwd()
        os.chdir(repo)  # HaMeR resolves ./_DATA relative to CWD
        try:
            from ultralytics import YOLO
            from hamer.datasets.vitdet_dataset import ViTDetDataset
            from hamer.models import load_hamer

            model, cfg = load_hamer(ckpt or "./_DATA/hamer_ckpts/checkpoints/hamer.ckpt")
            detector = YOLO(os.path.join(wilor_repo, "pretrained_models", "detector.pt"))
        finally:
            os.chdir(cwd)
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.detector = detector.to(self.device)
        self.cfg, self.ViTDetDataset = cfg, ViTDetDataset
        self.det_conf, self.det_iou = det_conf, det_iou
        self.rescale_factor, self.fast, self.batch_size = rescale_factor, False, batch_size
        self.name = "hamer"
