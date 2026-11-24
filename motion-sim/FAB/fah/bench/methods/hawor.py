"""Adapter: official HaWoR (https://github.com/ThunderVVV/HaWoR), camera-space hand motion.

Reproduces HaWoR's demo pipeline up to camera space (no SLAM / infiller):
detect_track (YOLO .track, conf 0.2, <=1 left + <=1 right per frame) -> merge tracks by majority
handedness -> interpolate boxes between first/last detection -> 16-frame temporal chunks through
HAWOR.inference(img_focal, img_center) -> MANO. Uses the TRUE intrinsics (focal, principal point).
Video method: frames are buffered by __call__ and predicted in finalize_clip.
"""
import contextlib
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict

import cv2
import numpy as np
import torch


@contextlib.contextmanager
def _in_dir(path):
    cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


class HaWoRMethod:
    name = "hawor"

    def __init__(self, repo=None, device="cuda", det_conf=0.2, **_):
        repo = repo or os.path.join(os.environ.get("THIRDPARTY", "thirdparty"), "HaWoR")  # checkout: docs/evaluation.md (baselines)
        self.repo = repo
        sys.path.insert(0, repo)
        with _in_dir(repo):
            from hawor.configs import get_config
            from lib.models.hawor import HAWOR
            from ultralytics import YOLO

            ckpt = "./weights/hawor/checkpoints/hawor.ckpt"
            cfg = get_config("./weights/hawor/model_config.yaml", update_cachedir=True)
            if cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in cfg.MODEL:
                cfg.defrost()
                cfg.MODEL.BBOX_SHAPE = [192, 256]
                cfg.freeze()
            self.model = HAWOR.load_from_checkpoint(ckpt, strict=False, cfg=cfg).to(device).eval()
            self.YOLO = YOLO
            self.det_path = os.path.join(repo, "weights/external/detector.pt")
        self.device = device
        self.det_conf = det_conf
        self.detector, self.tmp, self.imgfiles = None, None, []
        self.reset()

    def reset(self):
        if self.detector is not None and not self.imgfiles:
            return  # already fresh (run.py resets before every clip, finalize_clip after)
        self._cleanup()
        self.detector = self.YOLO(self.det_path)  # fresh tracker state per clip
        self.tmp = tempfile.mkdtemp(prefix="hawor_")
        self.imgfiles, self.tracks, self.K = [], {}, None

    def _cleanup(self):
        if self.tmp is not None:
            shutil.rmtree(self.tmp, ignore_errors=True)  # per-clip q95 JPEGs (HAWOR.inference reads files)
            self.tmp = None

    @torch.no_grad()
    def __call__(self, image_bgr, K):
        t = len(self.imgfiles)
        path = os.path.join(self.tmp, f"{t:05d}.jpg")
        cv2.imwrite(path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        self.imgfiles.append(path)
        self.K = K
        res = self.detector.track(image_bgr, conf=self.det_conf, persist=True, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        hand = res.boxes.cls.cpu().numpy()
        ids = res.boxes.id.cpu().numpy() if res.boxes.id is not None else [-1] * len(boxes)
        boxes = np.hstack([boxes, confs[:, None]])
        found = {True: False, False: False}
        for i in range(len(boxes)):
            is_r = bool(hand[i] > 0)
            tid = (10000 if is_r else 5000) if ids[i] == -1 else int(ids[i])
            if found[is_r]:
                continue
            self.tracks.setdefault(tid, []).append(
                dict(frame=t, det=True, det_box=boxes[[i]], det_handedness=hand[[i]]))
            found[is_r] = True
        return []  # predictions are filled in finalize_clip

    @torch.no_grad()
    def finalize_clip(self, frames):
        t0 = time.perf_counter()
        with _in_dir(self.repo):
            from hawor.utils.process import run_mano, run_mano_left
            from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
            from lib.eval_utils.custom_utils import interpolate_bboxes
            from lib.pipeline.tools import parse_chunks

            left, right = [], []
            for trk in self.tracks.values():
                valid = np.array([d["det"] for d in trk])
                is_r = np.concatenate([d["det_handedness"] for d in trk])[valid]
                (left if is_r.sum() / len(is_r) < 0.5 else right).extend(trk)
            final = {0: sorted(left, key=lambda d: d["frame"]), 1: sorted(right, key=lambda d: d["frame"])}
            imgfiles = np.array(self.imgfiles)
            K = self.K
            preds = defaultdict(list)
            for hid, trk in final.items():
                if len(trk) < 2:
                    continue
                # as in hawor_motion_estimation: per-detection boxes, chunks split at frame gaps
                boxes = np.concatenate([d["det_box"] for d in trk])
                valid = np.array([d["det"] for d in trk])
                nz = np.where(np.any(boxes != 0, axis=1))[0]
                first, last = nz[0], nz[-1]
                boxes[first:last + 1] = interpolate_bboxes(boxes[first:last + 1])
                valid[first:last + 1] = True
                boxes = boxes[first:last + 1]
                frame = np.array([d["frame"] for d in trk])[valid]
                frame_chunks, boxes_chunks = parse_chunks(frame, boxes, min_len=1)
                do_flip = hid == 0
                for fck, bck in zip(frame_chunks, boxes_chunks):
                    r = self.model.inference(imgfiles[fck], bck, img_focal=float(K[0, 0]),
                                             img_center=[float(K[0, 2]), float(K[1, 2])], do_flip=do_flip)
                    root = rotation_matrix_to_angle_axis(r["pred_rotmat"][None, :, 0])
                    hp = rotation_matrix_to_angle_axis(r["pred_rotmat"][None, :, 1:])
                    if do_flip:
                        root[..., 1:] *= -1
                        hp[..., 1:] *= -1
                    fn = run_mano_left if do_flip else run_mano
                    out = fn(r["pred_trans"][None, :, 0], root, hp, betas=r["pred_shape"][None])
                    J = out["joints"][0].cpu().numpy()
                    V = out["vertices"][0].cpu().numpy()
                    for k, fi in enumerate(fck):
                        j3 = J[k]
                        uv = j3[:, :2] / np.clip(j3[:, 2:], 1e-6, None) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
                        preds[int(fi)].append(dict(is_right=not do_flip, score=float(bck[k, 4]) if bck.shape[1] > 4 else 1.0,
                                                   bbox=bck[k, :4].astype(np.float32), joints3d=j3, verts3d=V[k], joints2d=uv))
        dt = (time.perf_counter() - t0) / max(len(frames), 1)
        for i, fr in enumerate(frames):
            fr["pred"] = preds.get(i, [])
            fr["latency"] += dt  # amortized temporal-model cost
        self._cleanup()
        self.imgfiles = []
        self.detector = None
