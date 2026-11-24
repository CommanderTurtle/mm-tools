"""Converted contact video clips (HOI4D, ARCTIC egocentric; train.build_hoi4d_contact / train.build_arctic_video): one directory
per clip, <root>/clips/<name>/ = meta.json (K, hw, fps, first / last frame, video "image.mp4" or frames "frames/%06d.jpg",
dataset, sequence, licence) + gt.npz (one row per GT hand-frame: frame, is_right, verts (778, 3) / joints (21, 3, HOT3D
fingertips, OpenPose order) camera frame m, sdf (778,) float16 mm to the GT object (train.contact_sdf), valid, T_obj[, vis]).
Frames cover first..last (every step-th frame, meta "step", default 1; HOI4D: between the first and the last hand
annotation, ARCTIC: the sequence's contiguous run of frames with an image); a hand without a fit in a frame is absent there
(not a negative). Hands as DexYCBClip: box = projected mesh inside the image (>= 10 vertices), scored if >= min_visible of it
is inside, else ignore. The clips are built locally from the datasets each user downloads under their own licences (HOI4D
CC BY-NC 4.0, ARCTIC non-commercial research); nothing of them is redistributed.

    python -m bench.run --dataset contactclip --split $DATA_ROOT/contact/video/hoi4d/split_test.json \
        --data_root $DATA_ROOT/contact/video/hoi4d --method fah ...
"""
import json
import os

import cv2
import numpy as np


class ContactClip:
    def __init__(self, path, min_visible=0.5):
        self.path = path
        self.meta = json.load(open(os.path.join(path, "meta.json")))
        z = np.load(os.path.join(path, "gt.npz"))
        self.gt = {k: z[k] for k in z.files}
        self.frame_keys = list(range(self.meta["first"], self.meta["last"] + 1, self.meta.get("step", 1)))
        self.rows = {}
        for r, f in enumerate(self.gt["frame"].tolist() if len(self.gt.get("frame", [])) else []):
            self.rows.setdefault(int(f), []).append(r)
        self.K = np.asarray(self.meta["K"], np.float32)
        self.size = tuple(self.meta["hw"])
        self.sequence_id = self.meta["sequence"]
        self.min_visible = min_visible
        self._cap, self._pos = None, 0

    def __len__(self):
        return len(self.frame_keys)

    def _image(self, key):
        if "frames" in self.meta:
            return cv2.imread(os.path.join(self.path, self.meta["frames"] % key), cv2.IMREAD_COLOR)
        if self._cap is None or key < self._pos:  # sequential decode; decoded frame i = annotation frame i
            if self._cap is not None:
                self._cap.release()
            self._cap, self._pos = cv2.VideoCapture(os.path.join(self.path, self.meta["video"])), 0
        img = None
        while self._pos <= key:
            ok, img = self._cap.read()
            self._pos += 1
            if not ok:
                return None
        return img

    def frame(self, i, image=True):
        key = self.frame_keys[i]
        img = self._image(key) if image else None
        H, W = self.size
        hands, ignore = [], []
        for r in self.rows.get(key, []):
            v, j = self.gt["verts"][r].astype(np.float64), self.gt["joints"][r].astype(np.float64)
            proj = lambda X: (X[:, :2] / np.clip(X[:, 2:], 1e-6, None)) * self.K[[0, 1], [0, 1]] + self.K[[0, 1], [2, 2]]
            uv = proj(v)
            inside = (v[:, 2] > 0.01) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if inside.sum() < 10:
                continue
            uv_in = uv[inside]
            hand = dict(is_right=bool(self.gt["is_right"][r]), joints3d=j, verts3d=v, joints2d=proj(j),
                        bbox=np.array([*uv_in.min(0), *uv_in.max(0)], dtype=np.float32), visible=float(inside.mean()),
                        sdf=self.gt["sdf"][r], valid=bool(self.gt["valid"][r]), T_obj=self.gt["T_obj"][r])
            if "vis" in self.gt:
                hand["vis"] = self.gt["vis"][r]
            (hands if inside.mean() >= self.min_visible else ignore).append(hand)
        return dict(image=img, K=self.K.copy(), hands=hands, ignore=ignore, key=f"{key:06d}", sequence_id=self.sequence_id,
                    device=self.meta["dataset"], annotated=bool(self.rows.get(key)))
