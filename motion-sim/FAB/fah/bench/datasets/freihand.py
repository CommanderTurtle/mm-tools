"""FreiHAND evaluation set (3960 right hands, 224x224, GT joints in OpenPose order + MANO verts)."""
import json
import os

import cv2
import numpy as np


class FreiHANDEval:
    def __init__(self, root, mano_dir=None):
        self.root = root
        self.K = json.load(open(os.path.join(root, "evaluation_K.json")))
        self.xyz = json.load(open(os.path.join(root, "evaluation_xyz.json")))
        verts_path = os.path.join(root, "evaluation_verts.json")
        self.verts = json.load(open(verts_path)) if os.path.exists(verts_path) else None
        self.ids = list(range(len(self.K)))

    def __len__(self):
        return len(self.ids)

    def subset(self, ids):
        sub = FreiHANDEval.__new__(FreiHANDEval)
        sub.__dict__.update(self.__dict__)
        sub.ids = list(ids)
        return sub

    def frame(self, i):
        idx = self.ids[i]
        img = cv2.imread(os.path.join(self.root, "evaluation", "rgb", f"{idx:08d}.jpg"))
        K = np.asarray(self.K[idx], dtype=np.float32)
        j = np.asarray(self.xyz[idx], dtype=np.float32)
        v = np.asarray(self.verts[idx], dtype=np.float32) if self.verts else None
        pts = v if v is not None else j
        uv = (pts[:, :2] / pts[:, 2:]) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
        juv = (j[:, :2] / j[:, 2:]) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
        hand = dict(is_right=True, joints3d=j, verts3d=v, joints2d=juv,
                    bbox=np.array([*uv.min(0), *uv.max(0)], dtype=np.float32), visible=1.0)
        return dict(image=img, K=K, hands=[hand], key=f"{idx:08d}", sequence_id="freihand", device="rgb")
