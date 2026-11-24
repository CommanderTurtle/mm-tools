"""Loader for the bench2d benchmark folder (images/ + annotations.json)."""
import json
import os

import cv2
import numpy as np


class Bench2D:
    def __init__(self, root, subsets=None):
        self.root = root
        data = json.load(open(os.path.join(root, "annotations.json")))
        self.meta = data["meta"]
        self.records = [r for r in data["images"] if not subsets or r["subset"] in subsets]

    def __len__(self):
        return len(self.records)

    def chunks(self, size=100):
        for i in range(0, len(self.records), size):
            yield i // size, _Chunk(self, self.records[i:i + size])


class _Chunk:
    def __init__(self, ds, records):
        self.ds, self.records = ds, records

    def __len__(self):
        return len(self.records)

    def frame(self, i):
        r = self.records[i]
        img = cv2.imread(os.path.join(self.ds.root, r["file"]))
        hands = [dict(is_right=h["is_right"], joints2d=np.asarray(h["kp2d"], np.float32), vis=np.asarray(h["vis"]),
                      bbox=np.asarray(h["bbox"], np.float32), joints3d=None, verts3d=None) for h in r["hands"]]
        K = np.asarray(r["K"], np.float32) if r.get("K") else None
        return dict(image=img, K=K, hands=hands, key=r["id"], subset=r["subset"],
                    fully_annotated=r["fully_annotated"], sequence_id=r.get("sequence", r["subset"]))
