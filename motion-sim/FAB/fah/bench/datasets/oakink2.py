"""OakInk2 egocentric clips (CVPR 2024, https://oakink.net/v2, CC BY-SA 4.0): the head camera of the bimanual
hand-object sequences, 848 x 480 at 30 fps (image frames = every 4th 120 Hz mocap frame), both hands with mocap MANO GT.
Labels as train.build_oakink2_crops (manotorch convention, left shapedirs as shipped, joints = kinematic MANO joints +
HOT3D fingertip vertices, OpenPose order, camera frame from the per-frame cam_extr). A clip is a window of `length`
consecutive image frames of one sequence; split entries are "<sequence>@<start>" (bench/splits/oakink2_ego_*.json).
Used for video-refiner training tracks (the per-frame model saw crops of these sequences: train-only data).
"""
import os
import tarfile

import cv2
import numpy as np

from train.build_oakink2_crops import hand_world, load_anno, mano_layer, seq_tar


class OakInk2Clip:
    def __init__(self, raw, entry, mano_dir, length=300, view="egocentric", min_visible=0.5):
        seq, start = entry.rsplit("@", 1)
        self.anno = load_anno(os.path.join(raw, "anno_preview", f"{seq}.pkl"))
        self.tar = tarfile.open(seq_tar(raw, seq, [view]))
        self.members = {m.name.split("/", 1)[1]: m for m in self.tar.getmembers() if m.name.endswith(".png")}
        self.serial = {v: k for k, v in self.anno["cam_def"].items()}[view]
        fids = [f for f in self.anno["frame_id_list"] if f in self.anno["raw_mano"] and f"{self.serial}/{f:06d}.png" in self.members]
        self.fids = fids[int(start):int(start) + length]
        self.view, self.sequence_id, self.min_visible = view, seq, min_visible
        self.ml = mano_layer(mano_dir)

    def __len__(self):
        return len(self.fids)

    def frame(self, i, image=True):
        fid = self.fids[i]
        img = None
        if image:
            img = cv2.imdecode(np.frombuffer(self.tar.extractfile(self.members[f"{self.serial}/{fid:06d}.png"]).read(), np.uint8),
                               cv2.IMREAD_COLOR)
        K = np.asarray(self.anno["cam_intr"][self.view][fid], np.float64)
        E = np.asarray(self.anno["cam_extr"][self.view][fid], np.float64)  # world -> camera
        H, W = (img.shape[:2] if img is not None else (480, 848))
        proj = lambda X: (X[:, :2] / np.clip(X[:, 2:], 1e-6, None)) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
        hands, ignore = [], []
        for is_right in (True, False):
            jw, vw = hand_world(self.ml, self.anno["raw_mano"][fid], is_right)
            j, v = (E[:3, :3] @ jw.T + E[:3, 3:]).T, (E[:3, :3] @ vw.T + E[:3, 3:]).T
            uv = proj(v)
            inside = (v[:, 2] > 0.01) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if inside.sum() < 10:
                continue
            box = np.array([*uv[inside].min(0), *uv[inside].max(0)], dtype=np.float32)
            hand = dict(is_right=is_right, joints3d=j, verts3d=v, joints2d=proj(j), bbox=box, visible=float(inside.mean()))
            (hands if inside.mean() >= self.min_visible else ignore).append(hand)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = E[:3, :3].T, -E[:3, :3].T @ E[:3, 3]
        return dict(image=img, K=K.astype(np.float32), hands=hands, ignore=ignore, key=f"{fid:06d}", sequence_id=self.sequence_id,
                    T_world_cam=T)
