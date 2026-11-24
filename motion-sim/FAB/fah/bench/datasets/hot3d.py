"""HOT3D-Clips loader (https://github.com/facebookresearch/hot3d/tree/main/hot3d/clips).

Each clip tar holds ~150 frames of an Aria/Quest3 headset with MANO GT for both hands.
Images are rectified from the fisheye model to an upright pinhole camera (Aria RGB is rotated
90 deg CW to be upright), and the GT is expressed in that camera's frame.
"""
import io
import json
import tarfile

import cv2
import numpy as np
import torch
from hand_tracking_toolkit import camera as htt_camera
from hand_tracking_toolkit.dataset import HandSide, decode_hand_pose

from bench.common import ManoLayer

ROT_CW = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # eye -> upright eye


class Hot3DClip:
    def __init__(self, tar_path, mano_dir, stream="214-1", focal_scale=1.0, out_size=None,
                 rotate_upright=None, min_visible=0.5, mano=None):
        self.tar = tarfile.open(tar_path)
        self.names = set(self.tar.getnames())
        self.frame_keys = sorted({n.split(".")[0] for n in self.names if n[:6].isdigit()})
        self.stream = stream
        self.focal_scale = focal_scale
        self.out_size = out_size
        self.min_visible = min_visible
        shapes = json.load(self.tar.extractfile("__hand_shapes.json__"))
        self.betas = np.asarray(shapes["mano"], dtype=np.float32)
        self.mano = mano or ManoLayer(mano_dir, num_pca_comps=15)
        info = self._json(self.frame_keys[0], "info")
        self.device = info["device"]
        self.sequence_id = info["sequence_id"]
        # Aria RGB is mounted sideways; Quest3 mono cameras are not
        self.rotate = (self.device == "Aria") if rotate_upright is None else rotate_upright
        self._maps = None

    def __len__(self):
        return len(self.frame_keys)

    def _json(self, key, kind):
        return json.load(self.tar.extractfile(f"{key}.{kind}.json"))

    def _setup_rectification(self, fisheye):
        W = self.out_size or fisheye.width
        s = W / fisheye.width
        f = fisheye.f[0] * self.focal_scale * s
        c = (np.asarray(fisheye.c) + 0.5) * s - 0.5
        pinhole = htt_camera.PinholePlaneCameraModel(W, W, (f, f), c, [])
        u, v = np.meshgrid(np.arange(W), np.arange(W))
        rays = pinhole.window_to_eye(np.stack([u.ravel(), v.ravel()], 1).astype(np.float64))
        src = fisheye.eye_to_window(rays).astype(np.float32).reshape(W, W, 2)
        self._maps = (src[..., 0], src[..., 1])
        K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]], dtype=np.float32)
        if self.rotate:  # rotated image: u' = W-1-v, v' = u  =>  K' with swapped centers
            K = np.array([[f, 0, W - 1 - c[1]], [0, f, c[0]], [0, 0, 1]], dtype=np.float32)
        self.K = K
        self.size = W

    def frame(self, i):
        key = self.frame_keys[i]
        cams = self._json(key, "cameras")
        fisheye = htt_camera.from_json(cams[self.stream])
        if self._maps is None:
            self._setup_rectification(fisheye)
        raw = cv2.imdecode(np.frombuffer(self.tar.extractfile(f"{key}.image_{self.stream}.jpg").read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.remap(raw, self._maps[0], self._maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if self.rotate:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        T_cam_from_world = np.linalg.inv(fisheye.T_world_from_eye)
        R = ROT_CW if self.rotate else np.eye(3)
        hands, ignore = [], []
        hand_json = self._json(key, "hands") if f"{key}.hands.json" in self.names else {}
        for side, pose in decode_hand_pose(hand_json).items():
            if pose.mano is None:
                continue
            is_right = side == HandSide.RIGHT
            xf = pose.mano.wrist_xform.numpy()
            j_w, v_w = self.mano(is_right, self.betas, pose.mano.mano_theta.numpy(), xf[:3], xf[3:])
            to_cam = lambda X: (R @ (T_cam_from_world[:3, :3] @ X.T + T_cam_from_world[:3, 3:])).T
            j, v = to_cam(j_w), to_cam(v_w)
            uv = (v[:, :2] / np.clip(v[:, 2:], 1e-6, None)) * self.K[[0, 1], [0, 1]] + self.K[[0, 1], [2, 2]]
            inside = (v[:, 2] > 0.01) & (uv[:, 0] >= 0) & (uv[:, 0] < self.size) & (uv[:, 1] >= 0) & (uv[:, 1] < self.size)
            if inside.sum() < 10:
                continue
            uv_in = uv[inside]
            box = np.array([*uv_in.min(0), *uv_in.max(0)], dtype=np.float32)
            juv = (j[:, :2] / np.clip(j[:, 2:], 1e-6, None)) * self.K[[0, 1], [0, 1]] + self.K[[0, 1], [2, 2]]
            hand = dict(is_right=is_right, joints3d=j, verts3d=v, joints2d=juv, bbox=box, visible=float(inside.mean()))
            # mostly out-of-view hands are not scored, but detections on them are not false positives
            (hands if inside.mean() >= self.min_visible else ignore).append(hand)
        T_cam = np.eye(4)  # world -> (upright) camera, for world-space evaluation
        T_cam[:3, :3], T_cam[:3, 3] = R @ T_cam_from_world[:3, :3], R @ T_cam_from_world[:3, 3]
        return dict(image=img, K=self.K.copy(), hands=hands, ignore=ignore, key=key, sequence_id=self.sequence_id,
                    device=self.device, T_world_cam=np.linalg.inv(T_cam))
