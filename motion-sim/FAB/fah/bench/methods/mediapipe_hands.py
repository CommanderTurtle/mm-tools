"""Adapter: MediaPipe HandLandmarker (https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
-> benchmark prediction format.

Official Tasks API + hand_landmarker.task (float16 bundle: palm detector + landmark model), default confidences
(0.5 / 0.5 / 0.5), num_hands=4. IMAGE mode (every frame independent: palm detector + landmarks) by default;
video=True uses VIDEO mode (MediaPipe's own tracking: landmarks of the previous frame give the ROI, the palm
detector only runs when hands are lost), a fresh landmarker per clip.

Keypoints are already in OpenPose hand order (wrist, thumb CMC..tip, index MCP..tip, ...). joints2d = normalized
image landmarks * (W, H). The world landmarks are metric but hand-centred (origin ~ hand centre, no translation)
and, being predicted from the crop's appearance, oriented relative to the viewing ray through the crop rather than
the optical axis (HOT3D: a systematic ~50 deg tilt for hands low in the wide-FOV image). They are made
wrist-relative, rotated by the minimal rotation taking the optical axis to the ray through the 2D landmark-box
centre (ray_align; joints3d_noray / ray_align=False keep the raw orientation), and placed in the camera frame by
the translation t that best reprojects them onto the 2D landmarks with the TRUE K (linear least squares on the
pinhole equations, reweighted by depth = reprojection error); MediaPipe's 3D shape / scale is kept. Unknown camera:
focal = max(H, W), centred. bbox = 2D landmark box padded by bbox_pad * its longer side on each side
(HOT3D GT boxes are mesh boxes). score = handedness confidence (the Tasks API exposes no presence score).
Handedness: the legacy Hands docs say labels assume mirrored (selfie) input, but the Tasks API labels are already
correct for unmirrored images (HOT3D check: handedness_acc 0.88 as is vs 0.12 swapped); flip_handedness=True swaps.
"""
import os

import cv2
import numpy as np


def solve_translation(rel, uv, K, iters=3):
    """t minimising the reprojection error of rel + t (wrist-relative 3D, metres) onto uv (px) under K."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    du, dv = uv[:, 0] - cx, uv[:, 1] - cy
    n = len(rel)
    A = np.zeros((2 * n, 3))
    A[:n, 0], A[:n, 2] = fx, -du
    A[n:, 1], A[n:, 2] = fy, -dv
    b = np.concatenate([du * rel[:, 2] - fx * rel[:, 0], dv * rel[:, 2] - fy * rel[:, 1]])
    w = np.ones(2 * n)
    for _ in range(iters):
        t = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)[0]
        z = np.clip(rel[:, 2] + t[2], 1e-3, None)
        w = np.concatenate([1 / z, 1 / z])  # algebraic -> (approximately) pixel residuals
    return t


def ray_rotation(d):
    """Minimal rotation taking the optical axis (0, 0, 1) to the direction d."""
    d = d / np.linalg.norm(d)
    v, c = np.cross([0.0, 0.0, 1.0], d), d[2]
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


class MediaPipeHandsMethod:
    name = "mediapipe"

    def __init__(self, model=None, num_hands=4, min_det_conf=0.5, min_presence_conf=0.5, min_tracking_conf=0.5,
                 video=False, fps=30.0, flip_handedness=False, ray_align=True, bbox_pad=0.05, delegate="cpu", **_):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self.model = model or os.path.join(os.environ.get("THIRDPARTY", "thirdparty"), "mediapipe",
                                           "hand_landmarker.task")  # docs/evaluation.md (baselines)
        self.vision, self.BaseOptions = vision, BaseOptions
        self.opts = dict(num_hands=num_hands, min_hand_detection_confidence=min_det_conf,
                         min_hand_presence_confidence=min_presence_conf, min_tracking_confidence=min_tracking_conf)
        self.delegate = BaseOptions.Delegate.GPU if delegate == "gpu" else BaseOptions.Delegate.CPU
        self.video, self.fps, self.flip, self.ray_align, self.bbox_pad = video, fps, flip_handedness, ray_align, bbox_pad
        self.lm, self.t = None, 0
        self.reset()
        self.name = "mediapipe-video" if video else "mediapipe"

    def reset(self):  # new clip (VIDEO mode needs monotonic timestamps and drops the previous clip's tracks)
        if self.lm is not None and not self.video:
            return
        if self.lm is not None:
            self.lm.close()
        v = self.vision
        self.lm = v.HandLandmarker.create_from_options(v.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model, delegate=self.delegate),
            running_mode=v.RunningMode.VIDEO if self.video else v.RunningMode.IMAGE, **self.opts))
        self.t = 0

    def __call__(self, image_bgr, K):
        import mediapipe as mp

        h, w = image_bgr.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))
        if self.video:
            res = self.lm.detect_for_video(img, int(round(self.t * 1000 / self.fps)))
            self.t += 1
        else:
            res = self.lm.detect(img)
        if K is None:
            f = float(max(h, w))
            K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32)
        out = []
        for lms, wlms, hd in zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness):
            uv = np.array([[p.x * w, p.y * h] for p in lms], dtype=np.float64)
            rel = np.array([[p.x, p.y, p.z] for p in wlms], dtype=np.float64)
            rel -= rel[:1]
            lo, hi = uv.min(0), uv.max(0)
            cam = rel
            if self.ray_align:
                cam = rel @ ray_rotation(np.linalg.solve(K, [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 1.0])).T
            t = solve_translation(cam, uv, K)
            pad = self.bbox_pad * (hi - lo).max()
            is_right = (hd[0].category_name == "Right") != self.flip
            out.append(dict(is_right=bool(is_right), score=float(hd[0].score),
                            bbox=np.array([*(lo - pad), *(hi + pad)], dtype=np.float32),
                            joints2d=uv.astype(np.float32), joints3d=(cam + t).astype(np.float32),
                            joints3d_noray=(rel + solve_translation(rel, uv, K)).astype(np.float32),
                            joints3d_rel=rel.astype(np.float32)))
        return out
