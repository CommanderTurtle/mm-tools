"""DexYCB loader (https://dex-ycb.github.io/, toolkit https://github.com/NVlabs/dex-ycb-toolkit; CC BY-NC 4.0, research
use only, cite Chao et al. CVPR 2021).

Third-person hand-object grasping: 10 subjects x 100 sequences (20 YCB objects x 5 trials, ~70 frames at 30 fps),
each recorded by the same 8 static RealSense cameras (640x480 color). One hand per sequence (meta.yml mano_sides) with
MANO GT per frame and camera: labels_%06d.npz pose_m (1, 51) = global rotation (3) + 45 PCA coefficients + translation,
in that camera's frame (metres); betas per subject (calibration/mano_*/mano.yml); pose_m == 0 where the hand has no
annotation (only leading frames, ~11 per clip on average; DexYCBClip trims them).
Toolkit MANO convention (manopth, flat_hand_mean=False, ncomps=45, use_pca): articulation = pca @ hands_components +
hands_mean, left model with its shapedirs as shipped. smplx ignores use_pca when num_pca_comps == 45, so DexYCBMano
expands the PCA itself and feeds axis-angle to bench.common.ManoLayer; with manopth's fingertip vertices
(DEXYCB_TIP_VERTS) this reproduces the label joint_3d to 2e-4 mm (both sides). Our GT joints use the HOT3D fingertips
(bench.common) like every other benchmark; the toolkit's are kept as joints3d_dexycb.
The subject's other hand is never annotated but sometimes in view: detections on it count as false positives, so
det_precision / det_f1 are pessimistic on DexYCB (pose and recall metrics are not affected).

Converted clips (bench.datasets.build_dexycb): <root>/clips/<subject>/<sequence>_<serial>.tar = one sequence x one
camera, holding info.json (K, side, betas, T_world_cam, meta), color_%06d.jpg and the original labels_%06d.npz
(depth dropped). Splits are the toolkit's setups (SPLITS); s0 test = sequences i % 5 == 4 of subjects 03-10.
"""
import io
import json
import pickle
import tarfile

import cv2
import numpy as np
import torch

from bench.common import ManoLayer

SUBJECTS = ["20200709-subject-01", "20200813-subject-02", "20200820-subject-03", "20200903-subject-04",
            "20200908-subject-05", "20200918-subject-06", "20200928-subject-07", "20201002-subject-08",
            "20201015-subject-09", "20201022-subject-10"]
SERIALS = ["836212060125", "839512060362", "840412060917", "841412060263", "932122060857", "932122060861",
           "932122061900", "932122062010"]
# dex_ycb_toolkit/dex_ycb.py: setup -> split -> (subject indices, serial indices, sequence index predicate)
SPLITS = {
    "s0": {"train": (range(10), range(8), lambda i: i % 5 != 4), "val": ([0, 1], range(8), lambda i: i % 5 == 4),
           "test": (range(2, 10), range(8), lambda i: i % 5 == 4)},
    "s1": {"train": ([0, 1, 2, 3, 4, 5, 9], range(8), lambda i: True), "val": ([6], range(8), lambda i: True),
           "test": ([7, 8], range(8), lambda i: True)},
    "s2": {"train": (range(10), range(6), lambda i: True), "val": (range(10), [6], lambda i: True),
           "test": (range(10), [7], lambda i: True)},
    "s3": {"train": (range(10), range(8), lambda i: i // 5 not in (3, 7, 11, 15, 19)),
           "val": (range(10), range(8), lambda i: i // 5 in (3, 19)),
           "test": (range(10), range(8), lambda i: i // 5 in (7, 11, 15))},
}
DEXYCB_TIP_VERTS = {True: [745, 317, 444, 556, 673], False: [745, 317, 445, 556, 673]}  # manopth fingertips per side


def split_clips(sequences, setup="s0", split="test"):
    """sequences: {subject: sorted list of its 100 sequence names} -> [(subject, sequence, serial)] of a toolkit split."""
    subj, ser, keep = SPLITS[setup][split]
    out = []
    for si in subj:
        seqs = sequences[SUBJECTS[si]]
        assert len(seqs) == 100, (SUBJECTS[si], len(seqs))
        out += [(SUBJECTS[si], q, SERIALS[c]) for i, q in enumerate(seqs) if keep(i) for c in ser]
    return out


class DexYCBMano:
    """DexYCB (manopth) MANO convention on top of ManoLayer: pose_m (51,) -> 21 OpenPose joints, 778 verts."""

    def __init__(self, model_dir):
        self.layer = ManoLayer(model_dir)  # full axis-angle articulation, hands_mean added
        pkl = {s: pickle.load(open(f"{model_dir}/MANO_{'RIGHT' if s else 'LEFT'}.pkl", "rb"), encoding="latin1")
               for s in (True, False)}
        self.comps = {s: np.asarray(d["hands_components"], np.float32) for s, d in pkl.items()}
        # the GT was fit with manopth, which uses the left shapedirs as shipped: undo ManoLayer's smplx issue-48 fix
        self.layer.layers[False].shapedirs[:] = torch.as_tensor(np.asarray(pkl[False]["shapedirs"], np.float32))

    def __call__(self, is_right, betas, pose_m):
        p = np.asarray(pose_m, np.float32).reshape(-1)
        return self.layer(is_right, betas, p[3:48] @ self.comps[bool(is_right)], p[:3], p[48:51])


class DexYCBClip:
    def __init__(self, tar_path, mano_dir=None, min_visible=0.5, mano=None):
        self.tar = tarfile.open(tar_path)
        self.info = json.load(self.tar.extractfile("info.json"))
        keys = [f"{i:06d}" for i in range(self.info["num_frames"])]
        ann = [i for i, k in enumerate(keys) if np.any(self._labels(k)["pose_m"] != 0)]
        # the hand is not tracked in the first frames (pose_m == 0; 11 on average, up to 33): the clip spans the annotated frames
        self.frame_keys = keys[ann[0] : ann[-1] + 1] if ann else []
        self.K = np.asarray(self.info["K"], np.float32)
        self.is_right = self.info["side"] == "right"
        self.betas = np.asarray(self.info["betas"], np.float32)
        self.T_world_cam = np.asarray(self.info["T_world_cam"], np.float64)
        self.sequence_id = self.info["sequence"]
        self.size = tuple(self.info["hw"])
        self.min_visible = min_visible
        self.mano = mano or DexYCBMano(mano_dir)

    def __len__(self):
        return len(self.frame_keys)

    def _labels(self, key):
        return np.load(io.BytesIO(self.tar.extractfile(f"labels_{key}.npz").read()))

    def frame(self, i, image=True):
        key = self.frame_keys[i]
        img = None
        if image:
            img = cv2.imdecode(np.frombuffer(self.tar.extractfile(f"color_{key}.jpg").read(), np.uint8), cv2.IMREAD_COLOR)
        lab = self._labels(key)
        pose_m = lab["pose_m"][0]
        hands, ignore = [], []
        if np.any(pose_m != 0):
            j, v = self.mano(self.is_right, self.betas, pose_m)
            H, W = self.size
            proj = lambda X: (X[:, :2] / np.clip(X[:, 2:], 1e-6, None)) * self.K[[0, 1], [0, 1]] + self.K[[0, 1], [2, 2]]
            uv = proj(v)
            inside = (v[:, 2] > 0.01) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if inside.sum() >= 10:
                uv_in = uv[inside]
                box = np.array([*uv_in.min(0), *uv_in.max(0)], dtype=np.float32)
                hand = dict(is_right=self.is_right, joints3d=j, verts3d=v, joints2d=proj(j), bbox=box,
                            visible=float(inside.mean()), joints3d_dexycb=lab["joint_3d"][0].astype(np.float64))
                # mostly out-of-view hands are not scored, but detections on them are not false positives
                (hands if inside.mean() >= self.min_visible else ignore).append(hand)
        return dict(image=img, K=self.K.copy(), hands=hands, ignore=ignore, key=key, sequence_id=self.sequence_id,
                    device=self.info["serial"], annotated=bool(np.any(pose_m != 0)), T_world_cam=self.T_world_cam.copy())
