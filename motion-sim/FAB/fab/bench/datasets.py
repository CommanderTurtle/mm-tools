"""Benchmark sample lists. Every sample is a dict:
image (path), bbox (xyxy, float32), K (3x3), gender, global_orient (3), body_pose (69), betas (10),
transl (3), extrinsic (4x4, world -> camera) and an id. Ground truth is posed in the camera frame by the caller.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def _bbox_from_keypoints(xy: np.ndarray, conf: np.ndarray, padding: float = 0.2, min_valid: int = 4):
    valid = conf > 0.1
    if valid.sum() < min_valid:
        return None
    x1, y1 = xy[valid].min(axis=0)
    x2, y2 = xy[valid].max(axis=0)
    w, h = x2 - x1, y2 - y1
    return np.array([x1 - w * padding, y1 - h * padding, x2 + w * padding, y2 + h * padding], dtype=np.float32)


def threedpw(root: str | Path, split: str = "test") -> list[dict]:
    """3DPW: every person-frame with a valid camera pose; box from the 2D poses with 20 % padding
    (the Fast SAM 3D Body protocol); gendered SMPL ground truth."""
    root = Path(root)
    samples = []
    for pkl in sorted((root / "sequenceFiles" / split).glob("*.pkl")):
        with open(pkl, "rb") as f:
            seq = pickle.load(f, encoding="latin1")
        name = pkl.stem
        K = np.asarray(seq["cam_intrinsics"], dtype=np.float32)
        for person in range(len(seq["poses"])):
            gender = {"m": "male", "f": "female"}.get(str(seq["genders"][person])[0], "neutral")
            betas = np.asarray(seq["betas"][person][:10], dtype=np.float32)
            for frame in range(len(seq["poses"][person])):
                if not seq["campose_valid"][person][frame]:
                    continue
                poses2d = np.asarray(seq["poses2d"][person][frame])  # [3, 18]
                bbox = _bbox_from_keypoints(poses2d[:2].T, poses2d[2])
                if bbox is None:
                    continue
                pose = np.asarray(seq["poses"][person][frame], dtype=np.float32)
                samples.append({
                    "id": f"{name}/{frame:05d}/{person}",
                    "image": str(root / "imageFiles" / name / f"image_{frame:05d}.jpg"),
                    "bbox": bbox,
                    "K": K,
                    "gender": gender,
                    "global_orient": pose[:3],
                    "body_pose": pose[3:72],
                    "betas": betas,
                    "transl": np.asarray(seq["trans"][person][frame], dtype=np.float32),
                    "extrinsic": np.asarray(seq["cam_poses"][frame], dtype=np.float32),
                })
    return samples


def emdb(root: str | Path, subset: str = "emdb1") -> list[dict]:
    """EMDB: sequences flagged `emdb1` (camera-space benchmark) or `emdb2` (world); good frames only; the
    released bounding boxes; gendered SMPL ground truth."""
    root = Path(root)
    samples = []
    for pkl in sorted(root.glob("P*/*/*_data.pkl")):
        with open(pkl, "rb") as f:
            seq = pickle.load(f, encoding="latin1")
        if subset == "dev":  # model selection: the sequences outside EMDB-1, so EMDB-1 stays a once-only test set
            if seq.get("emdb1", False):
                continue
        elif not seq.get(subset, False):
            continue
        name = f"{pkl.parent.parent.name}/{pkl.parent.name}"
        smpl = seq["smpl"]
        cam = seq["camera"]
        K = np.asarray(cam["intrinsics"], dtype=np.float32)
        good = np.asarray(seq["good_frames_mask"], dtype=bool)
        boxes = np.asarray(seq["bboxes"]["bboxes"], dtype=np.float32)
        betas = np.asarray(smpl["betas"], dtype=np.float32)[:10]
        for frame in np.flatnonzero(good):
            samples.append({
                "id": f"{name}/{frame:05d}",
                "image": str(pkl.parent / "images" / f"{frame:05d}.jpg"),
                "bbox": boxes[frame],
                "K": K,
                "gender": str(seq["gender"]),
                "global_orient": np.asarray(smpl["poses_root"][frame], dtype=np.float32),
                "body_pose": np.asarray(smpl["poses_body"][frame], dtype=np.float32),
                "betas": betas,
                "transl": np.asarray(smpl["trans"][frame], dtype=np.float32),
                "extrinsic": np.asarray(cam["extrinsics"][frame], dtype=np.float32),
            })
    return samples


def wds_test(root: str | Path) -> list[dict]:
    """Index-file test sets (<root>/index.pkl, one dict per person): SMPL ground truth (gender, global_orient, body_pose,
    betas, transl, extrinsic; or gt_verts_file / gt_verts_row) or MHR ground truth (model_params, shape_params, mapped to
    SMPL topology by the evaluator like a prediction)."""
    with open(Path(root) / "index.pkl", "rb") as f:
        return pickle.load(f)


def wds_index(name: str):
    """Another index file of a bench set directory (e.g. the same samples with other ground truth)."""
    def read(root):
        with open(Path(root) / name, "rb") as f:
            return pickle.load(f)
    return read


DATASETS = {"3dpw": threedpw, "emdb1": lambda root: emdb(root, "emdb1"), "emdb_dev": lambda root: emdb(root, "dev"),
            "athletepose3d": wds_test, "moyo_val": wds_test, "egobody_val": wds_test, "rich_val": wds_test,
            "aistpp": wds_test, "3dpw_validation": lambda root: threedpw(root, "validation")}
DATASETS["emdb2"] = lambda root: emdb(root, "emdb2")  # world-space sequences (contact evaluation)
# held-out test splits: bench/build_*_test.py, bench/eval_ssp3d.py build
DATASETS.update({"harmony4d_test": wds_test, "harmony4d_test_mhr": wds_index("index_mhr.pkl"), "rich_test": wds_test,
                 "egobody_test": wds_test, "egohumans_test": wds_test, "ssp3d": wds_test})
# FAB-Hard 3D items, root = $FAB_DATA/fab_hard/<category> (the FAB-Hard index files, docs/evaluation.md)
DATASETS.update({f"fabhard_{c}": wds_index("index3d.pkl") for c in ("lowlight", "extreme_pose", "acrobatics_martial",
                                                                    "occlusion", "dance")})
# Legacy reader policy. eval_hmr rejects AIST's placeholder meshes; calibrated camera-frame
# joint evaluation is separate in eval_aist_joints and does not use this classification.
WORLD_FRAME = {"aistpp"}
# robustness variants: same samples, image corrupted per bench/corruptions.py (suffix = corruption name)
DATASETS.update({f"{name}_dark": fn for name, fn in list(DATASETS.items())})
DATASETS.update({f"{name}_{c}": DATASETS[name] for c in ("blur", "lowres") for name in
                 ("3dpw", "emdb1", "rich_test", "egobody_test", "harmony4d_test")})


def loose_boxes(fn, scale=1.3, jitter=0.1):
    """Box-protocol robustness: every box enlarged by `scale` about its centre and
    shifted by up to `jitter` of its size, deterministic per sample id (a looser / misplaced detector box)."""
    import zlib

    def read(root):
        out = []
        for s in fn(root):
            b = np.asarray(s["bbox"], np.float32)
            rng = np.random.default_rng(zlib.crc32(str(s["id"]).encode()))
            c, wh = (b[:2] + b[2:]) / 2, (b[2:] - b[:2]) * scale
            c = c + rng.uniform(-jitter, jitter, 2).astype(np.float32) * (b[2:] - b[:2])
            out.append(dict(s, bbox=np.concatenate([c - wh / 2, c + wh / 2]).astype(np.float32)))
        return out
    return read


DATASETS.update({f"{name}_loose": loose_boxes(DATASETS[name]) for name in
                 ["3dpw", "emdb1", "rich_test", "egobody_test", "harmony4d_test"] + [k for k in DATASETS if k.startswith("fabhard_")
                                                                                  and not k.endswith("_dark")]})


def trunc_boxes(fn, mode, shift=0.3):
    """Truncated-body protocol (orig = the original box as the control): upper / lower = the top /
    bottom half of the box (the other half of the body leaves the crop), shift = the box moved by `shift` of its width
    or height left / right / up / down (direction fixed per sample id); the original box is kept as bbox_orig."""
    import zlib

    def read(root):
        out = []
        for s in fn(root):
            b = np.asarray(s["bbox"], np.float32)
            ym = (b[1] + b[3]) / 2
            if mode == "orig":  # the control: the same samples / stride with the original box
                nb = b
            elif mode == "upper":
                nb = np.array([b[0], b[1], b[2], ym], np.float32)
            elif mode == "lower":
                nb = np.array([b[0], ym, b[2], b[3]], np.float32)
            else:
                k = zlib.crc32(f"shift/{s['id']}".encode()) % 4
                d = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], np.float32)[k] * shift * (b[2:] - b[:2])
                nb = np.concatenate([b[:2] + d, b[2:] + d]).astype(np.float32)
            out.append(dict(s, bbox=nb, bbox_orig=b))
        return out
    return read


DATASETS.update({f"{name}_{m}": trunc_boxes(DATASETS[name], m) for m in ("orig", "upper", "lower", "shift")
                 for name in ("3dpw", "emdb1")})
