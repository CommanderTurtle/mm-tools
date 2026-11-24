"""Readers and geometry helpers shared by the test-split builders (bench/build_rich_test.py, build_egobody_test.py,
build_harmony4d_test.py). Every dataset is read from its official release, downloaded locally (datasets/README.md)."""

from __future__ import annotations

import io
import pickle
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

from bench.body_models import _rodrigues

# ----------------------------------------------------------------------------------------------------------------- RICH
PLY_RE = re.compile(rb"element vertex (\d+)")


def read_ply_vertices(data: bytes) -> np.ndarray:
    """Binary little-endian PLY (vertex x, y, z float32 first) -> (V, 3)."""
    end = data.index(b"end_header") + len(b"end_header")
    end += 2 if data[end : end + 2] == b"\r\n" else 1
    header = data[:end]
    n = int(PLY_RE.search(header).group(1))
    props = re.findall(rb"property (\w+) (\w+)\n", header.split(b"element face")[0])
    if not header.count(b"binary_little_endian") or [p[1] for p in props[:3]] != [b"x", b"y", b"z"] or \
            any(t != b"float" for t, _ in props):
        raise ValueError(f"unsupported ply header: {header[:300]!r}")
    stride = len(props)
    return np.frombuffer(data, np.float32, n * stride, end).reshape(n, stride)[:, :3].copy()


def read_camera(xml_path):
    fs = cv2.FileStorage(str(xml_path), cv2.FILE_STORAGE_READ)
    E, K, D = fs.getNode("CameraMatrix").mat(), fs.getNode("Intrinsics").mat(), fs.getNode("Distortion").mat()
    fs.release()
    return E[:, :3].astype(np.float64), E[:, 3].astype(np.float64), K.astype(np.float64), D.ravel().astype(np.float64)


def _save_frame(blob, dst, scale):
    img = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    shape = img.shape[:2]
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    return shape


# -------------------------------------------------------------------------------------------------------------- EgoBody
GENDER = {"male": 1, "female": 2}
# SMPL joints vs OpenPose BODY_25 keypoints: shoulders, elbows, wrists, knees, ankles (left, right)
SMPL_LIMBS = [16, 17, 18, 19, 20, 21, 4, 5, 7, 8]
OP_LIMBS = [5, 2, 6, 3, 7, 4, 13, 10, 14, 11]


def egobody_zip(root, name):
    """A zip of the EgoBody release, opened from the local download folder `root`."""
    return zipfile.ZipFile(Path(root) / name)


def rotmat_to_aa(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(R.astype(np.float64))[0][:, 0]


def to_camera(body, R, t, go, bp, betas, tr):
    """SMPL (global_orient, transl) given in frame A -> frame B with x_B = R x_A + t (rotation about the pelvis)."""
    t_ = lambda a: torch.from_numpy(np.asarray(a, np.float32))[None].cuda()  # noqa: E731
    v_shaped = body.v_template + torch.einsum("bl,vcl->bvc", t_(betas), body.shapedirs)
    J0 = (body.J_regressor[0] @ v_shaped[0]).cpu().numpy().astype(np.float64)
    R_w = _rodrigues(torch.from_numpy(np.asarray(go, np.float64)[None])).numpy()[0]
    go_c = rotmat_to_aa(R @ R_w)
    tr_c = R @ (J0 + np.asarray(tr, np.float64)) + t - J0
    pose = np.concatenate([go_c, np.asarray(bp, np.float64)])
    v, j = body(t_(pose[:3]), t_(pose[3:72]), t_(betas), t_(tr_c))
    return pose, tr_c, v[0].cpu().numpy().astype(np.float64), j[0].cpu().numpy().astype(np.float64)


def box_of(uv, W, H, min_visible, min_height):
    full = np.concatenate([uv.min(0), uv.max(0)])
    clip = np.array([np.clip(full[0], 0, W), np.clip(full[1], 0, H), np.clip(full[2], 0, W), np.clip(full[3], 0, H)])
    area = lambda b: max(b[2] - b[0], 0) * max(b[3] - b[1], 0)  # noqa: E731
    if area(clip) >= min_visible * area(full) and clip[3] - clip[1] >= min_height:
        return clip.astype(np.float32)
    return None


def limb_err(uv_gt, kp2d, idx_gt, idx_kp, conf=None):
    e = np.linalg.norm(uv_gt[idx_gt] - kp2d[idx_kp, :2], axis=1)
    if conf is not None:
        e = e[conf[idx_kp] > 0.2]
    return float(np.median(e)) if len(e) >= 4 else np.nan


# ------------------------------------------------------------------------------------------------------------ Harmony4D
def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_colmap(cameras_txt, images_txt):
    """Exo cameras of a COLMAP text model: {camXX: (R, t, (model, W, H, params))} (world -> camera)."""
    cams = {}
    for line in cameras_txt.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        cams[int(f[0])] = (f[1], int(f[2]), int(f[3]), np.array([float(x) for x in f[4:]]))
    exo = {}
    lines = [ln for ln in images_txt.splitlines() if not ln.startswith("#")]
    for line in lines[::2]:
        f = line.split()
        if len(f) < 10 or not f[9].startswith("cam"):
            continue
        exo.setdefault(f[9].split("/")[0], (qvec2rotmat([float(x) for x in f[1:5]]),
                                            np.array([float(x) for x in f[5:8]]), cams[int(f[8])]))
    return exo


def colmap_from_world(z, work, names):
    """4x4 similarity world (SMPL fits) -> COLMAP model: inverse of colmap/workplace/scale.npy where it is a 4x4,
    else colmap_from_aria["aria01"], else identity (checked by reprojection per unit)."""
    if f"{work}/scale.npy" in names:
        S = np.asarray(np.load(io.BytesIO(z.read(f"{work}/scale.npy")), allow_pickle=True), np.float64)
        if S.shape == (4, 4):
            return np.linalg.inv(S)
    if f"{work}/colmap_from_aria_transforms.pkl" in names:
        return np.asarray(pickle.loads(z.read(f"{work}/colmap_from_aria_transforms.pkl"))["aria01"], np.float64)
    return np.eye(4)


class DirSource:
    """An extracted release directory with the read() / namelist() of a ZipFile."""

    def __init__(self, root):
        self.root = Path(root)
        self.names = [str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file()]

    def namelist(self):
        return self.names

    def read(self, name):
        return (self.root / name).read_bytes()
