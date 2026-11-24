"""Train FAB: the body run (FAB-L / FAB-S) and the frozen-body head runs (contact, visibility, G1, parts).

    python -m train.train_fab --config configs/train/fab-l.yaml --out runs/fab-l-body
    torchrun --standalone --nproc_per_node <N> -m train.train_fab --config configs/train/fab-l.yaml --out runs/fab-l-body
    python -m train.train_fab --config configs/train/heads-contact.yaml --init runs/fab-l-body/model.pt --out runs/fab-l-contact

A config file (YAML) sets the defaults of every argument below; arguments given on the command line override it, and
${VAR} in its strings is expanded from the environment (e.g. ${FAB_DATA}). docs/training.md describes the data format
(pool tars + label sidecars), the order body -> heads -> bundle (scripts/build_bundle.py) and the released configs.

Targets per crop: the teacher's 70 keypoints (3D pelvis-relative and 2D), its mesh (re-generated from its MHR
parameters with the eager rig), its rig parameters and camera translation, plus the source's human labels (2D, and 3D /
SMPL where present). Augmentation: photometric (incl. low light), crop-window zoom / shift, in-plane rotation and flips
(geometry is updated, so the camera stays consistent). Checkpoints are preemption-safe (`last.pt`, resumed automatically).
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import io
import json
import math
import os
import random
import re
import sys
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from fab import paths
from fab.model import FABImage
from fab.metadata import MHR70_PERMUTATION
from fab.models.heads.mhr_fast import SubsetMHR
from fab.utils.config import get_config
from train.aist_geometry import aist_geometry_valid

PELVIS = (9, 10)  # MHR70 left / right hip


# ----------------------------------------------------------------------------------------------------- data
def low_light(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Dim exposure + gamma + shot / read noise + colour cast (images in [0, 1])."""
    x = img * rng.uniform(0.05, 0.4)
    x = np.power(np.clip(x, 0, 1), rng.uniform(0.8, 1.4))
    x = x + np.random.normal(0, rng.uniform(0.005, 0.03), x.shape) * np.sqrt(np.maximum(x, 1e-4))
    x = x + np.random.normal(0, rng.uniform(0.002, 0.01), x.shape)
    x = x * np.array([rng.uniform(0.8, 1.2) for _ in range(3)])
    return np.clip(x / max(x.mean() * 2.5, 1e-3) * 0.5 if rng.random() < 0.5 else x, 0, 1)


MOTION_BLUR_P = 0.25


def motion_blur(x: np.ndarray, rng: random.Random) -> np.ndarray:
    """Linear motion blur: a 1-px line of 3-21 px at a random angle (scaled with the crop size, defined at 512)."""
    n = int(round(rng.uniform(3, 21) * x.shape[0] / 512)) | 1
    if n < 3:
        return x
    k = np.zeros((n, n), np.float32)
    k[n // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D(((n - 1) / 2, (n - 1) / 2), rng.uniform(0, 180), 1.0)
    k = cv2.warpAffine(k, rot, (n, n))
    return cv2.filter2D(x, -1, k / max(k.sum(), 1e-6), borderType=cv2.BORDER_REFLECT)


def batch_occlude(img, p_occ=0.0, p_lowres=0.0):
    """GPU batch augmentation after FAH's --p_ego: (1) with p_lowres, a crop seen at 1/2-1/6 resolution (area down,
    bilinear up); (2) with p_occ, a rectangle (15-45 % of the side) cut from another crop of the batch pasted at a
    random place, with a feathered edge. Targets are unchanged: the body is still there, only hidden."""
    B, _, S, _ = img.shape
    img = img.clone()
    for i in range(B):
        if random.random() < p_lowres:
            s = max(16, int(S / random.uniform(2.0, 6.0)))
            small = F.interpolate(img[i : i + 1], size=(s, s), mode="area")
            img[i] = F.interpolate(small, size=(S, S), mode="bilinear", align_corners=False)[0]
        if B > 1 and random.random() < p_occ:
            j = (i + random.randint(1, B - 1)) % B
            h, w = (int(S * random.uniform(0.15, 0.45)) for _ in range(2))
            ys, xs, yd, xd = (random.randint(0, S - h), random.randint(0, S - w), random.randint(0, S - h),
                              random.randint(0, S - w))
            ry = torch.linspace(0, 1, h, device=img.device)
            rx = torch.linspace(0, 1, w, device=img.device)
            edge = lambda r: (torch.minimum(r, 1 - r) * 8).clamp(max=1)  # noqa: E731  (feathered border)
            a = (edge(ry)[:, None] * edge(rx)[None])[None]
            img[i, :, yd : yd + h, xd : xd + w] = (a * img[j, :, ys : ys + h, xs : xs + w]
                                                   + (1 - a) * img[i, :, yd : yd + h, xd : xd + w])
    return img


def label_silhouette(b, smpl_target, size, res=64):
    """(B, 1, size, size) approximate person silhouette in the crop from the 3D labels (GT SMPL where valid, else the
    MHR teacher's mesh): its SMPL-topology points projected into the crop, splatted on a res x res grid, closed (dilate
    5, erode 3) and upsampled. Flipped crops see the mirrored body; rolled crops the rolled one."""
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        roll = lambda v: torch.einsum("bij,bvj->bvi", b["rot"], v)  # noqa: E731
        t_cam = roll(smpl_target.pred(b["t_model_params"], b["t_shape"])) + b["t_cam_t"][:, None]
        g_cam = roll(smpl_target.gt(b["gt_smpl_pose"], b["gt_smpl_betas"], b["gt_smpl_trans"], b["gt_smpl_gender"],
                                    verts=b.get("gt_smpl_v"), verts_valid=b.get("gt_smpl_v_valid")))
        pts = torch.where(b["gt_smpl_valid"][:, None, None] > 0, g_cam, t_cam)
        pts = torch.where(b["flipped"][:, None, None] > 0, pts * torch.tensor([-1.0, 1.0, 1.0], device=pts.device), pts)
        uv = project_crop(pts, b["cam_int"], b["affine"], size)
        ok = (pts[..., 2] > 0.1) & (uv >= 0).all(-1) & (uv < 1).all(-1)
        ij = (uv * res).long().clamp(0, res - 1)
        flat = ij[..., 1] * res + ij[..., 0]
        grid = torch.zeros(len(uv), res * res, device=uv.device)
        grid.scatter_(1, flat, ok.float())
        grid = F.max_pool2d(grid.view(-1, 1, res, res), 5, 1, 2)
        grid = -F.max_pool2d(-grid, 3, 1, 1)
        return (F.interpolate(grid, size=(size, size), mode="bilinear", align_corners=False) > 0.5).float()


def box_masks(box, S):
    """(B, 4) boxes in crop units -> (B, 1, S, S) filled rectangles: the box as the decoder's mask prompt (--box_prompt_p;
    bench/eval_hmr.py --mask_prompt box at test time)."""
    g = (torch.arange(S, device=box.device, dtype=torch.float32) + 0.5) / S
    x = (g[None] >= box[:, 0:1]) & (g[None] <= box[:, 2:3])
    y = (g[None] >= box[:, 1:2]) & (g[None] <= box[:, 3:4])
    return (y[:, :, None] & x[:, None, :]).float()[:, None]


def noisy_masks(m):
    """Mask-head-style variation of label silhouettes (B, 1, S, S): random dilation / erosion, shift, dropped band."""
    out = m.clone()
    S = m.shape[-1]
    for i in range(len(m)):
        k = random.choice([1, 3, 5, 9, 15])
        x = out[i : i + 1]
        x = F.max_pool2d(x, k, 1, k // 2) if random.random() < 0.7 else -F.max_pool2d(-x, k, 1, k // 2)
        dy, dx = (random.randint(-S // 40, S // 40) for _ in range(2))
        x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
        if random.random() < 0.2:  # a missed band (as a mask head on an occluded limb)
            y0 = random.randint(0, S - S // 6)
            x[..., y0 : y0 + S // 6, :] = 0
        out[i : i + 1] = x
    return out


def paste_distractors(img, sil, p):
    """With probability p per crop, another crop's person (its label silhouette, feathered) is pasted over the target
    with a horizontal shift of 20-50 % of the crop (overlapping people), in front or (half the time) behind the target's
    silhouette. The target's labels are unchanged."""
    B, _, S, _ = img.shape
    img = img.clone()
    for i in range(B):
        if B < 2 or random.random() >= p:
            continue
        j = (i + random.randint(1, B - 1)) % B
        dx = int(S * random.uniform(0.2, 0.5)) * random.choice([-1, 1])
        dy = int(S * random.uniform(-0.08, 0.08))
        m = torch.zeros_like(sil[j])
        src = torch.zeros_like(img[j])
        ys, yd = (slice(0, S - dy), slice(dy, S)) if dy >= 0 else (slice(-dy, S), slice(0, S + dy))
        xs, xd = (slice(0, S - dx), slice(dx, S)) if dx >= 0 else (slice(-dx, S), slice(0, S + dx))
        m[:, yd, xd], src[:, yd, xd] = sil[j][:, ys, xs], img[j][:, ys, xs]
        if random.random() < 0.5:  # behind the target
            m = m * (1 - sil[i])
        m = F.avg_pool2d(m[None], 5, 1, 2)[0]  # feathered edge
        img[i] = m * src + (1 - m) * img[i]
    return img


def photometric(img: np.ndarray, rng: random.Random) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    if rng.random() < 0.8:
        x = x * rng.uniform(0.6, 1.4) + rng.uniform(-0.15, 0.15)
        mean = x.mean(axis=(0, 1), keepdims=True)
        x = (x - mean) * rng.uniform(0.6, 1.4) + mean
        gray = x.mean(axis=2, keepdims=True)
        x = (x - gray) * rng.uniform(0.5, 1.5) + gray
    if rng.random() < 0.1:
        x = np.repeat(x.mean(axis=2, keepdims=True), 3, axis=2)
    if rng.random() < 0.2:
        x = low_light(np.clip(x, 0, 1), rng)
    if rng.random() < 0.2:
        k = rng.choice([3, 5, 7])
        x = cv2.GaussianBlur(x, (k, k), 0)
    if rng.random() < MOTION_BLUR_P:  # handheld / fast-motion footage: line kernel, random direction and length
        x = motion_blur(x, rng)
    if rng.random() < 0.2:
        x = x + np.random.normal(0, rng.uniform(0.01, 0.05), x.shape)
    x = np.clip(x, 0, 1)
    if rng.random() < 0.3:
        q = rng.randint(30, 90)
        ok, enc = cv2.imencode(".jpg", (x * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
        x = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    return x.astype(np.float32)


GT_CAMERA, GT_OTHER_FRAME = 1, 2
WORLD_FRAME_SOURCES = {"aistpp_train"}  # teacher-agreement axes; --aist_camera_gt uses the verified camera convention


def sidecar_keys(archive, path):
    keys = archive["keys"].tolist()
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate sample keys in {path}; repair the pool and sidecars before training")
    return keys


def box_crop(rec, A, S):
    """The pool record's box (bbox x0, y0, x1, y1 in the image; the whole crop without one) in crop units under the
    crop affine A (3, 3, image -> crop px): the axis-aligned hull of its corners (rotation / flip safe). -> (4,) f32"""
    if "bbox" not in rec:
        return np.array([0, 0, 1, 1], np.float32)
    x0, y0, x1, y1 = np.asarray(rec["bbox"], np.float64)
    c = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]]) @ np.asarray(A, np.float64)[:2].T / S
    return np.array([c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()], np.float32)


def mask_kp_vis(fg, keys, vis_path, body=tuple(range(21)) + (41, 62)):
    """Teacher-mask gate from the visibility sidecar: per crop, the share of the body
    keypoints labelled visible that fall inside the 3-cell dilated mask (fg: packed 256 x 256); 1 when < 3 are
    visible (close-ups: nothing to check). -> (N,)"""
    with np.load(vis_path) as z:
        idx = {k: i for i, k in enumerate(z["keys"].tolist())}
        vis, uv = z["vis"], z["uv"]
    out = np.ones(len(keys), np.float32)
    for n, (key, f) in enumerate(zip(keys, fg)):
        i = idx.get(key)
        if i is None:
            continue
        sel = [k for k in body if vis[i, k] == 0]
        if len(sel) < 3:
            continue
        m = cv2.dilate(np.unpackbits(f)[: 256 * 256].reshape(256, 256), np.ones((7, 7), np.uint8)) > 0
        xy = np.clip((uv[i, sel].astype(np.float32) * 256).astype(int), 0, 255)
        out[n] = m[xy[:, 1], xy[:, 0]].mean()
    return out


def gt_keypoints_3d(gt, t_kp3d, world_frame=False, max_err=0.15, camera_frame_cm=False):
    """Human 3D keypoint labels in camera axes (m) and how to use them. The pools store them per source in
    camera axes or MHR axes (y, z negated). By default the axes are picked by teacher agreement and
    treats AIST++ as frame-free. Audited AIST++ labels instead use explicit camera coordinates in cm,
    bypassing the teacher-dependent frame heuristic."""
    g, w = gt[:, :3].astype(np.float64), (gt[:, 3] > 0).astype(np.float64)
    if w.sum() < 6 or not (w[PELVIS[0]] and w[PELVIS[1]]):
        return np.zeros_like(g), np.zeros_like(w), 0
    if camera_frame_cm:
        if not np.isfinite(g).all():
            raise ValueError("Nonfinite calibrated camera-frame ground truth")
        return g / 100, w, GT_CAMERA
    if np.ptp(g[w > 0], 0).max() > 10:  # cm
        g = g / 100
    v = w > 0

    def rel(x):
        return x - x[list(PELVIS)].mean(0)

    cands = [g, g * np.array([1.0, -1.0, -1.0])]
    errs = [np.linalg.norm(rel(c)[v] - rel(t_kp3d)[v], axis=1).mean() for c in cands]
    i = int(np.argmin(errs))
    return (cands[i], w, GT_CAMERA) if errs[i] < max_err and not world_frame else (g, w, GT_OTHER_FRAME)


class PoolDataset(IterableDataset):
    def __init__(self, tars: list[str], size: int, rank: int, world: int, train: bool = True, seed: int = 0,
                 shuffle_buffer: int = 2000, flip: float = 0.0, aug: bool = True, rot: float = 0.0, rot_full: float = 0.2,
                 g1_labels: str = "g1",
                 teacher_labels: str = "", contact_labels: str = "", contact_any_labels: str = "", contact_knee_labels: str = "", contact_phys_dev_max: float = 0.0,
                 contact_phys_dev_pools: str = "3dpw", aist_camera_gt: bool = False, aist_geometry_gate: bool = False,
                 min_in_frame: float = 0.0, pseudo_3d_sources: tuple = (), smpl_gt_off: tuple = (),
                 smpl_shape_off: tuple = (), mask_labels: str = "", mask_gate: dict | None = None, vis_labels: str = ""):
        self.tars, self.size, self.rank, self.world = tars, size, rank, world
        # person-mask sidecars <tar>.<name>.npz, comma-separated in priority order (e.g. human COCO masks, then
        # model-generated teacher masks, kept when they pass mask_gate: min score / kp_t / kp_gt,
        # max out_box; a keypoint share of -1 = too few keypoints in the crop, not gated)
        self.mask_labels = mask_labels
        self.mask_gate = mask_gate or {}
        # per-keypoint visibility sidecars <tar>.<vis_labels>.npz (docs/training.md): 70 MHR keypoints + J14
        self.vis_labels = vis_labels
        self.teacher_labels = teacher_labels
        self.aist_camera_gt = aist_camera_gt
        self.aist_geometry_gate = aist_geometry_gate
        self.aug, self.rot, self.rot_full = aug, rot, rot_full
        self.g1_labels = g1_labels
        self.contact_labels = contact_labels
        self.contact_any_labels = contact_any_labels
        self.contact_knee_labels = contact_knee_labels
        self.contact_phys_dev_max, self.contact_phys_dev_pools = contact_phys_dev_max, contact_phys_dev_pools
        self.flip = flip
        self.min_in_frame = min_in_frame
        self.pseudo_3d_sources = set(pseudo_3d_sources)
        self.smpl_gt_off = set(smpl_gt_off)
        self.smpl_shape_off = set(smpl_shape_off)
        self.train, self.seed, self.buffer = train, seed, shuffle_buffer
        self.epoch = 0

    def _samples(self, path):
        path = str(path)  # support Path inputs as well as glob-produced strings in optional label sidecars
        with tarfile.open(path) as tf:
            source = tf.pax_headers.get("pool.source", Path(path).parent.name)
            if "pool.source" not in tf.pax_headers:
                # Tars without a pool.source header named v1_<source>_<n>.tar (flattened copies) take <source>.
                # Their containing directory ("hard"/"mined") is not the label coordinate convention.
                for world_source in WORLD_FRAME_SOURCES:
                    if Path(path).stem.startswith(f"v1_{world_source}_"):
                        source = world_source
            # --foot_off_tars: tars whose GT feet are unreliable (e.g. Harmony4D, EgoBody egocentric): no foot losses
            foot_off = any(fnmatch.fnmatch(path, pat) if any(ch in pat for ch in "*?[") else pat in path
                           for pat in getattr(self, "foot_off_tars", ()))  # patterns, or plain substrings of the path
            # --gt2d_crop_tars: tars whose 2D labels outside the crop are projected GT of truncated parts (EgoBody-ego)
            crop2d_off = any(fnmatch.fnmatch(path, pat) if any(ch in pat for ch in "*?[") else pat in path
                             for pat in getattr(self, "gt2d_crop_tars", ()))
            g1 = {}  # G1 retargeting targets (<tar>.<g1_labels>.npz sidecar): key -> (q, R)
            side = Path(f"{path[:-4]}.{self.g1_labels}.npz")
            if side.exists():
                with np.load(side) as z:
                    keys = sidecar_keys(z, side)
                    columns = [z[k] for k in ("q", "R")]
                    g1 = dict(zip(keys, zip(*columns)))
            others = {}  # --neg_prompt_p: the frame's other people (<tar>.others.npz): key -> (K, 4) xyxy
            side = Path(f"{path[:-4]}.others.npz")
            if getattr(self, "neg_prompt", None) and side.exists():
                with np.load(side) as z:
                    others = dict(zip(sidecar_keys(z, side), z["boxes"]))
            detbox = {}  # --det_box_p: the person detector's box + the frame's other detections (det_boxes_wide.py)
            side = Path(f"{path[:-4]}.detbox.npz")
            if getattr(self, "det_box", None) and side.exists():
                with np.load(side) as z:
                    detbox = dict(zip(sidecar_keys(z, side), zip(z["det"], z["others"])))
            contact = {}  # contact labels from world GT motion (<tar>.<contact_labels>.npz sidecar): key -> (c, valid)
            names = [s for s in self.contact_labels.split(",") if s]  # "a,b": the first sidecar present per tar
            side = next((p_ for p_ in (Path(f"{path[:-4]}.{s}.npz") for s in names) if p_.exists()),
                        Path(f"{path[:-4]}.{self.contact_labels}.npz"))
            unstable = set()
            if self.contact_labels and side.exists():
                with np.load(side) as z:
                    keys = sidecar_keys(z, side)
                    h = z["h"] if "h" in z.files else np.full(z["c"].shape, np.nan, np.float16)
                    valid = z["valid"]
                    if "stable" in z.files:  # physics-simulation labels: 0 stable, 1 flight, 2 failed -> masked
                        bad = z["stable"] != 0
                        if self.contact_phys_dev_max > 0 and "dev_mm" in z.files and any(
                                t in str(path) for t in self.contact_phys_dev_pools.split(",") if t):
                            bad = bad | ~(z["dev_mm"].astype(np.float32) < self.contact_phys_dev_max)  # sim far off the GT
                        valid = valid * (~bad)[:, None].astype(valid.dtype)
                        unstable = {k for k, b_ in zip(keys, bad) if b_}
                    contact = dict(zip(keys, zip(z["c"], valid, h)))
            knee = {}  # --contact_knee_labels: the knee channels from this sidecar ("a,b": the first present per tar)
            side_k = next((p_ for p_ in (Path(f"{path[:-4]}.{s}.npz") for s in self.contact_knee_labels.split(",") if s)
                           if p_.exists()), Path(f"{path[:-4]}.{self.contact_knee_labels}.npz"))
            if self.contact_knee_labels and side_k.exists():
                with np.load(side_k) as z:
                    h = z["h"] if "h" in z.files else np.full(z["c"].shape, np.nan, np.float16)
                    knee = dict(zip(sidecar_keys(z, side_k), zip(z["c"], z["valid"], h)))
            contact_any = {}  # any-surface contact labels where a dataset ships them (RICH human-scene contact)
            for name in [s for s in self.contact_any_labels.split(",") if s]:  # "a,b": the first sidecar present
                side = Path(f"{path[:-4]}.{name}.npz")
                if side.exists():
                    with np.load(side) as z:
                        keys = sidecar_keys(z, side)
                        valid = z["valid"]
                        if name.startswith("phys"):  # simulated labels share the simulation's stability
                            valid = valid * np.array([k not in unstable for k in keys], valid.dtype)[:, None]
                        contact_any = dict(zip(keys, zip(z["c"], valid)))
                    break
            masks = {}  # person-mask labels: key -> (fg, unknown or None) packed 256 x 256 bits in the stored crop
            for name in [s for s in self.mask_labels.split(",") if s]:
                side = Path(f"{path[:-4]}.{name}.npz")
                if not side.exists():
                    continue
                with np.load(side) as z:
                    keys = sidecar_keys(z, side)
                    ok = np.ones(len(keys), bool)
                    for k, thr in self.mask_gate.items():
                        if k in z.files:
                            v = z[k]
                            ok &= (v <= thr) if k == "out_box" else ((v >= thr) | (v < 0))
                    vside = Path(f"{path[:-4]}.visibility.npz")  # visibility labels (docs/training.md) first
                    vside = vside if vside.exists() else Path(f"{path[:-4]}.vis.npz")
                    if "kp_vis" in self.mask_gate and "unk" not in z.files and vside.exists():
                        ok &= mask_kp_vis(z["fg"], keys, vside) >= self.mask_gate["kp_vis"]
                    unk = z["unk"] if "unk" in z.files else [None] * len(keys)
                    for key, f, u, keep in zip(keys, z["fg"], unk, ok):
                        if keep and key not in masks:
                            masks[key] = (f, u)
            vis = {}  # visibility labels: key -> (vis (84,) 0 visible / 1 occluded / 2 out of crop / 255, uv (84, 2))
            side = Path(f"{path[:-4]}.{self.vis_labels}.npz")
            for a_, b_ in getattr(self, "vis_alias", ()):  # --vis_alias: a re-cut pool with the same stored crops / keys
                if not side.exists() and a_ in path:
                    side = Path(f"{path.replace(a_, b_)[:-4]}.{self.vis_labels}.npz")
            if self.vis_labels and side.exists():
                with np.load(side) as z:
                    why = z["why"] if "why" in z.files else [None] * len(z["vis"])  # reason bits (8: a human v-flag)
                    vis = dict(zip(sidecar_keys(z, side), zip(z["vis"], z["uv"], why)))
            kp2d_side = {}  # --kp2d_labels: extra 2D labels key -> (70, 3) stored crop px + confidence (MHR70 order)
            for ni, name in enumerate([s for s in getattr(self, "kp2d_labels", "").split(",") if s]):
                # "name!": the sidecar's labels also replace the pool's own (e.g. human WholeBody feet over the COCO
                # pool's projected-fit feet); plain "name": only the channels the pool has no label for
                # "name@tol": per-foot confidence re-thresholded from the sidecar's dev (N, 2) (e.g. model-agreement labels:
                # the closest model's deviation / foot length; accepted where dev <= tol)
                name, tol = (name.split("@")[0], float(name.split("@")[1])) if "@" in name else (name, None)
                name, ovr = (name[:-1], True) if name.endswith("!") else (name, False)
                side = Path(f"{path[:-4]}.{name}.npz")
                for a_, b_ in getattr(self, "kp2d_alias", ()):  # --kp2d_alias: a re-cut pool with the same stored crops
                    if not side.exists() and a_ in path:
                        side = Path(f"{path.replace(a_, b_)[:-4]}.{name}.npz")
                if side.exists():
                    with np.load(side) as z:
                        # optional vis (N, 70): human visibility per channel, 0 visible / 1 occluded / 2 out of frame /
                        # 255 unknown (--vis_human); (N, 6): the feet only (MHR 15-20)
                        hvis = z["vis"] if "vis" in z.files else [None] * len(z["kp2d"])
                        if "vis" in z.files and hvis.shape[1] == 6:
                            full = np.full((len(hvis), 70), 255, hvis.dtype)
                            full[:, 15:21] = hvis
                            hvis = full
                        kp = z["kp2d"]
                        if tol is not None and "dev" in z.files:
                            kp = kp.copy()
                            for f_, ch in enumerate(((15, 16, 17), (18, 19, 20))):
                                kp[:, list(ch), 2] = (z["dev"][:, f_] <= tol)[:, None].astype(kp.dtype)
                        for key, k2, hv in zip(sidecar_keys(z, side), kp, hvis):
                            kp2d_side.setdefault(key, []).append((k2, ovr, hv, ni))  # per channel the first listed wins
            footfix = {}  # --gt_fix_labels: corrected GT foot rotations (<tar>.<gt_fix_labels>.npz) key -> rows
            side = Path(f"{path[:-4]}.{getattr(self, 'gt_fix_labels', '')}.npz")
            for a_, b_ in getattr(self, "gt_fix_alias", ()):  # --gt_fix_alias: a re-cut pool with the same sample keys
                if not side.exists() and a_ in path:
                    side = Path(f"{path.replace(a_, b_)[:-4]}.{self.gt_fix_labels}.npz")
            fix_tars = getattr(self, "gt_fix_tars", ())  # --gt_fix_tars: only these tars (substrings / patterns)
            fix_here = not fix_tars or any(fnmatch.fnmatch(path, q) if any(ch in q for ch in "*?[") else q in path
                                           for q in fix_tars)
            if getattr(self, "gt_fix_labels", "") and side.exists() and fix_here:
                with np.load(side) as z:  # full form: corrected gt_smpl_pose (N, 72) + has_ref; short form: R_foot + err_before
                    n_ = len(z["ok"])
                    err = z["err_before"] if "err_before" in z.files else np.full((n_, 2), np.inf)
                    ref = z["has_ref"] if "has_ref" in z.files else z["ok"]
                    pz = z["gt_smpl_pose"] if "gt_smpl_pose" in z.files else [None] * n_
                    rz = z["R_foot"] if "R_foot" in z.files else [None] * n_
                    footfix = dict(zip(sidecar_keys(z, side), zip(rz, z["ok"], z["reversed"], err, pz, ref)))
            relabel, rl_index = {}, {}  # labels of a stronger teacher (<tar>.<teacher_labels>.npz sidecar, e.g. FAB-H)
            side = Path(f"{path[:-4]}.{self.teacher_labels}.npz")
            if self.teacher_labels and side.exists():
                with np.load(side) as z:
                    relabel = {k: z[k] for k in z.files if k != "keys"}
                    rl_index = {k: i for i, k in enumerate(sidecar_keys(z, side))}
            skip_rx = getattr(self, "skip_rx", None)  # --skip_takes: benchmark takes, never loaded
            skip_keys = getattr(self, "skip_pairs", {}).get(path.split("/pools/", 1)[-1].rsplit("/", 1)[0])  # --skip_pairs
            members = {}
            for m in tf:
                if not m.isfile():
                    continue
                key, ext = m.name.rsplit(".", 1)
                if skip_rx is not None and skip_rx.search(key):
                    continue
                if skip_keys is not None and key in skip_keys:
                    continue
                members.setdefault(key, {})[ext] = tf.extractfile(m).read()
                if len(members[key]) == 2:
                    parts = members.pop(key)
                    parts["src"] = source
                    parts["key"] = key  # (label review tools)
                    if key in g1:
                        parts["g1"] = g1[key]
                    if key in others:
                        parts["others"] = others[key]
                    if key in detbox:
                        parts["detbox"] = detbox[key]
                    if key in contact:
                        parts["contact"] = contact[key]
                        if key in knee:  # knees (channels 6, 7) from another label set (e.g. the kinematic one)
                            c_, v_, h_ = (np.array(x) for x in contact[key])
                            ck, vk, hk = knee[key]
                            c_[6:8], v_[6:8], h_[6:8] = ck[6:8], vk[6:8], hk[6:8]
                            parts["contact"] = (c_, v_, h_)
                    if key in contact_any:
                        parts["contact_any"] = contact_any[key]
                        if key in knee:  # the knees' any-contact then follows their own height rule
                            ca_, va_ = (np.array(x) for x in contact_any[key])
                            va_[6:8] = 0
                            parts["contact_any"] = (ca_, va_)
                    if key in masks:
                        parts["mask"] = masks[key]
                    if key in vis:
                        parts["vis"] = vis[key]
                    if foot_off:
                        parts["foot_off"] = True
                    if crop2d_off:
                        parts["crop2d_off"] = True
                    if key in footfix:
                        parts["footfix"] = footfix[key]
                    if key in kp2d_side:
                        parts["kp2d_side"] = kp2d_side[key]
                    if key in rl_index:
                        parts["relabel"] = {k: v[rl_index[key]] for k, v in relabel.items()}
                    yield parts

    def _item(self, parts, rng):
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(parts["jpg"], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        rec = dict(np.load(io.BytesIO(parts["npz"])))
        rec.update(parts.get("relabel", {}))
        from_side = np.zeros(len(rec["gt_kp2d"]), bool)  # channels the --kp2d_labels sidecars supplied
        src2d = np.full(len(rec["gt_kp2d"]), -1, np.int64)  # which sidecar (index in --kp2d_labels) per channel; -1 pool
        if "kp2d_side" in parts:  # --kp2d_labels: extra 2D (whole-body feet / hands) on the channels the pool has none
            own = rec["gt_kp2d"].astype(np.float64)
            for k2, ovr, _, ni in parts["kp2d_side"]:  # in --kp2d_labels order: per channel the first with a label wins
                side = np.asarray(k2, np.float64)
                fill = ((own[:, 2] <= 0) | ovr) & ~from_side & (side[:, 2] > 0) & np.isfinite(side).all(1)
                if getattr(self, "sidecar_no_feet", False):  # --sidecar_no_feet: the foot channels stay the pool's own
                    fill[15:21] = False
                own[fill] = side[fill]
                from_side = from_side | fill
                src2d[fill] = ni
            rec["gt_kp2d"] = own
        if getattr(self, "human_feet_2d", False):  # --human_feet_2d: foot 2D (MHR70 15-20) only from the human sidecars
            k2 = rec["gt_kp2d"].astype(np.float64).copy()  # (the pools' own foot channels can be projected fits)
            ft = np.zeros(len(k2), bool)
            ft[15:21] = True
            k2[ft & ~from_side, 2] = 0.0
            rec["gt_kp2d"] = k2
        foot_off = bool(parts.get("foot_off"))
        if "footfix" in parts and "gt_smpl_pose" in rec:  # --gt_fix_labels: direction-fixed GT ankles (SMPL 7 / 8)
            Rf, okf, rev, err, pfix, ref = parts["footfix"]
            okf = np.asarray(okf, bool) & (np.asarray(err, np.float64) > getattr(self, "gt_fix_min_deg", 0.0))
            pose = np.asarray(rec["gt_smpl_pose"], np.float64).copy()
            for s_ in range(2):
                sl_ = slice(21 + 3 * s_, 24 + 3 * s_)  # SMPL joint 7 (left ankle) / 8 (right ankle)
                if okf[s_] and pfix is not None and np.isfinite(np.asarray(pfix)[sl_]).all():
                    pose[sl_] = np.asarray(pfix, np.float64)[sl_]
                elif okf[s_] and Rf is not None and np.isfinite(Rf[s_]).all():
                    pose[sl_] = cv2.Rodrigues(np.asarray(Rf[s_], np.float64))[0].ravel()
            rec["gt_smpl_pose"] = pose.astype(np.float32)
            # foot losses back on (a --foot_off_tars source) only where both feet had a usable 2D reference (refined
            # or verified within 10 deg); off on a reversed / swapped foot
            if foot_off and bool(np.asarray(ref, bool).all()):
                foot_off = False
            if np.asarray(rev, bool).any():
                foot_off = True
        if foot_off:  # --foot_off_tars: no foot 2D labels (MHR70 heels / toes 15-20) in any 2D term
            k2 = rec["gt_kp2d"].astype(np.float64).copy()
            off = np.zeros(len(k2), bool)
            off[15:21] = True
            if getattr(self, "foot_off_keep_side", False):  # --foot_off_keep_side: image-based sidecar feet stay
                off &= ~from_side
            k2[off, 2] = 0.0
            rec["gt_kp2d"] = k2
        stored = img.shape[0]
        S = self.size
        # crop-window jitter: new_px = s * (px - c) + c + t, then resize stored -> S
        s, tx, ty = 1.0, 0.0, 0.0
        if self.train and self.aug:
            s = rng.uniform(0.85, 1.2)
            tx, ty = (rng.uniform(-0.08, 0.08) * stored for _ in range(2))
        # --loose_box_p: annotation-style loose boxes, only on crops whose pool keeps real context around the
        # stored crop (wide_jpg, wide_off px per side), so the window never shows black
        # padding; the shift is clipped to stay inside that context. --wide_context: every window from the wide crop
        lb = getattr(self, "loose_box", None)
        use_wide = "wide_jpg" in rec and getattr(self, "wide_context", False)
        if self.train and self.aug and lb and "wide_jpg" in rec and rng.random() < lb[0]:
            s = rng.uniform(lb[1], lb[2])
            lim = max(0.0, s * (stored / 2.0 + float(rec["wide_off"])) - stored / 2.0)
            tx, ty = (rng.uniform(-1.0, 1.0) * min(lb[3] * stored, lim) for _ in range(2))
            use_wide = True
        # --det_box_p: the crop drawn from the person detector's box (group boxes included), as the inference pipeline
        # crops it: window = 1.25 x its longer side about its centre, inside the wide crop's context (zoom / shift clipped)
        det_crop = None
        dbp = getattr(self, "det_box", None)
        if (self.train and self.aug and dbp and "wide_jpg" in rec and "detbox" in parts
                and np.isfinite(parts["detbox"][0]).all() and rng.random() < dbp):
            db = np.asarray(parts["detbox"][0], np.float64)
            A0 = rec["affine"].astype(np.float64)  # orig -> stored crop
            q = np.array([[db[0], db[1], 1], [db[2], db[3], 1]]) @ A0.T
            side = 1.25 * max(abs(q[1, 0] - q[0, 0]), abs(q[1, 1] - q[0, 1]))
            off = float(rec["wide_off"])
            s = max(stored / max(side, 1.0), (stored / 2.0) / (stored / 2.0 + off))
            lim = max(0.0, s * (stored / 2.0 + off) - stored / 2.0)
            tx, ty = (float(np.clip(-s * ((q[0, i] + q[1, i]) / 2 - stored / 2.0), -lim, lim)) for i in range(2))
            use_wide, det_crop = True, db
        # --trunc_aug: a truncated body as the inference pipeline crops it: the window is 1.25 x the
        # longer side of the record's box with a fraction f cut off one side, centred on the kept part (clipped to the
        # stored / wide crop: no new black padding); all labels are kept, so the cut-off part stays supervised in 3D and
        # through the amodal 2D (--dense_uv_off; the soft-argmax 2D alone would pin it to the edge)
        tr = getattr(self, "trunc", None)
        if ((self.train and self.aug) or getattr(self, "trunc_always", False)) and tr and det_crop is None and "bbox" in rec \
                and rng.random() < tr[0]:  # (trunc_always: truncation without any other augmentation)
            A0 = rec["affine"].astype(np.float64)  # orig -> stored crop
            x0, y0, x1, y1 = np.asarray(rec["bbox"], np.float64)
            q = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]]) @ A0.T
            bx0, by0, bx1, by1 = q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()
            f = rng.uniform(tr[1], tr[2])
            # top / bottom / left / right of the box cut off, or a zoom-in x1.2-1.6 shifted to the context's limit (the
            # benchmark's shifted boxes; a cut along the box's shorter side keeps the same square crop without context)
            side = rng.choices((0, 1, 2, 3, 4), weights=(0.3, 0.3, 0.1, 0.1, 0.2))[0]
            bw, bh = bx1 - bx0, by1 - by0
            if side == 0:
                by0 += f * bh
            elif side == 1:
                by1 -= f * bh
            elif side == 2:
                bx0 += f * bw
            elif side == 3:
                bx1 -= f * bw
            off = float(rec["wide_off"]) if "wide_jpg" in rec else 0.0
            zmax = getattr(self, "trunc_zoom_max", 0.0) or 3.0  # --trunc_zoom_max: cap on the truncation crops' zoom-in
            if side == 4:
                s = min(rng.uniform(1.2, 1.6), max(zmax, 1.0))
                lim = max(0.0, s * (stored / 2.0 + off) - stored / 2.0)
                tx, ty = (rng.uniform(-1.0, 1.0) * lim for _ in range(2))
            else:
                s = min(stored / max(1.25 * max(bx1 - bx0, by1 - by0), 1.0), zmax)
                lim = max(0.0, s * (stored / 2.0 + off) - stored / 2.0)
                tx, ty = (float(np.clip(s * (stored / 2.0 - m), -lim, lim)) for m in ((bx0 + bx1) / 2, (by0 + by1) / 2))
            use_wide = use_wide or "wide_jpg" in rec
        c = stored / 2.0
        r = S / stored
        M = np.array([[s * r, 0, (c - s * c + tx) * r], [0, s * r, (c - s * c + ty) * r]], np.float64)
        # in-plane rotation = a camera roll: rotating the crop about any crop point is exactly the axis-aligned crop
        # of the image rotated about the principal point, whose camera is the original one rolled about its axis
        # (K unchanged); 2D targets / affine / box centre rotate in the image, 3D targets by R3 = R_z(theta)
        theta = 0.0
        if self.train and self.aug and self.rot > 0:
            u = rng.random()
            if u < self.rot * (1.0 - self.rot_full):
                theta = float(np.clip(rng.gauss(0.0, 25.0), -60.0, 60.0))
            elif u < self.rot:
                theta = rng.uniform(-180.0, 180.0)
        R3 = np.eye(3)
        if theta != 0.0:
            co, si = math.cos(math.radians(theta)), math.sin(math.radians(theta))
            R2 = np.array([[co, -si], [si, co]])
            R3[:2, :2] = R2
            u0 = np.full(2, (S - 1) / 2.0)
            Rc = np.eye(3)
            Rc[:2, :2], Rc[:2, 2] = R2, u0 - R2 @ u0
            M = (Rc @ np.vstack([M, [0, 0, 1]]))[:2]
        M_img = M
        if use_wide:  # the same window cut from the wide crop: wide px = stored px + wide_off
            img = cv2.cvtColor(cv2.imdecode(rec["wide_jpg"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            M_img = M.copy()
            M_img[:, 2] -= float(rec["wide_off"]) * M[:, :2].sum(1)
        img = cv2.warpAffine(img, M_img, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        img = photometric(img, rng) if self.train and self.aug else img.astype(np.float32) / 255.0

        def warp(xy):
            return xy @ M[:, :2].T + M[:, 2]

        A = np.vstack([rec["affine"].astype(np.float64), [0, 0, 1]])  # orig -> stored crop
        A_new = np.vstack([M, [0, 0, 1]]) @ A  # orig -> new crop (S px)
        bcrop = box_crop(rec, A_new, S)  # the record's box in the new crop (before the rotation's camera trick)
        # --box_prompt_p: the box the crop is "drawn from" (the eval pipeline's input box, --mask_prompt box): the
        # record's box at the stored crop's scale without the zoom / shift jitter (roll kept), so the jitter makes it
        # loose / tight / shifted on the person as detector and annotation boxes are; it keeps the box's aspect
        M0 = np.array([[r, 0, 0], [0, r, 0], [0, 0, 1]], np.float64)
        if theta != 0.0:
            M0 = Rc @ M0
        pbox = box_crop(rec, M0 @ A, S)
        if det_crop is not None:  # the drawn box is the detection (the window is built from it)
            pbox = box_crop({"bbox": det_crop}, A_new, S)
        # --neg_prompt_p: the frame's other people (their boxes follow the image, as detections do), with detector-like
        # noise: missed (dropped), jittered, and now and then a group box (a neighbour's box joined with the target's);
        # a crop drawn from a detection (--det_box_p) takes the frame's other detections as they are
        negs = np.full((8, 4), np.nan, np.float32)
        npar = getattr(self, "neg_prompt", None)  # (drop, jitter, group_p)
        src_o = parts["detbox"][1] if det_crop is not None else parts.get("others")
        if npar and src_o is not None and "bbox" in rec:
            j, tb = 0, np.asarray(rec["bbox"], np.float64)
            for ob in np.asarray(src_o, np.float64):
                if j >= len(negs) or not np.isfinite(ob).all():
                    continue
                if self.train and self.aug and det_crop is None:
                    if rng.random() < npar[0]:
                        continue
                    w, h = ob[2] - ob[0], ob[3] - ob[1]
                    sc = 1.0 + rng.uniform(-npar[1], npar[1])
                    cx, cy = (ob[0] + ob[2]) / 2 + rng.uniform(-npar[1], npar[1]) * w, (ob[1] + ob[3]) / 2 + rng.uniform(-npar[1], npar[1]) * h
                    ob = np.array([cx - sc * w / 2, cy - sc * h / 2, cx + sc * w / 2, cy + sc * h / 2])
                    if rng.random() < npar[2]:
                        ob = np.concatenate([np.minimum(ob[:2], tb[:2]), np.maximum(ob[2:], tb[2:])])
                negs[j] = box_crop({"bbox": ob}, A_new, S)
                j += 1
        K = rec["cam_int"].copy()
        if theta != 0.0:  # new "original" image = the image rotated about the principal point: compose its inverse
            pp = K[:2, 2].astype(np.float64)
            Ri_inv = np.eye(3)
            Ri_inv[:2, :2], Ri_inv[:2, 2] = R2.T, pp - R2.T @ pp
            A_new = A_new @ Ri_inv
        centre = np.linalg.solve(A_new, np.array([S / 2, S / 2, 1.0]))[:2]
        scale = rec["bbox_scale"] / s
        t_kp2d = warp(rec["t_kp2d"].astype(np.float64))
        gt_kp2d = np.concatenate([warp(rec["gt_kp2d"][:, :2].astype(np.float64)), rec["gt_kp2d"][:, 2:3]], 1)
        gate2d = getattr(self, "gt2d_gate", ())
        if "image" in gate2d:  # --gt2d_gate image: no 2D label outside the source image (invented / projected labels)
            src_px = rec["gt_kp2d"][:, :2].astype(np.float64) @ np.linalg.inv(A)[:2, :2].T + np.linalg.inv(A)[:2, 2]
            Wi, Hi = (float(x) for x in np.asarray(rec["ori_size"]).ravel()[:2])
            ins_img = (src_px[:, 0] >= 0) & (src_px[:, 0] <= Wi - 1) & (src_px[:, 1] >= 0) & (src_px[:, 1] <= Hi - 1)
            gt_kp2d[:, 2] = np.where(ins_img, gt_kp2d[:, 2], 0.0)
        if getattr(self, "gt2d_in_crop", False) or "crop" in gate2d or parts.get("crop2d_off"):
            # --gt2d_in_crop / --gt2d_gate crop / --gt2d_crop_tars: no 2D label outside the (augmented) crop window
            ins = ((gt_kp2d[:, :2] >= 0) & (gt_kp2d[:, :2] <= S - 1)).all(1)
            gt_kp2d[:, 2] = np.where(ins, gt_kp2d[:, 2], 0.0)
        t_kp3d, t_cam_t = rec["t_kp3d"].astype(np.float64), rec["t_cam_t"].astype(np.float64)
        is_aist = parts.get("src") == "aistpp_train"
        calibrated_aist = self.aist_camera_gt and is_aist
        gt_source = rec["gt_kp3d"]
        if self.aist_geometry_gate and is_aist and not aist_geometry_valid(gt_source):
            # Reject only this unreliable 3D target, retaining the image, 2D labels and teacher targets.
            # Clear coordinates too: NaN * zero weights would still poison alignment / loss operations.
            gt_source = np.zeros_like(gt_source)
        gt_kp3d, gt_w, gt_mode = gt_keypoints_3d(
            gt_source, t_kp3d, parts.get("src") in WORLD_FRAME_SOURCES, camera_frame_cm=calibrated_aist)
        t_kp3d, t_cam_t, gt_kp3d = t_kp3d @ R3.T, R3 @ t_cam_t, gt_kp3d @ R3.T
        flipped = self.train and self.aug and self.flip > 0 and rng.random() < self.flip
        if flipped:
            # The flipped crop is the crop of the mirrored image (X' = W - 1 - X, cv2 pixel centres) seen by the
            # mirrored camera (cx' = W - 1 - cx): 2D x' = S - 1 - x, 3D / translation x' = -x, left <-> right swapped.
            W1, S1 = float(rec["ori_size"][0]) - 1, S - 1.0
            img = img[:, ::-1].copy()
            F_img = np.array([[-1.0, 0, W1], [0, 1, 0], [0, 0, 1]])
            F_crop = np.array([[-1.0, 0, S1], [0, 1, 0], [0, 0, 1]])
            A_new = F_crop @ A_new @ F_img
            bcrop = np.array([(S1 - bcrop[2] * S) / S, bcrop[1], (S1 - bcrop[0] * S) / S, bcrop[3]], np.float32)
            pbox = np.array([(S1 - pbox[2] * S) / S, pbox[1], (S1 - pbox[0] * S) / S, pbox[3]], np.float32)
            negs = np.stack([(S1 - negs[:, 2] * S) / S, negs[:, 1], (S1 - negs[:, 0] * S) / S, negs[:, 3]], 1).astype(np.float32)
            centre = np.array([W1 - centre[0], centre[1]])
            K[0, 2] = W1 - K[0, 2]
            perm = MHR70_PERMUTATION
            t_kp2d = np.stack([S1 - t_kp2d[:, 0], t_kp2d[:, 1]], 1)[perm]
            gt_kp2d = np.concatenate([S1 - gt_kp2d[:, :1], gt_kp2d[:, 1:]], 1)[perm]
            src2d = src2d[perm]
            t_kp3d = (t_kp3d * np.array([-1.0, 1, 1]))[perm]
            t_cam_t = t_cam_t * np.array([-1.0, 1, 1])
            gt_kp3d, gt_w = (gt_kp3d * np.array([-1.0, 1, 1]))[perm], gt_w[perm]
        out = {
            "img": torch.from_numpy(img).permute(2, 0, 1).contiguous(),
            "affine": torch.from_numpy(A_new[:2].astype(np.float32)),
            "box_crop": torch.from_numpy(bcrop),
            "box_prompt": torch.from_numpy(pbox),
            "neg_boxes": torch.from_numpy(negs),
            "bbox_center": torch.from_numpy(centre.astype(np.float32)),
            "bbox_scale": torch.from_numpy(scale.astype(np.float32)),
            "ori_size": torch.from_numpy(rec["ori_size"]),
            "cam_int": torch.from_numpy(K.astype(np.float32)),
            "t_kp2d": torch.from_numpy(t_kp2d.astype(np.float32)),
            "gt_kp2d": torch.from_numpy(gt_kp2d.astype(np.float32)),
            "foot2d_src": torch.from_numpy(src2d[15:21].copy()),  # the foot labels' sidecar index (-1: the pool's own)
            "t_kp3d": torch.from_numpy(t_kp3d.astype(np.float32)),
            "t_cam_t": torch.from_numpy(t_cam_t.astype(np.float32)),
            "flipped": torch.tensor(float(flipped)),
            "rot": torch.from_numpy(R3.astype(np.float32)),  # camera roll applied before the flip
            "rotated": torch.tensor(float(theta != 0.0)),
            "gt_kp3d": torch.from_numpy(gt_kp3d.astype(np.float32)),
            "gt_kp3d_w": torch.from_numpy(gt_w.astype(np.float32)),
            "gt_mode": torch.tensor(gt_mode),
            # AIST's zero MHR parameters are placeholders despite its stored validity flag.
            # Enabling calibrated joints must never activate that bogus mesh target.
            "gt_mhr_valid": torch.tensor(float(np.asarray(rec["gt_mhr_valid"]).item() > 0 and not calibrated_aist)),
        }
        for k in ("t_pose_raw", "t_global_rot", "t_body_pose", "t_shape", "t_scale", "t_hand", "t_model_params",
                  "gt_model_params", "gt_shape"):
            out[k] = torch.from_numpy(rec[k].astype(np.float32))
        if out["gt_model_params"].numel() != 204 or out["gt_shape"].numel() != 45:  # empty labels (SA-1B, ~0.5 %)
            out["gt_model_params"], out["gt_shape"] = torch.zeros(204), torch.zeros(45)
            out["gt_mhr_valid"] = torch.tensor(0.0)
        if parts.get("src") in self.pseudo_3d_sources:  # their "GT" 3D is another model's fit: keep 2D labels only
            out["gt_mhr_valid"] = torch.tensor(0.0)
            out["gt_kp3d_w"] = torch.zeros_like(out["gt_kp3d_w"])
        for k, n in (("gt_smpl_pose", 72), ("gt_smpl_betas", 11), ("gt_smpl_trans", 3)):  # BEDLAM pools only
            out[k] = torch.from_numpy(np.asarray(rec.get(k, np.zeros(n)), np.float32))
        out["gt_smpl_valid"] = torch.tensor(float(np.asarray(rec.get("gt_smpl_valid", 0.0)).item()))
        out["gt_smpl_gender"] = torch.tensor(int(np.asarray(rec.get("gt_smpl_gender", 0)).item()))
        out["foot_off"] = torch.tensor(1.0 if foot_off else 0.0)  # --foot_off_tars (dense foot points; --gt_fix_labels)
        # exact SMPL-topology GT vertices (camera frame, unrolled) where a pool stores them (SMPL-X sources: the official
        # vertex transfer, no 10-beta refit error); they replace the vertices posed from gt_smpl_* in the SMPL losses
        gv = np.asarray(rec["gt_smpl_v"]) if "gt_smpl_v" in rec else None
        ok = gv is not None and gv.shape == (6890, 3) and float(np.asarray(rec.get("gt_smpl_v_valid", 0)).item()) > 0
        out["gt_smpl_v"] = torch.from_numpy(gv.astype(np.float16)) if ok else torch.zeros(6890, 3, dtype=torch.float16)
        out["gt_smpl_v_valid"] = torch.tensor(1.0 if ok else 0.0)
        if parts.get("src") in self.smpl_gt_off:  # SMPL GT with off-convention proportions: images + teacher labels only
            out["gt_smpl_valid"], out["gt_smpl_v_valid"] = torch.tensor(0.0), torch.tensor(0.0)
        # few-subject sources: their GT pose on the neutral mean shape in the root-relative 3D terms (gt_mesh)
        out["gt_shape_off"] = torch.tensor(1.0 if parts.get("src") in self.smpl_shape_off else 0.0)
        # --foot3d_sources: sources whose GT feet (orientation, toes) the foot terms trust (foot_losses)
        out["foot3d_ok"] = torch.tensor(1.0 if parts.get("src") in getattr(self, "foot3d_sources", ()) else 0.0)
        # Independent mocap uses an explicit, audited joint/camera convention; never infer it from the teacher.
        mocap = np.asarray(rec.get("gt_mocap_j14", np.zeros((14, 3))), np.float64)
        mocap_valid = float(np.asarray(rec.get("gt_mocap_valid", 0)).item())
        if mocap.shape != (14, 3) or not np.isfinite(mocap).all() or mocap_valid not in (0, 1):
            raise ValueError("Invalid explicit camera-frame mocap target")
        mocap = mocap @ R3.T
        if flipped:
            mocap = (mocap * [-1, 1, 1])[J14_FLIP]
        out["gt_mocap_j14"] = torch.from_numpy(mocap.astype(np.float32))
        out["gt_mocap_valid"] = torch.tensor(mocap_valid)
        q, Rg = parts.get("g1", (np.zeros(29), np.eye(3)))
        out["g1_q"] = torch.from_numpy(np.asarray(q, np.float32))
        out["g1_R"] = torch.from_numpy((R3 @ np.asarray(Rg, np.float64)).astype(np.float32))  # camera roll
        out["g1_valid"] = torch.tensor(float("g1" in parts))
        if self.mask_labels:  # person mask in the new crop at S / 2: 1 person, 0 background, 255 unknown
            Rm = S // 2
            lab = np.full((Rm, Rm), 255, np.uint8)
            if "mask" in parts:
                fg, unk = (np.unpackbits(a)[: 256 * 256].reshape(256, 256) if a is not None else None
                           for a in parts["mask"])
                src = fg.astype(np.uint8) if unk is None else np.where(unk > 0, 255, fg).astype(np.uint8)
                Mg = np.concatenate([M[:, :2] * (stored / 256.0), M[:, 2:]], 1) * (Rm / S)  # 256 grid -> new grid
                lab = cv2.warpAffine(src, Mg, (Rm, Rm), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=255)
                if flipped:
                    lab = lab[:, ::-1].copy()
            out["gt_mask"] = torch.from_numpy(lab)
            out["gt_mask_valid"] = torch.tensor(float("mask" in parts))
            out["gt_mask_teacher"] = torch.tensor(float("mask" in parts and parts["mask"][1] is None))  # teacher mask
        if self.vis_labels:  # visibility in the new crop: occlusion from the label, out-of-crop from the label's 2D
            lab = np.full(84, 255, np.int64)
            if "vis" in parts:
                # the 84 points (vis2 sidecars append 20 sole-point columns after them: not consumed)
                v0, uv0 = np.asarray(parts["vis"][0], np.int64)[:84], np.asarray(parts["vis"][1], np.float64)[:84]
                px = uv0 * stored
                q = px @ M[:, :2].T + M[:, 2]  # stored crop px -> new crop px (zoom / shift / roll)
                if flipped:
                    q[:, 0] = S - 1 - q[:, 0]
                inside = ((q >= 0) & (q <= S - 1)).all(1)
                lab = np.where(v0 == 255, 255, np.where(inside, np.where(v0 == 2, 255, v0), 2))
                if flipped:  # left <-> right: MHR70 and J14 permutations, as the 2D labels
                    lab = np.concatenate([lab[:70][MHR70_PERMUTATION], lab[70:][J14_FLIP]])
            out["gt_vis"] = torch.from_numpy(lab)
        if getattr(self, "vis_human", False):  # --vis_human: human-annotation visibility:
            # 0 = the annotator's v 2 (visible), 1 = v 1 (labelled, not visible), 2 = a labelled point outside the new crop
            # (or flagged out of frame), 255 = no human flag. Sources: the vis sidecars' points whose reason has bit 8
            # (COCO's own v-flags), then the --kp2d_labels sidecars' vis (WholeBody / Human-Art / other foot labels)
            def in_crop(px):  # stored crop px -> inside the new crop (zoom / shift / roll / flip)
                q = px @ M[:, :2].T + M[:, 2]
                if flipped:
                    q[:, 0] = S - 1 - q[:, 0]
                return ((q >= 0) & (q <= S - 1)).all(1)

            lab_h = np.full(84, 255, np.int64)
            if "vis" in parts and len(parts["vis"]) > 2 and parts["vis"][2] is not None:
                v0 = np.asarray(parts["vis"][0], np.int64)[:84]
                hum = ((np.asarray(parts["vis"][2], np.int64)[:84] & 8) > 0) & (v0 <= 1)
                ins = in_crop(np.asarray(parts["vis"][1], np.float64)[:84] * stored)
                lab_h[hum] = np.where(ins, v0, 2)[hum]
            for k2, _, hv, _ in parts.get("kp2d_side", []):
                if hv is None:
                    continue
                hv = np.asarray(hv, np.int64)[:70]
                ins = in_crop(np.asarray(k2, np.float64)[:70, :2])
                free = lab_h[:70] == 255
                lab_h[:70] = np.where(free & (hv <= 1), np.where(ins, hv, 2), np.where(free & (hv == 2), 2, lab_h[:70]))
            if flipped:
                lab_h = np.concatenate([lab_h[:70][MHR70_PERMUTATION], lab_h[70:][J14_FLIP]])
            out["gt_vis_h"] = torch.from_numpy(lab_h)
        cl, cv, ch = parts.get("contact", (np.zeros(8, np.uint8), np.zeros(8, np.uint8), np.full(8, np.nan)))
        if flipped:  # left <-> right channels (fab/contact_regions.CONTACT_NAMES order)
            cl, cv, ch = cl[[2, 3, 0, 1, 5, 4, 7, 6]], cv[[2, 3, 0, 1, 5, 4, 7, 6]], ch[[2, 3, 0, 1, 5, 4, 7, 6]]
        out["contact"] = torch.from_numpy(np.asarray(cl, np.float32))
        out["contact_valid"] = torch.from_numpy(np.asarray(cv, np.float32))
        # touching a support (height above the local floor only; a rolling board counts) vs static (height and speed)
        ch = np.asarray(ch, np.float64)
        out["contact_any"] = torch.from_numpy(((ch < CONTACT_H_THR) & np.isfinite(ch)).astype(np.float32))
        out["contact_any_valid"] = torch.from_numpy((np.asarray(cv, np.float32) * np.isfinite(ch)).astype(np.float32))
        if "contact_any" in parts:  # the dataset's own any-surface contact replaces the floor-height rule
            ca, va = (np.asarray(x) for x in parts["contact_any"])
            if flipped:
                ca, va = ca[[2, 3, 0, 1, 5, 4, 7, 6]], va[[2, 3, 0, 1, 5, 4, 7, 6]]
            ok = va > 0
            out["contact_any"] = torch.where(torch.from_numpy(ok), torch.from_numpy(ca.astype(np.float32)), out["contact_any"])
            out["contact_any_valid"] = torch.maximum(out["contact_any_valid"], torch.from_numpy(ok.astype(np.float32)))
        return out

    def _enough_body(self, parts):
        """Pseudo-labelled crops must show at least min_in_frame of the teacher's 15 MHR body keypoints (face,
        shoulders, elbows, hips, knees, ankles) inside the original image: hands / arms-only frames (common in
        web videos) only carry the teachers' guess of an off-screen body. Crops with any human label are kept."""
        rec = np.load(io.BytesIO(parts["npz"]))
        if (float(np.asarray(rec.get("gt_smpl_valid", 0)).item()) > 0 or float(np.asarray(rec["gt_mhr_valid"]).item()) > 0
                or (rec["gt_kp2d"][:, 2] > 0).any() or float(np.asarray(rec.get("gt_mocap_valid", 0)).item()) > 0):
            return True
        kp = parts.get("relabel", {}).get("t_kp2d", rec["t_kp2d"])[:15].astype(np.float64)
        A = np.vstack([rec["affine"].astype(np.float64), [0, 0, 1]])  # original image -> stored crop
        full = (np.linalg.inv(A) @ np.c_[kp, np.ones(len(kp))].T).T[:, :2]
        W, H = rec["ori_size"]
        inside = (full[:, 0] >= 0) & (full[:, 0] < W) & (full[:, 1] >= 0) & (full[:, 1] < H)
        return inside.mean() >= self.min_in_frame

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed + 1000 * self.epoch + 97 * self.rank + wid)
        while True:
            tars = list(self.tars)
            random.Random(self.seed + self.epoch).shuffle(tars)
            gid, total = self.rank * nw + wid, self.world * nw
            # shard by tar when there are enough of them, otherwise by sample (no worker may end up empty)
            by_tar = len(tars) >= total
            mine = tars[gid::total] if by_tar else tars
            buf = []
            for path in mine:
                try:
                    for i, parts in enumerate(self._samples(path)):
                        if not by_tar and i % total != gid:
                            continue
                        if self.min_in_frame > 0 and not self._enough_body(parts):
                            continue
                        buf.append(parts)
                        if len(buf) >= self.buffer:
                            yield self._item(buf.pop(rng.randrange(len(buf))), rng)
                except (tarfile.TarError, OSError) as err:
                    print(f"skip {path}: {err}", flush=True)
            while buf:
                yield self._item(buf.pop(rng.randrange(len(buf))), rng)
            if not self.train:
                return
            self.epoch += 1


# ----------------------------------------------------------------------------------------------------- losses
def subset_vertices(rig, model_params, shape):
    """Vertices (m, camera axes) of the rig's vertex subset from MHR model / shape parameters (fp32)."""
    with torch.autocast("cuda", enabled=False):
        joint_params, skel_state = rig.skeleton(model_params.float())
        verts = rig.vertices(shape.float(), joint_params, skel_state) / 100
    verts = verts.clone()
    verts[..., [1, 2]] *= -1  # camera system difference, as in MHRHead.forward
    return verts


def rotation_aligned_l1(pred, gt, w):
    """Per-sample L1 between pelvis-centred keypoints after rotating the labels onto the prediction (Kabsch, no
    gradient through the rotation): supervises articulation and shape when the labels' frame is unknown."""
    p = pred - pred[:, list(PELVIS)].mean(1, keepdim=True)
    g = gt - gt[:, list(PELVIS)].mean(1, keepdim=True)
    with torch.no_grad():
        H = torch.einsum("bk,bki,bkj->bij", w, g, p) + 1e-6 * torch.eye(3, device=p.device)
        U, _, Vh = torch.linalg.svd(H)
        V = Vh.transpose(1, 2)
        d = torch.sign(torch.det(V @ U.transpose(1, 2)))
        D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
        R = V @ D @ U.transpose(1, 2)
    g = torch.einsum("bij,bkj->bki", R, g)
    return ((p - g).abs().mean(-1) * w).sum(1) / w.sum(1).clamp_min(1.0)


def masked_mean(per, mask):
    return torch.where(mask > 0, per, torch.zeros_like(per)).sum() / mask.sum().clamp_min(1.0)


def uv_grad_gate(args, gt_vis, n, device):
    """The dense head's uv_grad_gate for a batch of n (--dense_25d_uvgrad_vis / --dense_25d_uvgrad_scale /
    --dense_25d_uvgrad_mesh; training only: the forward value is unchanged) -> (gate, uvg): gate None = leave the head's
    gate as it is (the lift's 2D detached unless --dense_25d_uvgrad); uvg = the visibility gate (logged) or None."""
    if not (args.dense_25d_uvgrad_vis or args.dense_25d_uvgrad_scale >= 0 or args.dense_25d_uvgrad_mesh):
        return None, None
    s_ = args.dense_25d_uvgrad_scale if args.dense_25d_uvgrad_scale >= 0 else 1.0
    if args.dense_25d_uvgrad_vis:  # visible (label 0) MHR / J14 keypoints x s, the rest detached
        uvg = ((gt_vis[:, :84] == 0).float() * s_)[..., None]  # 70 MHR + 14 J14 (dense_head order)
        return (uvg, 0.0), uvg
    if args.dense_25d_uvgrad_mesh:  # the 70 MHR keypoints detached, the mesh-defined points (J14 + surface) x s
        from fab.dense_head import N_MHR

        g = torch.zeros(n, N_MHR, 1, device=device)
        if getattr(args, "dense_25d_uvgrad_feet", False):  # + the MHR70 foot keypoints (big / small toe, heel; 15:21) x s
            g[:, 15:21] = s_
        return (g, s_), None
    return (None, s_), None


def identity_gate(b, w):
    """(B,) 1 on crops whose teacher labelled another person (--w idgate; on overlapping crops a teacher can
    reconstruct the other person): the teacher's 2D is more than w["idgate"] x the GT-2D extent from the GT 2D (>= 4
    labelled keypoints)."""
    with torch.no_grad():
        t2 = b["t_kp2d"]
        g = torch.zeros(t2.shape[0], device=t2.device)
        if w.get("idgate", 0) > 0:
            vis = b["gt_kp2d"][..., 2] > 0.1
            gk = b["gt_kp2d"][..., :2]
            lo = torch.where(vis[..., None], gk, torch.full_like(gk, 1e9)).amin(1)
            hi = torch.where(vis[..., None], gk, torch.full_like(gk, -1e9)).amax(1)
            ext = (hi - lo).amax(-1).clamp_min(1.0)
            d = torch.where(vis, (t2 - gk).norm(dim=-1), torch.full_like(gk[..., 0], float("nan")))
            med = torch.nanmedian(d, dim=1).values / ext
            g = torch.maximum(g, ((vis.sum(1) >= 4) & (med > w["idgate"])).float())
    return g


H36M_TO_J14 = [3, 2, 1, 4, 5, 6, 16, 15, 14, 11, 12, 13, 8, 10]  # as bench/eval_hmr.py
J14_FLIP = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]  # left <-> right
J24_FLIP = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]  # SMPL joints, L <-> R


class SMPLTarget(torch.nn.Module):
    """SMPL ground truth (BEDLAM) against the predicted MHR mesh in SMPL topology, scored as the evaluator does:
    MHR -> SMPL barycentric mapping, the 14 H36M-regressor joints, pelvis = hip midpoint. Only the SMPL vertices
    the 14 joints read plus a fixed random subset are formed (and only the MHR vertices those map from are skinned).
    Flipped crops get the mirrored GT joints (x negated, left <-> right: exact); their vertex term is masked by the
    caller (the SMPL template is only symmetric to ~13 mm, too loose for mirrored vertex targets)."""

    def __init__(self, mhr, bm_dir, n_verts=1024, seed=0, extra_idx=(), full=False, j24=False):
        from bench.body_models import SMPL
        from bench.mhr_smpl import MHRToSMPL

        super().__init__()
        bm = Path(bm_dir)
        self.smpl = SMPL(bm / "smpl" / "SMPL_NEUTRAL.pkl")
        self.smpl_g = torch.nn.ModuleList([SMPL(bm / "smpl" / f"SMPL_{g}.pkl") for g in ("MALE", "FEMALE")])
        mapping = MHRToSMPL(bm / "mhr")
        j14 = torch.from_numpy(np.load(bm / "smpl" / "J_regressor_h36m.npy")[H36M_TO_J14]).float()
        rand = torch.randperm(j14.shape[1], generator=torch.Generator().manual_seed(seed))[:n_verts]
        extra = torch.as_tensor(list(extra_idx), dtype=torch.long)
        # j24: also the 24 SMPL joints (the standard EMDB metric's joint set; SMPL's own regressor), their vertices kept
        jr24 = self.smpl.J_regressor.float() if j24 else torch.zeros(0, j14.shape[1])
        sub = torch.unique(torch.cat([torch.nonzero(j14.abs().sum(0) > 0).flatten(), rand, extra,
                                      torch.nonzero(jr24.abs().sum(0) > 0).flatten()]))
        self.register_buffer("j24", jr24[:, sub] if j24 else torch.zeros(0, len(sub)), persistent=False)
        # positions of the extra (dense-head surface) vertices inside the subset
        self.register_buffer("extra_pos", torch.searchsorted(sub, extra), persistent=False)
        mhr_ids, tri = torch.unique(mapping.vert_ids[sub].flatten(), return_inverse=True)
        self.register_buffer("sub", sub, persistent=False)
        self.register_buffer("in_rand", torch.isin(sub, rand), persistent=False)
        self.register_buffer("j14", j14[:, sub], persistent=False)
        self.register_buffer("tri", tri.view(-1, 3), persistent=False)
        self.register_buffer("bary", mapping.bary[sub], persistent=False)
        self.rig = SubsetMHR(mhr, mhr_ids)
        self.rig_size = len(mhr_ids)
        self.register_buffer("mirror", torch.tensor([-1.0, 1.0, 1.0]), persistent=False)
        self.register_buffer("yz_flip", torch.tensor([1.0, -1.0, -1.0]), persistent=False)
        self.rig_full = None
        if full:  # whole meshes (part-segmentation labels): every SMPL vertex, the faces and their parts
            from fab.body_parts import face_parts

            ids_f, tri_f = torch.unique(mapping.vert_ids.flatten(), return_inverse=True)
            self.rig_full = SubsetMHR(mhr, ids_f)
            self.register_buffer("tri_full", tri_f.view(-1, 3), persistent=False)
            self.register_buffer("bary_full", mapping.bary.clone(), persistent=False)
            self.register_buffer("faces", self.smpl.faces.long(), persistent=False)
            self.register_buffer("fpart", face_parts(self.smpl.weights, self.smpl.faces), persistent=False)
            from fab.body_parts import SMPL_JOINT_TO_PART

            self.register_buffer("vpart", torch.as_tensor(SMPL_JOINT_TO_PART)[self.smpl.weights.argmax(1)],
                                 persistent=False)

    def pred(self, model_params, shape, full=False):
        rig, tri, bary = (self.rig_full, self.tri_full, self.bary_full) if full else (self.rig, self.tri, self.bary)
        with torch.autocast("cuda", enabled=False):
            joint_params, skel_state = rig.skeleton(model_params.float())
            v = rig.vertices(shape.float(), joint_params, skel_state) / 100 * self.yz_flip
        return (v[:, tri] * bary[..., None]).sum(2)

    @torch.no_grad()
    def gt(self, pose, betas, trans, gender=None, full=False, verts=None, verts_valid=None):
        """gender (B,): 0 neutral (BEDLAM), 1 male, 2 female (3DPW). verts (B, 6890, 3) / verts_valid (B,): stored exact
        GT vertices that replace the posed ones where valid."""
        with torch.autocast("cuda", enabled=False):
            v, _ = self.smpl(pose[:, :3], pose[:, 3:], betas, trans)
            if gender is not None and (gender > 0).any():
                for k, body in enumerate(self.smpl_g, start=1):
                    vk, _ = body(pose[:, :3], pose[:, 3:], betas, trans)
                    v = torch.where((gender == k)[:, None, None], vk, v)
            if verts is not None and verts_valid is not None and (verts_valid > 0).any():
                v = torch.where(verts_valid[:, None, None] > 0, verts.float(), v)
        return v if full else v[:, self.sub]

    def errors24(self, pred, gt, flipped):
        """Per-sample L1 (m) of the 24 SMPL joints after pelvis alignment; flipped crops get the mirrored GT joints."""
        pj, gj = torch.einsum("js,bsc->bjc", self.j24, pred), torch.einsum("js,bsc->bjc", self.j24, gt)
        gj = torch.where(flipped[:, None, None] > 0, (gj * self.mirror)[:, J24_FLIP], gj)
        e = (pj - pj[:, :1] - (gj - gj[:, :1])).abs().mean(-1)
        w = getattr(self, "j24_w", None)  # per-joint weights (--w j24_feet=k: knees / ankles / feet x k)
        return e.mean(1) if w is None else (e * w).sum(1) / w.sum()

    def errors(self, pred, gt, flipped):
        """Per-sample L1 (m) of the 14 joints and of the random vertices, both after hip-midpoint alignment
        (the vertex error of flipped crops is meaningless: mask it)."""
        pj, gj = torch.einsum("js,bsc->bjc", self.j14, pred), torch.einsum("js,bsc->bjc", self.j14, gt)
        gj = torch.where(flipped[:, None, None] > 0, (gj * self.mirror)[:, J14_FLIP], gj)
        pp, gp = pj[:, [2, 3]].mean(1, keepdim=True), gj[:, [2, 3]].mean(1, keepdim=True)
        e_j = (pj - pp - (gj - gp)).abs().mean(dim=(1, 2))
        e_v = (pred[:, self.in_rand] - pp - (gt[:, self.in_rand] - gp)).abs().mean(dim=(1, 2))
        return e_j, e_v

    def errors_v_weighted(self, pred, gt, flipped, vw):
        """errors()'s vertex term with per-vertex weights vw (B, n_rand) (--foot_pitch_gate)."""
        pj, gj = torch.einsum("js,bsc->bjc", self.j14, pred), torch.einsum("js,bsc->bjc", self.j14, gt)
        gj = torch.where(flipped[:, None, None] > 0, (gj * self.mirror)[:, J14_FLIP], gj)
        pp, gp = pj[:, [2, 3]].mean(1, keepdim=True), gj[:, [2, 3]].mean(1, keepdim=True)
        e = (pred[:, self.in_rand] - pp - (gt[:, self.in_rand] - gp)).abs().mean(-1)
        return (e * vw).sum(1) / vw.sum(1).clamp_min(1e-6)


def to_crop(xy, affine):
    return torch.einsum("bij,bkj->bki", affine[:, :, :2], xy) + affine[:, None, :, 2]


def hard_weights(err, mask=None, gamma=0.0):
    """Per-sample weights emphasising hard samples: (err / mean err)^gamma, detached, clipped to [0.25, 4] and
    renormalised to mean 1 over the valid samples (gamma 0: uniform)."""
    m = torch.ones_like(err) if mask is None else mask
    if gamma <= 0:
        return m
    e = err.detach()
    mean = (e * m).sum() / m.sum().clamp_min(1.0)
    w = ((e / mean.clamp_min(1e-6)) ** gamma).clamp(0.25, 4.0) * m
    return w * m.sum().clamp_min(1.0) / w.sum().clamp_min(1e-6)


def mocap_joint_loss(pred, target, valid):
    """Metric camera-frame J14 supervision, hip aligned without rotation or scale fitting."""
    pred = pred - pred[:, [2, 3]].mean(1, keepdim=True)
    target = target - target[:, [2, 3]].mean(1, keepdim=True)
    return masked_mean((pred - target).abs().mean((1, 2)), valid)


def g1_losses(kin, out, b, keep):
    """G1 head: joint angles (L1) and pelvis-relative link positions of the mapped links (L1, root rotation
    included); flipped crops skipped (the targets are not mirrored)."""
    from fab.retarget.g1 import SMPL_TO_G1, rot6d_to_matrix

    m = b["g1_valid"] * keep
    names = list(SMPL_TO_G1.values())
    zeros = torch.zeros_like(b["g1_q"][:, :3])
    with torch.no_grad():
        pos_t, _ = kin(b["g1_R"], zeros, b["g1_q"])
        P_t = torch.stack([pos_t[n] for n in names], 1)
    pos_p, _ = kin(rot6d_to_matrix(out["g1_rot6d"].float()), zeros, out["g1_q"].float())
    P_p = torch.stack([pos_p[n] for n in names], 1)
    e_fk = ((P_p - P_p[:, :1]) - (P_t - P_t[:, :1])).abs().mean(dim=(1, 2))
    e_q = (out["g1_q"].float() - b["g1_q"]).abs().mean(-1)
    return {"g1_q": masked_mean(e_q, m), "g1_fk": masked_mean(e_fk, m)}


def closed_form_shift(X, uv_full, K, z0, w):
    """Translation (tx, ty, z0) placing root-relative points X (B, N, 3) onto pixels uv_full (B, N, 2) at the given
    root depth z0 (B,): per-point x / y solutions averaged with weights w (B, N) (no gradient)."""
    with torch.no_grad():
        fx, fy, cx, cy = K[:, 0, 0, None], K[:, 1, 1, None], K[:, 0, 2, None], K[:, 1, 2, None]
        z = (z0[:, None] + X[..., 2]).clamp_min(0.1)
        tx = (uv_full[..., 0] - cx) * z / fx - X[..., 0]
        ty = (uv_full[..., 1] - cy) * z / fy - X[..., 1]
        ws = w.sum(1).clamp_min(1e-6)
        return torch.stack([(tx * w).sum(1) / ws, (ty * w).sum(1) / ws, z0], -1)


def project_crop(P, K, affine, size):
    """Camera-frame points (B, N, 3) -> crop units in [0, 1]."""
    uv = torch.einsum("bij,bnj->bni", K, P)
    return to_crop(uv[..., :2] / uv[..., 2:3].clamp_min(1e-3), affine) / size


# J14 (SMPL evaluation joints) <-> MHR70 keypoints with a human 2D label convention: ankles, knees, hips, wrists,
# elbows, shoulders (the joint definitions differ by a few cm, alike for both teachers). NB the H36M regressor's leg
# joints 0-2 ("right ankle / knee / hip" in the SPIN order) lie on SMPL's LEFT side (+x on the template) and 3-5 on the
# right; the arms 6-11 are as labelled. Pairs by side:
J14_MHR = [(0, 13), (1, 11), (2, 9), (3, 10), (4, 12), (5, 14), (6, 41), (7, 8), (8, 6), (9, 5), (10, 7), (11, 62)]


@torch.no_grad()
def surface_visibility(b, smpl_target, size, roll, pts, hw=128, tol=0.03):
    """Which target surface points (B, S, 3) (camera frame, already mirrored for flipped crops) the crop shows (FAH's
    visibility labels): the full target mesh (GT, else the MHR teacher) z-buffered at
    hw x hw; a point is visible when it projects inside the crop within tol (m) of the nearest depth there."""
    from fab.body_parts import barycentric_grid, render_depth

    t_f = roll(smpl_target.pred(b["t_model_params"], b["t_shape"], full=True)) + b["t_cam_t"][:, None]
    g_f = roll(smpl_target.gt(b["gt_smpl_pose"], b["gt_smpl_betas"], b["gt_smpl_trans"], b["gt_smpl_gender"], full=True,
                              verts=b.get("gt_smpl_v"), verts_valid=b.get("gt_smpl_v_valid")))
    gv = b["gt_smpl_valid"][:, None, None] > 0
    tgt = torch.where(gv, g_f, t_f)
    tgt = torch.where(b["flipped"][:, None, None] > 0, tgt * smpl_target.mirror, tgt)
    depth = render_depth(project_crop(tgt, b["cam_int"], b["affine"], size), tgt[..., 2], smpl_target.faces, (hw, hw),
                         barycentric_grid(3).to(tgt.device))
    up = project_crop(pts, b["cam_int"], b["affine"], size)
    x, y = (up[..., 0] * hw).floor().long(), (up[..., 1] * hw).floor().long()
    inside = (x >= 0) & (x < hw) & (y >= 0) & (y < hw) & (pts[..., 2] > 0.1)
    d = depth.flatten(1).gather(1, y.clamp(0, hw - 1) * hw + x.clamp(0, hw - 1))
    return (inside & (pts[..., 2] <= d + tol)).float()


def dense_losses(out, b, smpl_target, size, keep, roll, w_pseudo=0.5, w=None):
    """Dense head. Image-defined MHR keypoints: 2D labels strongly, the teacher's 2D weakly, teacher 3D. Mesh-defined
    points (14 evaluation joints, SMPL surface): the SMPL-space target -- GT where present, else the MHR teacher
    (weight w_pseudo) -- 3D hip-aligned and 2D through the crop camera;
    flipped crops use the mirrored target (J14 left <-> right, the template's vertex mirror map). Weak agreement:
    the 3D points placed at a closed-form translation (target root depth) reproject onto the 2D labels (MHR
    keypoints) and onto the head's own (detached) 2D (mesh points)."""
    from fab.dense_head import N_J14, N_MHR

    uv, xyz = out["dense_uv"].float(), out["dense_xyz"].float()
    if "dense_anchor" in out:  # residual head: offsets on the detached parametric points
        xyz = xyz + out["dense_anchor"]
    t = {}
    conf = b["gt_kp2d"][..., 2:3].clamp(0, 1)
    t["dense_mhr2d_gt"] = ((uv[:, :N_MHR] - b["gt_kp2d"][..., :2] / size).abs() * conf).sum() / conf.sum().clamp_min(1.0) / 2
    if getattr(smpl_target, "heat2d_in_crop", False):  # --gt2d_gate heat: the heatmap 2D lives in the crop ([0, 1]);
        # labels outside it stay for the amodal projected / reprojected terms (kp2d_gt, dense_rep) only
        cin = conf * ((b["gt_kp2d"][..., :2] >= 0) & (b["gt_kp2d"][..., :2] <= size - 1)).all(-1, keepdim=True).float()
        t["dense_mhr2d_gt"] = ((uv[:, :N_MHR] - b["gt_kp2d"][..., :2] / size).abs() * cin).sum() / cin.sum().clamp_min(1.0) / 2
    t["dense_mhr2d_t"] = (uv[:, :N_MHR] - b["t_kp2d"] / size).abs().mean()
    ik = b.get("id_keep")  # identity gate (--w idgate): 0 where the teacher labelled another person
    if ik is not None:
        t["dense_mhr2d_t"] = ((uv[:, :N_MHR] - b["t_kp2d"] / size).abs().mean(dim=(1, 2)) * ik).mean()
    vg, gvis = getattr(smpl_target, "vis_gate_2d", None), None
    if vg is not None and "gt_vis" in b:  # --vis_gate_2d: the teacher's 2D of keypoints labelled occluded / out of the crop
        lv = b["gt_vis"].long()
        gvis = torch.ones(lv.shape, device=uv.device)
        gvis[lv == 1], gvis[lv == 2] = vg[0], vg[1]
        gm = gvis[:, :N_MHR] * (ik[:, None] if ik is not None else 1.0)
        t["dense_mhr2d_t"] = ((uv[:, :N_MHR] - b["t_kp2d"] / size).abs().mean(-1) * gm).sum() / gm.sum().clamp_min(1.0)
        t["vis_gate_frac"] = (gvis < 1).float().mean().detach()
    pk = xyz[:, :N_MHR]
    e_mhr = ((pk - pk[:, list(PELVIS)].mean(1, keepdim=True))
             - (b["t_kp3d"] - b["t_kp3d"][:, list(PELVIS)].mean(1, keepdim=True))).abs().mean(-1)
    t["dense_mhr3d"] = e_mhr.mean()
    if ik is not None:
        t["dense_mhr3d"] = (e_mhr * ik[:, None]).mean()
    with torch.no_grad():
        t_cam = roll(smpl_target.pred(b["t_model_params"], b["t_shape"])) + b["t_cam_t"][:, None]
        g_cam = roll(smpl_target.gt(b["gt_smpl_pose"], b["gt_smpl_betas"], b["gt_smpl_trans"], b["gt_smpl_gender"], verts=b.get("gt_smpl_v"), verts_valid=b.get("gt_smpl_v_valid")))
        gv = b["gt_smpl_valid"][:, None, None] > 0
        tgt = torch.where(gv, g_cam, t_cam)
        tj = torch.einsum("js,bsc->bjc", smpl_target.j14, tgt)
        tv = tgt[:, smpl_target.extra_pos]
        mirror = torch.tensor([-1.0, 1.0, 1.0], device=tgt.device)
        fl = b["flipped"][:, None, None] > 0  # the flipped crop sees the mirrored body through the mirrored camera
        tj = torch.where(fl, (tj * mirror)[:, J14_FLIP], tj)
        tv = torch.where(fl, (tv * mirror)[:, smpl_target.extra_mirror], tv)
        pts = torch.cat([tj, tv], 1)  # camera frame (m)
        uvt = project_crop(pts, b["cam_int"], b["affine"], size)
        ok2d = (pts[..., 2] > 0.1).float()
        sw = torch.where(b["gt_smpl_valid"] > 0, torch.ones_like(keep), torch.full_like(keep, w_pseudo))
        if ik is not None:  # pseudo targets (teacher) of an identity-gated crop: dropped
            sw = torch.where(b["gt_smpl_valid"] > 0, sw, sw * ik)
    pj, pv = xyz[:, N_MHR : N_MHR + N_J14], xyz[:, N_MHR + N_J14 :]
    hp, ht = pj[:, [2, 3]].mean(1, keepdim=True), tj[:, [2, 3]].mean(1, keepdim=True)
    if b.get("gt_shape_off") is not None and (b["gt_shape_off"] > 0).any():  # 3D targets: neutral shape there
        with torch.no_grad():
            g3 = roll(gt_mesh(smpl_target, b, neutral_off=True))
            tgt3 = torch.where(gv & (b["gt_shape_off"][:, None, None] > 0), g3, tgt)
            tj = torch.where(fl, (torch.einsum("js,bsc->bjc", smpl_target.j14, tgt3) * mirror)[:, J14_FLIP],
                             torch.einsum("js,bsc->bjc", smpl_target.j14, tgt3))
            tv = torch.where(fl, (tgt3[:, smpl_target.extra_pos] * mirror)[:, smpl_target.extra_mirror],
                             tgt3[:, smpl_target.extra_pos])
            ht = tj[:, [2, 3]].mean(1, keepdim=True)
    t["dense_j3d"] = (((pj - hp) - (tj - ht)).abs().mean(dim=(1, 2)) * sw).sum() / sw.sum().clamp_min(1e-6)
    t["dense_v3d"] = (((pv - hp) - (tv - ht)).abs().mean(dim=(1, 2)) * sw).sum() / sw.sum().clamp_min(1e-6)
    fpos = getattr(smpl_target, "foot_pos", None)
    if fpos is not None and "foot3d_ok" in b:  # --dense_foot_untrusted: foot points' 3D from trusted GT feet only
        fu = smpl_target.foot_untrusted
        pw = torch.ones_like(pv[..., 0])
        pw[:, fpos] = fu + (1 - fu) * (b["foot3d_ok"] * (b["gt_smpl_valid"] > 0).float())[:, None]
        if "foot_off" in b:  # --foot_off_tars: no foot points at all on those crops
            pw[:, fpos] = pw[:, fpos] * (1 - b["foot_off"])[:, None]
            t["foot_off_frac"] = b["foot_off"].mean().detach()
        ev = (((pv - hp) - (tv - ht)).abs().mean(-1) * pw).sum(1) / pw.sum(1).clamp_min(1e-6)
        t["dense_v3d"] = (ev * sw).sum() / sw.sum().clamp_min(1e-6)
    w = w or {}
    g2 = 1.0
    if w.get("dense_vis_gate", 0) > 0:  # FAH vis_gate: hidden surface points' 2D weight (their 2D follows the 3D)
        vis = surface_visibility(b, smpl_target, size, roll, tv)
        g2 = torch.cat([torch.ones_like(vis[:, :N_J14]), w["dense_vis_gate"] + (1 - w["dense_vis_gate"]) * vis], 1)
        t["dense_vis_frac"] = vis.mean().detach()
    if gvis is not None:  # --vis_gate_2d: the J14 2D of joints labelled occluded / out of the crop
        g2 = g2 * torch.cat([gvis[:, N_MHR:], torch.ones_like(uv[:, N_MHR + N_J14 :, 0])], 1)
    if fpos is not None and "foot_off" in b and (b["foot_off"] > 0).any():  # --foot_off_tars: the foot points' 2D too
        gf = torch.ones_like(uv[:, N_MHR:, 0])
        gf[:, N_J14 + fpos] = (1 - b["foot_off"])[:, None]
        g2 = g2 * gf
    e2 = (uv[:, N_MHR:] - uvt).abs().mean(-1) * ok2d * sw[:, None] * g2
    t["dense_2d"] = e2.sum() / (ok2d * sw[:, None] * g2).sum().clamp_min(1.0)
    # weak agreement 1: MHR keypoints' 3D at a closed-form translation onto the 2D labels (GT, else the teacher's)
    K, A = b["cam_int"], b["affine"]
    with torch.no_grad():
        lab = torch.where(conf > 0, b["gt_kp2d"][..., :2], b["t_kp2d"])  # crop px
        wl = torch.where(conf[..., 0] > 0, conf[..., 0], torch.full_like(conf[..., 0], 0.3))
        Ainv = torch.linalg.inv(torch.cat([A, torch.tensor([0.0, 0, 1], device=A.device).expand(len(A), 1, 3)], 1))
        lab_full = torch.einsum("bij,bnj->bni", Ainv[:, :2, :2], lab) + Ainv[:, None, :2, 2]
        t3 = b["t_kp3d"] + b["t_cam_t"][:, None]
        z0 = t3[:, list(PELVIS), 2].mean(1)
    Xk = pk - pk[:, list(PELVIS)].mean(1, keepdim=True)
    Tk = closed_form_shift(Xk, lab_full, K, z0, wl)
    rk = project_crop(Xk + Tk[:, None], K, A, size)
    t["dense_rep"] = ((rk - lab / size).abs().mean(-1) * wl).sum() / wl.sum().clamp_min(1.0)
    # weak agreement 2: mesh points' 3D onto the head's own 2D (detached) at the target hip depth
    with torch.no_grad():
        own = uv[:, N_MHR:].detach() * size
        own_full = torch.einsum("bij,bnj->bni", Ainv[:, :2, :2], own) + Ainv[:, None, :2, 2]
        z1 = tj[:, [2, 3], 2].mean(1)
    Xm = torch.cat([pj, pv], 1) - hp
    Tm = closed_form_shift(Xm, own_full, K, z1, ok2d)
    rm = project_crop(Xm + Tm[:, None], K, A, size)
    t["dense_rep_surf"] = ((rm - uv[:, N_MHR:].detach()).abs().mean(-1) * ok2d).sum() / ok2d.sum().clamp_min(1.0)
    fit_net = getattr(smpl_target, "fit_net", None)
    if fit_net is not None:  # fit in the loop (--dense_fit_loss): the one-pass SMPL fit of the surface vs the target
        with torch.autocast("cuda", enabled=False):
            R, betas = fit_net(pv.float(), smpl_target.smpl, smpl_target.fit_idx, detach_rot=True)
            z = torch.zeros(len(pv), 3, device=pv.device)
            vf, _ = smpl_target.smpl(z, z.repeat(1, 23), betas, None, rotmats=R)
        vf = vf[:, smpl_target.sub]
        tf = tgt3 if b.get("gt_shape_off") is not None and (b["gt_shape_off"] > 0).any() else tgt
        e_j, e_v = smpl_target.errors(vf, tf, b["flipped"])  # flipped crops: mirrored target joints, no vertex term
        t["dense_fit_j14"] = (e_j * sw).sum() / sw.sum()
        t["dense_fit_v"] = (e_v * sw * keep).sum() / (sw * keep).sum().clamp_min(1e-6)
        if len(smpl_target.j24):
            t["dense_fit_j24"] = (smpl_target.errors24(vf, tf, b["flipped"]) * sw).sum() / sw.sum()
    if w.get("dense_tprior", 0) > 0 or w.get("dense_tprior2d", 0) > 0:
        # FAH's surface prior (wsurf_prior): every mesh-defined point weakly follows the MHR teacher's own (image-aligned)
        # mesh on the crops without GT -- 3D hip-aligned and its projection in 2D -- on top of the SMPL-space target
        with torch.no_grad():
            pj_t = torch.einsum("js,bsc->bjc", smpl_target.j14, t_cam)
            pj_t = torch.where(fl, (pj_t * mirror)[:, J14_FLIP], pj_t)
            pv_t = torch.where(fl, (t_cam[:, smpl_target.extra_pos] * mirror)[:, smpl_target.extra_mirror],
                               t_cam[:, smpl_target.extra_pos])
            p_t = torch.cat([pj_t, pv_t], 1)
            uv_t = project_crop(p_t, b["cam_int"], b["affine"], size)
            ok_t = (p_t[..., 2] > 0.1).float()
            wp = 1.0 - b["gt_smpl_valid"]
        e3 = ((torch.cat([pj, pv], 1) - hp) - (p_t - pj_t[:, [2, 3]].mean(1, keepdim=True))).abs().mean(dim=(1, 2))
        t["dense_tprior"] = (e3 * wp).sum() / wp.sum().clamp_min(1.0)
        e2t = ((uv[:, N_MHR:] - uv_t).abs().mean(-1) * ok_t).sum(1) / ok_t.sum(1).clamp_min(1.0)
        t["dense_tprior2d"] = (e2t * wp).sum() / wp.sum().clamp_min(1.0)
    if "dense_xyz3d" in out:  # 2.5D head (--dense_25d): its regressed 3D P before the lift (depths, translation solve,
        # edge points) on the same hip-aligned targets as dense_mhr3d / dense_j3d / dense_v3d (w dense_p{mhr,j,v}3d)
        q = out["dense_xyz3d"].float()
        qk, qj, qv = q[:, :N_MHR], q[:, N_MHR : N_MHR + N_J14], q[:, N_MHR + N_J14 :]
        e_q = ((qk - qk[:, list(PELVIS)].mean(1, keepdim=True))
               - (b["t_kp3d"] - b["t_kp3d"][:, list(PELVIS)].mean(1, keepdim=True))).abs().mean(-1)
        t["dense_pmhr3d"] = (e_q * ik[:, None]).mean() if ik is not None else e_q.mean()
        hq = qj[:, [2, 3]].mean(1, keepdim=True)
        t["dense_pj3d"] = (((qj - hq) - (tj - ht)).abs().mean(dim=(1, 2)) * sw).sum() / sw.sum().clamp_min(1e-6)
        t["dense_pv3d"] = (((qv - hq) - (tv - ht)).abs().mean(dim=(1, 2)) * sw).sum() / sw.sum().clamp_min(1e-6)
        t["dense_lift_mm"] = ((xyz - q).norm(dim=-1).mean() * 1000).detach()  # how far the lift moves the points
    if "dense_abs_z" in out and w.get("dense_tz", 0) > 0:  # metric root depth of the 2.5D lift (depth
        # split): the lifted J14 hips' absolute depth vs the GT camera-frame hips, log ratio, GT-camera crops only
        zh = out["dense_abs_z"].float()[:, N_MHR + 2 : N_MHR + 4].mean(1).clamp_min(0.1)
        zg = pts[:, [2, 3], 2].mean(1).clamp_min(0.1)
        gt = (b["gt_smpl_valid"] > 0).float()
        e = (zh.log() - zg.log()).abs()
        t["dense_tz"] = (e * gt).sum() / gt.sum().clamp_min(1.0)
        t["dense_tz_ratio"] = (((zh / zg) * gt).sum() / gt.sum().clamp_min(1.0)).detach()
    if w.get("dense_bone", 0) > 0:  # compositional loss: the 14 joints' bone vectors (limb chains, shoulders / hips to
        # the neck, head) vs the target's, so a leg's error is not only its hip-aligned ankle
        c, p_ = [i for i, _ in J14_BONES], [j for _, j in J14_BONES]
        eb = ((pj[:, c] - pj[:, p_]) - (tj[:, c] - tj[:, p_])).abs().mean(dim=(1, 2))
        t["dense_bone"] = (eb * sw).sum() / sw.sum().clamp_min(1e-6)
    return t


# (child, parent) in the J14 order: legs, hip width, arms, shoulders / hips -> neck, head
J14_BONES = [(0, 1), (1, 2), (5, 4), (4, 3), (2, 3), (6, 7), (7, 8), (11, 10), (10, 9), (8, 12), (9, 12), (2, 12), (3, 12),
             (13, 12)]


# feet: heel, big toe, small toe (left, then right) as GT SMPL vertices and MHR70 keypoints
FOOT_SMPL_V = [3387, 3216, 3226, 6787, 6617, 6624]
FOOT_MHR = [17, 15, 16, 20, 18, 19]
FOOT_SWAP = [3, 4, 5, 0, 1, 2]
LEG_MHR = list(range(11, 21))  # knees, ankles, toes, heels
BODY_MHR = list(range(21)) + [41, 62]  # body keypoints and wrists
FOOT_TERMS = ("foot_kp", "foot_dir", "foot_2d", "foot_lab", "foot_reproj")
HEEL_MHR = [17, 20]  # MHR70 left / right heel


def heel_weights(tmpl, surface_idx, verts, k=8):
    """--heel2d_vertex: each SMPL vertex as the min-norm affine combination (weights sum to 1) of its k nearest dense
    surface points in the template, reproducing it exactly -> (dense channel ids (V, k), weights (V, k))."""
    from fab.dense_head import N_J14, N_MHR

    tmpl, si = np.asarray(tmpl, np.float64), np.asarray(surface_idx)
    ch, wt = [], []
    for v in verts:
        o = np.argsort(np.linalg.norm(tmpl[si] - tmpl[v], axis=1))[:k]
        m = np.vstack([tmpl[si[o]].T, np.ones((1, k))])
        wt.append(np.linalg.pinv(m) @ np.concatenate([tmpl[v], [1.0]]))
        ch.append(N_MHR + N_J14 + o)
    return np.stack(ch), np.stack(wt)


def heel_relabel(b):
    """--heel2d_vertex: the --kp2d_labels sidecars' heel 2D (human / RTMW, MHR70 17 / 20; feet: 0.04-0.11 foot lengths forward
    of SMPL's heel vertex) leave the MHR70 heel channels (conf 0 in every MHR70 2D term: kp2d_gt, dense_mhr2d_gt, foot_lab,
    dense_rep) for b["heel2d_lab"] (B, 2, 3), which foot_losses' heel_lab puts on the matched SMPL heel vertices."""
    side = (b["foot2d_src"][:, [h - 15 for h in HEEL_MHR]] >= 0).float()
    k2 = b["gt_kp2d"].clone()
    lab = k2[:, HEEL_MHR].clone()
    lab[..., 2] = lab[..., 2] * side
    k2[:, HEEL_MHR, 2] = k2[:, HEEL_MHR, 2] * (1 - side)
    return dict(b, gt_kp2d=k2, heel2d_lab=lab)


def foot_losses(out, b, smpl_target, size, roll, w):
    """Foot terms of the dense head's MHR70 foot keypoints (--w foot_*). GT feet only on the
    trusted sources (foot3d_ok: --foot3d_sources, with SMPL GT):
      foot_kp      hip-aligned L1 of the 6 foot points about the MHR hips 9 / 10; GT: the GT mesh's MHR keypoints
                   (--foot_reg regressor), else its foot vertices about the J14 hips; 3D terms on the neutral shape where
                   gt_shape_off, as gt_mesh
      foot_dir     1 - cos of heel -> big toe plus 1 - cos of the sole normal (heel, big toe, small toe) vs the GT foot
                   vertices: free of the MHR / SMPL point offsets
      foot_2d      the GT foot points (as foot_kp) through the GT camera into the crop vs the dense 2D of those channels
    every crop:
      foot_lab     the labelled foot channels of the human 2D (gt_kp2d conf > 0) once more at dense_mhr2d_gt's weight
                   and normalisation, i.e. foot_lab=2 -> x3
      foot_reproj  the dense 3D knees / ankles / toes / heels at the MHR head's hip depth, shifted in closed form onto
                   the head's own (detached) 2D of the body keypoints, vs that 2D (self-consistency: depth-tilted shins);
                   foot_reproj_vis > 0: only on the points the --vis_labels sidecars call visible"""
    uv, xyz = out["dense_uv"].float(), out["dense_xyz"].float()
    if "dense_anchor" in out:
        xyz = xyz + out["dense_anchor"]
    t = {}
    size_ = float(size)
    pk = xyz[:, :70]
    hp = pk[:, list(PELVIS)].mean(1, keepdim=True)
    if w.get("foot_kp", 0) > 0 or w.get("foot_dir", 0) > 0 or w.get("foot_2d", 0) > 0:
        rel = b.get("foot3d_ok", torch.zeros_like(b["flipped"])) * (b["gt_smpl_valid"] > 0).float()
        with torch.no_grad():
            fl = b["flipped"][:, None, None] > 0
            mirror = smpl_target.mirror

            def side(f):  # the flipped crop shows the mirrored body: left <-> right
                return torch.where(fl, (f * mirror)[:, FOOT_SWAP], f)

            g2 = roll(gt_mesh(smpl_target, b, full=True))
            g3 = roll(gt_mesh(smpl_target, b, neutral_off=True, full=True))
            fd = side(g3[:, FOOT_SMPL_V])  # directions: the GT mesh's own heel / toe vertices
            reg = getattr(smpl_target, "foot_reg", None)
            if reg is not None:  # positions in the MHR keypoint convention: GT mesh -> MHR body keypoints (--foot_reg)
                k3, k2 = torch.einsum("kv,bvc->bkc", reg, g3), torch.einsum("kv,bvc->bkc", reg, g2)
                gh = k3[:, list(PELVIS)].mean(1, keepdim=True)
                f3, p2 = side(k3[:, FOOT_MHR]) - torch.where(fl, gh * mirror, gh), side(k2[:, FOOT_MHR])
            else:  # SMPL vertices about the J14 hip midpoint (MHR / SMPL hip and toe definitions differ by cm)
                gh = torch.einsum("js,bsc->bjc", smpl_target.j14, g3[:, smpl_target.sub])[:, [2, 3]].mean(1, keepdim=True)
                f3, p2 = fd - torch.where(fl, gh * mirror, gh), side(g2[:, FOOT_SMPL_V])
            f2 = project_crop(p2, b["cam_int"], b["affine"], size_)
            ok2 = (p2[..., 2] > 0.1).float() * rel[:, None]
        pf = pk[:, FOOT_MHR] - hp
        n = rel.sum().clamp_min(1.0)
        t["foot_kp"] = ((pf - f3).abs().mean(dim=(1, 2)) * rel).sum() / n

        def dirs(p):
            hl, hr = p[:, 0], p[:, 3]
            d = torch.stack([p[:, 1] - hl, p[:, 4] - hr], 1)
            nrm = torch.stack([torch.cross(p[:, 1] - hl, p[:, 2] - hl, dim=-1),
                               torch.cross(p[:, 4] - hr, p[:, 5] - hr, dim=-1)], 1)
            return F.normalize(d, dim=-1), F.normalize(nrm, dim=-1)

        (dp, sp), (dg, sg) = dirs(pf), dirs(fd)
        e_dir = (1 - (dp * dg).sum(-1)) + (1 - (sp * sg).sum(-1))
        t["foot_dir"] = (e_dir.mean(1) * rel).sum() / n
        t["foot_2d"] = ((uv[:, FOOT_MHR] - f2).abs().mean(-1) * ok2).sum() / ok2.sum().clamp_min(1.0)
        t["foot_rel_frac"] = rel.mean().detach()
        t["foot_reg_on"] = torch.tensor(float(reg is not None), device=uv.device)
    if w.get("foot_lab", 0) > 0:
        conf = b["gt_kp2d"][..., 2:3].clamp(0, 1)
        if getattr(smpl_target, "heat2d_in_crop", False):  # --gt2d_gate heat: labels outside the crop out of the heatmap terms
            conf = conf * ((b["gt_kp2d"][..., :2] >= 0) & (b["gt_kp2d"][..., :2] <= size_ - 1)).all(-1, keepdim=True).float()
        e = (uv[:, FOOT_MHR] - b["gt_kp2d"][:, FOOT_MHR, :2] / size_).abs() * conf[:, FOOT_MHR]
        t["foot_lab"] = w.get("dense_mhr2d_gt", 0.0) * e.sum() / conf.sum().clamp_min(1.0) / 2
    if "heel2d_lab" in b and w.get("heel_lab", 0) > 0:  # --heel2d_vertex: the human / RTMW heel 2D on the matched SMPL heel
        lab = b["heel2d_lab"]
        conf = lab[..., 2:3].clamp(0, 1)
        if getattr(smpl_target, "heat2d_in_crop", False):
            conf = conf * ((lab[..., :2] >= 0) & (lab[..., :2] <= size_ - 1)).all(-1, keepdim=True).float()
        huv = (uv[:, smpl_target.heel_ch] * smpl_target.heel_w[None, ..., None]).sum(2)  # (B, 2 heels, 2)
        e = (huv - lab[..., :2] / size_).abs() * conf
        t["heel_lab"] = w.get("dense_mhr2d_gt", 0.0) * e.sum() / conf.sum().clamp_min(1.0) / 2
        t["heel_lab_frac"] = (conf > 0).float().mean().detach()
    if w.get("foot_reproj", 0) > 0:
        K, A = b["cam_int"], b["affine"]
        with torch.no_grad():
            Ainv = torch.linalg.inv(torch.cat([A, torch.tensor([0.0, 0, 1], device=A.device).expand(len(A), 1, 3)], 1))
            own = uv[:, BODY_MHR].detach() * size_
            own_full = torch.einsum("bij,bnj->bni", Ainv[:, :2, :2], own) + Ainv[:, None, :2, 2]
            z0 = (out["pred_keypoints_3d"].float()[:, list(PELVIS)].mean(1) + out["pred_cam_t"].float())[:, 2]
        X = pk - hp
        T = closed_form_shift(X[:, BODY_MHR], own_full, K, z0.clamp_min(0.5), torch.ones_like(own_full[..., 0]))
        P = X[:, LEG_MHR] + T[:, None]
        okr = (P[..., 2] > 0.1).float()
        if w.get("foot_reproj_vis", 0) > 0 and "gt_vis" in b:  # only points the visibility labels call visible (no
            okr = okr * (b["gt_vis"][:, LEG_MHR] == 0).float()  # label: dropped; the sidecars cover ~1/4 of the pools)
            t["foot_reproj_kept"] = okr.mean().detach()
        r = project_crop(P, K, A, size_)
        t["foot_reproj"] = ((r - uv[:, LEG_MHR].detach()).abs().mean(-1) * okr).sum() / okr.sum().clamp_min(1.0)
    return t


def mask_losses(logits, lab, valid):
    """Person mask (after FAH's mask loss): BCE + soft Dice of the logits (B, 1, h, w) vs human mask labels (B, h, w;
    1 person, 0 background, 255 unknown) on the crops that have labels (valid); unknown pixels excluded."""
    lg = logits[:, 0].float()
    soft = None
    if lab.shape[-2] < lg.shape[-2]:  # a finer decoder than the labels ("+full"): soft (bilinear) targets at the edges
        soft = F.interpolate((lab == 1).float()[:, None], size=lg.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    if lab.shape[-2:] != lg.shape[-2:]:
        lab = F.interpolate(lab[:, None].float(), size=lg.shape[-2:], mode="nearest")[:, 0]
    known = (lab != 255).float() * valid[:, None, None]
    t = ((lab == 1).float() if soft is None else soft) * known
    bce = (F.binary_cross_entropy_with_logits(lg, t, reduction="none") * known).sum((1, 2)) / known.sum((1, 2)).clamp_min(1.0)
    p = lg.sigmoid() * known
    dice = 1 - (2 * (p * t).sum((1, 2)) + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)
    n = valid.sum().clamp_min(1.0)
    return {"mask_bce": (bce * valid).sum() / n, "mask_dice": (dice * valid).sum() / n}


def part_losses(out, b, smpl_target, size, roll, w=None):
    """Part head CE (w partseg_limb: class weight of the 12 limb parts, after FAH's finger classes x3) and the
    predicted mesh's part alignment (w part_region / part_cover > 0) against the same rendered labels."""
    from fab.body_parts import N_PARTS

    w = w or {}
    t = {}
    logits = out.get("part_logits")
    hw = logits.shape[-2:] if logits is not None else (128, 128)
    lab = part_targets(b, smpl_target, size, roll, hw, fill=w.get("part_fill", 0) > 0)
    if w.get("part_clip", 0) > 0 and "gt_mask" in b:  # boundaries of the real person (clothing, hair), not the mesh
        valid = b["gt_mask_valid"]
        if w.get("mask_sil", 0) > 0:
            valid = valid * torch.where(b["gt_mask_teacher"] > 0, mask_sil_gate(b, smpl_target, size, roll, w).float(),
                                        torch.ones_like(valid))
        lab = clip_parts(lab, b["gt_mask"], valid)
    if logits is not None:
        cw = None
        if w.get("partseg_limb", 1.0) != 1.0:  # bg, head, torso x1; arms, hands, legs, feet x partseg_limb
            cw = torch.ones(N_PARTS, device=lab.device)
            cw[3:] = w["partseg_limb"]
        t["partseg"] = F.cross_entropy(logits.float(), lab, weight=cw, ignore_index=-100)
    if w.get("part_region", 0) > 0 or w.get("part_cover", 0) > 0:
        t.update(mesh_part_alignment(out, b, smpl_target, size, lab))
    return t


def gaussian_blur(x, sigma):
    """Separable Gaussian blur of x (N, 1, H, W), zero padding."""
    r = int(math.ceil(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2)
    k = k / k.sum()
    x = F.conv2d(x, k.view(1, 1, 1, -1), padding=(0, r))
    return F.conv2d(x, k.view(1, 1, -1, 1), padding=(r, 0))


def mesh_part_alignment(out, b, smpl_target, size, lab, sigmas=(1.5, 4.0, 12.0), n_cover=32, tol=0.03):
    """Part-aware alignment of the predicted mesh (after FAH's part_loss on its dense points): the SMPL-topology
    vertices of the predicted MHR mesh at the predicted camera translation, projected into the crop; the vertices its
    own z-buffer shows (detached) against the part labels lab (B, H, W; 0 background, -100 ignored band).
    region: a visible vertex outside its own part's region is pulled towards it (1 - the region blurred at several
    scales, zero inside the region, so vertices are not drawn to the region's centre); cover: sampled pixels of every
    labelled part to the nearest visible vertex of that part (crop units, smooth L1), so the projected mesh spans the
    labelled silhouette. Differentiable in the mesh's pose, shape and camera translation."""
    from fab.body_parts import N_PARTS, barycentric_grid, render_depth

    B, H, W = lab.shape
    P = smpl_target.pred(out["mhr_model_params"], out["shape"], full=True) + out["pred_cam_t"].float()[:, None]
    uv = project_crop(P, b["cam_int"], b["affine"], size)  # (B, V, 2) crop units
    vp = smpl_target.vpart
    with torch.no_grad():
        zmin = render_depth(uv, P[..., 2], smpl_target.faces, (H, W), barycentric_grid(3).to(uv.device))
        x, y = (uv[..., 0] * W).floor().long(), (uv[..., 1] * H).floor().long()
        inb = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (P[..., 2] > 0.05)
        zpix = zmin.flatten(1).gather(1, (y.clamp(0, H - 1) * W + x.clamp(0, W - 1)))
        vis = (inb & (P[..., 2] <= zpix + tol)).float()
        ign = (lab == -100).float()
        oh0 = F.one_hot(lab.clamp_min(0), N_PARTS).permute(0, 3, 1, 2).float() * (1 - ign[:, None])
        oh = sum(gaussian_blur(oh0.reshape(B * N_PARTS, 1, H, W), s) for s in sigmas).view(B, N_PARTS, H, W)
        oh = oh / len(sigmas)
        ig = gaussian_blur(ign[:, None], sigmas[0])
        has = ((lab > 0).flatten(1).any(1)).float()
    g = (uv * 2 - 1)[:, None]  # (B, 1, V, 2) in [-1, 1]
    idx = vp[None, None].expand(B, 1, -1)
    own = F.grid_sample(oh, g, align_corners=False)[:, :, 0].gather(1, idx)[:, 0]  # (B, V)
    with torch.no_grad():
        inside = F.grid_sample(oh0, g.detach(), mode="nearest", align_corners=False)[:, :, 0].gather(1, idx)[:, 0]
        wv = (1 - F.grid_sample(ig, g.detach(), align_corners=False)[:, 0, 0]) * vis * has[:, None]
    region = ((1 - own) * (1 - inside) * wv).sum() / wv.sum().clamp_min(1.0)
    flat, cover, n = lab.flatten(1), uv.new_zeros(()), 0
    scale = torch.tensor([1.0 / W, 1.0 / H], device=uv.device)
    for k in range(1, N_PARTS):
        sel = vp == k
        mk = (flat == k).float()
        pv = vis[:, sel]
        ok = (mk.sum(1) > 0) & (pv.sum(1) > 0)
        if not sel.any() or not ok.any():
            continue
        pix = torch.multinomial(mk + (~ok).float()[:, None], n_cover, replacement=True)  # (B, n_cover) label pixels
        xy = (torch.stack([pix % W, pix // W], -1).float() + 0.5) * scale
        d = torch.cdist(xy, uv[:, sel]) + (1 - pv)[:, None] * 1e4  # hidden vertices do not count
        dm = d.min(-1).values
        hub = torch.where(dm < 0.02, 0.5 * dm ** 2 / 0.02, dm - 0.01)
        cover = cover + (hub.mean(1) * ok).sum() / ok.sum()
        n += 1
    return {"part_region": region, "part_cover": cover / max(n, 1)}


@torch.no_grad()
def part_targets(b, smpl_target, size, roll, hw, band=5, fill=False):
    """Part segmentation labels (B, H, W): the SMPL-space target mesh (GT, else the MHR teacher; as the dense points) z-buffered into the crop with its part ids (fab/body_parts.py); mirrored
    crops render the mirrored mesh with left / right swapped. A band just outside the silhouette is ignored (-100):
    the mesh is not the person's mask (clothing, hair), the two only agree weakly."""
    from fab.body_parts import adaptive_bary, barycentric_grid, render_parts

    H, W = hw
    t_cam = roll(smpl_target.pred(b["t_model_params"], b["t_shape"], full=True)) + b["t_cam_t"][:, None]
    g_cam = roll(smpl_target.gt(b["gt_smpl_pose"], b["gt_smpl_betas"], b["gt_smpl_trans"], b["gt_smpl_gender"], verts=b.get("gt_smpl_v"), verts_valid=b.get("gt_smpl_v_valid"),
                                full=True))
    gv = b["gt_smpl_valid"][:, None, None] > 0
    tgt = torch.where(gv, g_cam, t_cam)
    fl = b["flipped"] > 0
    tgt = torch.where(fl[:, None, None], tgt * smpl_target.mirror, tgt)
    uv = project_crop(tgt, b["cam_int"], b["affine"], size)
    bary = adaptive_bary(uv, smpl_target.faces, (H, W)) if fill else barycentric_grid(3).to(uv.device)  # fill: close-ups
    lab = render_parts(uv, tgt[..., 2], smpl_target.faces, smpl_target.fpart, (H, W), bary, flip_parts=fl)
    body = (lab > 0).float()[:, None]
    ring = (F.max_pool2d(body, band, 1, band // 2) > 0)[:, 0] & (lab == 0)
    lab = lab.masked_fill(ring, -100)
    return lab


@torch.no_grad()
def mask_sil_gate(b, smpl_target, size, roll, w):
    """Teacher person masks vs the target's rendered body silhouette (the part labels' target mesh) at 128 x 128: keep a
    mask when IoU >= w mask_sil and at most w mask_silout of it lies outside the silhouette dilated by w mask_sildil cells
    (clothing / hair fit inside; another person's body does not) -> (B,) bool."""
    sil = (part_targets(b, smpl_target, size, roll, (128, 128), fill=w.get("part_fill", 0) > 0) > 0).float()
    lab = F.interpolate(b["gt_mask"][:, None].float(), size=(128, 128), mode="nearest")[:, 0]
    m, known = (lab == 1).float(), (lab != 255).float()
    iou = (m * sil).sum((1, 2)) / (((m + sil) > 0).float() * known).sum((1, 2)).clamp_min(1.0)
    r = int(w.get("mask_sildil", 8))
    sil_d = F.max_pool2d(sil[:, None], 2 * r + 1, 1, r)[:, 0]
    out = (m * (1 - sil_d)).sum((1, 2)) / m.sum((1, 2)).clamp_min(1.0)
    return (iou >= w["mask_sil"]) & (out <= w.get("mask_silout", 1.0))


def clip_parts(lab, mask, valid, grow=12):
    """Part labels (B, H, W; 0 background, -100 ignored) clipped to person masks (B, h, w; 1 person, 0 background, 255
    unknown) on the crops with a mask (valid): background outside the mask, mask pixels the mesh misses (clothing, hair)
    -> the neighbouring part grown over up to `grow` cells (3 x 3 dilations), farther ones and the mask border ignored."""
    H, W = lab.shape[-2:]
    m = F.interpolate(mask[:, None].float(), size=(H, W), mode="nearest")[:, 0]
    fill = lab.clamp_min(0)
    for _ in range(grow):
        grown = F.max_pool2d(fill.float()[:, None], 3, 1, 1)[:, 0].long()
        fill = torch.where((fill == 0) & (m == 1), grown, fill)
    out = torch.where(m == 1, torch.where(fill > 0, fill, torch.full_like(fill, -100)), torch.zeros_like(fill))
    fg = (m == 1).float()[:, None]
    edge = (F.max_pool2d(fg, 3, 1, 1) + F.max_pool2d(-fg, 3, 1, 1))[:, 0] > 0  # dilated minus eroded
    out = out.masked_fill(edge | (m == 255), -100)
    return torch.where(valid[:, None, None] > 0, out, lab)


def gt_mesh(smpl_target, b, neutral_off=False, full=False):
    """GT SMPL mesh (camera frame). neutral_off: crops flagged gt_shape_off (--smpl_shape_off) get their GT pose on the
    neutral mean shape; used by the root-relative 3D terms only (2D targets and the teacher gate keep the real body)."""
    betas, gender, vv = b["gt_smpl_betas"], b["gt_smpl_gender"], b.get("gt_smpl_v_valid")
    off = b.get("gt_shape_off")
    if neutral_off and off is not None and (off > 0).any():
        keep = 1.0 - off
        betas, gender = betas * keep[:, None], (gender * keep).long()
        vv = vv * keep if vv is not None else None
    return smpl_target.gt(b["gt_smpl_pose"], betas, b["gt_smpl_trans"], gender, full=full, verts=b.get("gt_smpl_v"),
                          verts_valid=vv)


def losses(model, out, b, size, w, rig, smpl_target=None, hard_gamma=0.0, g1_kin=None):
    out = {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in out.items()}
    # Independent metric GT may disagree with the teacher. Preserve the original denominators:
    # normalising by the attenuated weights would cancel the treatment in all-GT batches.
    trusted = torch.zeros_like(b["flipped"])
    if smpl_target is not None:
        if w.get("smpl_j14", 0) > 0 or w.get("smpl_v", 0) > 0:
            smpl_active = torch.ones_like(trusted) if w.get("smpl_j14", 0) > 0 else 1.0 - b["flipped"]
            trusted = torch.maximum(trusted, b["gt_smpl_valid"] * smpl_active)
        if w.get("mocap_j14", 0) > 0:
            trusted = torch.maximum(trusted, b["gt_mocap_valid"])
    teacher_body = 1.0 - trusted * (1.0 - w.get("teacher_body_on_gt", 1.0))
    terms = {}
    if smpl_target is not None and w.get("teacher_gate", 0) > 0:  # a teacher that contradicts the GT mesh (e.g. the
        with torch.no_grad():  # other person of a close interaction) is not distilled on that crop
            e_t, _ = smpl_target.errors(
                smpl_target.pred(b["t_model_params"], b["t_shape"]),
                smpl_target.gt(b["gt_smpl_pose"], b["gt_smpl_betas"], b["gt_smpl_trans"], b["gt_smpl_gender"], verts=b.get("gt_smpl_v"), verts_valid=b.get("gt_smpl_v_valid")),
                torch.zeros_like(b["flipped"]))
            gated = b["gt_smpl_valid"] * (e_t > w["teacher_gate"]).float()
        teacher_body = teacher_body * (1.0 - gated)
        terms["teacher_gated"] = gated.sum() / b["gt_smpl_valid"].sum().clamp_min(1.0)
    if w.get("idgate", 0) > 0:  # identity gate: a teacher of the other person is not
        idg = identity_gate(b, w)  # distilled on that crop (body, 2D, camera and dense targets)
        teacher_body = teacher_body * (1.0 - idg)
        b["id_keep"] = 1.0 - idg
        terms["id_gated"] = idg.mean()
    pelvis_p = out["pred_keypoints_3d"][:, list(PELVIS)].mean(1, keepdim=True)
    pelvis_t = b["t_kp3d"][:, list(PELVIS)].mean(1, keepdim=True)
    e_kp = (out["pred_keypoints_3d"] - pelvis_p - (b["t_kp3d"] - pelvis_t)).abs().mean(-1).mean(1)
    terms["kp3d"] = (e_kp * hard_weights(e_kp, gamma=hard_gamma) * teacher_body).mean()
    if out.get("aux_keypoints_3d"):  # deep supervision of the intermediate refinement passes
        terms["kp3d_aux"] = sum(
            (((a.float() - a.float()[:, list(PELVIS)].mean(1, keepdim=True) -
               (b["t_kp3d"] - pelvis_t)).abs().mean(-1)).mean(1) * teacher_body).mean()
            for a in out["aux_keypoints_3d"]) / len(out["aux_keypoints_3d"])
    keep = 1.0 - b["flipped"]  # teacher rig parameters / mesh cannot be mirrored: those terms skip flipped crops
    n_keep = keep.sum().clamp_min(1.0)
    pred_verts = subset_vertices(rig, out["mhr_model_params"], out["shape"])
    roll = lambda v: torch.einsum("bij,bvj->bvi", b["rot"], v)  # noqa: E731  (targets from rig parameters)
    with torch.no_grad():
        t_verts = roll(subset_vertices(rig, b["t_model_params"], b["t_shape"]))
    per = (pred_verts - pelvis_p - (t_verts - pelvis_t)).abs().mean(dim=(1, 2))
    terms["verts"] = (per * keep * teacher_body).sum() / n_keep
    pred2d = to_crop(out["pred_keypoints_2d"], b["affine"]) / size
    terms["kp2d_t"] = (pred2d - b["t_kp2d"] / size).abs().mean()
    if "id_keep" in b:  # identity gate (--w idgate)
        terms["kp2d_t"] = ((pred2d - b["t_kp2d"] / size).abs().mean(dim=(1, 2)) * b["id_keep"]).mean()
    conf = b["gt_kp2d"][..., 2:3].clamp(0, 1)
    terms["kp2d_gt"] = ((pred2d - b["gt_kp2d"][..., :2] / size).abs() * conf).sum() / conf.sum().clamp_min(1.0) / 2
    if "pred_keypoints_2d_ind" in out:  # independent 2D head: 2D labels strongly, the teacher weakly
        ind2d = to_crop(out["pred_keypoints_2d_ind"], b["affine"]) / size
        terms["kp2d_ind_gt"] = ((ind2d - b["gt_kp2d"][..., :2] / size).abs() * conf).sum() / conf.sum().clamp_min(1.0) / 2
        terms["kp2d_ind_t"] = (ind2d - b["t_kp2d"] / size).abs().mean()
        if "id_keep" in b:
            terms["kp2d_ind_t"] = ((ind2d - b["t_kp2d"] / size).abs().mean(dim=(1, 2)) * b["id_keep"]).mean()

    def masked_mse(a, t, strength=1.0):
        return (((a - t) ** 2).mean(dim=1) * keep * strength).sum() / n_keep

    keep_g = keep * (1.0 - b["rotated"])  # the global orientation of a rolled crop is not the teacher's
    g6 = ((out["pred_pose_raw"][:, :6] - b["t_pose_raw"][:, :6]) ** 2).mean(dim=1)
    b260 = ((out["pred_pose_raw"][:, 6:] - b["t_pose_raw"][:, 6:]) ** 2).mean(dim=1)
    terms["pose"] = (6 * (g6 * keep_g * teacher_body).sum() / keep_g.sum().clamp_min(1.0)
                     + 260 * (b260 * keep * teacher_body).sum() / n_keep) / 266
    terms["shape"] = (masked_mse(out["shape"], b["t_shape"], teacher_body) +
                      masked_mse(out["scale"], b["t_scale"], teacher_body))
    terms["hand"] = masked_mse(out["hand"], b["t_hand"])
    terms["cam_t"] = (out["pred_cam_t"] - b["t_cam_t"]).abs().mean()
    if "id_keep" in b:
        terms["cam_t"] = ((out["pred_cam_t"] - b["t_cam_t"]).abs().mean(-1) * b["id_keep"]).mean()
    # human labels, where present: camera-frame keypoints, mesh (unflipped crops) and frame-free keypoints
    gw = b["gt_kp3d_w"]
    if getattr(smpl_target, "foot_pitch_gate", False) and "foot3d_ok" in b:  # --foot_pitch_gate: GT heels / toes (MHR70
        gw = gw.clone()  # 15-20) only on --foot3d_sources crops
        gw[:, 15:21] = gw[:, 15:21] * b["foot3d_ok"][:, None]
    cam = (b["gt_mode"] == GT_CAMERA).float()
    pelvis_g = b["gt_kp3d"][:, list(PELVIS)].mean(1, keepdim=True)
    per = ((out["pred_keypoints_3d"] - pelvis_p - (b["gt_kp3d"] - pelvis_g)).abs().mean(-1) * gw).sum(1) \
        / gw.sum(1).clamp_min(1.0)
    terms["kp3d_gt"] = masked_mean(per, cam)
    mesh = cam * b["gt_mhr_valid"] * keep
    with torch.no_grad():  # the stored label keypoints drop the parameters' translation: centroid-align the mesh
        g_verts = roll(subset_vertices(rig, b["gt_model_params"], b["gt_shape"]))
        g_verts = g_verts - g_verts.mean(1, keepdim=True)
    per = (pred_verts - pred_verts.mean(1, keepdim=True) - g_verts).abs().mean(dim=(1, 2))
    rf = getattr(smpl_target, "rig_foot", None)
    if rf is not None and "foot3d_ok" in b:  # --foot_pitch_gate: the GT MHR mesh's foot vertices only on trusted feet
        wv = torch.ones_like(pred_verts[..., 0])
        wv[:, rf] = b["foot3d_ok"][:, None]
        per = ((pred_verts - pred_verts.mean(1, keepdim=True) - g_verts).abs().mean(-1) * wv).sum(1) / wv.sum(1)
    terms["verts_gt"] = masked_mean(per, mesh)
    other = (b["gt_mode"] == GT_OTHER_FRAME).float()
    terms["kp3d_gt_rot"] = masked_mean(rotation_aligned_l1(out["pred_keypoints_3d"], b["gt_kp3d"], gw), other)
    if smpl_target is not None:  # exact SMPL meshes (BEDLAM), in the evaluator's topology and joints
        smpl_pred = smpl_target.pred(out["mhr_model_params"], out["shape"])
        e_j, e_v = smpl_target.errors(
            smpl_pred,
            roll(gt_mesh(smpl_target, b, neutral_off=True)),
            b["flipped"])
        terms["smpl_j14"] = masked_mean(e_j * hard_weights(e_j, b["gt_smpl_valid"], hard_gamma), b["gt_smpl_valid"])
        frd = getattr(smpl_target, "foot_rand", None)
        if frd is not None and "foot3d_ok" in b:  # --foot_pitch_gate: the GT SMPL foot vertices only on trusted feet
            vw = torch.ones(len(e_v), int(smpl_target.in_rand.sum()), device=e_v.device)
            vw[:, frd] = b["foot3d_ok"][:, None]
            e_v = smpl_target.errors_v_weighted(smpl_pred, roll(gt_mesh(smpl_target, b, neutral_off=True)), b["flipped"], vw)
        terms["smpl_v"] = masked_mean(e_v, b["gt_smpl_valid"] * keep)
        if len(smpl_target.j24) and w.get("smpl_j24", 0) > 0:  # 24 SMPL joints, pelvis-aligned
            g24 = roll(gt_mesh(smpl_target, b, neutral_off=True))
            terms["smpl_j24"] = masked_mean(smpl_target.errors24(smpl_pred, g24, b["flipped"]), b["gt_smpl_valid"])
        if w.get("mocap_j14", 0) > 0:
            joints = torch.einsum("js,bsc->bjc", smpl_target.j14, smpl_pred)
            terms["mocap_j14"] = mocap_joint_loss(joints, b["gt_mocap_j14"], b["gt_mocap_valid"])
    if smpl_target is not None and "dense_uv" in out and int(getattr(model, "dense_head").residual):
        with torch.no_grad():  # anchors: MHR keypoints, mesh-regressed J14 and mapped mesh vertices (camera axes)
            sp = smpl_target.pred(out["mhr_model_params"], out["shape"])
            out["dense_anchor"] = torch.cat([out["pred_keypoints_3d"].float(),
                                             torch.einsum("js,bsc->bjc", smpl_target.j14, sp),
                                             sp[:, smpl_target.extra_pos]], 1)
    if smpl_target is not None and "dense_uv" in out:
        terms.update(dense_losses(out, b, smpl_target, size, keep, roll, w=w))
    if smpl_target is not None and "dense_uv" in out and (any(w.get(k, 0) > 0 for k in FOOT_TERMS) or w.get("heel_lab", 0) > 0):
        terms.update(foot_losses(out, b, smpl_target, size, roll, w))
    if smpl_target is not None and ((out.get("part_logits") is not None and w.get("partseg", 0) > 0)
                                    or w.get("part_region", 0) > 0 or w.get("part_cover", 0) > 0):
        terms.update(part_losses(out, b, smpl_target, size, roll, w))
    if out.get("mask_logits") is not None and "gt_mask" in b:
        valid = b["gt_mask_valid"]
        if smpl_target is not None and w.get("mask_sil", 0) > 0:  # teacher masks must agree with the target's body
            with torch.no_grad():
                keep = mask_sil_gate(b, smpl_target, size, roll, w)
                valid = valid * torch.where(b["gt_mask_teacher"] > 0, keep.float(), torch.ones_like(valid))
                terms["mask_kept"] = valid.sum() / b["gt_mask_valid"].sum().clamp_min(1.0)
        terms.update(mask_losses(out["mask_logits"], b["gt_mask"], valid))
    if "vis_logits" in out and "gt_vis" in b:  # per-keypoint visibility (CE, 255 = unknown)
        # the rendered labels train the rendered-label classifier (vis_logits_rendered when a human head is attached)
        lg, lab = out.get("vis_logits_rendered", out["vis_logits"]).float().reshape(-1, 3), b["gt_vis"].reshape(-1)
        cw = torch.tensor([1.0, w.get("vis_occ_w", 1.0), 1.0], device=lg.device)
        if (lab != 255).any():
            terms["vis"] = F.cross_entropy(lg, lab, weight=cw, ignore_index=255)
            with torch.no_grad():
                ok = lab != 255
                terms["vis_acc"] = (lg.argmax(-1)[ok] == lab[ok]).float().mean()
        else:
            terms["vis"] = lg.sum() * 0.0
    if "vis_logits_rendered" in out and "gt_vis_h" in b:  # --vis_human: the human-annotation classifier (vis_logits)
        lg, lab = out["vis_logits"].float().reshape(-1, 3), b["gt_vis_h"].reshape(-1)
        cw = torch.tensor([1.0, w.get("vis_h_occ_w", 1.0), 1.0], device=lg.device)
        if (lab != 255).any():
            terms["vis_h"] = F.cross_entropy(lg, lab, weight=cw, ignore_index=255)
            with torch.no_grad():
                ok = lab != 255
                terms["vis_h_acc"] = (lg.argmax(-1)[ok] == lab[ok]).float().mean()
                terms["vis_h_frac"] = ok.float().mean()
        else:
            terms["vis_h"] = lg.sum() * 0.0
    names2d = getattr(smpl_target, "kp2d_names", None)
    if names2d and "foot2d_src" in b:  # --kp2d_labels: the foot 2D labels' sources (share of their confidence, foot_lab)
        with torch.no_grad():
            cf = b["gt_kp2d"][:, 15:21, 2].clamp(0, 1)
            tot = cf.sum().clamp_min(1e-6)
            for k_, nm in enumerate(names2d):
                terms[f"foot2d_{nm}"] = (cf * (b["foot2d_src"] == k_)).sum() / tot
            terms["foot2d_pool"] = (cf * (b["foot2d_src"] < 0)).sum() / tot
    if g1_kin is not None and "g1_q" in out:
        terms.update(g1_losses(g1_kin, out, b, keep))
    if "contact_logits" in out:
        terms.update(contact_losses(out, b))
    total = sum(w.get(k, 0.0) * v for k, v in terms.items())
    return total, terms


CONTACT_H_THR = np.array([0.04, 0.04, 0.04, 0.04, 0.05, 0.05, 0.05, 0.05])  # m above the local floor (any-contact rule)


def contact_losses(out, b):
    """Masked BCE of the contact head's logits against the world-GT contact labels (docs/training.md); a 16-way
    head (--contact_split): static contact (height and speed) then any contact (height only)."""
    lg = out["contact_logits"].float()
    y, m = b["contact"], b["contact_valid"]
    if lg.shape[1] == 2 * y.shape[1]:
        y, m = torch.cat([y, b["contact_any"]], 1), torch.cat([m, b["contact_any_valid"]], 1)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(lg, y, reduction="none")
    return {"contact": (bce * m).sum() / m.sum().clamp_min(1.0)}


WEIGHTS = {"kp3d": 5.0, "verts": 2.0, "kp2d_t": 2.0, "kp2d_gt": 1.0, "pose": 0.1, "shape": 0.05, "hand": 0.02,
           "cam_t": 0.5, "kp3d_gt": 5.0, "verts_gt": 2.0, "kp3d_gt_rot": 2.0, "smpl_j14": 5.0, "smpl_v": 2.0,
           "kp3d_aux": 2.0, "g1_q": 0.5, "g1_fk": 5.0, "kp2d_ind_gt": 2.0, "kp2d_ind_t": 0.5,
           "dense_mhr2d_gt": 2.0, "dense_mhr2d_t": 0.5, "dense_mhr3d": 1.0, "dense_j3d": 3.0, "dense_v3d": 1.0,
           "dense_2d": 1.0, "dense_rep": 1.0, "dense_rep_surf": 0.5, "partseg": 1.0, "contact": 1.0}


def trainable_state(model):
    """State dict without the frozen TorchScript rig (~700 MB; it loads from mhr_model.pt)."""
    return {k: v for k, v in model.state_dict().items()
            if not (k.startswith("head_pose.mhr.") or ".head_pose.mhr." in k or ".head_pose_hand.mhr." in k)}


def init_random_encoder(backbone):
    """--no_teacher_weights: SAM 3D Body's DINOv3 encoder without its checkpoint has uninitialised (torch.empty / NaN)
    tokens and qkv bias masks; DINOv3's own init_weights() fills them."""
    with torch.no_grad():
        backbone.encoder.init_weights()


def load_feature_teacher(backbone, checkpoint):
    """Replace only the frozen feature target with a fine-tuned SAM backbone (EMA when present)."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if state["args"]["arch"] != "fabt":
        raise ValueError("Feature teacher must be a fabt checkpoint")
    weights = {**state["model"], **(state.get("ema") or {})}
    weights = {k.removeprefix("body.backbone."): v for k, v in weights.items()
               if k.startswith("body.backbone.")}
    if not weights or any(not torch.isfinite(v).all() for v in weights.values()):
        raise ValueError("Feature teacher has missing or nonfinite backbone weights")
    backbone.load_state_dict(weights, strict=True)
    backbone.eval().requires_grad_(False)


# ----------------------------------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pools = p.add_mutually_exclusive_group()
    pools.add_argument("--pools", help="glob(s) of pool tars, comma-separated")
    pools.add_argument("--pool_list", help="JSON list of exact tar paths; duplicates preserve sampling weights")
    p.add_argument("--config", default="", help="YAML file of argument defaults (configs/train/*.yaml); the command line wins")
    p.add_argument("--teacher_ckpt", default="", help="SAM 3D Body's model.ckpt (its folder also holds model_config.yaml and "
                   "assets/mhr_model.pt); default: $FAB_SAM3D/model.ckpt")
    p.add_argument("--no_teacher_weights", action="store_true",
                   help="smoke tests only: build SAM 3D Body from its config without loading model.ckpt (its encoder "
                   "randomly initialised)")
    p.add_argument("--no_pretrained", action="store_true",
                   help="smoke tests only: no pretrained DINOv3 weights for the student backbone (no download)")
    p.add_argument("--out", required=True)
    p.add_argument("--backbone", default="dinov3_vitb16")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--batch", type=int, default=32, help="per GPU")
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--backbone_lr", type=float, default=4e-5)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--save_steps", default="", help="comma list of steps: also keep step<N>.pt there (multiples of --save_every)")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--vert_subset", type=int, default=2048, help="vertices in the mesh loss (fixed random subset)")
    p.add_argument("--flip", type=float, default=0.5, help="horizontal flip probability (keypoint / camera losses only)")
    p.add_argument("--rot", type=float, default=0.0,
                   help="in-plane rotation probability (80 %%: N(0, 25 deg) clipped to 60; 20 %%: uniform 360)")
    p.add_argument("--rot_full", type=float, default=0.2, help="share of the rotated crops rolled uniformly over 360 deg")
    p.add_argument("--p_occ", type=float, default=0.0, help="probability of a pasted occluder per crop (batch_occlude)")
    p.add_argument("--p_lowres", type=float, default=0.0, help="probability of a low-resolution crop (batch_occlude)")
    p.add_argument("--mask_prompt_p", type=float, default=0.0,
                   help="fabd: probability per crop of feeding the target's silhouette (its GT / teacher mesh splatted "
                   "into the crop, randomly dilated / eroded / shifted) as the decoder's mask prompt")
    p.add_argument("--box_prompt_p", type=float, default=0.0,
                   help="fabd: probability per crop of feeding the input box's extent (filled rectangle in crop "
                   "coordinates, the record's box without the zoom / shift jitter: loose / tight / shifted on the person) "
                   "as the decoder's mask prompt; evaluate with bench/eval_hmr.py --mask_prompt box")
    p.add_argument("--neg_prompt_p", type=float, default=0.0,
                   help="fabd: probability per crop of a signed mask prompt: +1 in the input box, -1 in the frame's other "
                   "people's boxes (<tar>.others.npz), 0 where both / neither; evaluate with "
                   "bench/eval_hmr.py --mask_prompt neg (the image's other boxes)")
    p.add_argument("--neg_prompt_aug", default="0.3,0.1,0.1",
                   help="--neg_prompt_p noise: drop probability per other person, box jitter (scale / shift fraction), "
                   "group-box probability (a neighbour's box joined with the target's)")
    p.add_argument("--det_box_p", type=float, default=0.0,
                   help="probability per crop (with a <tar>.detbox.npz detection and wide context) of cropping from the person "
                   "detector's box, as the inference pipeline does (group boxes included; <tar>.detbox.npz); "
                   "the box prompt / negatives then are that detection / the frame's other detections")
    p.add_argument("--loose_box_p", type=float, default=0.0,
                   help="probability per crop of an annotation-style loose box: crop zoom from --loose_box_zoom (< 1 = a "
                   "larger box: Harmony4D's annotation boxes are 1.19x the tight box, zoom 0.84, centred) and a shift of up "
                   "to --loose_box_shift of the stored crop, only on crops with wide context (wide_jpg / wide_off in the record)")
    p.add_argument("--loose_box_zoom", default="0.78,0.92")
    p.add_argument("--loose_box_shift", type=float, default=0.04)
    p.add_argument("--wide_context", action="store_true",
                   help="cut every crop window from the pool's wide-context crop where it has one (no black padding)")
    p.add_argument("--foot3d_sources", default="bedlam,agora_train,moyo",
                   help="pool sources whose GT feet the foot terms trust (--w foot_kp / foot_dir / foot_2d; GT feet elsewhere "
                   "are often mis-tilted)")
    p.add_argument("--foot_reg", default="",
                   help="SMPL mesh -> MHR body keypoint regressor for the GT foot positions (foot_kp / foot_2d); '' or "
                   "missing: the SMPL foot vertices about the J14 hips")
    p.add_argument("--p_paste", type=float, default=0.0,
                   help="probability per crop of a pasted distractor person: another crop's person (its label "
                   "silhouette, feathered) shifted to overlap the target, in front or behind; targets unchanged")
    p.add_argument("--init", default="", help="warm start from this checkpoint (weights only)")
    p.add_argument("--args_from", default="",
                   help="a checkpoint whose saved args become the defaults: every argument not given on the command line "
                   "takes the checkpoint's value (e.g. a released head's recipe on another body); arguments added since "
                   "keep their defaults")
    p.add_argument("--g1_head", action="store_true", help="fabd: add the G1 retargeting head (G1 label sidecars, docs/training.md)")
    p.add_argument("--g1_only", action="store_true", help="train only the G1 head (everything else frozen)")
    p.add_argument("--g1_labels", default="g1", help="G1 target sidecars <tar>.<g1_labels>.npz (keys, q (N, 29), R (N, 3, 3))")
    p.add_argument("--g1_joints", action="store_true", help="G1 head also reads the mesh path's 3D keypoints")
    p.add_argument("--vis_head", action="store_true",
                   help="fabd + --dense_head: add the per-keypoint visibility head (fab/vis_head.py; loss w vis)")
    p.add_argument("--vis_only", action="store_true", help="--vis_head: train only the visibility head (frozen model)")
    p.add_argument("--vis_labels", default="vis", help="--vis_head: visibility sidecars <tar>.<name>.npz")
    p.add_argument("--vis_attn", type=int, default=128, help="--vis_head: image cross-attention width (0 = off)")
    p.add_argument("--vis_human", action="store_true",
                   help="--vis_head: + a human-annotation classifier (COCO / WholeBody / Human-Art v-flags: the vis "
                   "sidecars' bit-8 points and the --kp2d_labels sidecars' vis; w vis_h, vis_h_occ_w), output as "
                   "vis_logits; the rendered labels keep training the first classifier (vis_logits_rendered)")
    p.add_argument("--vis_gate_2d", default="",
                   help="'occ,out': weights of the teacher 2D (dense_mhr2d_t) and the J14 2D (dense_2d) on keypoints the "
                   "--vis_labels sidecars call occluded / outside the crop (default: off, weight 1)")
    p.add_argument("--kp2d_labels", default="",
                   help="comma-separated 2D label sidecars <tar>.<name>.npz (keys, kp2d (N, 70, 3): stored crop px + "
                   "confidence, MHR70 order), e.g. wholebody (COCO-WholeBody feet / hands), filling the channels without a "
                   "human 2D label (the first listed wins); a trailing ! (wholebody!) also replaces the pool's own labels "
                   "where the sidecar has one (the COCO pool's 70-point labels include projected-fit feet); a trailing "
                   "@tol (foot2d_rtmw!@0.2) re-thresholds the per-foot confidence from the sidecar's dev (N, 2) <= tol")
    p.add_argument("--dense_extra", default="",
                   help="--dense_head: named surface points appended after the set: sole, tips (fab/dense_points.py)")
    p.add_argument("--dense_foot_untrusted", type=float, default=1.0,
                   help="--dense_head: weight of the foot surface points' 3D (dense_v3d) on crops outside --foot3d_sources")
    p.add_argument("--gt2d_in_crop", action="store_true",
                   help="human 2D labels outside the augmented crop window get confidence 0 (every gt_kp2d consumer: "
                   "dense_mhr2d_gt, kp2d_gt, foot_lab, dense_rep); = --gt2d_gate crop")
    p.add_argument("--gt_fix_labels", default="",
                   help="GT-foot fix sidecars <tar>.<name>.npz (keys, R_foot (N, 4, 3, 3) "
                   "local SMPL 7 / 8 / 10 / 11, ok / reversed (N, 2), err_before): the ankles' rotations replace "
                   "gt_smpl_pose[21:27] where ok; on --foot_off_tars tars the foot losses come back for samples with both "
                   "feet ok; samples with a reversed foot lose them")
    p.add_argument("--gt_fix_min_deg", type=float, default=0.0,
                   help="--gt_fix_labels: apply a foot's fix only where its GT was this far (deg) off the 2D reference")
    p.add_argument("--gt_fix_tars", default="",
                   help="--gt_fix_labels only on these tars (comma-separated substrings / fnmatch patterns, e.g. harmony4d: "
                   "Harmony4D only); default: every tar with the sidecar")
    p.add_argument("--gt_fix_alias", default="",
                   help="--gt_fix_labels: a=b,... path substitutions to find the sidecar of a re-cut pool with the same "
                   "sample keys, e.g. /pools/a/harmony4d/=/pools/b/harmony4d/")
    p.add_argument("--kp2d_alias", default="",
                   help="--kp2d_labels: a=b,... path substitutions to find a sidecar of a re-cut pool whose stored crops "
                   "are identical (same jpg and labels), e.g. "
                   "/pools/a/harmony4d/=/pools/b/harmony4d/,/pools/a/hard/=/pools/b/hard/")
    p.add_argument("--sidecar_no_feet", action="store_true",
                   help="--kp2d_labels sidecars never write the MHR70 foot channels (15-20): the feet keep the pool's own 2D "
                   "(body / hand channels as before)")
    p.add_argument("--heel2d_vertex", default="",
                   help="'l,r' SMPL vertices (e.g. 3427,6827, the heel vertices best matching the 2D heel labels): the --kp2d_labels "
                   "sidecars' heel 2D (MHR70 17 / 20) leave the MHR70 heel channels and supervise those vertices' 2D, an "
                   "affine combination of their 8 nearest dense surface points (--w heel_lab, foot_lab's normalisation); "
                   "the MHR70 heels keep only their 3D / teacher terms")
    p.add_argument("--human_feet_2d", action="store_true",
                   help="'trusted foot 2D': foot 2D labels (MHR70 heels / toes 15-20) only where a --kp2d_labels sidecar "
                   "supplied them -- human (COCO-WholeBody, Human-Art) or RTMW-x agreement pseudo-labels "
                   "; the pools' own foot channels (partly projected fits) get "
                   "conf 0. The per-source share of the foot labels' confidence is logged as foot2d_<name> / foot2d_pool")
    p.add_argument("--gt2d_gate", default="",
                   help="comma-separated gates of the human 2D labels: image (labels outside the source image: conf 0 "
                   "everywhere), heat (labels outside the crop only out of the heatmap term dense_mhr2d_gt, kept for the "
                   "amodal kp2d_gt / dense_rep), crop (outside the crop: conf 0 everywhere)")
    p.add_argument("--gt2d_crop_tars", default="",
                   help="substrings / patterns of tar paths whose labels outside the crop are projected GT of truncated "
                   "parts (e.g. EgoBody egocentric): conf 0 there")
    p.add_argument("--teacher_strict", action="store_true",
                   help="assert at startup that --teacher_labels is the only MHR teacher: no "
                   "--feature_teacher, every tar has its <tar>.<teacher>.npz sidecar or labels baked into its records, and each "
                   "round's crops labelled by another teacher (manifest label) are all in --skip_pairs")
    p.add_argument("--dense_25d_ramp", type=int, default=0,
                   help="--dense_25d: the lift's output share (dense_head.p25d_mix) and the dense_pj3d / pv3d / pmhr3d weights "
                   "ramp linearly from 0 to --dense_25d / their --w values over the first K steps (from-scratch runs: the lift at "
                   "full share diverged at lr 5e-4); 0: off")
    p.add_argument("--skip_pairs", default="",
                   help="file of '<pool dir>|<sample key>' lines (# comments, pool dir relative to pools/, e.g. "
                   "mined/r1|<key>): those samples are never loaded")
    p.add_argument("--skip_takes", default="",
                   help="file of sample-key regexes (one per line, # comments): matching samples of every pool are never "
                   "loaded (e.g. the takes of a benchmark's test split, any camera)")
    p.add_argument("--foot_pitch_gate", action="store_true",
                   help="GT foot 3D (pitch) only on --foot3d_sources crops in every 3D GT term: dense foot points (as "
                   "--dense_foot_untrusted 0), MHR70 heels / toes in kp3d_gt, foot vertices in smpl_v and verts_gt "
                   "(e.g. pitch-biased GT feet)")
    p.add_argument("--foot_off_tars", default="",
                   help="comma-separated fnmatch patterns or plain substrings (e.g. harmony4d,egobody_ego) of tar paths "
                   "whose GT feet are unreliable: no foot 2D labels (MHR70 15-20) and no dense foot points there")
    p.add_argument("--foot_off_keep_side", action="store_true",
                   help="--foot_off_tars drops only the pools' own foot 2D: foot channels a --kp2d_labels sidecar supplied "
                   "(image-based labels) stay; the dense foot points stay off")
    p.add_argument("--contact_head", action="store_true",
                   help="fabd + --dense_head: add the explicit contact head (fab/contact_head.py)")
    p.add_argument("--contact_only", action="store_true",
                   help="train only the contact head (everything else frozen; only the contact loss)")
    p.add_argument("--contact_split", action="store_true",
                   help="contact head: 8 static-contact + 8 any-contact (support touched, may move) outputs")
    p.add_argument("--contact_phys_dev_max", type=float, default=0.0,
                   help="physics contact labels: also mask samples whose simulation deviates from the GT pose by >= this "
                   "many mm (the sidecar's dev_mm; e.g. 30 on 3DPW), on tars whose path contains --contact_phys_dev_pools")
    p.add_argument("--contact_phys_dev_pools", default="3dpw")
    p.add_argument("--contact_knee_labels", default="",
                   help="contact head: the knee channels (static and their height-rule any-contact) from this sidecar "
                   "name instead (e.g. contact, the kinematic labels, with --contact_labels physc: physics knees are weak)")
    p.add_argument("--contact_any_labels", default="",
                   help="any-contact sidecars <tar>.<name>.npz (e.g. contact_hsc: RICH human-scene contact) where present; "
                   "comma-separated: the first present per tar (contact_hsc,physany)")
    p.add_argument("--contact_attn", type=int, default=0,
                   help="contact head: width of the keypoints' cross-attention to the image tokens (0 = off)")
    p.add_argument("--contact_labels", default="contact",
                   help="contact label sidecars <tar>.<contact_labels>.npz (docs/training.md); comma-separated: the "
                   "first present per tar (e.g. physics labels, the kinematic ones where missing)")
    p.add_argument("--dense_head", type=int, default=0,
                   help="fabd: direct dense keypoint head with this many SMPL surface points (0 = off)")
    p.add_argument("--smpl_shape_off", default="", help="comma-separated few-subject pool sources (e.g. moyo): "
                   "their GT pose on the neutral mean shape in the root-relative 3D SMPL terms (2D terms keep the body)")
    p.add_argument("--smpl_gt_off", default="", help="comma-separated pool sources whose SMPL GT is ignored (e.g. "
                   "egobody,egobody_ego: fits with shorter legs than the other pools); teacher / 2D labels stay")
    p.add_argument("--pseudo_3d_sources", default="",
                   help="comma-separated pool sources whose 3D labels are model fits (e.g. sa1b_train): 2D labels only")
    p.add_argument("--min_in_frame", type=float, default=0.0,
                   help="drop pseudo-labelled crops with fewer of the teacher's body keypoints inside the image")
    p.add_argument("--parts_only", action="store_true",
                   help="train only the part-segmentation head (dense_head.part_*) on an otherwise frozen --init model")
    p.add_argument("--mask_depth", type=int, default=0,
                   help="dense head: + a person-mask head with this many 3x3 convs (fab/dense_head.add_mask_head)")
    p.add_argument("--mask_query", default="box", choices=["box", "token", "box+pts", "token+pts", "box+pool", "box+pts+pool", "token+pool", "token+pts+pool",
                            "box+extent", "box+extent+pool", "box+pts+extent+pool", "box+extent+pool+full",
                            "box+extent+pool+corners", "box+extent+pool+corners+full", "box+extent+pool+kp"],
                   help="person-mask query: box = one learned query (the person the crop is centred on; usable as a "
                        "prompt), token = the person's pose token; +pts: + the model's own projected dense points; +extent: + the "
                        "pool box as a channel (its aspect, lost in the square crop); +kp: + the model's own predicted 2D body "
                        "keypoints as point-prompt channels")
    p.add_argument("--mask_only", action="store_true",
                   help="train only the person-mask head (dense_head.mask_*) on an otherwise frozen --init model")
    p.add_argument("--mask_labels", default="",
                   help="person-mask sidecars <tar>.<name>.npz, comma-separated in priority order (human COCO masks: "
                        "then model-generated teacher masks)")
    p.add_argument("--mask_gate", default="",
                   help="teacher-mask sidecar gates, e.g. score=0.85,kp_t=0.8,kp_gt=0.9,out_box=0.15 (+ --w mask_sil=0.5: "
                        "IoU with the rendered target silhouette, mask_silout / mask_sildil)")
    p.add_argument("--dense_only", action="store_true",
                   help="train only the dense head (dense_head.*) on an otherwise frozen --init model (backbone in eval mode)")
    p.add_argument("--init_dense", default="",
                   help="warm start the dense head from this checkpoint's dense_head.* (e.g. a FAB-L dense head on a FAB-H body)")
    p.add_argument("--dense_xyz_layers", type=int, default=0,
                   help="dense head: transformer layers (self-attention among points, cross to image tokens) before xyz")
    p.add_argument("--dense_xyz_dim", type=int, default=256,
                   help="dense head: width of the xyz MLP / point transformer")
    p.add_argument("--dense_xyz_mem", default="img", choices=["img", "img+dec"],
                   help="dense head: point-transformer memory = image tokens (+ the decoder's pose / keypoint tokens)")
    p.add_argument("--dense_25d", type=float, default=0.0,
                   help="dense head: 2.5D points, this weight on the back-projection of each point's heatmap 2D at its "
                        "regressed depth (1 = pure 2.5D, 0 = off; fab/dense_head.lift25d; w dense_p*3d on the "
                        "regressed 3D)")
    p.add_argument("--dense_25d_border", type=float, default=0.02,
                   help="--dense_25d: points this close to the crop edge (crop units) keep the regressed 3D")
    p.add_argument("--dense_25d_uvgrad", action="store_true",
                   help="--dense_25d: the 3D losses also move the heatmap 2D (default: detached)")
    p.add_argument("--dense_25d_uvgrad_vis", action="store_true",
                   help="--dense_25d: the 3D losses' gradient reaches the heatmap 2D only for the 70 MHR + 14 J14 keypoints the "
                   "visibility labels (--vis_labels, e.g. vis2) call visible (the surface points stay detached); training "
                   "only: the forward and inference are unchanged")
    p.add_argument("--vis_alias", default="",
                   help="--vis_labels: a=b,... path substitutions to find the visibility sidecar of a re-cut pool with the "
                   "same stored crops and keys, e.g. /pools/a/harmony4d/=/pools/b/harmony4d/,/pools/a/hard/=/pools/b/hard/")
    p.add_argument("--dense_25d_uvgrad_scale", type=float, default=-1.0,
                   help="--dense_25d: the 3D losses' gradient into the heatmap 2D x this share for every point (with "
                   "--dense_25d_uvgrad_vis: for the visible keypoints); training only; < 0: off (--dense_25d_uvgrad decides)")
    p.add_argument("--dense_25d_uvgrad_mesh", action="store_true",
                   help="--dense_25d: the 3D losses' gradient reaches the heatmap 2D through the lift only for the mesh-defined "
                   "points (J14 + surface, x --dense_25d_uvgrad_scale, default 1); the 70 MHR keypoints stay detached (their "
                   "2D only from the 2D terms); training only: the forward and inference are unchanged")
    p.add_argument("--dense_25d_uvgrad_feet", action="store_true",
                   help="with --dense_25d_uvgrad_mesh: the MHR70 foot keypoints (15:21: big / small toe, heel) also get the lift's "
                   "gradient (x the same scale); the other 64 MHR keypoints stay detached")
    p.add_argument("--dense_25d_vis_gate", action="store_true",
                   help="--dense_25d + --vis_head: each keypoint's lift weighted by the visibility head's P(visible)")
    p.add_argument("--dense_ray_in", action="store_true",
                   help="dense head: each point's viewing ray (its 2D through the crop camera) as a zero-initialised 3D input")
    p.add_argument("--dense_uv_off", type=float, default=0.0,
                   help="dense head: amodal 2D for truncated bodies, a zero-initialised offset on the soft-argmax 2D of "
                   "points within this distance of the crop edge (crop units; weight 1 at the edge), so out-of-crop points "
                   "do not pin to the edge; 0: off")
    p.add_argument("--uv_off_lr_mult", type=float, default=1.0,
                   help="--dense_uv_off: the offset layer (dense_head.uv_off) in its own optimizer group at this x --lr")
    p.add_argument("--trunc_aug", default="",
                   help="'p,lo,hi': with probability p the crop window is the inference crop (1.25 x the longer side) of the "
                   "record's box with a fraction lo-hi of it cut off one side (top / bottom 30 %% each, left / right 10 %%) or "
                   "a zoom-in x1.2-1.6 shifted to the edge of the context (20 %%), every label kept (3D and the amodal 2D "
                   "of the cut-off part); empty: off")
    p.add_argument("--trunc_zoom_max", type=float, default=0.0,
                   help="--trunc_aug: cap on the truncation crops' zoom-in (window scale; the default 3.0 / the x1.2-1.6 "
                   "zoom-ins), e.g. 1.6 against upsampled low-resolution / low-light crops; 0: no extra cap")
    p.add_argument("--ray_lr_mult", type=float, default=1.0,
                   help="--dense_ray_in: the ray layer (dense_head.pt_ray) in its own optimizer group at this x --lr (ray "
                   "arm 4K: it grew only to norm 0.12 at x1)")
    p.add_argument("--sidecar_rep", default="",
                   help="name=N: every pool tar with a <tar>.<name>.npz sidecar repeated exactly N times, replacing its @k "
                   "(e.g. wholebody=4: the tars with human WholeBody labels at @4)")
    p.add_argument("--init_vis", default="",
                   help="warm start the visibility head from this checkpoint's vis_head.*")
    p.add_argument("--dense_parts", action="store_true",
                   help="dense head: + part segmentation queries (fab/body_parts.py), rendered target labels")
    p.add_argument("--part_depth", type=int, default=0,
                   help="--dense_parts: FAH's d6-style part head with this many 3x3 convs and its own queries (0 = off)")
    p.add_argument("--dense_hr", action="store_true",
                   help="dense head: the 70 MHR keypoints' 2D from a 2x finer heatmap (zero-initialised refinement)")
    p.add_argument("--dense_residual", action="store_true",
                   help="dense 3D = zero-initialised offsets on the (detached) parametric mesh's points")
    p.add_argument("--dense_fit_loss", default="",
                   help="one-pass SMPL fitter checkpoint: losses on its fit of the dense surface (w dense_fit_j14 / "
                        "dense_fit_v / dense_fit_j24; fit in the loop)")
    p.add_argument("--dense_remap", action="store_true",
                   help="--init with a different --dense_head size: keep the trained point queries (MHR keypoints, "
                        "J14, surface points in both sets) and start new surface points from their nearest old one")
    p.add_argument("--kp2d_head", action="store_true",
                   help="fabd: independent 2D keypoint head (residual on the projected mesh keypoints, 2D labels)")
    p.add_argument("--teacher_labels", default="",
                   help="read t_* labels from <tar>.<name>.npz sidecars where present (e.g. FAB-H's labels)")
    p.add_argument("--aist_camera_gt", action="store_true",
                   help="use audited AIST camera-frame cm joints and disable its placeholder MHR mesh labels")
    p.add_argument("--aist_geometry_gate", action="store_true",
                    help="mask grossly implausible AIST 3D targets with fixed teacher-independent geometry checks")
    p.add_argument("--w", nargs="*", default=[], help="loss-weight overrides, e.g. smpl_j14=10 kp3d=3")
    p.add_argument("--arch", default="fab", choices=["fab", "fabd", "fabt"],
                   help="fab: decoder-free image model; fabd: SAM 3D Body's decoder on the student backbone")
    p.add_argument("--decoder_lr", type=float, default=2e-5, help="fabd: LR of the pretrained decoder / heads")
    p.add_argument("--feat_w", type=float, default=1.0, help="fabd: weight of the teacher-feature loss")
    p.add_argument("--feature_teacher", default="",
                   help="fabd: fine-tuned fabt checkpoint for frozen feature targets (default: original SAM)")
    p.add_argument("--grad_ckpt", action="store_true", help="fabd: gradient checkpointing in the student ViT")
    p.add_argument("--ft_blocks", type=int, default=8, help="fabt: trainable last blocks of SAM 3D Body's backbone")
    p.add_argument("--dec_layers", type=int, default=0, help="fabd: keep the first k decoder layers (0 = all 6)")
    p.add_argument("--feat_n", type=int, default=4, help="fabd: samples per GPU batch in the feature loss (0 = all)")
    p.add_argument("--pool_layers", type=int, default=2)
    p.add_argument("--iters", type=int, default=1, help="refinement passes of the pose token (1 = one shot)")
    p.add_argument("--clip", type=float, default=1.0, help="gradient-norm clip")
    p.add_argument("--hard_gamma", type=float, default=0.0,
                   help="per-sample hard-example weighting of the main 3D terms (0 = uniform)")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights (0 = off); evaluators prefer it")
    p.add_argument("--no_aug", action="store_true", help="no photometric / crop / flip augmentation (overfit tests)")
    p.add_argument("--bm_dir", default="", help="body_models dir (smpl/, mhr/): enables the SMPL loss of BEDLAM pools")
    p.add_argument("--wandb_project", default="", help="Weights & Biases project (default: off; needs `wandb login`)")
    p.add_argument("--run_name", default="")
    p.add_argument("--wandb_id", default="", help="Stable unique run ID for resumption across job attempts")
    p.add_argument("--seed", type=int, default=0, help="model, augmentation and data-order seed for paired studies")
    pre, _ = p.parse_known_args()
    os.environ.setdefault("FAB_HOME", str(paths.home()))  # for ${FAB_HOME} / ${FAB_DATA} in config strings
    os.environ.setdefault("FAB_DATA", str(paths.home() / "data"))
    if pre.config:  # the config file's values become the defaults (unknown keys are an error)
        import yaml

        cfg_file = yaml.safe_load(Path(pre.config).read_text()) or {}
        acts = {a.dest: a for a in p._actions}
        bad = sorted(k for k in cfg_file if k not in acts)
        if bad:
            p.error(f"{pre.config}: unknown keys {bad}")
        vals = {}
        for k, v in cfg_file.items():
            v = os.path.expandvars(v) if isinstance(v, str) else v
            if isinstance(v, str) and acts[k].type in (int, float):  # YAML reads e.g. 5e-4 as a string
                v = acts[k].type(v)
            vals[k] = v
        p.set_defaults(**vals)
    args = p.parse_args()
    if not (args.pools or args.pool_list):
        p.error("one of --pools / --pool_list is required (here or in --config)")
    args.teacher_ckpt = args.teacher_ckpt or str(paths.sam3d_dir() / "model.ckpt")
    if args.bm_dir == "auto":  # SMPL + MHR mapping files ($FAB_HOME/body_models)
        args.bm_dir = str(paths.body_models_dir())
    if args.foot_reg == "auto":  # the 23-joint regressor of the foot terms (docs/training.md)
        args.foot_reg = str(paths.home() / "extras" / "mhr23_regressor.npz")
    if args.args_from:  # --args_from: the checkpoint's saved args for everything not on the command line
        saved = torch.load(args.args_from, map_location="cpu", weights_only=False, mmap=True)["args"]
        saved = saved if isinstance(saved, dict) else vars(saved)
        given = set()
        for a in sys.argv[1:]:
            if a.startswith("--"):
                k = a[2:].split("=")[0]
                act = next((x for x in p._actions if f"--{k}" in x.option_strings), None)
                given.add(act.dest if act else k.replace("-", "_"))
        for k, v in saved.items():
            if k not in given and k != "args_from" and hasattr(args, k):
                setattr(args, k, v)
    if args.feature_teacher and (args.arch != "fabd" or args.feat_w <= 0):
        p.error("--feature_teacher requires --arch fabd and a positive --feat_w")
    if args.dense_25d_uvgrad_mesh and (args.dense_25d_uvgrad_vis or args.dense_25d <= 0):
        p.error("--dense_25d_uvgrad_mesh needs --dense_25d > 0 and excludes --dense_25d_uvgrad_vis")
    if args.dense_25d_uvgrad_feet and not args.dense_25d_uvgrad_mesh:
        p.error("--dense_25d_uvgrad_feet needs --dense_25d_uvgrad_mesh")
    if args.heel2d_vertex and not (args.dense_head and args.kp2d_labels):
        p.error("--heel2d_vertex needs --dense_head and --kp2d_labels")

    for kv in args.w:
        k, v = kv.split("=")
        WEIGHTS[k] = float(v)
    if not 0 <= WEIGHTS.get("teacher_body_on_gt", 1.0) <= 1:
        p.error("teacher_body_on_gt must be between zero and one")
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed)
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = get_config(str(Path(args.teacher_ckpt).parent / "model_config.yaml"))
    cfg.defrost()
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = str(Path(args.teacher_ckpt).parent / "assets" / "mhr_model.pt")
    cfg.freeze()
    teacher_backbone = None
    if args.arch == "fabd":  # SAM 3D Body's decoder / heads on a student backbone (fab/decoder_model.py)
        from fab.decoder_model import build_fab_decoder

        model, teacher_backbone = build_fab_decoder(args.teacher_ckpt, args.backbone, args.size, device=str(device),
                                                    dec_layers=args.dec_layers, pretrained=not args.no_pretrained,
                                                    teacher_weights=not args.no_teacher_weights)
        if args.no_teacher_weights:
            init_random_encoder(teacher_backbone)
        if args.feature_teacher:
            load_feature_teacher(teacher_backbone, args.feature_teacher)
            if rank == 0:
                print(f"Frozen feature teacher: {args.feature_teacher}", flush=True)
        if args.grad_ckpt:  # recompute the student ViT's activations in backward (ViT-L on 24 GB GPUs)
            model.body.backbone.vit.set_grad_checkpointing(True)
        head = model.body.head_pose
    elif args.arch == "fabt":  # fine-tune SAM 3D Body itself (its backbone's last --ft_blocks blocks + decoder / heads)
        from fab.decoder_model import build_fab_decoder

        model, tb = build_fab_decoder(args.teacher_ckpt, args.backbone, args.size, device=str(device),
                                      dec_layers=args.dec_layers, pretrained=False, teacher_weights=not args.no_teacher_weights)
        if args.no_teacher_weights:
            init_random_encoder(tb)
        tb = tb.float().train()
        blocks = tb.encoder.blocks
        for i, blk in enumerate(blocks):
            for q in blk.parameters():
                q.requires_grad_(i >= len(blocks) - args.ft_blocks)
        for name, q in tb.named_parameters():
            if not name.startswith("encoder.blocks."):
                q.requires_grad_(name.startswith("encoder.norm"))
        model.body.backbone = tb
        model.body.backbone_dtype = torch.float32
        teacher_backbone = None
        head = model.body.head_pose
    else:
        model = FABImage(cfg, backbone=args.backbone, size=args.size, pool_layers=args.pool_layers,
                         iters=args.iters)
        left = model.load_teacher_heads(args.teacher_ckpt)
        if rank == 0:
            print("teacher heads loaded; not initialised:", left, flush=True)
        model = model.to(device)
        head = model.head_pose
    head.enable_fast()
    if args.kp2d_head:
        model.add_kp2d_head().to(device)
    g1_kin = None
    if args.g1_head:
        from fab.retarget.g1 import G1Kinematics

        model.add_g1_head(joints_input=args.g1_joints).to(device)
        from fab.retarget.g1_files import urdf

        g1_kin = G1Kinematics(paths.require(urdf(), "the Unitree G1 URDF", "Run scripts/download_g1.sh.")).to(device)
        if args.g1_only:
            for n, q in model.named_parameters():
                q.requires_grad_(n.startswith("g1_head."))
    num_verts = head.mhr.character_torch.mesh.rest_vertices.shape[0]
    subset = torch.randperm(num_verts, generator=torch.Generator().manual_seed(0))[: args.vert_subset].sort().values
    rig = SubsetMHR(head.mhr, subset).to(device)
    smpl_target = None
    if args.bm_dir:
        dense_idx = ()
        if args.dense_head:
            from bench.body_models import SMPL as _SMPL
            from fab.dense_head import symmetric_subset

            tmpl = _SMPL(Path(args.bm_dir) / "smpl" / "SMPL_NEUTRAL.pkl").v_template.numpy()
            dense_idx, dense_mirror = symmetric_subset(tmpl, args.dense_head)  # closed under left <-> right
            if args.dense_extra:  # named points (sole, fingertips) appended after the set (fab/dense_points.py)
                from fab.dense_points import extra_points

                n0 = len(dense_idx)
                _w = _SMPL(Path(args.bm_dir) / "smpl" / "SMPL_NEUTRAL.pkl").weights.numpy()
                dense_idx, dense_mirror = extra_points(tmpl, _w, dense_idx, args.dense_extra.split(","))
                if rank == 0:
                    print(f"dense extra points {args.dense_extra}: {n0} -> {len(dense_idx)} surface points", flush=True)
            model.add_dense_head(dense_idx, residual=args.dense_residual, parts=args.dense_parts,
                                 xyz_layers=args.dense_xyz_layers, part_depth=args.part_depth,
                                 hr=70 if args.dense_hr else 0, mask_depth=args.mask_depth,
                                 mask_query=args.mask_query, xyz_dim=args.dense_xyz_dim, xyz_mem=args.dense_xyz_mem,
                                 p25d=args.dense_25d, p25d_border=args.dense_25d_border,
                                 p25d_vis=args.dense_25d_vis_gate, ray_in=args.dense_ray_in,
                                 uv_off=args.dense_uv_off).to(device)
            model.dense_head.p25d_uvgrad = args.dense_25d_uvgrad
            if args.dense_25d_vis_gate and not (args.dense_25d > 0 and args.vis_head):
                raise ValueError("--dense_25d_vis_gate needs --dense_25d and --vis_head")
        full = (args.dense_parts or WEIGHTS.get("part_region", 0) > 0 or WEIGHTS.get("part_cover", 0) > 0
                or WEIGHTS.get("mask_sil", 0) > 0)
        full = full or WEIGHTS.get("dense_vis_gate", 0) > 0  # visibility of the dense points: whole-mesh z-buffer
        smpl_target = SMPLTarget(head.mhr, args.bm_dir, extra_idx=dense_idx, full=full,
                                 j24=WEIGHTS.get("smpl_j24", 0) > 0).to(device)
        if WEIGHTS.get("j24_feet", 0) > 0:  # the EMDB-1 gap vs SAM 3D Body is in the knees / ankles / feet (SMPL 4 5 7 8 10 11)
            w24 = torch.ones(24, device=device)
            w24[[4, 5, 7, 8, 10, 11]] = WEIGHTS["j24_feet"]
            smpl_target.register_buffer("j24_w", w24, persistent=False)
        if any(WEIGHTS.get(k, 0) > 0 for k in FOOT_TERMS) and args.foot_reg and not Path(args.foot_reg).exists():
            raise FileNotFoundError(f"--foot_reg {args.foot_reg} not found (docs/training.md: the 23-joint regressor); "
                                    "pass --foot_reg '' to use the SMPL foot vertices instead (not the released recipe)")
        if any(WEIGHTS.get(k, 0) > 0 for k in FOOT_TERMS) and args.foot_reg and Path(args.foot_reg).exists():
            # rows: the MHR70 body keypoints 0-20, 41, 62 (so MHR ids 0-20 index them directly)
            smpl_target.foot_reg = torch.from_numpy(np.load(args.foot_reg)["reg"]).float().to(device)
        if args.dense_head:
            smpl_target.register_buffer("extra_mirror", torch.as_tensor(dense_mirror, device=device), persistent=False)
            if args.heel2d_vertex:  # the matched SMPL heel vertices (l, r) as affine combinations of their 8 nearest surface points
                hc, hw = heel_weights(tmpl, dense_idx, [int(x) for x in args.heel2d_vertex.split(",")])
                smpl_target.register_buffer("heel_ch", torch.as_tensor(hc, device=device), persistent=False)
                smpl_target.register_buffer("heel_w", torch.as_tensor(hw, dtype=torch.float32, device=device), persistent=False)
                if rank == 0:
                    print(f"heel 2D labels -> SMPL {args.heel2d_vertex}: dense channels {hc.tolist()}, weights "
                          f"{np.round(hw, 3).tolist()} (template residual 0)", flush=True)
            if args.dense_foot_untrusted < 1.0 or args.foot_off_tars:  # foot surface points: --dense_foot_untrusted
                # (3D full weight only on --foot3d_sources) / --foot_off_tars (no foot points on those tars)
                from fab.dense_points import foot_positions

                fpos = foot_positions(tmpl, smpl_target.smpl.weights.cpu().numpy(), dense_idx)
                smpl_target.register_buffer("foot_pos", torch.as_tensor(fpos, device=device), persistent=False)
                smpl_target.foot_untrusted = args.dense_foot_untrusted
                if rank == 0:
                    print(f"dense foot points: {len(fpos)} of {len(dense_idx)}, 3D weight {args.dense_foot_untrusted} "
                          f"outside {args.foot3d_sources}; off on {args.foot_off_tars or '-'}", flush=True)
            if args.vis_gate_2d:  # (occluded, out of crop) weights of the 2D terms (dense_losses)
                smpl_target.vis_gate_2d = tuple(float(x) for x in args.vis_gate_2d.split(","))
        if "heat" in args.gt2d_gate.split(","):  # labels outside the crop: not in the heatmap 2D term (dense_losses)
            smpl_target.heat2d_in_crop = True
        if args.kp2d_labels:  # the foot 2D labels' per-source share (logged)
            smpl_target.kp2d_names = [s.split("@")[0].rstrip("!") for s in args.kp2d_labels.split(",") if s]
        if args.foot_pitch_gate:  # GT foot 3D (pitch) only on --foot3d_sources crops, in every 3D GT term
            from bench.mhr_smpl import MHRToSMPL
            from fab.dense_points import foot_positions

            smpl_target.foot_pitch_gate = True  # kp3d_gt / kp3d_gt_rot heels / toes
            tm, wts = smpl_target.smpl.v_template.cpu().numpy(), smpl_target.smpl.weights.cpu().numpy()
            rand_idx = smpl_target.sub[smpl_target.in_rand].cpu().numpy()
            smpl_target.register_buffer("foot_rand", torch.as_tensor(foot_positions(tm, wts, rand_idx), device=device),
                                        persistent=False)  # smpl_v
            foot_all = foot_positions(tm, wts, np.arange(len(tm)))
            mhr_ids = torch.unique(MHRToSMPL(Path(args.bm_dir) / "mhr").vert_ids[torch.as_tensor(foot_all)].flatten())
            smpl_target.register_buffer("rig_foot", torch.nonzero(torch.isin(subset.cpu(), mhr_ids)).flatten().to(device),
                                        persistent=False)  # verts_gt
            if args.dense_head:  # dense foot points (as --dense_foot_untrusted 0)
                fpos = foot_positions(tmpl, wts, dense_idx)
                smpl_target.register_buffer("foot_pos", torch.as_tensor(fpos, device=device), persistent=False)
                smpl_target.foot_untrusted = 0.0
            if rank == 0:
                print(f"foot pitch gate: GT foot 3D only on {args.foot3d_sources}: {len(smpl_target.foot_rand)} smpl_v "
                      f"vertices, {len(smpl_target.rig_foot)} MHR subset vertices, MHR70 15-20", flush=True)
        if args.dense_fit_loss:  # fit in the loop: the frozen one-pass fitter (fab/smpl_fitter.py) on the surface
            from fab.smpl_fitter import load_fitter

            fit_net = load_fitter(args.dense_fit_loss, device)
            for q in fit_net.parameters():
                q.requires_grad_(False)
            st_fit = torch.load(args.dense_fit_loss, map_location="cpu", weights_only=False)
            if not np.array_equal(np.asarray(st_fit["idx"]), np.asarray(dense_idx)):
                raise ValueError("--dense_fit_loss: the fitter's points are not the dense head's surface points")
            smpl_target.fit_net = fit_net
            smpl_target.register_buffer("fit_idx", torch.as_tensor(np.asarray(dense_idx), device=device), persistent=False)
        if rank == 0:
            print(f"SMPL loss: {len(smpl_target.sub)} SMPL vertices from {smpl_target.rig_size} MHR vertices",
                  flush=True)
    if args.parts_only:
        for n, q in model.named_parameters():
            q.requires_grad_(n.startswith("dense_head.part_"))
    if args.dense_only:
        for n, q in model.named_parameters():
            q.requires_grad_(n.startswith("dense_head."))
    if args.mask_only:  # with --parts_only too: both separate heads train together, everything else frozen
        for n, q in model.named_parameters():
            q.requires_grad_(n.startswith("dense_head.mask_") or (args.parts_only and n.startswith("dense_head.part_")))
    if args.vis_head:
        model.add_vis_head(attn_dim=args.vis_attn, human=args.vis_human).to(device)
        if args.vis_only:
            for n, q in model.named_parameters():
                q.requires_grad_(n.startswith("vis_head."))
    if args.contact_head:
        model.add_contact_head(attn_dim=args.contact_attn, n_out=16 if args.contact_split else 8).to(device)
        if args.contact_only:
            for n, q in model.named_parameters():
                q.requires_grad_(n.startswith("contact_head."))

    is_vit = lambda n: (n.startswith("vit.") or n.startswith("body.backbone.vit.")  # noqa: E731
                        or (args.arch == "fabt" and n.startswith("body.backbone.")))
    # fabd / fabt: SAM 3D Body's pretrained decoder / heads get --decoder_lr, the new projection --lr
    is_dec = lambda n: (args.arch in ("fabd", "fabt") and not n.startswith("body.backbone.")  # noqa: E731
                        and not n.startswith(("g1_head.", "kp2d_head.", "dense_head.", "contact_head.", "vis_head.")))
    is_ray = lambda n: args.ray_lr_mult != 1.0 and n.startswith("dense_head.pt_ray.")  # noqa: E731
    is_uvo = lambda n: args.uv_off_lr_mult != 1.0 and n.startswith("dense_head.uv_off.")  # noqa: E731
    vit_params = [p_ for n, p_ in model.named_parameters() if is_vit(n) and p_.requires_grad]
    other = [p_ for n, p_ in model.named_parameters()
             if not is_vit(n) and not is_dec(n) and not is_ray(n) and not is_uvo(n) and p_.requires_grad]
    dec_params = [p_ for n, p_ in model.named_parameters() if is_dec(n) and p_.requires_grad]
    ray_params = [p_ for n, p_ in model.named_parameters() if is_ray(n) and p_.requires_grad]
    groups = [{"params": vit_params, "lr": args.backbone_lr}, {"params": other, "lr": args.lr}]
    if dec_params:
        groups.append({"params": dec_params, "lr": args.decoder_lr})
    if ray_params:  # --ray_lr_mult: the zero-initialised viewing-ray input of the points head at x mult the head LR
        groups.append({"params": ray_params, "lr": args.lr * args.ray_lr_mult})
    uvo_params = [p_ for n, p_ in model.named_parameters() if is_uvo(n) and p_.requires_grad]
    if uvo_params:  # --uv_off_lr_mult: the zero-initialised amodal 2D offset at x mult the head LR
        groups.append({"params": uvo_params, "lr": args.lr * args.uv_off_lr_mult})
    opt = torch.optim.AdamW(groups,
                            weight_decay=0.05)
    base_lrs = [g["lr"] for g in opt.param_groups]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    last = out_dir / "last.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        opt.load_state_dict(state["opt"])
        step = state["step"]
        if rank == 0:
            print(f"resumed from {last} at step {step}", flush=True)
    elif args.init:  # warm start: weights (EMA when kept) of another run, fresh optimizer / schedule
        st = torch.load(args.init, map_location="cpu", weights_only=False)
        w = dict(st["model"])
        w.update(st.get("ema") or {})
        own = model.state_dict()
        skipped = [k for k, v in w.items() if k in own and own[k].shape != v.shape]  # e.g. a re-shaped new head
        if args.dense_remap and "dense_head.point" in skipped and "dense_head.surface_idx" in w and smpl_target is not None:
            with torch.no_grad():  # more / other surface points: trained queries kept, new ones from the nearest old point
                old_idx, new_idx = w["dense_head.surface_idx"].long(), model.dense_head.surface_idx.cpu().long()
                old_pt, new_pt = w["dense_head.point"], own["dense_head.point"].detach().cpu().clone()
                n0 = old_pt.shape[0] - len(old_idx)  # MHR keypoints + J14
                tmpl = smpl_target.smpl.v_template.cpu()
                near = torch.cdist(tmpl[new_idx], tmpl[old_idx]).argmin(1)
                new_pt[:n0] = old_pt[:n0]
                new_pt[n0:] = old_pt[n0:][near]
                w["dense_head.point"] = new_pt
                skipped.remove("dense_head.point")
            if rank == 0:
                print(f"init: dense points remapped {len(old_idx)} -> {len(new_idx)} surface points "
                      f"({int(torch.isin(new_idx, old_idx).sum())} shared)", flush=True)
        w = {k: v for k, v in w.items() if k not in skipped}
        missing, unexpected = model.load_state_dict(w, strict=False)
        if skipped and rank == 0:
            print(f"init: {len(skipped)} shape-mismatched tensors left at their fresh values: {skipped[:4]}", flush=True)
        if args.dense_residual and "dense_head.residual" not in w and getattr(model, "dense_head", None) is not None:
            with torch.no_grad():  # absolute -> residual 3D: restart the offsets at zero (heatmaps / 2D kept)
                model.dense_head.residual.fill_(1)
                model.dense_head.xyz[2].weight.zero_()
                model.dense_head.xyz[2].bias.zero_()
        if (not args.dense_residual and int(w.get("dense_head.residual", torch.tensor(0)))
                and getattr(model, "dense_head", None) is not None):
            with torch.no_grad():  # residual -> absolute 3D (dense-first): fresh last layer, heatmaps / 2D kept
                model.dense_head.residual.fill_(0)
                model.dense_head.xyz[2].reset_parameters()
        new_pt = sorted({int(k.split(".")[3]) for k in missing if k.startswith("dense_head.pt.layers.")})
        if new_pt and getattr(model, "dense_head", None) is not None and model.dense_head.pt is not None:
            with torch.no_grad():  # point-transformer layers the init lacks: start as identity (zero residual branches)
                for i in new_pt:
                    lay = model.dense_head.pt.layers[i]
                    for m in (lay.self_attn.out_proj, lay.multihead_attn.out_proj, lay.linear2):
                        m.weight.zero_()
                        m.bias.zero_()
            if rank == 0:
                print(f"init: point-transformer layers {new_pt} start as identity", flush=True)
        if rank == 0:
            print(f"initialised from {args.init} (step {st.get('step')}); {len(missing)} missing, "
                  f"{len(unexpected)} unexpected keys", flush=True)
        if args.init_dense:  # the dense head of another run (same surface points; EMA when kept)
            sd = torch.load(args.init_dense, map_location="cpu", weights_only=False)
            wd = {**sd["model"], **(sd.get("ema") or {})}
            own = model.state_dict()
            wd = {k: v for k, v in wd.items() if k.startswith("dense_head.") and k in own and own[k].shape == v.shape}
            model.load_state_dict(wd, strict=False)
            if rank == 0:
                print(f"dense head from {args.init_dense}: {len(wd)} tensors", flush=True)
        if args.init_vis:  # a visibility head trained on a frozen model (a warm start)
            sv = torch.load(args.init_vis, map_location="cpu", weights_only=False)
            wv = {**sv["model"], **(sv.get("ema") or {})}
            own = model.state_dict()
            wv = {k: v for k, v in wv.items() if k.startswith("vis_head.") and k in own and own[k].shape == v.shape}
            model.load_state_dict(wv, strict=False)
            if rank == 0:
                print(f"visibility head from {args.init_vis}: {len(wv)} tensors", flush=True)
        if getattr(model, "vis_head", None) is not None and getattr(model.vis_head, "mlp_h", None) is not None:
            loaded = set(w) | (set(wv) if args.init_vis else set())
            if not any(k.startswith("vis_head.mlp_h.") for k in loaded):  # --vis_human from a rendered-only head
                with torch.no_grad():
                    model.vis_head.mlp_h.load_state_dict(model.vis_head.mlp.state_dict())
                if rank == 0:
                    print("visibility: the human classifier starts from the rendered-label one", flush=True)
    # EMA of the trainable weights (fp32, on every rank: DDP keeps the ranks identical); saved as "ema"
    ema = None
    if args.ema > 0:
        ema = {k: v.detach().float().clone() for k, v in trainable_state(model).items() if v.is_floating_point()}
        if last.exists() and state.get("ema"):
            ema = {k: v.to(device) for k, v in state["ema"].items()}
    ddp = model
    if distributed:
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

        # bf16 gradients and big buckets: about half the all-reduce time on slow interconnects
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=True,
                                                        bucket_cap_mb=100, gradient_as_bucket_view=True)
        ddp.register_comm_hook(None, default_hooks.bf16_compress_hook)

    tars = []  # "<glob>@N" repeats that pool's tars N times (oversampling, e.g. mined hard poses)
    if args.pool_list:
        tars = json.loads(Path(args.pool_list).read_text())
        if (not isinstance(tars, list) or not tars
                or any(not isinstance(path, str) or not Path(path).is_file() for path in tars)):
            raise ValueError("Pool list must contain existing tar paths")
    else:
        for item in args.pools.split(","):
            pattern, rep = (item.rsplit("@", 1) + ["1"])[:2] if "@" in item else (item, "1")
            tars += sorted(glob.glob(pattern)) * int(rep)
    if args.sidecar_rep:  # --sidecar_rep name=N: every tar with a <tar>.<name>.npz sidecar exactly N times
        nm_, n_ = args.sidecar_rep.split("=")
        has = {t for t in set(tars) if Path(f"{t[:-4]}.{nm_}.npz").exists()}
        tars = [t for t in tars if t not in has] + sorted(has) * int(n_)
        if rank == 0:
            print(f"--sidecar_rep: {len(has)} tars with a {nm_} sidecar at x{n_}", flush=True)
    tars = sorted(tars)
    if rank == 0:
        print(f"{len(tars)} pool tars", flush=True)
    if args.teacher_strict:  # one MHR teacher (--teacher_labels) only
        assert args.teacher_labels, "--teacher_strict needs --teacher_labels"
        assert not args.feature_teacher, "--teacher_strict: --feature_teacher set"
        # every tar's t_* come from the teacher: its <tar>.<teacher>.npz sidecar, or baked into the records (pools under an
        # /engine/ folder, with a manifest_keys.npz / manifest_crops.npz giving each crop's label source)
        miss = sorted({t for t in tars if "/engine/" not in t and not Path(f"{t[:-4]}.{args.teacher_labels}.npz").exists()})
        if rank == 0:
            print(f"--teacher_strict: {args.teacher_labels} only; {len(set(tars)) - len(miss)} / "
                  f"{len(set(tars))} tars carry its labels", flush=True)
        assert not miss, f"--teacher_strict: {len(miss)} tars without {args.teacher_labels} labels, e.g. {miss[:3]}"
    skip_pairs = {}
    if args.skip_pairs:
        for ln in Path(args.skip_pairs).read_text().splitlines():
            if ln.strip() and not ln.startswith("#"):
                d_, k_ = ln.strip().split("|", 1)
                skip_pairs.setdefault(d_, set()).add(k_)
        if rank == 0:
            print(f"--skip_pairs {args.skip_pairs}: {sum(map(len, skip_pairs.values()))} samples in "
                  f"{len(skip_pairs)} pool dirs", flush=True)
    if args.teacher_strict:  # baked-label pools: the crops another teacher labelled must all be in --skip_pairs
        for d_ in sorted({t.split("/pools/", 1)[-1].rsplit("/", 1)[0] for t in tars if "/engine/" in t}):
            root_ = Path(tars[0].split("/pools/", 1)[0]) / "pools" / d_
            mf = root_ / "manifest_keys.npz" if (root_ / "manifest_keys.npz").exists() else root_ / "manifest_crops.npz"
            with np.load(mf, allow_pickle=False) as z:
                lab = z["label"].astype(str)
            other = int(sum(not str(l_).startswith(args.teacher_labels) for l_ in lab))
            if rank == 0:
                print(f"--teacher_strict: {d_}: {other} / {len(lab)} crops labelled by another teacher, "
                      f"{len(skip_pairs.get(d_, ()))} skipped", flush=True)
            assert other == len(skip_pairs.get(d_, ())), f"--teacher_strict: {d_} has {other} non-{args.teacher_labels} crops"
    data = PoolDataset(tars, args.size, rank, world, train=True, seed=args.seed + step, flip=args.flip, aug=not args.no_aug,
                       rot=args.rot, rot_full=args.rot_full, g1_labels=args.g1_labels, teacher_labels=args.teacher_labels, aist_camera_gt=args.aist_camera_gt,
                       contact_labels=args.contact_labels if args.contact_head else "",
                       contact_any_labels=args.contact_any_labels if args.contact_head else "",
                       contact_knee_labels=args.contact_knee_labels if args.contact_head else "",
                       contact_phys_dev_max=args.contact_phys_dev_max, contact_phys_dev_pools=args.contact_phys_dev_pools,
                       aist_geometry_gate=args.aist_geometry_gate, min_in_frame=args.min_in_frame,
                       pseudo_3d_sources=tuple(x for x in args.pseudo_3d_sources.split(",") if x),
                       smpl_gt_off=tuple(x for x in args.smpl_gt_off.split(",") if x),
                       smpl_shape_off=tuple(x for x in args.smpl_shape_off.split(",") if x),
                       mask_labels=args.mask_labels, vis_labels=args.vis_labels if args.vis_head or args.vis_gate_2d
                       or WEIGHTS.get("foot_reproj_vis", 0) > 0 or args.dense_25d_uvgrad_vis else "",
                       mask_gate={k: float(v) for k, v in (kv.split("=") for kv in args.mask_gate.split(",") if kv)})
    if args.loose_box_p > 0:  # (p, zoom lo, zoom hi, shift) for PoolDataset._item
        data.loose_box = (args.loose_box_p, *[float(x) for x in args.loose_box_zoom.split(",")], args.loose_box_shift)
    data.wide_context = args.wide_context
    data.kp2d_labels = args.kp2d_labels
    data.foot_off_tars = tuple(s for s in args.foot_off_tars.split(",") if s)
    data.foot_off_keep_side = args.foot_off_keep_side
    data.gt2d_in_crop = args.gt2d_in_crop
    data.gt2d_gate = tuple(s for s in args.gt2d_gate.split(",") if s)
    data.gt2d_crop_tars = tuple(s for s in args.gt2d_crop_tars.split(",") if s)
    data.skip_pairs = skip_pairs
    if args.skip_takes:
        pats = [ln.strip() for ln in Path(args.skip_takes).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        data.skip_rx = re.compile("|".join(f"(?:{q})" for q in pats))
        if rank == 0:
            print(f"--skip_takes {args.skip_takes}: {len(pats)} patterns", flush=True)
    data.human_feet_2d = args.human_feet_2d
    data.sidecar_no_feet = args.sidecar_no_feet
    data.vis_human =args.vis_human and args.vis_head
    data.gt_fix_labels, data.gt_fix_min_deg = args.gt_fix_labels, args.gt_fix_min_deg
    data.gt_fix_alias = tuple(tuple(s.split("=", 1)) for s in args.gt_fix_alias.split(",") if "=" in s)
    data.gt_fix_tars = tuple(s for s in args.gt_fix_tars.split(",") if s)
    data.kp2d_alias = tuple(tuple(s.split("=", 1)) for s in args.kp2d_alias.split(",") if "=" in s)
    data.vis_alias = tuple(tuple(s.split("=", 1)) for s in args.vis_alias.split(",") if "=" in s)
    data.foot3d_sources = {s for s in args.foot3d_sources.split(",") if s}
    if args.neg_prompt_p > 0:  # (drop, jitter, group_p) for PoolDataset._item; loads the <tar>.others.npz sidecars
        data.neg_prompt = tuple(float(x) for x in args.neg_prompt_aug.split(","))
    if args.det_box_p > 0:  # loads the <tar>.detbox.npz sidecars
        data.det_box = args.det_box_p
    if args.trunc_aug:  # (p, lo, hi) for PoolDataset._item
        data.trunc = tuple(float(x) for x in args.trunc_aug.split(","))
        data.trunc_zoom_max = args.trunc_zoom_max
    loader = DataLoader(data, batch_size=args.batch, num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=True, prefetch_factor=4)

    wandb = None
    if rank == 0 and args.wandb_project:
        import wandb as _wandb

        wandb = _wandb
        wandb.init(project=args.wandb_project, entity=os.environ.get("WANDB_ENTITY"),
                   name=args.run_name or out_dir.name, config=vars(args), resume="allow",
                   id=args.wandb_id or out_dir.name)

    model.train()
    if args.dense_only:  # frozen body: deterministic backbone features (no drop-path)
        model.body.backbone.eval()
    t0 = time.time()
    ramp_w0 = {k_: WEIGHTS[k_] for k_ in ("dense_pj3d", "dense_pv3d", "dense_pmhr3d") if WEIGHTS.get(k_, 0) > 0}
    if args.dense_25d_ramp > 0 and rank == 0:
        print(f"--dense_25d_ramp {args.dense_25d_ramp}: p25d_mix 0 -> {args.dense_25d}, {ramp_w0} 0 -> full", flush=True)
    for b in loader:
        if step >= args.steps:
            break
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        if args.dense_25d_ramp > 0:  # the 2.5D lift (output share + its dense_p* terms) ramps to full over the first K steps
            r_ = min(1.0, max(1e-3, step / args.dense_25d_ramp))
            model.dense_head.p25d_mix = args.dense_25d * r_
            WEIGHTS.update({k_: v_ * r_ for k_, v_ in ramp_w0.items()})
        if args.heel2d_vertex:  # the sidecar (human / RTMW) heel 2D labels off the MHR70 heels, onto SMPL --heel2d_vertex
            b = heel_relabel(b)
        if args.p_occ > 0 or args.p_lowres > 0:
            b["img"] = batch_occlude(b["img"], args.p_occ, args.p_lowres)
        sil = None
        if (args.mask_prompt_p > 0 or args.p_paste > 0) and smpl_target is not None:  # person silhouettes from labels
            sil = label_silhouette(b, smpl_target, args.size)
            if args.p_paste > 0:
                b["img"] = paste_distractors(b["img"], sil, args.p_paste)
        for g, base in zip(opt.param_groups, base_lrs):
            warm = min(1.0, (step + 1) / args.warmup)
            g["lr"] = base * warm * 0.5 * (1 + math.cos(math.pi * min(step / args.steps, 1.0)))
        # per-point share of the lift's gradient into the heatmap 2D
        gate_, uvg = uv_grad_gate(args, b.get("gt_vis"), len(b["img"]), b["img"].device)
        if gate_ is not None:
            model.dense_head.uv_grad_gate = gate_
        with torch.autocast("cuda", dtype=torch.bfloat16):
            extra = {"affine": b["affine"]} if args.arch in ("fabd", "fabt") else {}
            if args.mask_prompt_p > 0 and sil is not None:  # the target's mask as the decoder's prompt, part of the time
                use = (torch.rand(len(sil), device=sil.device) < args.mask_prompt_p).float()
                extra["mask"], extra["mask_score"] = noisy_masks(sil) * use[:, None, None, None], use
            if args.box_prompt_p > 0 and "box_prompt" in b:  # the input box's extent as the decoder's mask prompt
                use = (torch.rand(len(b["box_prompt"]), device=b["img"].device) < args.box_prompt_p).float()
                extra["mask"] = box_masks(b["box_prompt"], b["img"].shape[-1]) * use[:, None, None, None]
                extra["mask_score"] = use
            if args.neg_prompt_p > 0 and "neg_boxes" in b:  # +1 the target's box, -1 the other people's, 0 both / neither
                S_ = b["img"].shape[-1]
                use = (torch.rand(len(b["box_prompt"]), device=b["img"].device) < args.neg_prompt_p).float()
                nb = b["neg_boxes"]
                neg = box_masks(nb.flatten(0, 1).nan_to_num(-1.0), S_).view(len(nb), -1, S_, S_).amax(1, keepdim=True)
                extra["mask"] = (box_masks(b["box_prompt"], S_) - neg) * use[:, None, None, None]
                extra["mask_score"] = use
            if ("+extent" in args.mask_query or "+corners" in args.mask_query) and "box_crop" in b:  # the pool's box
                extra["mask_box"] = b["box_crop"]
            out = ddp(b["img"], b["bbox_center"], b["bbox_scale"], b["ori_size"], b["cam_int"], return_vertices=False,
                      **extra)
        if args.contact_only:
            terms = contact_losses(out, b)
            total = WEIGHTS["contact"] * terms["contact"]
        else:
            total, terms = losses(model, out, b, args.size, WEIGHTS, rig, smpl_target, args.hard_gamma, g1_kin)
            if uvg is not None:  # --dense_25d_uvgrad_vis: share of keypoints whose 2D the 3D losses move
                terms["uvgrad_vis_frac"] = (uvg > 0).float().mean()
        if teacher_backbone is not None and args.feat_w > 0:  # token-level distillation of the teacher backbone
            n = args.feat_n or b["img"].shape[0]  # the teacher backbone costs ~2x the step: a subset of the batch
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                x = model.body.data_preprocess(b["img"][:n])
                t_feat = teacher_backbone(x.type(torch.bfloat16), extra_embed=None)
                t_feat = (t_feat[-1] if isinstance(t_feat, tuple) else t_feat).float()
            # The teacher above returns raw backbone features. image_embeddings already includes the learned
            # mask/no-mask prompt, so using it here would train the student to cancel decoder conditioning.
            s_feat = out["backbone_embeddings"][:n].float()
            terms["feat"] = ((s_feat - t_feat) ** 2).mean() / (t_feat ** 2).mean().clamp_min(1e-6)
            total = total + args.feat_w * terms["feat"]
        opt.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        step += 1
        if ema is not None:
            decay = min(args.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                cur = model.state_dict()
                torch._foreach_lerp_(list(ema.values()), [cur[k].float() for k in ema], 1 - decay)

        if rank == 0 and step % args.log_every == 0:
            rate = args.log_every * args.batch * world / (time.time() - t0)
            t0 = time.time()
            msg = {f"loss/{k}": float(v) for k, v in terms.items()}
            msg.update({"loss/total": float(total), "grad_norm": float(grad_norm), "lr": opt.param_groups[1]["lr"],
                        "samples_per_s": rate})
            print(f"step {step} " + " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in msg.items()), flush=True)
            if wandb:
                wandb.log(msg, step=step)
        if rank == 0 and (step % args.save_every == 0 or step == args.steps):
            state = {"model": trainable_state(model), "opt": opt.state_dict(), "step": step, "args": vars(args),
                     "ema": ema}
            torch.save(state, out_dir / "last.pt.tmp")
            os.replace(out_dir / "last.pt.tmp", last)
            if step % (args.save_every * 10) == 0 or step == args.steps or str(step) in args.save_steps.split(","):
                torch.save({"model": trainable_state(model), "ema": ema, "step": step, "args": vars(args)},
                           out_dir / f"step{step}.pt")
    if rank == 0:
        torch.save({"model": trainable_state(model), "ema": ema, "step": step, "args": vars(args)}, out_dir / "model.pt")
        print("done", step, flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
