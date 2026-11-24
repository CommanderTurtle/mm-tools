"""Stage-2: GT-supervised fine-tuning of the hand decoder (+ distilled student encoder), regularised by
output distillation from the frozen teacher.

Encoder: --student <train/distill.py checkpoint> (trainable, 256 crop) or --encoder teacher (the frozen
DINOv3-H+ backbone at --size; only the decoder trains). Losses on the final decoder layer, all in the
right-hand (mirrored) frame, on MANO-convention joints = fixed MHR->MANO regressor on the skinned mesh:
  2d  smooth-L1 of reprojected joints vs GT (crop pixels / crop size), labelled joints only
  3d  root-relative joints (m) where 3D GT exists (HOT3D-train), + root position (true intrinsics)
  kd  root-relative 3D + 2D vs the frozen teacher (native crop at --size) on every sample
Held-out 2D val = 2% of each labelled source (by source image hash); plus FreiHAND (decoder-in-the-loop).

    torchrun --nproc_per_node "${TRAIN_PROCESSES:?set TRAIN_PROCESSES}" -m train.finetune --batch_size "${TRAIN_BATCH_SIZE:?set TRAIN_BATCH_SIZE}" --crops $DATA_ROOT/crops --sources hot3d_lab,coco_wb,cmu_lab \
        --student $CKPT_ROOT/distill/<run>/best.pt --out $CKPT_ROOT/finetune/<run> --run_name <run>
"""
import argparse
import glob
import json
import math
import os
import time
import zlib

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train.distill import CONTEXT, CROP, eval_freihand, photometric, seed_worker

VAL_MOD = 50  # 1 / VAL_MOD of each source (by source image) is held out


def default_K(h, w, is_right):
    f = (h ** 2 + w ** 2) ** 0.5
    cx = w / 2.0 if is_right else w - 1 - w / 2.0  # the estimator mirrors the default K for left hands
    return np.array([[f, 0, cx], [0, f, h / 2.0], [0, 0, 1]], np.float64)


class LabeledCrops(Dataset):
    """train.build_crops / build_hot3d_crops pools; returns a size x size view (random square window of
    the 640 crop, no rotation so the camera model stays exact), its affine from the (mirrored) original
    image, intrinsics and GT in view pixels / camera metres."""

    def __init__(self, root, sources, size=384, aug=True, split="train", min_mask_score=0.0, arm_sources=(), p_wrong=0.0,
                 box_jitter=1.0, min_crop=0.0, p_shadow=0.0, raw_tip_sources=(), raw_tips=False, tip_shift=0.0, tip_mask_ext=0.0, parts=False, part_sources=None, part_routes=None, tip_from_parts=0.0, part_dirs=None,
                 joint_mask=None, no3d=()):
        self.jmask = {k: joint_mask[s] for k, s in enumerate(sources) if s in (joint_mask or {})}  # per pool: joints without 2D / 3D targets
        self.no3d = {k for k, s in enumerate(sources) if s in no3d}  # pools trained 2D-only (--no3d_sources)
        self.tip_mask_ext = tip_mask_ext  # > 0: labelled tips moved out to where the hand mask ends (fah.tips), at most x the distal phalanx
        self.tip_from_parts = tip_from_parts  # > 0: visible tip ends from the union of the part labels (3D pools without masks; dense targets only)
        self.parts = parts  # load <pool>/masks_parts/ (hand part segmentation, train.part_masks)
        self.part_src = None if not part_sources else {k for k, s in enumerate(sources) if s in part_sources}  # pools whose part labels are used
        self.part_dir = {k: (part_dirs or {}).get(s, "masks_parts") for k, s in enumerate(sources)}  # per pool: label version dir
        self.part_ok = None  # part_routes: only labels made by these train.part_masks routes (e.g. the silhouette-clipped ones)
        if parts and part_routes:
            self.part_ok = set()
            for k, s in enumerate(sources):
                for f in glob.glob(os.path.join(root, s, self.part_dir[k] + "_index.*.jsonl")):
                    self.part_ok.update(os.path.join(s, q["file"]) for q in map(json.loads, open(f)) if q.get("route") in part_routes)
        self.raw_tips, self.tip_shift = raw_tips, tip_shift  # fingertip-end convention arms (see --raw_tip_sources)
        self.raw = {}  # per raw-tip source: crop file -> the human-clicked tips (5, 3) of its BASE_POOL (before the _kin shift)
        self.p_wrong = p_wrong  # prob. of a mirrored "wrong handedness" view (verification head)
        self.p_shadow = p_shadow  # prob. of a hand-shaped cast shadow on a negative view (still a negative)
        self.box_jitter = box_jitter  # x the default scale / shift jitter (detector boxes vs label boxes)
        self.min_crop = min_crop  # tall label boxes: view covers min_crop x the long side (HandEstimator min_crop)
        self.items, self.src = [], []
        self.ext = {}  # per source: external 3D KD targets (<pool>/ext_kd.npz: files, root-relative j3d (21) / v3d (778))
        for k, s in enumerate(sources):
            if s in raw_tip_sources:
                base = open(os.path.join(root, s, "BASE_POOL")).read().strip()
                self.raw[k] = {r["file"]: np.asarray(r["kp2d"], np.float64)[TIPS] for r in map(json.loads, open(os.path.join(root, base, "index.jsonl")))
                               if r.get("kp2d") is not None}
            kind = 1 if s in arm_sources else 0  # mask channel: 0 hand, 1 hand + arm (VISOR convention)
            ef = os.path.join(root, s, "ext_kd.npz")
            if os.path.exists(ef):
                z = np.load(ef)
                vf = os.path.join(root, s, "ext_kd_v3d.npy")  # big pools: vertices as a separate memory-mapped array
                v3 = np.load(vf, mmap_mode="r") if os.path.exists(vf) else z["v3d"]
                self.ext[k] = ({f: n for n, f in enumerate(z["files" if "files" in z.files else "file"].tolist())}, z["j3d"], v3)
            mscore = {}  # SAM pseudo-mask scores (train.sam_masks), if the pool has masks
            for mf in glob.glob(os.path.join(root, s, "masks_index.*.jsonl")):
                for line in open(mf):
                    m = json.loads(line)
                    mscore[m["file"]] = m["score"]
            qc = os.path.join(root, s, "mask_qc.jsonl")  # train.qc_masks: drop wrong-instance / broken masks
            if MASK_QC and os.path.exists(qc):
                for line in open(qc):
                    q = json.loads(line)
                    if (q["kp_in"] is not None and q["kp_in"] < MASK_QC[0]) or q["mesh_iou"] < MASK_QC[1]:
                        mscore.pop(q["file"], None)
            qf = glob.glob(os.path.join(root, s, "qc.jsonl")) + glob.glob(os.path.join(root, s, "qc.[0-9]*.jsonl"))
            keep = None  # qf: the train.qc_crops label QC (EgoDex; one file or its shards)
            if LABEL_QC and qf:
                keep = {q["file"] for f in qf for q in map(json.loads, open(f)) if q["err2d"] < LABEL_QC}
            for line in open(os.path.join(root, s, "index.jsonl")):
                r = json.loads(line)
                if keep is not None and r["file"] not in keep:
                    continue
                is_val = zlib.crc32(r.get("src", r["file"]).split(":")[0].encode()) % VAL_MOD == 0
                if split != "all" and (split == "val") != is_val:
                    continue
                r["path"] = os.path.join(root, s, r["file"])
                if mscore.get(r["file"], -1) >= min_mask_score and r["file"] in mscore:
                    r["mask_path"] = os.path.join(root, s, "masks", os.path.splitext(r["file"])[0] + ".png")
                    r["mask_kind"] = kind
                self.items.append(r)
                self.src.append(k)
        self.src = np.asarray(self.src)
        self.size, self.aug = size, aug
        self.mask_items = [i for i, r in enumerate(self.items) if "mask_path" in r and r.get("mask_kind", 0) == 0]

    def __len__(self):
        return len(self.items)

    def _shadow(self, view):
        """A hand-shaped cast shadow on a negative view (shadow_v1 type B false positives: a hand's shadow on a wall):
        the instance mask of a random hand crop under a random scale / rotation / shear / shift (+ mirror), blurred,
        darkening the view by 30-75 % (sometimes tinted)."""
        S = self.size
        m = cv2.imread(self.items[self.mask_items[np.random.randint(len(self.mask_items))]]["mask_path"], cv2.IMREAD_GRAYSCALE)
        if m is None:
            return view
        m = cv2.resize(m, (S, S)).astype(np.float32) / 255.0
        ang, sc, sh = np.random.uniform(-np.pi, np.pi), np.random.uniform(0.6, 1.2), np.random.uniform(-0.4, 0.4)
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) @ np.array([[1.0, sh], [0.0, 1.0]]) * sc
        t = S / 2 - R @ np.array([S / 2, S / 2]) + np.random.uniform(-0.15, 0.15, 2) * S
        m = cv2.warpAffine(m, np.hstack([R, t[:, None]]), (S, S))
        if np.random.rand() < 0.5:
            m = np.ascontiguousarray(m[:, ::-1])
        m = cv2.GaussianBlur(m, (0, 0), np.random.uniform(1.0, 6.0))
        tint = np.ones(3) if np.random.rand() < 0.6 else np.random.uniform(0.85, 1.1, 3)
        dark = 1 - np.random.uniform(0.3, 0.75) * m[..., None] * tint
        return np.clip(view.astype(np.float32) * dark, 0, 255).astype(np.uint8)

    def __getitem__(self, i):
        r = self.items[i]
        img = cv2.imread(r["path"])
        img = np.zeros((CROP, CROP, 3), np.uint8) if img is None else img[:, :, ::-1]
        side, c = CROP / CONTEXT, np.array([CROP / 2, CROP / 2])
        if self.min_crop > 0 and "box" in r:  # same tall-box widening as inference (teacher region = 0.9 x height)
            bw, bh = r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]
            ts = max(bw * 0.9 / 0.75, bh * 0.9) if bw > 0.75 * bh else bh * 0.9
            if bh > bw and ts < self.min_crop * bh:
                side *= self.min_crop * bh / ts
        if self.aug:
            side *= np.exp(np.random.uniform(np.log(0.85), np.log(1.15)) * self.box_jitter)
            c = c + np.random.uniform(-0.08, 0.08, 2) * self.box_jitter * side
        s = self.size / side
        V = np.array([[s, 0, self.size / 2 - s * c[0]], [0, s, self.size / 2 - s * c[1]]])
        view = cv2.warpAffine(np.ascontiguousarray(img), V, (self.size, self.size), flags=cv2.INTER_LINEAR)
        if self.aug and self.p_shadow > 0 and r.get("negative") and self.mask_items and np.random.rand() < self.p_shadow:
            view = self._shadow(view)
        if self.aug:
            view = photometric(view)
        M = np.asarray(r["M"], np.float64)
        A = np.concatenate([V[:, :2] @ M[:, :2], (V[:, :2] @ M[:, 2] + V[:, 2])[:, None]], 1)  # original -> view
        h, w = r["hw"]
        K = np.asarray(r["K"], np.float64) if "K" in r else default_K(h, w, r["is_right"])
        kp = np.zeros((21, 3))
        if "kp2d" in r:
            kp = np.asarray(r["kp2d"], np.float64)
            kp[:, :2] = kp[:, :2] @ V[:, :2].T + V[:, 2]
            kp[:, 2] *= (kp[:, 0] >= 0) & (kp[:, 0] < self.size) & (kp[:, 1] >= 0) & (kp[:, 1] < self.size)
        j3 = np.asarray(r["joints3d"], np.float64) if "joints3d" in r else np.zeros((21, 3))
        tip_raw = np.zeros((5, 3))
        raw = self.raw.get(int(self.src[i]))
        if raw is not None and r["file"] in raw:  # human-clicked tips in view pixels (the visible end of each finger)
            tip_raw = raw[r["file"]].copy()
            tip_raw[:, :2] = tip_raw[:, :2] @ V[:, :2].T + V[:, 2]
            tip_raw[:, 2] *= (tip_raw[:, 0] >= 0) & (tip_raw[:, 0] < self.size) & (tip_raw[:, 1] >= 0) & (tip_raw[:, 1] < self.size)
            if self.raw_tips:
                kp[TIPS] = np.where(tip_raw[:, 2:] > 0, tip_raw, kp[TIPS])
        elif self.tip_shift:  # every other label: tips moved out along the distal phalanx (MANO tip vertex -> finger end)
            ok = (kp[TIPS, 2] > 0) & (kp[DIPS, 2] > 0)
            kp[TIPS, :2] = np.where(ok[:, None], extend_tips(kp[:, :2], self.tip_shift)[TIPS], kp[TIPS, :2])
            if "joints3d" in r:
                j3 = extend_tips(j3, self.tip_shift)
        v3 = np.load(os.path.join(os.path.dirname(r["path"]), os.path.basename(r["verts_file"]))).astype(np.float64) \
            if "verts_file" in r else np.zeros((778, 3))
        parts, has_parts = np.zeros((self.size, self.size), np.uint8), 0.0
        pf = os.path.join(r["path"][: -len(r["file"])], self.part_dir[int(self.src[i])], os.path.splitext(r["file"])[0] + ".png")  # = train.part_masks
        if self.parts and (self.part_src is None or int(self.src[i]) in self.part_src) and \
                (self.part_ok is None or os.path.join(os.path.basename(r["path"][: -len(r["file"]) - 1]), r["file"]) in self.part_ok) and os.path.exists(pf):  # 0 bg, 1 palm, 2-16 fingers, 255 ignore
            pm = cv2.imread(pf, cv2.IMREAD_GRAYSCALE)
            if pm is not None:
                parts = cv2.warpAffine(pm, V, (self.size, self.size), flags=cv2.INTER_NEAREST, borderValue=255)
                has_parts = 1.0
        mask, has_mask = np.zeros((self.size, self.size), np.uint8), 0.0
        if "mask_path" in r:
            m = cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE)
            if m is not None:
                mask = cv2.warpAffine(m, V, (self.size, self.size), flags=cv2.INTER_LINEAR)
                has_mask = 1.0
        negative = bool(r.get("negative", False))
        wrong = (not negative) and self.aug and np.random.rand() < self.p_wrong
        if wrong:  # same hand seen with the opposite handedness claim: a consistent mirrored camera, no GT
            S = self.size
            Fv = np.array([[-1.0, 0, S - 1], [0, 1, 0], [0, 0, 1]])
            Fw = np.array([[-1.0, 0, w - 1], [0, 1, 0], [0, 0, 1]])
            A = (Fv @ np.vstack([A, [0, 0, 1]]) @ Fw)[:2]
            view = np.ascontiguousarray(view[:, ::-1])
            K = K.copy()
            K[0, 2] = w - 1 - K[0, 2]
        if wrong or negative:
            kp[:] = 0
            has_mask = has_parts = 0.0
        tip_ext = np.zeros(5)
        if self.tip_mask_ext and has_mask > 0 and r.get("mask_kind", 0) == 0:  # visible finger ends from the hand-only mask
            from fah.tips import extend_tips as mask_extend

            ok, k2 = (kp[TIPS, 2] > 0) & (kp[DIPS, 2] > 0), kp[:, :2]
            if not ok.any() and "joints3d" in r and (j3[:, 2] > 0.05).all():  # 3D-only labels (EgoDex): their projection
                uvo = (j3 @ K.T)[:, :2] / j3[:, 2:]
                k2 = uvo @ A[:, :2].T + A[:, 2]
                inv = lambda q: (q[:, 0] >= 0) & (q[:, 0] < self.size) & (q[:, 1] >= 0) & (q[:, 1] < self.size)
                ok = inv(k2[TIPS]) & inv(k2[DIPS])
            if ok.any():
                new, ext = mask_extend(k2, mask > 127, max_ext=self.tip_mask_ext)
                tip_ext = np.where(ok, ext, 0.0)
                kp[TIPS, :2] = np.where((tip_ext[:, None] > 0) & (kp[TIPS, 2:] > 0), new[TIPS], kp[TIPS, :2])
                if "joints3d" in r:  # 3D tips by the same fraction of the distal phalanx
                    j3[TIPS] = j3[TIPS] + tip_ext[:, None] * (j3[TIPS] - j3[DIPS])
        if self.tip_from_parts and has_parts > 0 and not tip_ext.any():  # silhouette-clipped part labels as the hand mask: tip_ext only
            from fah.tips import extend_tips as mask_extend  # (kp / joints keep their convention; --tip_mask_dense stretches the surfaces)

            ok, k2 = (kp[TIPS, 2] > 0) & (kp[DIPS, 2] > 0), kp[:, :2]
            if ok.any():
                _, ext = mask_extend(k2, (parts > 0) & (parts < 255), max_ext=self.tip_from_parts)
                tip_ext = np.where(ok, ext, 0.0)
        ej, ev, he = np.zeros((21, 3)), np.zeros((778, 3)), 0.0
        e = self.ext.get(int(self.src[i]))
        if e is not None and r["file"] in e[0] and not (wrong or negative):
            n = e[0][r["file"]]
            ej, ev, he = e[1][n], e[2][n], 1.0
        jw = np.ones(21)
        js = self.jmask.get(int(self.src[i]))
        if js:  # --joint_mask_sources: no 2D / 3D target for these joints (teacher KD stays)
            kp[js, 2] = 0
            jw[js] = 0
        g3 = int(self.src[i]) not in self.no3d  # --no3d_sources: no GT 3D joints / mesh (the dense points get the teacher prior)
        f32 = lambda x: torch.as_tensor(np.asarray(x, np.float32))
        return dict(img=torch.from_numpy(np.ascontiguousarray(view)).permute(2, 0, 1), A=f32(A), K=f32(K),
                    ext_j=f32(ej), ext_v=f32(ev), has_ext=torch.tensor(he),
                    wh=f32([w, h]), kp2d=f32(kp), has3d=torch.tensor(float("joints3d" in r and g3 and not (wrong or negative))),
                    j3d=f32(j3 if g3 else np.zeros_like(j3)), present=torch.tensor(0.0 if negative else 1.0), side_ok=torch.tensor(0.0 if wrong else 1.0),
                    src=torch.tensor(int(self.src[i])), mask=torch.from_numpy(mask), has_mask=torch.tensor(has_mask),
                    tip_raw=f32(tip_raw * (0.0 if (wrong or negative) else 1.0)), tip_ext=f32(tip_ext),
                    parts=torch.from_numpy(np.ascontiguousarray(parts)), has_parts=torch.tensor(has_parts),
                    mask_kind=torch.tensor(r.get("mask_kind", 0)), v3d=f32(v3 if g3 else np.zeros_like(v3)),
                    has_v=torch.tensor(float("verts_file" in r and g3 and not (wrong or negative))), j3w=f32(jw))


def make_batch(b, S):
    """Model batch in the HandEstimator._make_batch layout for views of size S (A: original -> view)."""
    B = b["img"].shape[0]
    A = b["A"]
    sc = A[:, 0, 0]
    center = torch.stack([(S / 2 - A[:, 0, 2]) / sc, (S / 2 - A[:, 1, 2]) / sc], 1)
    side = (S / sc)[:, None].expand(B, 2)
    dev = A.device
    return dict(
        img=b["img_f"][:, None], img_size=torch.full((B, 1, 2), float(S), device=dev), ori_img_size=b["wh"][:, None],
        bbox_center=center[:, None], bbox_scale=side[:, None].contiguous(),
        bbox=torch.cat([center - side / 2, center + side / 2], 1)[:, None], affine_trans=A[:, None],
        cam_int=b["K"], mask=torch.zeros(B, 1, 1, S, S, device=dev), mask_score=torch.zeros(B, 1, device=dev),
        mask_prompt=False, img_unit_range=True)


def project(pc, K, A):
    """Camera-frame points (B, N, 3) -> view pixels via intrinsics K (B, 3, 3) and original -> view affine A."""
    uv = pc[..., :2] / pc[..., 2:].clamp(min=1e-4) * torch.stack([K[:, 0, 0], K[:, 1, 1]], 1)[:, None] + K[:, :2, 2][:, None]
    return uv @ A[:, :, :2].mT + A[:, :, 2][:, None]


def fit_translation(X, uv, K, A, w, tz):
    """Translation t (B, 3) with the given depth tz (B,) whose x / y best reproject root-relative points X (B, N, 3)
    onto view pixels uv (B, N, 2) with weights w (B, N) (closed form, in the original image; view -> original via A)."""
    Ai = torch.linalg.inv(torch.cat([A, torch.tensor([0.0, 0.0, 1.0], device=A.device).expand(A.shape[0], 1, 3)], 1))
    uo = uv @ Ai[:, :2, :2].mT + Ai[:, :2, 2][:, None]
    f, c = torch.stack([K[:, 0, 0], K[:, 1, 1]], 1)[:, None], K[:, :2, 2][:, None]
    txy = (uo - c) / f * (X[..., 2:] + tz[:, None, None]) - X[..., :2]  # per point: fx (X + tx) = (u - cx) (Z + tz)
    txy = (txy * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1e-6)
    return torch.cat([txy, tz[:, None]], 1)


class ManoJoints(nn.Module):
    """MANO-convention joints from the hand sub-mesh (the estimator's sparse convex regressor)."""

    def __init__(self, est):
        super().__init__()
        self.register_buffer("W", torch.as_tensor(est.mano_regressor, dtype=torch.float32), False)
        self.register_buffer("idx", torch.as_tensor(est.mano_regressor_vert_idx, dtype=torch.long), False)

    def forward(self, out, K, A):
        j = torch.einsum("jv,bvc->bjc", self.W, out["pred_vertices"][:, self.idx])  # (B, 21, 3) model frame
        jc = j + out["pred_cam_t"][:, None]
        uv = jc[..., :2] / jc[..., 2:].clamp(min=1e-4) * torch.stack([K[:, 0, 0], K[:, 1, 1]], 1)[:, None] \
            + K[:, :2, 2][:, None]
        uv = uv @ A[:, :, :2].mT + A[:, :, 2][:, None]  # original image -> view pixels
        return jc, uv


class Trainable(nn.Module):
    def __init__(self, model, encoder, train_backbone=False):
        super().__init__()
        self.model, self.encoder, self.train_backbone = model, encoder, train_backbone

    def forward(self, batch, x_enc, emb=None):
        if self.encoder is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb = self.encoder(x_enc)
            emb = emb.float()
        elif self.train_backbone:  # partially trainable fp32 teacher backbone, bf16 autocast forward
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb = self.model.encode(batch)
            emb = emb.float()
        return self.model(batch, image_embeddings=emb)


def mask_loss(logits, target, w, kind=None):
    """BCE + soft Dice on (B, h, w) logits -- or (B, K, h, w) with per-sample channel `kind` -- vs (B, H, W)
    uint8 masks (area-downsampled), weighted per sample."""
    if logits.ndim == 4:
        logits = logits.gather(1, kind.long().view(-1, 1, 1, 1).expand(-1, 1, *logits.shape[-2:]))[:, 0]
    t = F.adaptive_avg_pool2d(target[:, None].float() / 255.0, logits.shape[-2:])[:, 0]
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none").mean((1, 2))
    p = logits.sigmoid()
    dice = 1 - (2 * (p * t).sum((1, 2)) + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)
    return ((bce + dice) * w).sum() / w.sum().clamp(min=1.0)


def smooth_l1(x, y, w, beta, norm=None):
    loss = F.smooth_l1_loss(x, y, reduction="none", beta=beta).sum(-1)
    return (loss * w).sum() / (w.sum() if norm is None else norm).clamp(min=1.0)


TIPS = [4, 8, 12, 16, 20]
DIPS = [3, 7, 11, 15, 19]


def part_seg_loss(logits, parts, has, finger_weight=1.0):
    """Weighted pixel CE, with differentiable zero for missing/all-ignored labels (including on a DDP rank)."""
    target = F.interpolate(parts[:, None].float(), size=logits.shape[-2:], mode="nearest")[:, 0].long()
    target = target.masked_fill(has[:, None, None] <= 0, 255)
    valid = target != 255
    weight = logits.new_ones(17)
    weight[2:] = finger_weight
    denom = (weight[target.masked_fill(~valid, 0)] * valid).sum()
    total = F.cross_entropy(logits, target, weight=weight, ignore_index=255, reduction="sum")
    return total / denom.clamp(min=torch.finfo(logits.dtype).tiny)


def part_loss(uv, parts, has, pid, S, vis=None, res=96, sigmas=(1.5, 4.0, 12.0), n_cover=24):
    """Part-aware alignment of the dense surface points uv (B, N, 2, view px; pid (N,): 1 palm, 2-16 finger segments) with
    the part map (B, S, S; 0 background, 255 ignore), on samples with part labels (has) and points the view shows (vis):
    region term = a point outside its own part's region is pulled towards it (1 - the region's indicator blurred at several
    scales, sampled at the point; zero anywhere inside the region, so points are not drawn to the region's centre);
    cover term = sampled pixels of every labelled part region to the nearest visible point of that part (view-normalised,
    smooth L1), so the points span their region (e.g. the distal points reach the visible fingertip end).
    -> (region, cover)"""
    B, N = uv.shape[:2]
    vis = torch.ones(B, N, device=uv.device) if vis is None else vis
    lab = F.interpolate(parts[:, None].float(), size=(res, res), mode="nearest")[:, 0].long()
    ign = (lab == 255).float()
    oh0 = (F.one_hot(lab.clamp(max=16), 17).permute(0, 3, 1, 2).float() * (1 - ign[:, None]))
    # multi-scale blur: the wide scales pull points that are far from their part, the narrow one places them precisely
    oh = sum(gaussian_blur(oh0.reshape(B * 17, 1, res, res), s) for s in sigmas).reshape(B, 17, res, res) / len(sigmas)
    ig = gaussian_blur(ign[:, None], sigmas[0])
    g = (uv / S * 2 - 1)[:, None]  # (B, 1, N, 2) in [-1, 1]
    idx = pid[None, None].expand(B, 1, -1)
    own = F.grid_sample(oh, g, align_corners=False)[:, :, 0].gather(1, idx)[:, 0]  # (B, N)
    inside = F.grid_sample(oh0, g, mode="nearest", align_corners=False)[:, :, 0].gather(1, idx)[:, 0]
    present = oh0.flatten(2).sum(-1).gather(1, pid[None].expand(B, -1)) > 0
    # Validity weights are targets, not an escape route: differentiating the sampled ignore mask pulls points into
    # ignored regions to reduce their penalty. A part absent from the labels supplies no region target either.
    w = (1 - F.grid_sample(ig, g, align_corners=False)[:, 0, 0]) * has[:, None] * vis
    w = (w * present).detach()
    region = ((1 - own) * (1 - inside) * w).sum() / w.sum().clamp(min=1)
    flat, cover, n = lab.flatten(1), uv.new_zeros(()), 0
    for k in range(1, 17):
        sel = pid == k
        mk = (flat == k).float()
        pv = vis[:, sel]
        ok = (mk.sum(1) > 0) & (has > 0) & (pv.sum(1) > 0)
        if not sel.any() or not ok.any():
            continue
        pix = torch.multinomial(mk + (~ok).float()[:, None], n_cover, replacement=True)  # (B, n_cover) region pixels
        xy = torch.stack([pix % res, pix // res], -1).float().add(0.5) * (S / res)
        d = torch.cdist(xy, uv[:, sel].float()) + (1 - pv)[:, None] * 1e4  # hidden points do not count
        dm = d.min(-1).values / S  # (B, n_cover)
        hub = torch.where(dm < 0.02, 0.5 * dm ** 2 / 0.02, dm - 0.01)
        cover = cover + (hub.mean(1) * ok).sum() / ok.sum()
        n += 1
    return region, cover / max(n, 1)


COARSE = (0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6)  # part id -> coarse class: 0 background, 1 palm, 2-6 thumb .. pinky


def part_finger_loss(uv, parts, has, pid, S, vis=None, res=96, sigmas=(1.5, 4.0, 12.0)):
    """Finger-level term of the dense finger points (pid 2-16) on the coarse part map (palm + 5 fingers), next to
    part_loss's segment terms: part_loss's region term on the point's whole finger, so a point on another segment of its
    own finger costs nothing here and one on another finger (or the background) does (--w_part_finger). Validity weights
    as part_loss (detached: ignore, has, vis, finger present)."""
    B, N = uv.shape[:2]
    vis = torch.ones(B, N, device=uv.device) if vis is None else vis
    grp = torch.as_tensor(COARSE, device=uv.device)
    lab = F.interpolate(parts[:, None].float(), size=(res, res), mode="nearest")[:, 0].long()
    ign = (lab == 255).float()
    oh0 = F.one_hot(grp[lab.clamp(max=16)], 7).permute(0, 3, 1, 2).float() * (1 - ign[:, None])  # (B, 7, res, res)
    oh = sum(gaussian_blur(oh0.reshape(B * 7, 1, res, res), s) for s in sigmas).reshape(B, 7, res, res) / len(sigmas)
    ig = gaussian_blur(ign[:, None], sigmas[0])
    g = (uv / S * 2 - 1)[:, None]  # (B, 1, N, 2) in [-1, 1]
    cid = grp[pid]  # (N,) coarse class of each point
    idx = cid[None, None].expand(B, 1, -1)
    own = F.grid_sample(oh, g, align_corners=False)[:, :, 0].gather(1, idx)[:, 0]  # (B, N)
    inside = F.grid_sample(oh0, g, mode="nearest", align_corners=False)[:, :, 0].gather(1, idx)[:, 0]
    present = oh0.flatten(2).sum(-1).gather(1, cid[None].expand(B, -1)) > 0
    w = ((1 - F.grid_sample(ig, g, align_corners=False)[:, 0, 0]) * has[:, None] * vis * present * (pid >= 2)[None]).detach()
    return ((1 - own) * (1 - inside) * w).sum() / w.sum().clamp(min=1)


def extend_tips_by(x, ext):
    """Joints (B, 21, C) with fingertip f moved ext[:, f] x its distal phalanx further along DIP -> tip."""
    y = x.clone()
    y[:, TIPS] = x[:, TIPS] + ext[..., None] * (x[:, TIPS] - x[:, DIPS])
    return y


def stretch_distal(v, j, ext, distal_idx):
    """MANO-topology vertices (B, 778, 3) with each finger's distal-phalanx vertices stretched along DIP -> tip so that the tip
    moves ext[:, f] x the phalanx (0 at the DIP joint, 1 at the tip); j (B, 21, 3) the joints the axes come from."""
    v = v.clone()
    for f, idx in enumerate(distal_idx):
        a, t = j[:, DIPS[f]], j[:, TIPS[f]]
        d = t - a
        L = d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        u = d / L
        sc = (((v[:, idx] - a[:, None]) * u[:, None]).sum(-1) / L).clamp(min=0)  # (B, n)
        v[:, idx] = v[:, idx] + (ext[:, f, None] * L) [..., None] * sc[..., None] * u[:, None]
    return v


FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]


def finger_id_loss(p, g, vis, margin):
    """Margin loss on finger identity (thumb / index confusions on pointing hands): for every labelled finger j and every
    other predicted finger i, the distance of predicted finger j to labelled finger j (mean over their labelled joints)
    must be smaller by `margin` than the distance of predicted finger i to labelled finger j. p, g (B, 21, 2) in crop
    units, vis (B, 21)."""
    idx = torch.as_tensor(FINGERS, device=p.device)
    pf, gf, vf = p[:, idx], g[:, idx], vis[:, idx]  # (B, 5, 4, 2), (B, 5, 4)
    d = (pf[:, :, None] - gf[:, None]).norm(dim=-1)  # (B, 5 pred, 5 gt, 4)
    w = vf[:, None].expand_as(d)
    e = (d * w).sum(-1) / w.sum(-1).clamp(min=1)  # (B, i pred, j gt)
    ok = (vf.sum(-1) >= 2).float()  # labelled fingers
    own = torch.diagonal(e, dim1=1, dim2=2)  # (B, 5)
    hinge = F.relu(margin + own[:, None, :] - e)  # (B, i, j): predicted finger i vs the true one, for labelled finger j
    m = ok[:, None, :] * (1 - torch.eye(5, device=p.device))[None]
    return (hinge * m).sum() / m.sum().clamp(min=1)


def extend_tips(x, d):
    """Joints (..., 21, C) with each fingertip moved d x the distal phalanx length further along DIP -> tip."""
    y = x.clone() if torch.is_tensor(x) else np.array(x, copy=True)
    y[..., TIPS, :] = x[..., TIPS, :] + d * (x[..., TIPS, :] - x[..., DIPS, :])
    return y
MASK_QC = (0.5, 0.2)  # min fraction of GT keypoints inside the mask, min IoU with the model mesh silhouette (set in main)
LABEL_QC = 0.12  # max label-vs-model 2D disagreement (fraction of hand size) for pools with a qc.jsonl / qc.<i>.jsonl


def mask_terms(uv, mask, n_contour=192, prior_uv=None, wrist=None):
    """Instance-mask agreement of projected surface points. uv (B, P, 2) in [0, 1] view units, mask (B, H, W)
    uint8. Returns (inside, coverage): points should lie inside the (blurred) mask, and every mask-boundary pixel
    should have a surface point nearby (one-sided Chamfer, in view units).
    prior_uv (B, P, 2): the same points from a mesh (GT / teacher); points whose mesh position falls outside the
    mask are taken as occluded (object in front, modal mask) and get no inside pull.
    wrist (B, 2, 2): wrist and middle-MCP 2D; mask pixels more than 0.25 hand lengths behind the wrist (forearm,
    where pseudo-masks disagree) are removed before both terms."""
    m = F.interpolate(mask[:, None].float() / 255.0, size=(96, 96), mode="area")
    if wrist is not None:
        H, W = m.shape[-2:]
        ys, xs = torch.meshgrid((torch.arange(H, device=m.device) + 0.5) / H, (torch.arange(W, device=m.device) + 0.5) / W, indexing="ij")
        pix = torch.stack([xs, ys], -1)[None]  # (1, H, W, 2)
        d = wrist[:, 1] - wrist[:, 0]
        L = d.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        s = ((pix - wrist[:, None, None, 0]) * (d / L)[:, None, None]).sum(-1)
        m = m * (s >= -0.25 * L[:, :, None])[:, None].float()
    soft = 0.5 * (gaussian_blur(m, 1.5) + gaussian_blur(m, 6.0))
    g = uv[:, :, None] * 2 - 1  # grid_sample coordinates
    out = 1 - F.grid_sample(soft, g, align_corners=False)[:, 0, :, 0]
    if prior_uv is not None:
        w = (F.grid_sample(soft, prior_uv[:, :, None] * 2 - 1, align_corners=False)[:, 0, :, 0] > 0.3).float()
        inside = (out * w).sum() / w.sum().clamp(min=1.0)
    else:
        inside = out.mean()
    contour = ((m - (-F.max_pool2d(-m, 3, 1, 1))) > 0.25).float()[:, 0]
    B, H, W = contour.shape
    score = (contour * torch.rand_like(contour)).flatten(1)
    val, idx = score.topk(n_contour, dim=1)
    cxy = torch.stack([(idx % W).float() + 0.5, (idx // W).float() + 0.5], -1) / torch.tensor([W, H], device=uv.device)
    d = torch.cdist(cxy, uv).amin(-1)  # (B, M)
    ok = (val > 0).float()
    return inside, (d * ok).sum() / ok.sum().clamp(min=1.0) * 10


@torch.no_grad()
def vis_labels(v, faces, K, A, S, mask=None, has_mask=None, grid=64):
    """1 for MANO-topology vertices of mesh v (B, V, 3, camera frame) seen in the view: front-facing (outward vertex
    normal towards the camera), not behind the mesh's own nearer surface (point z-buffer on a grid x grid raster,
    min-pooled 3 x 3, 1.5 cm tolerance) and, where an instance mask exists (has_mask), projected inside it (outside =
    hidden by an object)."""
    uv = (project(v, K, A) / S).clamp(0, 1 - 1e-6)
    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
    fn = torch.cross(v[:, f1] - v[:, f0], v[:, f2] - v[:, f0], dim=-1)
    vn = torch.zeros_like(v)
    for fi in (f0, f1, f2):
        vn.index_add_(1, fi, fn)
    out_sign = torch.sign((vn * (v - v.mean(1, keepdim=True))).sum(-1).mean(1, keepdim=True))
    front = (vn * v).sum(-1) * out_sign < 0
    B = v.shape[0]
    cell = (uv[..., 1] * grid).long() * grid + (uv[..., 0] * grid).long()
    z = v[..., 2]
    zb = torch.full((B, grid * grid), 1e3, device=v.device, dtype=z.dtype).scatter_reduce(1, cell, z, "amin")
    zb = -F.max_pool2d(-zb.view(B, 1, grid, grid), 3, 1, 1).view(B, -1)
    vis = front & (z <= zb.gather(1, cell) + 0.015)
    if mask is not None and has_mask is not None and has_mask.any():
        m = F.max_pool2d(mask[:, None].float() / 255.0, 5, 1, 2)
        inside = F.grid_sample(m, uv[:, :, None] * 2 - 1, align_corners=False)[:, 0, :, 0] > 0.5
        vis = torch.where(has_mask[:, None], vis & inside, vis)
    return vis.float()


def bone_coords(surf, joints, a_idx, b_idx):
    """Surface points (B, V, 3) relative to their anchor bones (joint pairs a_idx -> b_idx per point): position
    along the bone (fraction t) and distance from it (m). Fixes where on the hand a point is (no sliding)."""
    a, bj = joints[:, a_idx], joints[:, b_idx]
    d = bj - a
    t = ((surf - a) * d).sum(-1) / (d * d).sum(-1).clamp(min=1e-8)
    r = (surf - (a + t[..., None] * d)).norm(dim=-1)
    return t, r, d.norm(dim=-1)


def gaussian_blur(x, sigma):
    r = int(3 * sigma + 0.5)
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


def low_light(x, p, lo=0.015, hi=0.25):
    """Per sample with prob p: linear-light exposure drop (x lo-hi), shot + read noise at the reduced signal,
    weak gain with a slight colour cast, sRGB gamma (as bench/bench2d/make_dark.py). x (B, 3, H, W) in [0, 1]."""
    B, dev = x.shape[0], x.device
    u = lambda a, b, *shape: torch.empty(*(shape or (B, 1, 1, 1)), device=dev).uniform_(a, b)
    sel = (torch.rand(B, 1, 1, 1, device=dev) < p).float()
    fw = u(800.0, 3000.0)
    sig = x.clamp(0, 1) ** 2.2 * torch.exp(u(math.log(lo), math.log(hi))) * fw
    noisy = sig + torch.randn_like(sig) * sig.sqrt() + torch.randn_like(sig) * u(1.0, 4.0)
    out = ((noisy / fw).clamp(0, 1) * u(0.8, 1.5) * u(0.9, 1.1, B, 3, 1, 1)).clamp(0, 1) ** (1 / 2.2)
    return sel * out + (1 - sel) * x


def ego_degrade(x, p):
    """Per sample, each with prob p: directional motion blur (5-25 px of 384, random angle), 1-3 object-like occluders
    (patches of another sample's view, soft edges, 10-35 % side), low resolution (x 1/2-1/4 and back), sensor noise.
    x (B, 3, H, W) in [0, 1]; student-only (clean-view KD); labels unchanged."""
    B, _, H, W = x.shape
    dev = x.device
    out = x
    sel = torch.rand(B, device=dev) < p
    if sel.any():  # motion blur: a line kernel per sample, grouped conv
        k = 2 * int(12 * H / 384) + 1
        L = torch.empty(B, device=dev).uniform_(5, 25) * H / 384
        th = torch.rand(B, device=dev) * math.pi
        r = torch.arange(k, device=dev) - k // 2
        yy, xx = torch.meshgrid(r, r, indexing="ij")
        d = (xx[None] * th.sin()[:, None, None] - yy[None] * th.cos()[:, None, None]).abs()
        a = (xx[None] * th.cos()[:, None, None] + yy[None] * th.sin()[:, None, None]).abs()
        ker = ((1 - d).clamp(min=0) * (a <= L[:, None, None] / 2)).float()
        ker = ker / ker.sum((1, 2), keepdim=True)
        bl = F.conv2d(F.pad(out, (k // 2,) * 4, mode="replicate").reshape(1, B * 3, H + k - 1, W + k - 1),
                      ker[:, None].repeat_interleave(3, 0), groups=B * 3).reshape(B, 3, H, W)
        out = torch.where(sel[:, None, None, None], bl, out)
    sel = torch.rand(B, device=dev) < p
    if sel.any():  # occluders: soft-edged rectangles cut from another sample of the batch
        donor = out[torch.randperm(B, device=dev)]
        ys = torch.linspace(0, 1, H, device=dev)[None, :, None]
        xs = torch.linspace(0, 1, W, device=dev)[None, None, :]
        m = torch.zeros(B, H, W, device=dev)
        for _ in range(int(torch.randint(1, 4, (1,)))):
            s = torch.empty(B, 2, device=dev).uniform_(0.10, 0.35)
            c = torch.rand(B, 2, device=dev)
            e = 0.02
            my = ((s[:, :1, None] / 2 - (ys - c[:, :1, None]).abs()) / e).clamp(0, 1)
            mx = ((s[:, 1:, None] / 2 - (xs - c[:, 1:, None]).abs()) / e).clamp(0, 1)
            m = torch.maximum(m, my * mx)
        m = m[:, None] * sel[:, None, None, None]
        out = out * (1 - m) + donor * m
    sel = torch.rand(B, device=dev) < p
    if sel.any():  # low resolution
        f = float(np.random.uniform(0.25, 0.5))
        lo = F.interpolate(F.interpolate(out, scale_factor=f, mode="bilinear", antialias=True), size=(H, W), mode="bilinear")
        out = torch.where(sel[:, None, None, None], lo, out)
    sel = torch.rand(B, device=dev) < p
    if sel.any():  # sensor noise
        sig = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.01, 0.05)
        out = torch.where(sel[:, None, None, None], (out + torch.randn_like(out) * sig).clamp(0, 1), out)
    return out


def hand_occlude(x, kp, p):
    """Hand-aware occlusion (EgoStandard: wrist hardware, objects over the fingertips), student-only, labels unchanged.
    x (B, 3, H, W) in [0, 1]; kp (B, 21, 2) view pixels (GT where labelled, else teacher). Per sample with prob p each:
    a wrist band (strap / sleeve edge / tracker pod: dark, grey or white, soft edges) across the forearm just behind
    the wrist; an occluder patch from another sample over 1-2 fingertips."""
    B, _, H, W = x.shape
    dev = x.device
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    P = torch.stack([xs, ys], -1)[None]  # (1, H, W, 2)
    u = lambda a, b: torch.empty(B, device=dev).uniform_(a, b)
    w, m = kp[:, 0], kp[:, 9]
    L = (m - w).norm(dim=-1).clamp(min=8.0)
    d = (m - w) / L[:, None]
    n = torch.stack([-d[:, 1], d[:, 0]], -1)
    out = x
    sel = torch.rand(B, device=dev) < p
    if sel.any():  # wrist band (+ a rounded pod on half of them)
        c = w - d * (L * u(0.0, 0.35))[:, None]
        rel = P - c[:, None, None]
        a, b = (rel * d[:, None, None]).sum(-1).abs(), (rel * n[:, None, None]).sum(-1).abs()
        ht, hl, e = L * u(0.1, 0.3), L * u(0.4, 0.7), 2.0
        band = ((ht[:, None, None] - a) / e).clamp(0, 1) * ((hl[:, None, None] - b) / e).clamp(0, 1)
        pod = (torch.rand(B, device=dev) < 0.5).float()
        pr = L * u(0.12, 0.25)
        cp = c + n * (L * u(-0.3, 0.3))[:, None]
        disc = ((pr[:, None, None] - (P - cp[:, None, None]).norm(dim=-1)) / e).clamp(0, 1) * pod[:, None, None]
        mk = torch.maximum(band, disc) * sel[:, None, None]
        tone = torch.where(torch.rand(B, device=dev) < 0.4, u(0.03, 0.2), torch.where(torch.rand(B, device=dev) < 0.5, u(0.35, 0.6), u(0.8, 1.0)))
        col = (tone[:, None] * torch.empty(B, 3, device=dev).uniform_(0.9, 1.1)).clamp(0, 1)[:, :, None, None]
        out = out * (1 - mk[:, None]) + col * mk[:, None]
    sel = torch.rand(B, device=dev) < p
    if sel.any():  # object-like patch from another sample over fingertips
        donor = out[torch.randperm(B, device=dev)]
        tips = kp[:, [4, 8, 12, 16, 20]]
        k = torch.randint(0, 5, (B,), device=dev)
        c = tips[torch.arange(B, device=dev), k] + d * (L * u(-0.1, 0.25))[:, None]
        rel = P - c[:, None, None]
        a, b = (rel * d[:, None, None]).sum(-1).abs(), (rel * n[:, None, None]).sum(-1).abs()
        ha, hb = L * u(0.2, 0.5), L * u(0.15, 0.6)
        mk = ((ha[:, None, None] - a) / 3.0).clamp(0, 1) * ((hb[:, None, None] - b) / 3.0).clamp(0, 1) * sel[:, None, None]
        out = out * (1 - mk[:, None]) + donor * mk[:, None]
    return out


def hand_over_hand(x, kp, mask, has_mask, p):
    """Another hand over the hand (interacting / overlapping hands: the crop then holds two hands and the model must stay
    on the centred one of the given side), student-only, labels unchanged (the teacher sees the clean view). Per sample
    with prob p: the hand of another sample of the batch -- its instance mask where it has one (has_mask), else its
    keypoint skeleton widened to a hand silhouette -- scaled to 0.7-1.1x this hand's size, mirrored half of the time,
    centred 0.35-0.75 hand lengths away from this hand's centre in a random direction; soft edges.
    x (B, 3, H, W) in [0, 1]; kp (B, 21, 2) view pixels; mask (B, H, W) uint8; has_mask (B,)."""
    B, _, H, W = x.shape
    dev = x.device
    sel = torch.rand(B, device=dev) < p
    if not sel.any():
        return x
    perm = torch.randperm(B, device=dev)
    u = lambda a, b: torch.empty(B, device=dev).uniform_(a, b)
    # donor silhouettes at 1/4 resolution: SAM mask, or bones of width 0.25 hand lengths
    h4, w4 = H // 4, W // 4
    ys, xs = torch.meshgrid(torch.arange(h4, device=dev, dtype=torch.float32), torch.arange(w4, device=dev, dtype=torch.float32), indexing="ij")
    P = torch.stack([xs, ys], -1).reshape(1, 1, -1, 2) * 4 + 1.5  # (1, 1, h4 * w4, 2) view pixels
    kd = kp[perm]
    Ld = (kd[:, 9] - kd[:, 0]).norm(dim=-1).clamp(min=8.0)
    par = torch.tensor([0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], device=dev)
    a, b = kd[:, par][:, :, None], kd[:, 1:][:, :, None]  # (B, 20, 1, 2) bone ends
    ab = b - a
    t = (((P - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-6)).clamp(0, 1)
    dist = (P - (a + t[..., None] * ab)).norm(dim=-1).min(1).values  # (B, h4 * w4)
    skel = ((0.25 * Ld[:, None] - dist) / 2.0).clamp(0, 1).reshape(B, 1, h4, w4)
    sam = F.adaptive_avg_pool2d(mask[perm][:, None].float() / 255.0, (h4, w4))
    md = torch.where(has_mask[perm].view(B, 1, 1, 1) > 0, sam, skel)
    md = F.interpolate(md, size=(H, W), mode="bilinear", align_corners=False)
    # placement: donor centre -> this hand's centre + offset; scale s; mirror m (affine_grid samples donor coordinates)
    ct, cd = kp.mean(1), kd.mean(1)
    Lt = (kp[:, 9] - kp[:, 0]).norm(dim=-1).clamp(min=8.0)
    s = Lt / Ld * u(0.7, 1.1)
    ang = u(0, 2 * math.pi)
    ct = ct + (Lt * u(0.35, 0.75))[:, None] * torch.stack([ang.cos(), ang.sin()], -1)
    m = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    nrm = lambda q: torch.stack([2 * q[:, 0] / (W - 1) - 1, 2 * q[:, 1] / (H - 1) - 1], -1)
    A = torch.zeros(B, 2, 3, device=dev)
    A[:, 0, 0], A[:, 1, 1] = m / s, 1.0 / s
    A[:, :, 2] = nrm(cd) - A[:, :, :2].matmul(nrm(ct)[..., None])[..., 0]
    grid = F.affine_grid(A, (B, 3, H, W), align_corners=True)
    don = F.grid_sample(x[perm], grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    mk = F.grid_sample(md, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    mk = F.avg_pool2d(mk, 5, 1, 2) * sel[:, None, None, None]
    return x * (1 - mk) + don * mk


@torch.no_grad()
def eval_val(trainable, mano, loader, S_enc, S, dev, pid=None):
    trainable.eval()
    errs, ious = {}, {}
    for b in loader:
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
        b["img_f"] = F.interpolate(b["img"].float() / 255.0, size=(S, S), mode="bilinear", antialias=True) \
            if S != b["img"].shape[-1] else b["img"].float() / 255.0
        Ab = b["A"].clone()
        Ab[:, :, :] *= S / b["img"].shape[-1]
        bb = dict(b, A=Ab)
        x_enc = F.interpolate(b["img_f"], size=(S_enc, S_enc), mode="bilinear", antialias=True) if S_enc != S else b["img_f"]
        out = trainable(make_batch(bb, S), x_enc)
        uv = out["pt_uv"][:, :21] * S if "pred_vertices" not in out else mano(out, b["K"], Ab)[1]
        gt = b["kp2d"].clone()
        gt[..., :2] *= S / b["img"].shape[-1]
        preds = {"": uv}
        if "pt_uv" in out:  # direct keypoint head
            preds["_head"] = out["pt_uv"][:, :21] * S
        for tag, pu in preds.items():
            e = (pu - gt[..., :2]).norm(dim=-1) / S
            for s in b["src"].unique().tolist():
                m = (b["src"] == s)[:, None] & (gt[..., 2] > 0)
                errs.setdefault(("pck05" + tag, s), []).append(e[m].cpu())
                errs.setdefault(("tip05" + tag, s), []).append(e[:, TIPS][m[:, TIPS]].cpu())
        if "kp_dense" in out and int(out["kp_dense"][0, 1]) > 0:  # dense surface points vs hand instance masks
            nb = 21 + int(out["kp_dense"][0, 0])
            su = out["pt_uv"][:, nb:]
            for s in b["src"].unique().tolist():
                sel = (b["src"] == s) & (b["has_mask"] > 0) & (b["mask_kind"] == 0)
                if sel.any():
                    m = b["mask"][sel][:, None].float() / 255.0
                    inside = F.grid_sample(m, su[sel][:, :, None] * 2 - 1, align_corners=False)[:, 0, :, 0] > 0.5
                    _, cov = mask_terms(su[sel], b["mask"][sel])
                    errs.setdefault(("surf_in", s), []).append(inside.float().mean(1).cpu())  # fraction of points inside
                    errs.setdefault(("surf_cov", s), []).append(torch.full((int(sel.sum()),), float(cov) * 10).cpu())  # % of view
        if (b["has_parts"] > 0).any():  # part labels: the part head's pixel accuracy / the dense points inside their own part
            lab = b["parts"].long()
            for s in b["src"].unique().tolist():
                sel = (b["src"] == s) & (b["has_parts"] > 0)
                if not sel.any():
                    continue
                if "part_logits" in out:
                    pl = out["part_logits"][sel].float()
                    t = F.interpolate(lab[sel][:, None].float(), size=pl.shape[-2:], mode="nearest")[:, 0].long()
                    ok = t != 255
                    errs.setdefault(("part_acc", s), []).append(((pl.argmax(1) == t) & ok).sum((1, 2)).float().cpu()
                                                               / ok.sum((1, 2)).clamp(min=1).float().cpu())
                if pid is not None and "kp_dense" in out and int(out["kp_dense"][0, 1]) > 0:
                    nb = 21 + int(out["kp_dense"][0, 0])
                    oh = F.one_hot(lab[sel].clamp(max=16), 17).permute(0, 3, 1, 2).float()
                    g = (out["pt_uv"][sel][:, nb:] * 2 - 1)[:, None]
                    inn = F.grid_sample(oh, g, mode="nearest", align_corners=False)[:, :, 0]
                    own = inn.gather(1, pid[None, None].expand(inn.shape[0], 1, -1))[:, 0]
                    errs.setdefault(("pt_in_part", s), []).append(own.mean(1).cpu())
        if "mask_logits" in out:
            ml = out["mask_logits"]
            if ml.ndim == 4:
                ml = ml.gather(1, b["mask_kind"].long().view(-1, 1, 1, 1).expand(-1, 1, *ml.shape[-2:]))[:, 0]
            t = F.adaptive_avg_pool2d(b["mask"][:, None].float() / 255.0, ml.shape[-2:])[:, 0] > 0.5
            pm = ml > 0
            iou = (pm & t).sum((1, 2)) / (pm | t).sum((1, 2)).clamp(min=1)
            for s in b["src"].unique().tolist():
                sel = (b["src"] == s) & (b["has_mask"] > 0)
                if sel.any():
                    ious.setdefault(s, []).append(iou[sel].cpu())
    trainable.train()
    res = {k: (torch.cat(v).mean() if k[0].startswith(("surf", "part", "pt_in")) else (torch.cat(v) < 0.05).float().mean()).item()
           for k, v in errs.items()}
    return res, {s: torch.cat(v).mean().item() for s, v in ious.items()}


def main(args):
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device("cuda")
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    from fah import HandEstimator, load_hand_model
    from fah.student import StudentEncoder

    teacher, cfg = load_hand_model(args.checkpoint, device=dev, backbone_dtype=torch.bfloat16, verbose=False)
    teacher.eval().requires_grad_(False)
    if args.teacher_ckpt:  # distil from a fine-tuned teacher (finetune checkpoint: decoder + trained backbone blocks)
        tk = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
        tsd = {k[6:]: v for k, v in tk["trainable"].items() if k.startswith("model.")}
        _, t_unexp = teacher.load_state_dict(tsd, strict=False)
        if rank == 0:
            print(f"teacher from {args.teacher_ckpt}: {len(tsd)} tensors ({len(t_unexp)} unused, e.g. heads)", flush=True)
        del tk, tsd
    t_est = HandEstimator(teacher, cfg, input_size=args.size,
                          **({"mano_regressor_path": args.teacher_regressor} if args.teacher_regressor else {}))  # enables the fast rig, builds the regressor
    model, _ = load_hand_model(args.checkpoint, device=dev,
                               backbone_dtype=torch.float32 if args.train_blocks else torch.bfloat16, verbose=False)
    HandEstimator(model, cfg)
    mano = ManoJoints(t_est).to(dev)

    encoder, S = None, args.size
    if args.student:
        ck = torch.load(args.student, map_location="cpu", weights_only=False)
        encoder = StudentEncoder(ck["student_name"], pretrained=False, size=args.student_size or None)
        sd = ck["student"] if "student" in ck else {k[8:]: v for k, v in ck["trainable"].items() if k.startswith("encoder.")}
        missing, unexpected = encoder.load_state_dict(sd, strict=False)  # RoPE / pos buffers are size dependent
        if rank == 0 and (missing or unexpected):
            print("student load: missing", missing[:4], "unexpected", unexpected[:4], flush=True)
        if args.grad_ckpt:  # activation checkpointing of the student ViT blocks (large backbones / inputs)
            encoder.vit.set_grad_checkpointing(True)
        encoder = encoder.to(dev)
        S = 16 * encoder.grid  # decoder geometry for the student's 16x16 tokens
        model.backbone = None  # not used (embeddings come from the student)
    else:
        model.backbone.requires_grad_(False)
        if args.train_blocks:  # unfreeze the last N transformer blocks + final norm
            enc = model.backbone.encoder
            for blk in enc.blocks[-args.train_blocks:]:
                blk.requires_grad_(True)
            enc.norm.requires_grad_(True)
    if args.slim_decoder:  # CPU path: small decoder replaces the 6-layer one (which is frozen / unused)
        from fah.models.decoders.slim import SlimHandDecoder

        model.slim_decoder = SlimHandDecoder(dim=args.slim_dim, layers=args.slim_layers, heads=args.slim_dim // 64,
                                             kp_queries=args.slim_kp_queries).to(dev)
    if args.mask_head:
        from fah.models.heads.mask_head import HandMaskHead

        model.mask_head = HandMaskHead(img_dim=args.slim_dim if args.slim_decoder else 1280).to(dev)
    if args.verify_head:
        from fah.models.heads.mask_head import HandVerifyHead

        model.verify_head = HandVerifyHead().to(dev)
    if args.kp_head:  # direct 2D / 3D joints + dense surface points on the shared latent
        from fah.models.heads.keypoint_head import HandKeypointHead, farthest_points

        rest = model.head_pose_hand.mhr.character_torch.mesh.rest_vertices[torch.as_tensor(t_est.hand_vert_idx)]
        surf = farthest_points(rest.float().cpu(), args.kp_surface) if args.kp_surface and not args.kp_dense else []
        dense_vidx = np.arange(778)
        if args.kp_dense and args.dense_n:  # select template vertices for the dense surface
            from bench.common import ManoLayer
            from fah.models.heads.keypoint_head import dense_subset

            rj_, rv_ = ManoLayer(args.mano_dir)(True, np.zeros(10), np.zeros(45), np.zeros(3), np.zeros(3))
            dense_vidx = dense_subset(rv_, args.dense_n, args.dense_sel, rj_)
        dense = (40, len(dense_vidx)) if args.kp_dense else (0, 0)  # MHR-free: joints + bone points + MANO-topology vertices
        model.keypoint_head = HandKeypointHead(img_dim=args.slim_dim if args.slim_decoder else 1280, surface_idx=surf, parts=17 if (args.w_partseg > 0 or args.w_partmask > 0) else 0,
                                               pool=args.kp_only, dense=dense, dim=args.kp_dim, stride=args.kp_stride,
                                               depth=args.kp_depth, vis=args.vis_head,
                                               dense_vidx=dense_vidx if args.dense_n else None, part_hr=args.part_hr, part_depth=args.part_depth).to(dev)
        model.keypoint_head.part_grad = args.partseg_grad
        sub = args.kp_dense and len(dense_vidx) < 778
        vidx_t = torch.as_tensor(dense_vidx, device=dev)
        if args.kp_dense:  # teacher MHR hand vertices -> MANO-topology vertices (sparse convex regressor)
            from fah.estimator import load_joint_regressor

            n_full = model.head_pose_hand.mhr.character_torch.mesh.rest_vertices.shape[0]
            W778 = load_joint_regressor("fah/assets/mhr_to_mano778_regressor.npz", n_full)
            cols = np.nonzero(W778.any(0))[0]
            remap = -np.ones(n_full, dtype=np.int64)
            remap[t_est.hand_vert_idx] = np.arange(len(t_est.hand_vert_idx))
            assert (remap[cols] >= 0).all()
            dense_W = torch.as_tensor(W778[:, cols], device=dev)
            dense_idx = torch.as_tensor(remap[cols], device=dev)
            # template connectivity (MANO-topology surface) for the edge-length regulariser only
            from bench.common import ManoLayer

            ml = ManoLayer(args.mano_dir)
            faces = ml.faces[True]
            e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
            dense_edges = torch.as_tensor(np.unique(np.sort(e, 1), axis=0), device=dev)
            # anchor bone per template point (nearest bone segment in the template's rest pose)
            from fah.models.heads.keypoint_head import BONES

            rj, rv = ml(True, np.zeros(10), np.zeros(45), np.zeros(3), np.zeros(3))
            A_, B_ = rj[[p for p, _ in BONES]], rj[[c for _, c in BONES]]
            dd = B_ - A_
            tt = np.clip(((rv[:, None] - A_[None]) * dd[None]).sum(-1) / (dd * dd).sum(-1)[None], 0, 1)
            dseg = np.linalg.norm(rv[:, None] - (A_[None] + tt[..., None] * dd[None]), axis=-1)
            nb_ = dseg.argmin(1)
            dense_anchor = (torch.as_tensor([BONES[i][0] for i in nb_], device=dev), torch.as_tensor([BONES[i][1] for i in nb_], device=dev))
            import pickle as _pk

            _mm = _pk.load(open(os.path.join(args.mano_dir, "MANO_RIGHT.pkl"), "rb"), encoding="latin1")
            _wj = np.asarray(_mm["weights"]).argmax(1)  # MANO joints: index 1-3, middle 4-6, pinky 7-9, ring 10-12, thumb 13-15
            distal_idx = [torch.as_tensor(np.where(_wj == jd)[0], device=dev) for jd in (15, 3, 6, 12, 9)]  # thumb..pinky
            _jp = {0: 1, 13: 2, 14: 3, 15: 4, 1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10, 10: 11, 11: 12, 12: 13, 7: 14, 8: 15, 9: 16}  # = train.part_masks
            dense_part = torch.as_tensor(np.array([_jp[j] for j in _wj])[np.asarray(dense_vidx)], device=dev)  # part of each dense point
            dense_tipj = torch.as_tensor([{4: 4, 7: 8, 10: 12, 13: 16, 16: 20}.get(q, -1) for q in dense_part.tolist()], device=dev)  # distal points -> tip joint
            kept = np.asarray(dense_vidx)  # fingertip-end points: the kept surface points nearest the MANO tip vertices (thumb..pinky)
            tip_rows = torch.as_tensor([int(np.argmin(np.linalg.norm(rv[kept] - rv[v], axis=1))) for v in (744, 320, 443, 554, 671)], device=dev)
            if sub:  # subset: anchors of the kept points, edges = 6-NN graph of the kept points in the rest pose
                dense_anchor = (dense_anchor[0][vidx_t], dense_anchor[1][vidx_t])
                rs = rv[dense_vidx]
                nn6 = np.argsort(np.linalg.norm(rs[:, None] - rs[None], axis=-1), 1)[:, 1:7]
                e = np.stack([np.repeat(np.arange(len(rs)), 6), nn6.reshape(-1)], 1)
                dense_edges = torch.as_tensor(np.unique(np.sort(e, 1), axis=0), device=dev)
            dense_faces = torch.as_tensor(faces.astype(np.int64), device=dev)
            if args.kd_mesh_joints:  # MANO J_regressor (16) + HOT3D tip vertices, OpenPose order: (21, 778)
                from bench.common import MANO21_TO_OPENPOSE, MANO_TIP_VERTS

                Jr = ml.layers[True].J_regressor
                Jr = np.asarray(Jr.cpu() if hasattr(Jr, "cpu") else Jr, np.float32)
                J21 = torch.as_tensor(np.concatenate([Jr, np.eye(778, dtype=np.float32)[MANO_TIP_VERTS]])[MANO21_TO_OPENPOSE], device=dev)
            if args.wsmooth > 0:  # low-frequency surface basis (cotangent Laplacian of the template, K modes)
                from fah.models.heads.spectral import smooth_surface, template_basis

                spec_basis = template_basis(args.mano_dir, args.smooth_k)
            if args.mano_head:  # parametric MANO mesh from the model's own 2D points (fah/models/heads/mano_head.py)
                from fah.models.heads.mano_head import ManoParamHead

                model.mano_head = ManoParamHead(pt_dim=args.kp_dim, mano_dir=args.mano_dir).to(dev)
        if args.kp_only:  # decoder-free CPU path: encoder -> keypoint head only
            model.kp_only = True
            for mod in (model.decoder_hand, model.head_pose_hand, model.head_camera_hand):
                mod.requires_grad_(False)
    if args.init_ckpt:  # start from a train/finetune.py checkpoint (decoder, heads, trained backbone blocks)
        ck0 = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        msd = model.state_dict()
        sd = {k[6:]: v for k, v in ck0["trainable"].items() if k.startswith("model.")}
        pk = "keypoint_head.point"
        if args.kp_dense and args.dense_n and pk in sd and sd[pk].shape[0] == 21 + 40 + 778:  # subset: keep the kept rows
            sd[pk] = torch.cat([sd[pk][:61], sd[pk][61:][torch.as_tensor(dense_vidx)]])
        if args.kp_dense and args.dense_n:  # the layout buffers describe this model, not the init
            sd.pop("keypoint_head.dense", None)
            sd.pop("keypoint_head.dense_vidx", None)
        skipped = [k for k, v in sd.items() if k in msd and msd[k].shape != v.shape]  # e.g. slim-width mask head
        sd = {k: v for k, v in sd.items() if k not in skipped}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if rank == 0 and skipped:
            print("init: skipped shape-mismatched", skipped[:6], flush=True)
        if rank == 0:
            print(f"init from {args.init_ckpt} (step {ck0['step']}): {len(sd)} tensors, {len(unexpected)} unexpected", flush=True)
    if args.slim_decoder:
        model.decoder_hand.requires_grad_(False)
    if args.decoder_depth:  # early-exit decoder: the first N layers are trained to give the final output
        model.decoder_hand.active_depth = args.decoder_depth
        model.decoder_hand.layers[args.decoder_depth:].requires_grad_(False)
    if args.freeze_decoder:  # only the new heads train on the frozen shared latent
        model.requires_grad_(False)
        for hd in ("mask_head", "verify_head", "keypoint_head"):
            if getattr(model, hd, None) is not None:
                getattr(model, hd).requires_grad_(True)
        if encoder is not None:
            encoder.requires_grad_(False)
    if args.train_only_parts:  # part head stage: everything else frozen, so joints / points / mesh cannot change
        model.requires_grad_(False)
        for n, p_ in model.keypoint_head.named_parameters():
            p_.requires_grad_(n.startswith("part_"))
        if encoder is not None:
            encoder.requires_grad_(False)
    S_enc = encoder.input_size if encoder is not None else S
    trainable = Trainable(model, encoder, train_backbone=bool(args.train_blocks) and encoder is None).to(dev)
    params = [p for p in trainable.parameters() if p.requires_grad]
    is_enc = lambda n: n.startswith("encoder.") or n.startswith("model.backbone.")
    enc_params = [p for n, p in trainable.named_parameters() if p.requires_grad and is_enc(n)]
    dec_params = [p for n, p in trainable.named_parameters() if p.requires_grad and not is_enc(n)]
    groups = [dict(params=dec_params, lr=args.lr)]
    if enc_params:
        groups.append(dict(params=enc_params, lr=args.lr * args.enc_lr_mult))
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    base_lrs = [g["lr"] for g in opt.param_groups]

    start_step, wandb_id = 0, None
    ckpt_path = os.path.join(args.out, "last.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        trainable.load_state_dict(ck["trainable"], strict=False)
        opt.load_state_dict(ck["opt"])
        start_step, wandb_id = ck["step"], ck.get("wandb_id")
    net = torch.nn.parallel.DistributedDataParallel(trainable, device_ids=[dev.index], find_unused_parameters=True) \
        if ddp else trainable

    sources = args.sources.split(",")
    part_dirs = dict(x.split(":") for x in args.part_dirs.split(",")) if args.part_dirs else None
    arm = tuple(args.arm_mask_sources.split(",")) if args.arm_mask_sources else ()
    jmask = {q[0]: [int(j) for j in q[1].split("+")] for q in (x.split(":") for x in args.joint_mask_sources.split(","))} \
        if args.joint_mask_sources else None
    ds = LabeledCrops(args.crops, sources, size=args.size, min_mask_score=args.min_mask_score, arm_sources=arm,
                      p_wrong=args.p_wrong if args.verify_head else 0.0, box_jitter=args.box_jitter, min_crop=args.min_crop,
                      p_shadow=args.p_shadow, raw_tip_sources=tuple(args.raw_tip_sources.split(",")) if args.raw_tip_sources else (),
                      raw_tips=args.raw_tips, tip_shift=args.tip_shift, tip_mask_ext=args.tip_mask_ext, tip_from_parts=args.tip_from_parts,
                      parts=args.w_part > 0 or args.w_part_cover > 0 or args.w_partseg > 0 or args.w_part_finger > 0, part_sources=args.part_sources.split(",") if args.part_sources else None,
                      part_routes=args.part_routes.split(",") if args.part_routes else None, part_dirs=part_dirs, joint_mask=jmask,
                      no3d=tuple(args.no3d_sources.split(",")) if args.no3d_sources else ())
    val_ds = LabeledCrops(args.crops, sources, size=args.size, aug=False, split="val", min_mask_score=args.min_mask_score,
                          arm_sources=arm, min_crop=args.min_crop, parts=args.w_part > 0 or args.w_part_cover > 0 or args.w_partseg > 0 or args.w_part_finger > 0,
                          part_sources=args.part_sources.split(",") if args.part_sources else None,
                          part_routes=args.part_routes.split(",") if args.part_routes else None, part_dirs=part_dirs)
    weights = [float(w) for w in args.weights.split(",")] if args.weights else [1.0] * len(sources)
    counts = np.bincount(ds.src, minlength=len(sources))
    w = np.asarray(weights)[ds.src] / counts[ds.src]
    s3d = dict(kv.split(":") for kv in args.w3d_sources.split(",")) if args.w3d_sources else {}
    src_w3d = torch.tensor([float(s3d.get(s, 1.0)) for s in sources], device=dev)  # per-source 3D trust
    skd = dict(kv.split(":") for kv in args.kd_sources.split(",")) if args.kd_sources else {}
    src_kd = torch.tensor([float(skd.get(s, 1.0)) for s in sources], device=dev)  # per-source teacher-KD trust
    val_loader = DataLoader(torch.utils.data.Subset(val_ds, np.linspace(0, len(val_ds) - 1, min(len(val_ds), args.val_num)).astype(int)),
                            batch_size=args.batch_size, num_workers=2)

    run = None
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        print(f"train {len(ds)} / val {len(val_ds)} crops; per source {dict(zip(sources, counts.tolist()))}; mix {weights}; "
              f"encoder {'student ' + args.student if args.student else 'teacher (frozen)'}; S={S} S_enc={S_enc}", flush=True)
        if args.wandb:
            import wandb

            run = wandb.init(project=os.environ.get("WANDB_PROJECT", "FAH"),
                             entity=os.environ.get("WANDB_ENTITY"), name=args.run_name,
                             id=wandb_id, resume="allow", config=vars(args), dir=args.out)
            wandb_id = run.id

    def save(step, final=False):
        if rank == 0:
            sd = {k: v for k, v in trainable.state_dict().items()
                  if not k.startswith("model.backbone.") or args.train_blocks}
            ck = dict(trainable=sd, opt=opt.state_dict(), step=step, wandb_id=wandb_id, args=vars(args),
                      student_name=ck_student_name)
            torch.save(ck, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
            if final:  # the finished run (what the next stage and bench.methods.fah load): its weights, no optimiser state
                ck.pop("opt")
                torch.save(ck, os.path.join(args.out, "final.pt.tmp"))
                os.replace(os.path.join(args.out, "final.pt.tmp"), os.path.join(args.out, "final.pt"))

    ck_student_name = encoder.name if encoder is not None else None

    def evaluate(step):
        if rank == 0:
            t0 = time.time()
            pck, miou = eval_val(trainable, mano, val_loader, S_enc, S, dev, dense_part if args.kp_dense else None)
            m = {f"val_{k}/{sources[s]}": v for (k, s), v in pck.items()}
            m.update({f"val_mask_iou/{sources[s]}": v for s, v in miou.items()})
            if args.freihand and not args.kp_only:
                m.update(eval_freihand(model, cfg, encoder, args.freihand, args.eval_num) if encoder is not None
                         else eval_freihand_teacher(model, cfg, args.freihand, args.eval_num, S))
            m["eval_s"] = time.time() - t0
            print(f"[eval {step}] {m}", flush=True)
            if run:
                run.log({f"eval/{k}": v for k, v in m.items()}, step=step)
        if ddp:
            dist.barrier()

    if start_step == 0:
        evaluate(0)
    step, epoch = start_step, 0
    t_last, seen = time.time(), 0
    while step < args.steps:
        g = torch.Generator().manual_seed(args.seed * 1000 + epoch)
        idx = torch.multinomial(torch.as_tensor(w), args.batch_size * world * 500, replacement=True, generator=g)[rank::world]
        loader = DataLoader(torch.utils.data.Subset(ds, idx.tolist()), batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=True, drop_last=True, prefetch_factor=4, worker_init_fn=seed_worker)
        for b in loader:
            if step >= args.steps:
                break
            lr_scale = min(1.0, (step + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(step / args.steps, 1.0)))
            for gr, lr in zip(opt.param_groups, base_lrs):
                gr["lr"] = lr * lr_scale
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            x_t = b["img"].float() / 255.0  # (B, 3, size, size) view at the teacher resolution
            # frozen teacher on the native crop -> distillation targets (view pixels of this resolution)
            with torch.no_grad():
                b["img_f"] = x_t
                tb = make_batch(b, args.size)
                feat = teacher.encode(tb)  # frozen backbone, shared with the trainable decoder in teacher mode
                t_j = None
                if not args.freeze_decoder or args.kp_head:  # distillation targets (not needed for mask-only)
                    t_out = teacher(tb, image_embeddings=feat)
                    t_j, t_uv = mano(t_out, b["K"], b["A"])
                    he = b["has_ext"][:, None, None] > 0  # external 3D KD (e.g. a MANO model): its shape at the teacher's root
                    t_j = torch.where(he, b["ext_j"] + t_j[:, :1], t_j)
                    if args.kd_mesh_joints:  # teacher joints in the MANO mesh convention (J_regressor + tip vertices of its MANO-topology mesh)
                        tv_ = torch.einsum("vk,bkc->bvc", dense_W, t_out["pred_vertices"][:, dense_idx]) + t_out["pred_cam_t"][:, None]
                        t_j = torch.einsum("jv,bvc->bjc", J21, tv_.float())
                        t_uv = project(t_j, b["K"], b["A"])
                    if args.tip_shift:  # teacher / external targets in the same fingertip-end convention as the labels
                        t_j, t_uv = extend_tips(t_j, args.tip_shift), extend_tips(t_uv, args.tip_shift)
                    t_j0 = t_j
                    if args.tip_mask_ext:  # the visible finger end measured on this sample's mask
                        t_j, t_uv = extend_tips_by(t_j, b["tip_ext"]), extend_tips_by(t_uv, b["tip_ext"])
            # student-side geometry (S x S) and encoder input
            ratio = S / args.size
            bs = dict(b, A=b["A"] * ratio)
            x_s = low_light(x_t, args.p_dark) if args.p_dark > 0 else x_t  # student-only low light: clean-view KD
            x_s = ego_degrade(x_s, args.p_ego) if args.p_ego > 0 else x_s
            if args.p_hocc > 0 and t_j is not None:  # wrist hardware / fingertip occluders at the hand's own keypoints
                kp_h = torch.where(b["kp2d"][..., 2:] > 0, b["kp2d"][..., :2], t_uv).float()
                x_s = hand_occlude(x_s, kp_h, args.p_hocc)
            if args.p_hoh > 0 and t_j is not None:  # another hand over the hand: stay on the centred hand
                kp_h = torch.where(b["kp2d"][..., 2:] > 0, b["kp2d"][..., :2], t_uv).float()
                x_s = hand_over_hand(x_s, kp_h, b["mask"], b["has_mask"] * (b["mask_kind"] == 0), args.p_hoh)
            bs["img_f"] = F.interpolate(x_s, size=(S, S), mode="bilinear", antialias=True) if S != args.size else x_s
            x_enc = F.interpolate(x_s, size=(S_enc, S_enc), mode="bilinear", antialias=True) if S_enc != args.size else x_s
            out = net(make_batch(bs, S), x_enc, emb=None if (encoder is not None or args.train_blocks) else feat)
            if args.kp_only:  # decoder-free: the keypoint head's joints play the MHR joints in every loss (root = GT)
                j, uv = out["pt_xyz"][:, :21] + b["j3d"][:, :1], out["pt_uv"][:, :21] * S
            else:
                j, uv = mano(out, b["K"], bs["A"])
            gt2d = b["kp2d"][..., :2] * ratio
            vis = b["kp2d"][..., 2]
            l2d = smooth_l1(uv / S, gt2d / S, vis, 0.02)
            rel, g_rel = j - j[:, :1], b["j3d"] - b["j3d"][:, :1]
            t_rel = t_j - t_j[:, :1] if t_j is not None else None
            w3 = b["has3d"] * src_w3d[b["src"]]
            w3j = w3[:, None] * b["j3w"]  # per-joint 3D weight (--joint_mask_sources)
            n3 = b["has3d"].sum()  # normalise by the number of 3D samples, so source scales down-weight
            l3d = smooth_l1(rel, g_rel, w3j, 0.01, norm=n3 * 21)
            lroot = smooth_l1(j[:, 0], b["j3d"][:, 0], w3, 0.02, norm=n3)
            valid = (b["present"] * b["side_ok"])[:, None].expand(-1, 21)  # KD only on real, correctly mirrored hands
            kdw = src_kd[b["src"]]
            valid = valid * kdw[:, None]  # sources where the teacher is wrong by construction (mined hard examples): no KD
            if "pt_depth" in out:  # decoder-free root depth (log z * size / crop focal): teacher on every hand, GT where 3D
                fz = b["K"][:, 0, 0] * b["A"][:, 0, 0] / args.size
                d_t = torch.log(t_j[:, 0, 2].clamp(min=0.05) / fz)
                d_g = torch.log(b["j3d"][:, 0, 2].clamp(min=0.05) / fz)
                vd = valid[:, 0]
                lroot = (F.smooth_l1_loss(out["pt_depth"], d_t, beta=0.02, reduction="none") * vd).sum() / vd.sum().clamp(min=1) \
                    + (F.smooth_l1_loss(out["pt_depth"], d_g, beta=0.02, reduction="none") * w3).sum() / n3.clamp(min=1)
            lkd = smooth_l1(rel, t_rel, valid, 0.01) + smooth_l1(uv / S, t_uv * ratio / S, valid, 0.02) \
                if not args.freeze_decoder else torch.zeros((), device=dev)
            if args.slim_decoder and "hand_box" in out and not args.freeze_decoder:  # distil the hand-box head too
                lkd = lkd + F.l1_loss(out["hand_box"], t_out["hand_box"])
            kp_uv = out.get("kp_uv")
            if kp_uv is not None:  # keypoint queries: auxiliary crop-normalised 2D (GT + teacher)
                lkd = lkd + smooth_l1(kp_uv, b["kp2d"][..., :2] / args.size, vis, 0.02) \
                    + smooth_l1(kp_uv, t_uv / args.size, valid, 0.02)
            lver = torch.zeros((), device=dev)
            if "verify_logits" in out:
                vl = out["verify_logits"]
                lver = F.binary_cross_entropy_with_logits(vl[:, 0], b["present"]) + \
                    (F.binary_cross_entropy_with_logits(vl[:, 1], b["side_ok"], reduction="none") * b["present"]).sum() / \
                    b["present"].sum().clamp(min=1.0)
            lmask = mask_loss(out["mask_logits"], b["mask"], b["has_mask"], b["mask_kind"]) if "mask_logits" in out \
                else torch.zeros((), device=dev)
            lpt = torch.zeros((), device=dev)
            lpart, lpseg = torch.zeros((), device=dev), torch.zeros((), device=dev)  # logged part terms
            lfing = torch.zeros((), device=dev)  # logged finger-level part term (--w_part_finger)
            if "pt_uv" in out:  # direct keypoints: GT 2D / 3D joints + teacher KD (joints, dense surface points)
                puv, pxyz = out["pt_uv"] * S, out["pt_xyz"]
                lpt = smooth_l1(puv[:, :21] / S, gt2d / S, vis, 0.02) \
                    + args.w3d * smooth_l1(pxyz[:, :21], g_rel, w3j, 0.01, norm=n3 * 21) \
                    + 0.5 * (smooth_l1(puv[:, :21] / S, t_uv * ratio / S, valid, 0.02) + smooth_l1(pxyz[:, :21], t_rel, valid, 0.01))
                if args.w_finger_id > 0:  # finger identity: each predicted finger closer to its own labelled finger than to any other
                    lpt = lpt + args.w_finger_id * finger_id_loss(puv[:, :21] / S, gt2d / S, vis, args.finger_id_margin)
                if args.wrep > 0:  # 3D-2D consistency: xyz joints at their best translation reproject onto the GT 2D
                    tgt = torch.where(vis[..., None] > 0, gt2d, puv[:, :21].detach())  # (own heatmap 2D where unlabelled)
                    wr = torch.where(vis > 0, torch.ones_like(vis), 0.5 * valid)
                    wr = wr * ((wr > 0).sum(1, keepdim=True) >= 4)
                    tz = torch.where(b["has3d"] > 0, b["j3d"][:, 0, 2], t_j[:, 0, 2]).float().clamp(min=0.1)  # GT / teacher root depth
                    tr = fit_translation(pxyz[:, :21].detach().float(), tgt.float(), b["K"].float(), bs["A"].float(), wr.float(), tz)
                    pc = pxyz[:, :21] + tr[:, None]
                    pc = torch.cat([pc[..., :2], pc[..., 2:].clamp(min=0.05)], -1)
                    lpt = lpt + args.wrep * smooth_l1(project(pc, b["K"], bs["A"]) / S, tgt / S, wr, 0.02)
                if args.kp_dense:  # bones from the joint targets, MANO-topology vertices from the teacher
                    from fah.models.heads.keypoint_head import bone_points

                    nb = 21 + 40
                    bvis = bone_points(vis[..., None], 2)[..., 0] * (bone_points(vis[..., None], 2)[..., 0] > 0.99)
                    lpt = lpt + smooth_l1(puv[:, 21:nb] / S, bone_points(gt2d, 2) / S, bvis, 0.02) \
                        + args.w3d * smooth_l1(pxyz[:, 21:nb], bone_points(g_rel, 2), w3[:, None] * (bone_points(b["j3w"][..., None], 2)[..., 0] > 0.99), 0.01, norm=n3 * 40) \
                        + 0.5 * (smooth_l1(puv[:, 21:nb] / S, bone_points(t_uv * ratio, 2) / S, valid[:, :1].expand(-1, 40), 0.02)
                                 + smooth_l1(pxyz[:, 21:nb], bone_points(t_rel, 2), valid[:, :1].expand(-1, 40), 0.01))
                    # surface points: a weak prior from the teacher's mesh + the image's own evidence (instance mask)
                    tv = torch.einsum("vk,bkc->bvc", dense_W, t_out["pred_vertices"][:, dense_idx]) + t_out["pred_cam_t"][:, None]
                    tv_img = tv  # the teacher's own mesh lines up with the image: every 2D surface term uses it
                    tv = torch.where(b["has_ext"][:, None, None] > 0, b["ext_v"] + t_j[:, :1], tv)  # 3D shape target
                    if args.tip_mask_dense:  # the target surfaces' distal phalanges stretched to the same visible ends
                        tv_img = stretch_distal(tv_img, t_j0, b["tip_ext"], distal_idx)
                        tv = stretch_distal(tv, t_j0, b["tip_ext"], distal_idx)
                    hv = b["has_v"][:, None]  # annotated mesh (HOT3D / FreiHAND): correspondence + 3D for every point
                    tm = torch.where(hv[..., None] > 0, b["v3d"], tv_img)  # full target mesh (visibility needs the faces)
                    tv_m = torch.where(hv[..., None] > 0, b["v3d"], tv)  # full 3D shape target (GT / external MANO / teacher)
                    gv_all = b["v3d"]
                    if args.tip_mask_dense and args.tip_from_parts:  # GT surfaces: distal phalanges out to the visible ends too
                        gv_all = torch.where(hv[..., None] > 0, stretch_distal(b["v3d"], b["j3d"], b["tip_ext"], distal_idx), gv_all)
                    if sub:
                        tv, tv_img, gv_all = tv[:, vidx_t], tv_img[:, vidx_t], gv_all[:, vidx_t]
                    vk = (b["present"] * b["side_ok"])[:, None].expand(-1, tv.shape[1])
                    su, sx = puv[:, nb:], pxyz[:, nb:]
                    if args.tipend_w > 0:  # fingertip-end points follow the human-clicked finger ends
                        tw = b["tip_raw"][..., 2] * (b["present"] * b["side_ok"])[:, None]
                        lpt = lpt + args.tipend_w * smooth_l1(su[:, tip_rows] / S, b["tip_raw"][..., :2] * ratio / S, tw, 0.02)
                    if args.w_partseg > 0 and "part_logits" in out:  # zero still connects the part head on ranks without labels
                        lpseg = part_seg_loss(out["part_logits"].float(), b["parts"], b["has_parts"], args.partseg_finger_w)
                        lpt = lpt + args.w_partseg * lpseg
                    if args.w_partmask > 0 and "part_logits" in out:  # the union of the parts = the hand mask (and nothing on negatives)
                        pl = out["part_logits"].float()
                        hand = (1 - pl.softmax(1)[:, 0]).clamp(1e-4, 1 - 1e-4)  # (B, h, w)
                        hm = (b["has_mask"] > 0) & (b["mask_kind"] == 0)
                        neg = b["present"] == 0
                        tmk = (F.adaptive_avg_pool2d(b["mask"][:, None].float() / 255.0, pl.shape[-2:])[:, 0] > 0.5).float()
                        tmk = torch.where(neg[:, None, None], torch.zeros_like(tmk), tmk)
                        sel = hm | neg
                        if sel.any():
                            lpm = F.binary_cross_entropy(hand[sel], tmk[sel])
                            if hm.any():  # + soft Dice on the hand-masked samples: thin fingers count as much as the palm
                                ph, th = hand[hm].flatten(1), tmk[hm].flatten(1)
                                lpm = lpm + args.partmask_dice * (1 - (2 * (ph * th).sum(1) + 1) / (ph.sum(1) + th.sum(1) + 1)).mean()
                            lpt = lpt + args.w_partmask * lpm
                            lpseg = lpseg + lpm.detach()
                    vl, g2 = None, 1.0
                    if args.vis_head or args.vis_gate:  # which target-mesh points the view shows (self / object occlusion)
                        vl = vis_labels(tm.float(), dense_faces, b["K"].float(), bs["A"].float(), S, b["mask"],
                                        (b["has_mask"] > 0) & (b["mask_kind"] == 0))
                        if sub:
                            vl = vl[:, vidx_t]
                        if args.vis_gate:  # hidden points: their 2D follows the 3D, the heatmap is not pulled to them
                            g2 = 0.2 + 0.8 * vl
                        if "pt_vis" in out:
                            vkk = vk * kdw[:, None]
                            lvis = (F.binary_cross_entropy_with_logits(out["pt_vis"][:, nb:].float(), vl, reduction="none") * vkk).sum() \
                                / vkk.sum().clamp(min=1.0)
                            lpt = lpt + args.wvis * lvis
                    if (args.w_part > 0 or args.w_part_cover > 0) and (b["has_parts"] > 0).any():  # dense points vs the part regions
                        hp = b["has_parts"] * (1 - hv[:, 0]) if args.part_no_gt else b["has_parts"]  # GT-mesh samples: the GT already places every point
                        pr, pc = part_loss(su, b["parts"], hp, dense_part, S, (vl if vl is not None else 1.0) * vk)
                        lpt = lpt + args.w_part * pr + args.w_part_cover * pc
                        lpart = (pr + pc).detach()
                    if args.w_part_finger > 0 and (b["has_parts"] > 0).any():  # finger-level dense term (coarse part map)
                        hp = b["has_parts"] * (1 - hv[:, 0]) if args.part_no_gt else b["has_parts"]
                        pf = part_finger_loss(su, b["parts"], hp, dense_part, S, (vl if vl is not None else 1.0) * vk)
                        lpt = lpt + args.w_part_finger * pf
                        lfing = lfing + pf.detach()
                    if args.wrep_surf > 0 and args.wrep > 0:  # the 3D surface at the joints' translation projects onto its own visible 2D
                        sv = (vl if vl is not None else torch.ones_like(vk)) * vk
                        pcs = sx + tr[:, None]
                        pcs = torch.cat([pcs[..., :2], pcs[..., 2:].clamp(min=0.05)], -1)
                        lpt = lpt + args.wrep_surf * smooth_l1(project(pcs, b["K"], bs["A"]) / S, su.detach() / S, sv, 0.02)
                    if hv.any():
                        gv = gv_all
                        wv = (hv * vk).expand(-1, gv.shape[1])
                        if args.joint_mask_sources:  # masked tips: no GT-mesh target on their distal phalanx
                            wv = wv * torch.where(dense_tipj >= 0, b["j3w"][:, dense_tipj.clamp(min=0)], torch.ones_like(b["j3w"][:, :1]))
                        lpt = lpt + args.wsurf_gt * (smooth_l1(su / S, project(gv, b["K"], bs["A"]) / S, wv * g2, 0.02)
                                                     + args.w3d * smooth_l1(sx, gv - b["j3d"][:, :1], wv, 0.01))
                        tv = torch.where(hv[..., None] > 0, gv, tv)  # anchors / edges / occlusion prior from the GT mesh
                        tv_img = torch.where(hv[..., None] > 0, gv, tv_img)
                    tj_ = torch.where(hv[..., None] > 0, b["j3d"], t_j)  # joints consistent with that mesh
                    wp = vk * (1 - hv) * kdw[:, None]
                    lpt = lpt + args.wsurf_prior * (smooth_l1(su / S, project(tv_img, b["K"], bs["A"]) / S, wp * g2, 0.02)
                                                    + smooth_l1(sx, tv - t_j[:, :1], wp, 0.01))
                    if args.wsmooth > 0:  # high-frequency energy of the predicted surface (3D and 2D): no spikes
                        c3, c2 = sx.mean(1, keepdim=True), (su / S).mean(1, keepdim=True)
                        if sub:  # weighted fit of the basis to the kept points only, read back at those points
                            def lowpass(x):
                                full = x.new_zeros(x.shape[0], 778, x.shape[2])
                                full[:, vidx_t] = x
                                w = torch.zeros(x.shape[0], 778, device=dev)
                                w[:, vidx_t] = 1.0
                                return smooth_surface(full, spec_basis, w, lam=1e-3)[:, vidx_t]
                        else:
                            lowpass = lambda x: smooth_surface(x, spec_basis)
                        lpt = lpt + args.wsmooth * (smooth_l1(sx, lowpass(sx - c3) + c3, vk, 0.01)
                                                    + smooth_l1(su / S, lowpass(su / S - c2) + c2, vk, 0.02))
                    ea, eb = dense_edges[:, 0], dense_edges[:, 1]  # keep the teacher's edge lengths (hand-like surface)
                    el = ((sx[:, ea] - sx[:, eb]).norm(dim=-1) - ((tv[:, ea] - tv[:, eb]).norm(dim=-1))).abs()
                    lpt = lpt + args.wsurf_edge * (el * vk[:, :1]).mean() * 100
                    # anchor: each point keeps the teacher's position along / distance from its bone
                    pt_, pr_, pl_ = bone_coords(sx, pxyz[:, :21], *dense_anchor)
                    tt_, tr_, _ = bone_coords(tv - tj_[:, :1], tj_ - tj_[:, :1], *dense_anchor)
                    lanc = ((pt_ - tt_).abs() * pl_ + (pr_ - tr_).abs()) * vk
                    lpt = lpt + args.wsurf_anchor * lanc.sum() / vk.sum().clamp(min=1.0) * 100
                    msel = (b["has_mask"] > 0) & (b["mask_kind"] == 0) & (b["present"] > 0)  # hand-only instance masks
                    if msel.any():
                        tuv = project(tv_img, b["K"], bs["A"]) / S  # mesh prior (teacher) in the same view
                        jw = torch.where(vis[:, [0, 9], None] > 0, gt2d[:, [0, 9]], t_uv[:, [0, 9]] * ratio) / S
                        lin, lcov = mask_terms(su[msel] / S, b["mask"][msel], prior_uv=tuv[msel], wrist=jw[msel])
                        lpt = lpt + args.wsurf_in * lin + args.wsurf_cov * lcov
                        if args.wjoint_in > 0:  # 2D joints + bone points inside their hand's mask (occlusion-aware via the teacher)
                            tj2 = t_uv * ratio
                            tjb = torch.cat([tj2, bone_points(tj2, 2)], 1) / S
                            jin, _ = mask_terms(puv[msel, :nb] / S, b["mask"][msel], prior_uv=tjb[msel], wrist=jw[msel])
                            lpt = lpt + args.wjoint_in * jin
                    if "mano_verts" in out:  # parametric mesh: teacher / GT mesh + joints, the 2D joints, the instance mask
                        mv, mj, mp = out["mano_verts"], out["mano_joints"], out["mano_params"]
                        wm = (b["present"] * b["side_ok"])[:, None].expand(-1, mv.shape[1]) * (1 + hv) * (kdw[:, None] + hv).clamp(max=1)
                        lm = smooth_l1(mv, tv_m - tj_[:, :1], wm, 0.01) + smooth_l1(mj, t_rel, valid, 0.01) \
                            + args.w3d * smooth_l1(mj, g_rel, w3j, 0.01, norm=n3 * 21)
                        tz = torch.where(b["has3d"] > 0, b["j3d"][:, 0, 2], t_j[:, 0, 2]).float().clamp(min=0.1)
                        tgt = torch.where(vis[..., None] > 0, gt2d, puv[:, :21].detach())
                        wr = torch.where(vis > 0, torch.ones_like(vis), 0.5 * valid)
                        wr = wr * ((wr > 0).sum(1, keepdim=True) >= 4)
                        tr = fit_translation(mj.detach().float(), tgt.float(), b["K"].float(), bs["A"].float(), wr.float(), tz)
                        pj = mj + tr[:, None]
                        pj = torch.cat([pj[..., :2], pj[..., 2:].clamp(min=0.05)], -1)
                        lm = lm + smooth_l1(project(pj, b["K"], bs["A"]) / S, tgt / S, wr, 0.02)
                        msel = (b["has_mask"] > 0) & (b["mask_kind"] == 0) & (b["present"] > 0)
                        if msel.any():  # silhouette of the placed mesh vs the instance mask (occlusion-aware)
                            pv = mv + tr[:, None]
                            pv = torch.cat([pv[..., :2], pv[..., 2:].clamp(min=0.05)], -1)
                            tuv = project(tm, b["K"], bs["A"]) / S
                            jw = torch.where(vis[:, [0, 9], None] > 0, gt2d[:, [0, 9]], t_uv[:, [0, 9]] * ratio) / S
                            lin_m, lcov_m = mask_terms(project(pv, b["K"], bs["A"])[msel] / S, b["mask"][msel], prior_uv=tuv[msel], wrist=jw[msel])
                            lm = lm + args.wsil * (lin_m + lcov_m)
                        lpt = lpt + args.wmano * (lm + 1e-3 * mp[:, 6:].pow(2).mean())
                si = model.keypoint_head.surface_idx
                if len(si):
                    tv = t_out["pred_vertices"][:, si] + t_out["pred_cam_t"][:, None]  # teacher surface, camera frame
                    vk = (b["present"] * b["side_ok"])[:, None].expand(-1, len(si))
                    lpt = lpt + 0.5 * (smooth_l1(puv[:, 21:] / S, project(tv, b["K"], bs["A"]) / S, vk, 0.02)
                                       + smooth_l1(pxyz[:, 21:], tv - t_j[:, :1], vk, 0.01))
            loss = args.w2d * l2d + args.w3d * l3d + args.wroot * lroot + args.wkd * lkd + args.wmask * lmask \
                + args.wverify * lver + args.wpt * lpt
            if args.freeze_decoder:
                loss = args.wmask * lmask + args.wverify * lver + args.wpt * lpt
            finite = torch.isfinite(loss.detach()).float()
            if ddp:  # one decision for every rank: a rank-local skip would leave the others waiting in backward's all-reduce
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if finite.item() < 1:
                if rank == 0:
                    print(f"step {step}: non-finite loss, skipping", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(gn) or (args.skip_gn and gn > args.skip_gn):  # non-finite grads / loss spike (grads are all-reduced, so every rank skips)
                if rank == 0:
                    print(f"step {step}: grad norm {gn.item():.3g}, skipping update", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            step += 1
            seen += x_t.shape[0] * world
            if rank == 0 and step % 50 == 0:
                dt = time.time() - t_last
                log = dict(loss=loss.item(), l2d=l2d.item(), l3d=l3d.item(), lroot=lroot.item(), lkd=lkd.item(),
                           lmask=lmask.item(), lver=lver.item(), lpt=lpt.item(), lpart=lpart.item(), lpseg=lpseg.item(),
                           **({"lfing": lfing.item()} if args.w_part_finger > 0 else {}),
                           grad_norm=gn.item(), lr=opt.param_groups[0]["lr"], img_per_s=seen / dt)
                print(f"step {step} " + " ".join(f"{k}={v:.4g}" for k, v in log.items()), flush=True)
                if run:
                    run.log({f"train/{k}": v for k, v in log.items()}, step=step)
                t_last, seen = time.time(), 0
            if step % args.save_every == 0:
                save(step)
            if step % args.eval_every == 0:
                evaluate(step)
        epoch += 1
    save(step, final=step >= args.steps)
    if step % args.eval_every:
        evaluate(step)
    if run:
        run.finish()
    if ddp:
        dist.destroy_process_group()


@torch.no_grad()
def eval_freihand_teacher(model, cfg, root, num, size):
    """FreiHAND PA-MPJPE of a (fine-tuned) decoder on its own frozen backbone at `size`."""
    from bench.common import procrustes
    from bench.datasets.freihand import FreiHANDEval
    from fah import HandEstimator

    ds = FreiHANDEval(root)
    ds = ds.subset(np.linspace(0, len(ds) - 1, num).astype(int))
    model.eval()
    est = HandEstimator(model, cfg, input_size=size)
    errs = []
    for i in range(len(ds)):
        fr = ds.frame(i)
        g = fr["hands"][0]
        h = est.process_image(fr["image"], boxes=g["bbox"][None], is_right=[True], cam_int=fr["K"])[0]
        errs.append(np.linalg.norm(procrustes(h["keypoints_3d_mano"], g["joints3d"]) - g["joints3d"], axis=1).mean() * 1000)
    model.train()
    return dict(freihand_pa_mpjpe=float(np.mean(errs)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crops", required=True)
    p.add_argument("--sources", default="hot3d_lab,coco_wb,cmu_lab")
    p.add_argument("--weights", default="")
    p.add_argument("--student", default="", help="stage-1 (or stage-2) checkpoint; empty = frozen teacher backbone")
    p.add_argument("--student_size", type=int, default=0, help="student input size override (multiple of 16)")
    p.add_argument("--slim_decoder", action="store_true", help="train the CPU-friendly slim decoder instead")
    p.add_argument("--slim_dim", type=int, default=384)
    p.add_argument("--slim_layers", type=int, default=2)
    p.add_argument("--slim_kp_queries", type=int, default=21)
    p.add_argument("--decoder_depth", type=int, default=0, help="train / run only the first N decoder layers (0: all)")
    p.add_argument("--kp_head", action="store_true", help="direct keypoint head (21 joints + dense surface points)")
    p.add_argument("--kp_surface", type=int, default=64, help="dense surface points (farthest-point MHR hand vertices)")
    p.add_argument("--wpt", type=float, default=1.0)
    p.add_argument("--mano_head", action="store_true", help="dense layout: + parametric MANO mesh head (learned IK from the 2D points)")
    p.add_argument("--wmano", type=float, default=1.0)
    p.add_argument("--wsil", type=float, default=1.0, help="MANO mesh silhouette vs the instance mask")
    p.add_argument("--vis_head", action="store_true", help="dense layout: per-point visibility head")
    p.add_argument("--wjoint_in", type=float, default=0.0, help="dense layout: 2D joints / bone points inside the instance mask")
    p.add_argument("--wsmooth", type=float, default=0.0, help="dense surface: penalty on the part outside the K lowest Laplacian modes")
    p.add_argument("--smooth_k", type=int, default=256)
    p.add_argument("--wvis", type=float, default=0.5)
    p.add_argument("--vis_gate", action="store_true", help="down-weight the 2D surface terms of hidden points (0.2)")
    p.add_argument("--wrep", type=float, default=0.0, help="dense keypoints: reprojection of the 3D joints (LS translation) onto the 2D")
    p.add_argument("--kp_only", action="store_true", help="decoder-free: train encoder + keypoint head only (with --kp_head)")
    p.add_argument("--kp_dense", action="store_true", help="dense MHR-free layout: 21 joints + 40 bone points + 778 MANO vertices")
    p.add_argument("--kp_dim", type=int, default=96, help="keypoint-head feature width")
    p.add_argument("--kp_stride", type=int, default=4, choices=[4, 8], help="keypoint heatmap stride (8: light CPU head)")
    p.add_argument("--kp_depth", action="store_true", help="decoder-free: learned root depth (teacher KD + GT) for the translation")
    p.add_argument("--wsurf_prior", type=float, default=0.3, help="dense surface points: weak prior to the teacher mesh")
    p.add_argument("--wsurf_gt", type=float, default=1.0, help="dense surface points: annotated mesh vertices (2D + 3D)")
    p.add_argument("--wsurf_edge", type=float, default=0.3, help="dense surface points: template edge-length consistency")
    p.add_argument("--wsurf_in", type=float, default=1.0, help="dense surface points: inside the instance mask")
    p.add_argument("--wsurf_cov", type=float, default=1.0, help="dense surface points: cover the instance-mask boundary")
    p.add_argument("--wsurf_anchor", type=float, default=0.5, help="dense surface points: keep position along / distance from the anchor bone")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"), help="template connectivity for the edge regulariser")
    p.add_argument("--p_dark", type=float, default=0.0, help="prob. of low-light student views (teacher sees the clean view)")
    p.add_argument("--teacher_regressor", default="", help="MHR -> MANO21 joint regressor for the teacher's KD joints (e.g. the kinematic refit)")
    p.add_argument("--wrep_surf", type=float, default=0.0, help="3D surface points (at the --wrep translation) reproject onto the model's own visible 2D surface points")
    p.add_argument("--dense_n", type=int, default=0, help="number of template vertices as surface points (0 = all 778)")
    p.add_argument("--dense_sel", default="fps", choices=["fps", "fingers"], help="surface vertex selection")
    p.add_argument("--p_ego", type=float, default=0.0, help="per-degradation prob. of ego_degrade student views (blur, occluders, low res, noise)")
    p.add_argument("--p_hocc", type=float, default=0.0, help="per-type prob. of hand-aware student occlusion (wrist band / pod, fingertip occluder)")
    p.add_argument("--p_shadow", type=float, default=0.0, help="prob. of a hand-shaped cast shadow on negative views (LabeledCrops._shadow)")
    p.add_argument("--raw_tip_sources", default="", help="human-clicked 2D pools (<pool>_kin with a BASE_POOL): their raw tips = the "
                   "visible finger end, used by --raw_tips / --tipend_w")
    p.add_argument("--raw_tips", action="store_true", help="fingertip arm: those pools' 2D tips = the raw clicks (no _kin tip shift)")
    p.add_argument("--tip_shift", type=float, default=0.0, help="fingertip arm: all other tip targets (2D / 3D labels, teacher, "
                   "external KD) moved this x the distal phalanx further along DIP -> tip")
    p.add_argument("--w_partseg", type=float, default=0.0, help="part segmentation head (17 classes, keypoint-head queries) + its CE loss")
    p.add_argument("--w_part", type=float, default=0.0, help="part-aware dense alignment (part_loss) on samples with masks_parts labels")
    p.add_argument("--w_part_cover", type=float, default=0.0, help="part_loss cover term: every part region spanned by its visible points")
    p.add_argument("--w_part_finger", type=float, default=0.0, help="part_finger_loss: a dense finger point outside its whole finger's "
                   "region (e.g. on another finger) is pulled to it, on top of part_loss's segment term")
    p.add_argument("--part_dirs", default="", help="pool:dir,... part-label version per pool (default <pool>/masks_parts)")
    p.add_argument("--part_sources", default="", help="pools whose part labels are used (default: every pool that has them)")
    p.add_argument("--part_no_gt", action="store_true", help="part region / cover terms only on samples without a GT mesh")
    p.add_argument("--part_routes", default="", help="only part labels of these train.part_masks routes, e.g. mesh+mask,model+mask")
    p.add_argument("--partseg_finger_w", type=float, default=1.0, help="part head CE weight of the 15 finger-segment classes")
    p.add_argument("--partseg_grad", type=float, default=1.0, help="scale of the part head losses' gradient into the shared heatmap features")
    p.add_argument("--part_hr", action="store_true", help="part head refined to half the crop resolution from a light image stem")
    p.add_argument("--part_depth", type=int, default=2, help="3x3 convs in the part head (on the 2x2-pooled heatmap features)")
    p.add_argument("--tip_from_parts", type=float, default=0.0, help="fingertip study: visible tip ends (<= x the distal phalanx) from the silhouette-clipped part labels")
    p.add_argument("--train_only_parts", action="store_true", help="train the part head only (part_* parameters of the keypoint head)")
    p.add_argument("--w_partmask", type=float, default=0.0, help="part head: BCE of the union of the parts vs the hand mask (all masked samples; negatives empty)")
    p.add_argument("--partmask_dice", type=float, default=1.0, help="weight of the soft Dice next to that BCE (0 = BCE only)")
    p.add_argument("--tip_mask_ext", type=float, default=0.0, help="fingertip study: labelled tips (2D, 3D, teacher / external KD) "
                   "moved out to where the hand-only mask ends, at most x the distal phalanx (samples with masks)")
    p.add_argument("--tip_mask_dense", action="store_true", help="fingertip study: also stretch the target surfaces' distal phalanges")
    p.add_argument("--w_finger_id", type=float, default=0.0, help="finger-identity margin loss on the 2D joints (finger_id_loss)")
    p.add_argument("--finger_id_margin", type=float, default=0.02, help="margin in crop units")
    p.add_argument("--tipend_w", type=float, default=0.0, help="fingertip arm: 2D loss of the dense surface points nearest the MANO "
                   "tip vertices to the raw clicked tips (fingertip-end points; the 21 joints keep their convention)")
    p.add_argument("--p_hoh", type=float, default=0.0, help="prob. of another hand pasted over the student view's hand (train.finetune.hand_over_hand)")
    p.add_argument("--min_crop", type=float, default=0.0, help="train views of tall boxes cover min_crop x the long side (match the estimator's min_crop)")
    p.add_argument("--box_jitter", type=float, default=1.0, help="scale of the crop scale / shift jitter (detector-box mismatch)")
    p.add_argument("--checkpoint", default="checkpoints/sam-3d-body-dinov3/model.ckpt")
    p.add_argument("--size", type=int, default=384, help="view / teacher crop resolution")
    p.add_argument("--freihand", default="")
    p.add_argument("--eval_num", type=int, default=500)
    p.add_argument("--val_num", type=int, default=2000)
    p.add_argument("--out", required=True)
    p.add_argument("--run_name", default=None)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, required=True, help="training samples per process; choose for your setup")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--enc_lr_mult", type=float, default=1.0)
    p.add_argument("--grad_ckpt", action="store_true", help="activation checkpointing in the student encoder")
    p.add_argument("--train_blocks", type=int, default=0, help="teacher mode: also train the last N backbone blocks")
    p.add_argument("--w3d_sources", default="", help="per-source 3D loss scale, e.g. egodex_test:0.3")
    p.add_argument("--joint_mask_sources", default="", help="pool:j+j,... joints (OpenPose order) without 2D / 3D targets in that pool, "
                   "e.g. arctic_ego_lab:4 (thumb tip); a masked tip also has no GT-mesh target on its distal phalanx")
    p.add_argument("--no3d_sources", default="", help="pools trained 2D-only: no GT 3D joint / root / depth / GT-mesh targets "
                   "(--w3d_sources scales the joint terms only); their dense points get the teacher prior like the 2D pools")
    p.add_argument("--kd_mesh_joints", action="store_true", help="teacher-KD joints = MANO J_regressor joints of its MANO-topology mesh")
    p.add_argument("--kd_sources", default="", help="per-source teacher-KD scale, e.g. hard_lab:0 (mined teacher failures)")
    p.add_argument("--mask_head", action="store_true", help="train the hand instance-mask head (SAM pseudo-masks)")
    p.add_argument("--init_ckpt", default="", help="initialise the decoder from a finetune checkpoint")
    p.add_argument("--freeze_decoder", action="store_true", help="train only the new heads (mask / verify)")
    p.add_argument("--verify_head", action="store_true", help="presence + handedness-consistency head")
    p.add_argument("--p_wrong", type=float, default=0.25, help="prob. of mirrored wrong-handedness views")
    p.add_argument("--wverify", type=float, default=0.5)
    p.add_argument("--wmask", type=float, default=1.0)
    p.add_argument("--min_mask_score", type=float, default=0.7, help="drop SAM masks below this predicted IoU")
    p.add_argument("--arm_mask_sources", default="visor_lab", help="pools whose masks are hand + arm (VISOR convention)")
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--skip_gn", type=float, default=0, help="skip updates whose pre-clip grad norm exceeds this (0: off)")
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--w2d", type=float, default=1.0)
    p.add_argument("--w3d", type=float, default=3.0)
    p.add_argument("--wroot", type=float, default=0.3)
    p.add_argument("--wkd", type=float, default=0.5)
    p.add_argument("--teacher_ckpt", default="", help="finetune checkpoint for the KD teacher (default: off-the-shelf)")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases (opt-in; WANDB_PROJECT / WANDB_ENTITY from the environment)")
    main(p.parse_args())
