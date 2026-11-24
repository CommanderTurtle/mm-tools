"""Do the outputs of one forward pass agree with each other? Dense points predicted directly (FAH: joints + MANO-topology
surface points) vs parameter-first outputs (MANO / MHR parameters -> forward kinematics -> mesh -> joints -> projection);
the paper's dense-points-versus-parameters table and its supplementary paragraph.

On bench.run pickles with camera-frame GT (HOT3D, FreiHAND, DexYCB); per matched hand; reported values are medians over
hands of per-hand medians over points, 2D distances in units of the GT hand size (the projected GT joints' box):
  A  3D joints vs the direct 2D: the 3D joints projected at their own translation (A_own) and at the translation that best
     fits the direct 2D (A_fit: shape / articulation only) vs the direct 2D joints (joints2d_head if saved, else joints2d).
  B  the output mesh vs the image evidence: its joints (MANO regressor + fingertip vertices of verts3d) projected vs the direct
     2D (B_joints); with GT masks, the mesh silhouette IoU / boundary F vs the mask (B_iou, B_bf) and the same for the
     direct dense surface rendered as a mesh (B_iou_dense, B_bf_dense).
  C  the dense points vs the mesh fitted to them (kind "dense"): 3D residual of verts3d_dense at the dense subset's vertices
     after a rigid alignment (C_3d, mm) and 2D residual of the direct 2D surface points (surface2d_head) vs the mesh projected
     (C_2d).
  D  the dense joints vs the joints of the output mesh (kind "dense"): 3D after a rigid alignment (D_3d, mm), 2D (D_2d).
  plus accuracy: median 2D error of the direct 2D joints (e2d) and of the projected mesh joints (emesh), PA-MPJPE of the
  3D joints (pa3d), split into fingertips (tip3d) / other joints (nontip3d).
"""
import glob
import json
import os
import pickle

import numpy as np

from bench.common import match_hands, procrustes

TIPS = [4, 8, 12, 16, 20]
KEYS = ("A_own", "A_fit", "B_joints", "B_iou", "B_bf", "B_iou_dense", "B_bf_dense", "C_3d", "C_2d", "D_3d", "D_2d", "e2d", "emesh", "pa3d",
        "tip3d", "nontip3d")


def proj(X, K):
    X = np.asarray(X, np.float64)
    return (X @ K.T)[:, :2] / np.clip(X[:, 2:], 1e-4, None)


def best_translation(X, uv, K):
    """Root-relative 3D points X translated so that they best reproject onto uv (linear least squares)."""
    X = X - X[:1]
    d = uv - K[:2, 2]
    n = len(X)
    M = np.zeros((2 * n, 3))
    M[:n, 0], M[n:, 1], M[:n, 2], M[n:, 2] = K[0, 0], K[1, 1], -d[:, 0], -d[:, 1]
    r = np.concatenate([d[:, 0] * X[:, 2] - K[0, 0] * X[:, 0], d[:, 1] * X[:, 2] - K[1, 1] * X[:, 1]])
    return X + np.linalg.lstsq(M, r, rcond=None)[0]


def rigid(A, B):
    """A rigidly (rotation + translation, no scale) aligned to B."""
    a, b = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - a).T @ (B - b))
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    return (A - a) @ R.T + b


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan


class Scorer:
    """Accumulates the per-hand measures of one run. mesh_joints: (778, 3) -> (21, 3) (bench.common.VertexJoints); dense: the
    dense subset's MANO vertex indices; faces / masks / silhouette: only for the mask measures (masks: frame key -> RLE dict;
    silhouette(verts, faces, K, hw) -> bool mask, e.g. fah.vis.render_instance_masks(...) == 1)."""

    def __init__(self, mesh_joints, dense, kind="dense", faces=None, masks=None, silhouette=None):
        self.vj, self.dense, self.kind, self.faces, self.masks, self.sil = mesh_joints, np.asarray(dense), kind, faces, masks, silhouette
        self.acc = {k: [] for k in KEYS}

    def add_frame(self, fr):
        acc = self.acc
        K = np.asarray(fr["K"], np.float64)
        pairs, _ = match_hands(fr["pred"], fr["gt"])
        for gi, pi in pairs:
            if pi is None:
                continue
            g, p = fr["gt"][gi], fr["pred"][pi]
            gj = np.asarray(g["joints3d"], np.float64)
            g2 = proj(gj, K)
            S = max(np.ptp(g2[:, 0]), np.ptp(g2[:, 1]), 1.0)
            d2 = p.get("joints2d_head") if p.get("joints2d_head") is not None else p.get("joints2d")
            j3 = p.get("joints3d")
            if d2 is None or j3 is None:
                continue
            d2, j3 = np.asarray(d2, np.float64), np.asarray(j3, np.float64)
            acc["e2d"].append(np.median(np.linalg.norm(d2 - g2, axis=1) / S))
            ej = np.linalg.norm(procrustes(j3, gj) - gj, axis=1) * 1000
            acc["pa3d"].append(ej.mean())
            acc["tip3d"].append(ej[TIPS].mean())
            acc["nontip3d"].append(np.delete(ej, TIPS).mean())
            acc["A_own"].append(np.median(np.linalg.norm(proj(j3, K) - d2, axis=1) / S))
            acc["A_fit"].append(np.median(np.linalg.norm(proj(best_translation(j3, d2, K), K) - d2, axis=1) / S))
            v = p.get("verts3d")
            if v is None or np.shape(v)[0] != 778 or not np.isfinite(v).all():
                continue
            v = np.asarray(v, np.float64)
            mj3 = self.vj(v)
            um = proj(mj3, K)
            acc["B_joints"].append(np.median(np.linalg.norm(um - d2, axis=1) / S))
            acc["emesh"].append(np.median(np.linalg.norm(um - g2, axis=1) / S))
            vd = p.get("verts3d_dense")
            if self.masks is not None:
                key = fr["key"].split("/")[-1]
                if key in self.masks:
                    from bench.seg.evaluate import boundary_f
                    from bench.seg.rle import decode

                    gm = decode(self.masks[key]) > 0
                    pm = self.sil(v, self.faces, K, gm.shape)
                    acc["B_iou"].append(iou(pm, gm))
                    acc["B_bf"].append(boundary_f(gm, pm))
                    if vd is not None and np.shape(vd)[0] == 778:  # the direct dense surface as a mesh (MANO faces)
                        pd = self.sil(np.asarray(vd, np.float64), self.faces, K, gm.shape)
                        acc["B_iou_dense"].append(iou(pd, gm))
                        acc["B_bf_dense"].append(boundary_f(gm, pd))
            if self.kind != "dense":
                continue
            acc["D_3d"].append(np.median(np.linalg.norm(rigid(j3, mj3) - mj3, axis=1)) * 1000)
            acc["D_2d"].append(np.median(np.linalg.norm(proj(j3, K) - um, axis=1) / S))
            if vd is not None and np.shape(vd)[0] == 778:
                vd = np.asarray(vd, np.float64)[self.dense]
                vm = v[self.dense]
                acc["C_3d"].append(np.median(np.linalg.norm(rigid(vd, vm) - vm, axis=1)) * 1000)
                s2 = p.get("surface2d_head")
                if s2 is not None and len(s2) >= 40 + len(self.dense):
                    s2 = np.asarray(s2, np.float64)[-len(self.dense):]
                    acc["C_2d"].append(np.median(np.linalg.norm(proj(vm, K) - s2, axis=1) / S))

    def summary(self):
        out = {}
        for k, x in self.acc.items():
            if x:
                x = np.asarray(x, np.float64)
                x = x[np.isfinite(x)]
                out[k] = dict(median=float(np.median(x)), mean=float(np.mean(x)), n=int(len(x)),
                              **({"share_gt_0p05": float((x > 0.05).mean())} if k.startswith(("A_", "B_j", "C_2")) else {}))
        return out


def score_run(run_dir, scorer, every=1):
    for path in sorted(glob.glob(os.path.join(run_dir, "*.pkl"))):
        for fr in pickle.load(open(path, "rb"))["frames"][::every]:
            scorer.add_frame(fr)
    return scorer.summary()


def load_masks(path):
    """bench2d-format mask annotations (one hand per image) -> {image key: mask RLE}."""
    ann = json.load(open(path))
    return {r["id"].split("/")[-1]: r["hands"][0]["mask_rle"] for r in ann["images"] if r.get("hands")}
