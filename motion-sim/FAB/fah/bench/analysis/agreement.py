"""How well each output modality of a method agrees with the ground truth and with its own other outputs, on a 3D benchmark
run (bench.run pickles with camera-frame GT, e.g. HOT3D; the supplementary table of direct and projected outputs).

Per matched hand (bench.common.match_hands, box IoU >= 0.3, as bench.evaluate):
- 2D output: `joints2d` vs the GT joints projected with the frame K;
- 3D joints projected: the method's camera-frame `joints3d` projected with K;
- mesh joints projected: the mesh joints (MANO regressor + fingertip vertices, bench.common.VertexJoints) of `verts3d`
  (778 vertices only), projected;
- PA-MPJPE (mm) of `joints3d` and of the mesh joints;
- internal disagreement: per hand, the median distance between the 2D output and the projected 3D joints / mesh joints.
2D errors are in units of the GT hand size (the projected GT joints' box); PCK@0.05 / @0.1 over all 21 joints.
"""
import glob
import os
import pickle

import numpy as np

from bench.common import match_hands, procrustes


def proj(X, K):
    X = np.asarray(X, np.float64)
    return (X @ K.T)[:, :2] / np.clip(X[:, 2:], 1e-4, None)


def score_frames(frames, mesh_joints, every=1):
    """frames: bench.run frames (K, gt, pred); mesh_joints: (778, 3) vertices -> (21, 3) joints (bench.common.VertexJoints)."""
    acc = {k: [] for k in ("e2d", "e3d", "emesh", "pa3d", "pamesh", "d_2d_3d", "d_2d_mesh")}
    for fr in frames[::every]:
        K = np.asarray(fr["K"], np.float64)
        pairs, _ = match_hands(fr["pred"], fr["gt"])
        for gi, pi in pairs:
            if pi is None:
                continue
            g, p = fr["gt"][gi], fr["pred"][pi]
            gj = np.asarray(g["joints3d"], np.float64)
            g2 = proj(gj, K)
            S = max(np.ptp(g2[:, 0]), np.ptp(g2[:, 1]), 1.0)
            n2 = lambda u: np.linalg.norm(np.asarray(u, np.float64) - g2, axis=1) / S  # noqa: E731
            j2 = p.get("joints2d")
            if j2 is not None:
                acc["e2d"].append(n2(j2))
            j3 = p.get("joints3d")
            if j3 is not None:
                u3 = proj(j3, K)
                acc["e3d"].append(n2(u3))
                acc["pa3d"].append(np.linalg.norm(procrustes(np.asarray(j3, np.float64), gj) - gj, axis=1).mean() * 1000)
                if j2 is not None:
                    acc["d_2d_3d"].append(np.median(np.linalg.norm(u3 - np.asarray(j2), axis=1) / S))
            v = p.get("verts3d")
            if v is not None and np.shape(v)[0] == 778 and np.isfinite(v).all():
                mj = mesh_joints(np.asarray(v, np.float64))
                um = proj(mj, K)
                acc["emesh"].append(n2(um))
                acc["pamesh"].append(np.linalg.norm(procrustes(mj, gj) - gj, axis=1).mean() * 1000)
                if j2 is not None:
                    acc["d_2d_mesh"].append(np.median(np.linalg.norm(um - np.asarray(j2), axis=1) / S))
    return acc


def summarize(acc):
    out = {}
    for k in ("e2d", "e3d", "emesh"):
        if acc[k]:
            e = np.concatenate(acc[k])
            out[k] = dict(pck05=float((e < 0.05).mean()), pck10=float((e < 0.1).mean()), median=float(np.median(e)), hands=len(acc[k]))
    for k in ("pa3d", "pamesh"):
        if acc[k]:
            out[k] = float(np.mean(acc[k]))
    for k in ("d_2d_3d", "d_2d_mesh"):
        if acc[k]:
            out[k] = dict(median=float(np.median(acc[k])), share_gt_0p05=float((np.asarray(acc[k]) > 0.05).mean()))
    return out


def score_run(run_dir, mesh_joints, every=1):
    """A bench.run directory (one pickle per clip) -> summarize(...); `every` subsamples each clip's frames."""
    acc = None
    for path in sorted(glob.glob(os.path.join(run_dir, "*.pkl"))):
        a = score_frames(pickle.load(open(path, "rb"))["frames"], mesh_joints, every)
        acc = a if acc is None else {k: acc[k] + a[k] for k in acc}
    return summarize(acc or score_frames([], mesh_joints))
