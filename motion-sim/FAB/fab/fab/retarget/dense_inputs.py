"""Inputs of fab-retarget's points mode (the dense-point term), built from a fab-world output:

register_points: FAB's dense points (the predictor's dense_surface_cam, camera frame; fab-world's work/ours/fab/<clip>_raw.npz: pts,
  surface_idx) registered on the reconstruction's world body per frame by a similarity at the same SMPL vertex ids; the per-point residual is smoothed (Gaussian,
  sigma 2 input frames) and added to the body's own vertices -> pts (n, 468, 3), pts_idx, pts_info.
An input without FAB's dense points (e.g. motion capture) gives the term the body's own vertices at the same ids;
points_unavailable names the cause (fab-retarget prints it as a WARNING and records it in <clip>_g1.json).
"""
import json

import numpy as np


def umeyama(src, dst):
    """Similarity (s, R, t) minimising |s R src + t - dst|^2; src / dst (N, 3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (xs ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def points_unavailable(raw, surface_idx):
    """Why the raw gives no dense points for these SMPL vertex ids, or None when it does."""
    from pathlib import Path

    if raw is None or not Path(raw).exists():
        return f"no FAB output {raw} (an input that is not a fab-world output)"
    r = np.load(raw, allow_pickle=True)
    if "pts" not in r.files or "surface_idx" not in r.files:
        return f"{raw} has no dense points (no pts / surface_idx)"
    n, m = len(np.asarray(r["surface_idx"])), len(surface_idx)
    if n == 0:
        return f"{raw} has no dense points (written by a model without the dense head)"
    if not np.array_equal(np.asarray(r["surface_idx"]), surface_idx):
        return (f"surface_idx mismatch: the {n} dense point ids of {raw} are not the correspondence's {m}"
                + (" (other ids or order)" if n == m else "") + "; rebuild the correspondence for the model that wrote it "
                "(python -m fab.retarget.dense_corr --ckpt <its model.pt>)")
    return None


def register_points(recon, raw, surface_idx, sigma=2.0):
    """{pts, pts_idx, pts_info} for the reconstruction's frames, or None when the raw has no dense points for these ids
    (points_unavailable: the cause)."""
    from scipy.ndimage import gaussian_filter1d

    from bench.videomimic.recon import fill_invalid

    if points_unavailable(raw, surface_idx) is not None:
        return None
    r = np.load(raw, allow_pickle=True)
    z = np.load(recon, allow_pickle=True)
    V = np.asarray(z["verts"], np.float64)[:, surface_idx]
    frames = np.asarray(z["frames"], int)
    k = np.clip(np.searchsorted(np.asarray(r["frames"], int), frames), 0, len(r["frames"]) - 1)
    X = np.asarray(r["pts"], np.float64)[k]
    valid = np.asarray(r["valid"], bool)[k] & np.asarray(z["valid"], bool)[: len(frames)]
    res, sc = np.zeros_like(V), np.ones(len(V))
    for i in np.flatnonzero(valid):
        sc[i], R, t = umeyama(X[i], V[i])
        res[i] = sc[i] * X[i] @ R.T + t - V[i]
    res = fill_invalid(res, valid)
    if sigma > 0:
        res = gaussian_filter1d(res, sigma, axis=0, mode="nearest")
    rn = np.linalg.norm(res[valid], axis=-1) if valid.any() else np.zeros(1)
    return {"pts": (V + res).astype(np.float32), "pts_idx": np.asarray(surface_idx),
            "pts_info": json.dumps({"residual_mm_median": float(np.median(rn) * 1000), "residual_mm_p95": float(np.percentile(rn, 95) * 1000),
                                    "sim_scale_median": float(np.median(sc[valid])) if valid.any() else 1.0, "pts_sigma": sigma,
                                    "valid_share": float(valid.mean())})}
