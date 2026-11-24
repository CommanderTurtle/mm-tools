"""World-mode helpers shared by the world reconstruction: the GT-free bridge scale of camera centres and the world up
direction from GeoCalib's per-frame camera up vectors."""
import numpy as np


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-12)


def angle(a, b):
    return np.degrees(np.arccos(np.clip((unit(a) * unit(b)).sum(-1), -1, 1)))


def bridge_scale(ctr, refs, win=60, step=10, min_disp=0.3):
    """GT-free scale correction of camera centres ctr (n, 3) from reference metric cameras (e.g. DPVO + Depth Pro at the
    static patches, list of (n, 3) centres on the same frames): per reference the displacement-weighted median of the
    window displacement ratios |d ref| / |d ctr| (windows of `win` frames where ctr moves > min_disp m), then the
    median over the references (a broken reference only spoils its own windows / its own vote)."""
    out = []
    for c in refs:
        i = np.arange(0, len(ctr) - win, step)
        dv = np.linalg.norm(ctr[i + win] - ctr[i], axis=1)
        dr = np.linalg.norm(c[i + win] - c[i], axis=1)
        m = dv > min_disp
        if m.sum() < 3:
            continue
        r, w = dr[m] / dv[m], dv[m]
        o = np.argsort(r)
        out.append(float(r[o][np.searchsorted(np.cumsum(w[o]), 0.5 * w.sum())]))
    return float(np.median(out)) if out else float("nan")


def geocalib_up(T_wc, geo, frames):
    """World up from GeoCalib's per-frame camera up vectors (bench/world_geocalib.py) on the frames of T_wc: rotated to
    the world, robust mean (frames > 10 deg off the running mean dropped) -> (up or None, T_wc rows, geo rows)."""
    pos = {int(f): i for i, f in enumerate(frames)}
    k = [(pos[int(f)], j) for j, f in enumerate(geo["frames"]) if int(f) in pos]
    if len(k) < 5:
        return None, None, None
    ii, jj = np.array(k).T
    gw = unit(np.einsum("nij,nj->ni", T_wc[ii, :3, :3], geo["up_cam"][jj]))
    m = unit(np.median(gw, 0))
    for _ in range(3):
        keep = angle(gw, m) < 10
        m = unit(gw[keep].mean(0)) if keep.sum() >= 5 else m
    return m, ii, jj
