"""Planar regularisation of a reconstructed heightfield (2.5D), used by the terrain clean (scripts/g1_terrain_clean_pkg.py):
region growing over the edge-smoothed heights (non-steep cells; neighbours within grow_m in height and grad_tol in gradient),
each region -> its robust plane (horizontal: tilt < horiz_deg -> one level; else the sloped plane; residual p90 > flat_tol:
kept); riser / break cells -> the neighbouring plane closest to their own height (sharp steps); stair runs (>= 3 treads,
monotone heights along one horizontal direction) -> regular rise / tread boundaries.
Heightfields use the reconstruction's format (x0, y0, res, height (ny, nx), valid).
"""

import json

import numpy as np
from scipy import ndimage

# ---------- heightfield helpers
def grid_xy(hf):
    ny, nx = hf["height"].shape
    r = float(hf["res"])
    return np.meshgrid(float(hf["x0"]) + r * np.arange(nx), float(hf["y0"]) + r * np.arange(ny))


def to_cell(hf, xy):
    r = float(hf["res"])
    return (np.round((xy[..., 1] - float(hf["y0"])) / r).astype(int), np.round((xy[..., 0] - float(hf["x0"])) / r).astype(int))


def near_path(hf, path_xy, dist):
    from scipy.spatial import cKDTree

    X, Y = grid_xy(hf)
    d = cKDTree(path_xy).query(np.stack([X.ravel(), Y.ravel()], 1))[0].reshape(X.shape)
    return d < dist


def bilateral(H, sigma_s=2.0, sigma_h=0.02, iters=3, radius=3):
    """Iterated bilateral filter of a heightfield (cells, m)."""
    Hc = H.astype(np.float64)
    offs = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    ws = {o: np.exp(-(o[0] ** 2 + o[1] ** 2) / (2 * sigma_s ** 2)) for o in offs}
    for _ in range(iters):
        P = np.pad(Hc, radius, mode="edge")
        num, den = np.zeros_like(Hc), np.zeros_like(Hc)
        for (dy, dx), w0 in ws.items():
            S = P[radius + dy : radius + dy + Hc.shape[0], radius + dx : radius + dx + Hc.shape[1]]
            w = w0 * np.exp(-((S - Hc) ** 2) / (2 * sigma_h ** 2))
            num += w * S
            den += w
        Hc = num / den
    return Hc


def slope(H, res):
    gy, gx = np.gradient(H, res)
    return np.hypot(gx, gy), gx, gy


def regions(mask, H, tol, G=None, gtol=None):
    """Connected components of `mask` cells where 4-neighbours differ by < tol in H (and, with G (ny, nx, 2), by < gtol in
    the gradient) (union-find) -> label (ny, nx), 0 = none."""
    ny, nx = H.shape
    parent = np.arange(ny * nx)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    idx = np.arange(ny * nx).reshape(ny, nx)
    for a, b in ((idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])):
        ok = mask.ravel()[a.ravel()] & mask.ravel()[b.ravel()] & (np.abs(H.ravel()[a.ravel()] - H.ravel()[b.ravel()]) < tol)
        if G is not None:
            Gf = G.reshape(-1, 2)
            ok &= np.linalg.norm(Gf[a.ravel()] - Gf[b.ravel()], axis=1) < gtol
        for i, j in zip(a.ravel()[ok], b.ravel()[ok]):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri
    roots = np.array([find(i) for i in range(ny * nx)])
    lab = np.zeros(ny * nx, int)
    m = mask.ravel()
    _, inv = np.unique(roots[m], return_inverse=True)
    lab[m] = inv + 1
    return lab.reshape(ny, nx)


def edge_variant(hf, a):
    out = dict(hf)
    H = bilateral(np.asarray(hf["height"], np.float64), a.sigma_s, a.sigma_h, a.bil_iters)
    out["height"] = ndimage.median_filter(H, size=3, mode="nearest").astype(np.float32)
    return out


def planar_variant(hf, a, path_xy):
    """Planar primitives over the edge-smoothed heights (2.5D)."""
    res = float(hf["res"])
    He = np.asarray(edge_variant(hf, a)["height"], np.float64)
    s, gx, gy = slope(He, res)
    Hp = He.copy()
    lab_out = np.zeros(He.shape, int)
    # support regions: non-steep cells grown over small height / gradient steps, each fitted by a robust plane
    # (IRLS z = ax + by + c); horizontal (tilt < horiz_deg) -> its median level, else the plane; regions whose
    # residual p90 exceeds flat_tol are not planes and keep the smoothed heights
    cand = s < np.tan(np.radians(a.ramp_max_deg))
    lab = regions(cand, He, a.grow_m, np.stack([gx, gy], -1), a.grad_tol)
    min_cells = max(int(a.min_area / res ** 2), 4)
    planes = {}
    Xg, Yg = grid_xy(hf)
    for k in range(1, lab.max() + 1):
        m = lab == k
        if m.sum() < min_cells:
            continue
        A_ = np.stack([Xg[m], Yg[m], np.ones(m.sum())], 1)
        w = np.ones(m.sum())
        for _ in range(5):
            sol = np.linalg.lstsq(A_ * w[:, None], He[m] * w, rcond=None)[0]
            r_ = np.abs(A_ @ sol - He[m])
            w = 1.0 / np.maximum(r_ / 0.01, 1.0)
        if np.percentile(r_, 90) > a.flat_tol:
            continue
        if np.degrees(np.arctan(np.hypot(sol[0], sol[1]))) < a.horiz_deg:
            planes[k] = float(np.median(He[m]))
            Hp[m] = planes[k]
        else:
            Hp[m] = A_ @ sol
        lab_out[m] = k
    # riser / break cells (steep, between planes) -> the neighbouring plane closest to their own height; cells of
    # regions that are not planes keep their smoothed heights
    un = (lab_out == 0) & ~cand
    if un.any() and (lab_out > 0).any():
        cand = np.full(He.shape + (25,), np.nan)
        P = np.pad(np.where(lab_out > 0, Hp, np.nan), 2, constant_values=np.nan)
        for i, (dy, dx) in enumerate([(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]):
            cand[..., i] = P[2 + dy : 2 + dy + He.shape[0], 2 + dx : 2 + dx + He.shape[1]]
        dif = np.abs(cand - He[..., None])
        has = np.isfinite(dif).any(-1)
        pick = np.nanargmin(np.where(np.isfinite(dif), dif, np.inf), -1)
        best = np.take_along_axis(cand, pick[..., None], -1)[..., 0]
        sel = un & has
        Hp[sel] = best[sel]
    stairs = regularize_stairs(Hp, lab_out, planes, hf, a, path_xy)
    out = dict(hf)
    out["height"] = stairs["height"].astype(np.float32)
    out["plane_label"] = lab_out.astype(np.int32)
    out["stair_runs"] = json.dumps(stairs["runs"])
    return out


def regularize_stairs(Hp, lab, planes, hf, a, path_xy):
    """Stair runs among the horizontal planes near the path: >= 3 treads whose heights increase monotonically along one
    horizontal direction d (the mean riser gradient direction); tread boundaries -> straight lines across d at the
    riser midpoints, tread heights -> a regular rise where they deviate by < stair_snap_m."""
    res = float(hf["res"])
    near = near_path(hf, path_xy, 1.0)
    keys = [k for k in planes if (lab == k).sum() and ((lab == k) & near).sum() >= 4]
    runs = []
    if len(keys) < 3:
        return {"height": Hp, "runs": runs}
    s, gx, gy = slope(Hp, res)
    riser = (s > 1.0) & near
    if riser.sum() < 4:
        return {"height": Hp, "runs": runs}
    g = np.stack([gx[riser], gy[riser]], 1)
    # direction: principal axis of the riser gradients (sign = uphill)
    u, _, _ = np.linalg.svd(g.T @ g)
    d = u[:, 0]
    if (g @ d).sum() < 0:
        d = -d
    X, Y = grid_xy(hf)
    S = X * d[0] + Y * d[1]
    tr = sorted(((float(np.median(S[(lab == k) & near])), planes[k], k) for k in keys))
    hs = [t[1] for t in tr]
    # the longest monotone (increasing) chain along S with steps between rise_min and rise_max
    best, cur = [], [tr[0]]
    for t in tr[1:]:
        if a.rise_min < t[1] - cur[-1][1] < a.rise_max:
            cur.append(t)
        else:
            best = max(best, cur, key=len)
            cur = [t]
    best = max(best, cur, key=len)
    if len(best) < 3:
        return {"height": Hp, "runs": runs}
    rises = np.diff([t[1] for t in best])
    rise = float(np.median(rises))
    z0 = best[0][1]
    reg = [z0 + i * rise for i in range(len(best))]
    snap = all(abs(r - t[1]) < a.stair_snap_m for r, t in zip(reg, best))
    heights = reg if snap else [t[1] for t in best]
    # boundaries at the midpoints between consecutive treads' S extents
    ext = [(float(np.percentile(S[lab == t[2]], 95)), float(np.percentile(S[lab == t[2]], 5))) for t in best]
    cuts = [(ext[i][0] + ext[i + 1][1]) / 2 for i in range(len(best) - 1)]
    span = np.zeros(Hp.shape, bool)
    for t in best:
        span |= lab == t[2]
    span = ndimage.binary_closing(span, iterations=2) | span
    H2 = Hp.copy()
    band = span & (S >= ext[0][1]) & (S <= ext[-1][0])
    idx = np.searchsorted(np.array(cuts), S[band])
    H2[band] = np.array(heights)[idx]
    runs.append({"treads": len(best), "rise_m": rise, "regular": bool(snap), "dir": d.tolist(),
                 "heights": [float(h) for h in heights], "cuts_m": cuts})
    return {"height": H2, "runs": runs}
