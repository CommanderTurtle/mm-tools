"""Terrain clean on the PACKAGE terrain of simulation packages (bench/scene_r2s_simpkg.py; fab-retarget step 4), on the
package's own 2 cm grid (no resampling), in the package frame (G1 world + dz):
  reliability  2 = observed (the reconstruction heightfield's valid cells, the package cell centre mapped back by 1 / s),
               1 = filled inside (unobserved holes <= 0.25 m^2 away from the grid border),
               0 = border-extrapolated (the export's nearest fill beyond the scan / recon border: the long ridges)
  clean        despike (> 3 cm off both the 10 cm median and a 3 x 3 grey open-close) -> holes from observed cells, border
               cells unchanged (flagged) -> simready.planar_variant (plane residual p90 <= 1 cm; level only within 1 deg) ->
               each plane back to its observed cells' median -> the RISE CAP: within 15 cm of any G1 foot path (ankle-roll
               links, every frame), and within 30 cm (xy) of any other robot link's path (seated pelvis, hands, knees, leaning torso),
               the clean surface may not rise more than 3 mm above the package terrain (lowering is free)
  structure    non-walkable cells: not reachable from the stance cells (feet within 8 cm of the terrain) over neighbours that
               differ <= max(3 cm, one cell's 45 deg) and more than 0.5 m above the nearest walkable cell (walls, railings,
               buildings: kept in collision, flagged for depth renders)
-> <out>/<clip>/terrain_hfield_2cm_clean.npz (x0, y0, res, height, reliability uint8, structure bool) + clean_report.json
(changed-cell shares, flat-patch bumpiness RMS before / after, structure / reliability shares, max rise near the feet).
    python scripts/g1_terrain_clean_pkg.py --v <packages root (clips/<clip>/package.json)> --out <dir> [--only clip] [--workers N]
fab-retarget runs it with --slivers 3 --sliver_obj_h 0.15 --pit_close 1 on the pinned reference (--ref p_pre).
"""

import argparse
import json
import os
import sys
import types
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PA = types.SimpleNamespace(sigma_s=2.0, sigma_h=0.02, bil_iters=3, horiz_deg=1.0, grow_m=0.015, min_area=0.04, ramp_max_deg=35.0,
                           grad_tol=0.04, flat_tol=0.01, rise_min=0.08, rise_max=0.25, stair_snap_m=0.025)


def observed_mask(d, pj, T, clip):
    """The package grid's observed cells (see the module docstring) or None if the source mask is not found."""
    H = np.asarray(T["height"])
    s = float(pj["scale_s"])
    f = Path(pj.get("recon", "")) / f"{clip}_hfield.npz"
    if f.exists():
        r = np.load(f)
        if "valid" in r.files:
            v = np.asarray(r["valid"], bool)
            ny, nx = H.shape
            xs = float(T["x0"]) + float(T["res"]) * np.arange(nx)
            ys = float(T["y0"]) + float(T["res"]) * np.arange(ny)
            ix = np.round((xs / s - float(r["x0"])) / float(r["res"])).astype(int)
            iy = np.round((ys / s - float(r["y0"])) / float(r["res"])).astype(int)
            okx, oky = (ix >= 0) & (ix < v.shape[1]), (iy >= 0) & (iy < v.shape[0])
            out = np.zeros(H.shape, bool)
            sub = v[np.clip(iy, 0, v.shape[0] - 1)][:, np.clip(ix, 0, v.shape[1] - 1)]
            out[np.ix_(oky, okx)] = sub[np.ix_(oky, okx)]
            return out
    return None


def structure_mask(Hc, seeds, res):
    """Walkable = reachable from the stance cells over 4-neighbours differing <= max(3 cm, one cell's 45 deg); structure = not
    walkable and > 0.5 m above the nearest walkable cell."""
    from collections import deque

    ny, nx = Hc.shape
    step = max(0.03, res * 1.0)
    walk = np.zeros(Hc.shape, bool)
    if not seeds.any():
        return walk, np.zeros(Hc.shape, bool)
    q = deque(zip(*np.nonzero(seeds)))
    walk[seeds] = True
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a_, b_ = i + di, j + dj
            if 0 <= a_ < ny and 0 <= b_ < nx and not walk[a_, b_] and abs(Hc[a_, b_] - Hc[i, j]) <= step:
                walk[a_, b_] = True
                q.append((a_, b_))
    _, (wy, wx) = ndimage.distance_transform_edt(~walk, return_indices=True)
    return walk, ~walk & (Hc - Hc[wy, wx] > 0.5)


def needle_share(H, res, w=0.10, h=0.10):
    """Needle metric: cells > h above a grey opening of width w."""
    k = max(3, int(round(w / res)) | 1)
    return float(np.mean(H - ndimage.grey_opening(H, size=(k, k)) > h))


def bumpiness(H, mask, res, scale_m=0.06):
    sm = ndimage.gaussian_filter(H, scale_m / res)
    return float(np.sqrt(np.mean((H - sm)[mask] ** 2))) if mask.any() else None


def mesh_points(pj, clip, H0, x0, y0, res):
    """The recon scene mesh / scan sample (<recon>/<clip>_scene.ply) in the package frame, if it lines up with the package
    terrain (median |max z - raw height| <= 3 cm on the cells it covers); else None."""
    import trimesh

    f = Path(pj.get("recon", "")) / f"{clip}_scene.ply"
    if not f.exists():
        return None, None
    g = trimesh.load(f, process=False)
    if not hasattr(g, "vertices"):  # an empty ply loads as a Scene
        return None, None
    P = np.asarray(g.vertices, np.float64) * float(pj["scale_s"])
    P[:, 2] += float(pj["dz_m"])
    ix, iy = np.round((P[:, 0] - x0) / res).astype(int), np.round((P[:, 1] - y0) / res).astype(int)
    ok = (ix >= 0) & (ix < H0.shape[1]) & (iy >= 0) & (iy < H0.shape[0])
    P, ix, iy = P[ok], ix[ok], iy[ok]
    if not len(P):
        return None, None
    D = np.full(H0.shape, np.inf)  # per cell, the vertex nearest the terrain height (ceilings / overhangs don't count against it)
    np.minimum.at(D, (iy, ix), np.abs(P[:, 2] - H0[iy, ix]))
    off = float(np.median(D[np.isfinite(D)]))
    return ((P[:, 2], iy, ix) if off <= 0.03 else None), off


SUPPORT_CACHE = {}


def support_patch(d, ref, r, names, H0, x0, y0, res, dmax, tol=0.03, reach=0.5, groups=None):
    """--support_keep: the raw surface patches the G1 rests on. For every frame and every contact group the refs label in
    contact (refs/<ref>_motion.npz contacts: feet / hands / knees / pelvis), the group's support points (the solver's own:
    fab.retarget.contact.support_points + scene_retarget.pelvis_support with the hip links, its seat_hips set) placed by
    the refs' link poses seed the cell below
    them when they lie within dmax of the raw surface (-0.05 <= z - H0 <= dmax); patch = the raw cells 4-connected to a seed,
    within tol of that seed's height and reach of it. -> (patch, seeds, {group: seeded frames})"""
    import mujoco  # noqa: F401  (support_points / pelvis_support load the MJCF with it)

    from bench.videomimic.scene_retarget import pelvis_support
    from fab.retarget.contact import support_points

    from fab.retarget import g1_files

    mjcf = os.environ.get("G1_WBT_MJCF", str(g1_files.mjcf()))
    if mjcf not in SUPPORT_CACHE:
        sp = support_points(mjcf)
        sp["pelvis"] = pelvis_support(mjcf, hips=True)  # the G1 sits on its hip motors (they hang below the pelvis shell)
        SUPPORT_CACHE[mjcf] = sp
    sp = SUPPORT_CACHE[mjcf]
    mo = np.load(d / "refs" / f"{ref}_motion.npz", allow_pickle=True)
    con = json.loads(str(mo["contacts"]))
    P, Q = np.asarray(r["body_pos_w"], np.float64), np.asarray(r["body_quat_w"], np.float64)  # (T, 51, 3), wxyz
    T = P.shape[0]
    ny, nx = H0.shape
    seeds = np.zeros(H0.shape, bool)
    used = {}
    for g, pts in sp.items():
        if groups and g not in groups:  # --support_groups (rule 2: the non-foot groups only)
            continue
        lab = np.asarray(con.get(g, []), bool)[:T]
        if not lab.any():
            continue
        tt = np.flatnonzero(lab)
        W = []
        for link, local, rad in pts:
            i = names.index(link)
            w, x, y, z = Q[tt, i].T
            R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                          np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                          np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
            p = P[tt, i][:, None] + np.einsum("tij,pj->tpi", R, np.asarray(local, np.float64))
            p[..., 2] -= rad
            W.append(p)
        W = np.concatenate(W, 1).reshape(-1, 3)
        cx, cy = np.round((W[:, 0] - x0) / res).astype(int), np.round((W[:, 1] - y0) / res).astype(int)
        ok = (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
        dz = np.full(len(W), np.inf)
        dz[ok] = W[ok, 2] - H0[cy[ok], cx[ok]]
        hit = ok & (dz >= -0.05) & (dz <= dmax)
        seeds[cy[hit], cx[hit]] = True
        used[g] = int(len(np.unique(np.repeat(tt, W.shape[0] // len(tt))[hit]))) if hit.any() else 0
    if not seeds.any():
        return seeds, seeds, used
    dist, (iy, ix) = ndimage.distance_transform_edt(~seeds, return_indices=True)
    cand = (dist * res <= reach) & (np.abs(H0 - H0[iy, ix]) <= tol)
    lab, n = ndimage.label(cand)  # 4-connected
    keep_ids = np.unique(lab[seeds & cand])
    patch = np.isin(lab, keep_ids[keep_ids > 0])
    return patch, seeds, used


def spikes_v3(H, rel, structure, walk, seeds, near_robot, res, mesh, sliver_w, sliver_h, sliver_min_m2, obj_h, rel2_min, pit, post_h,
              post_max_m2=0.02, post_cov=0.3, iters=3):
    """--slivers 3: drop isolated, low-support spikes and keep connected thin vertical structures.
    - Pits: 1-2 cell pits deeper than sliver_h closed (raise only), never within the robot's no-rise zone.
    - Slivers: cells > sliver_h above the opening (min of a sliver_w disk and a 10 cm square: the needle metric's own element),
      8-connected components, up to `iters` passes (lowering a spike can leave its shoulder standing).
    - Kept: a stance foot within 6 cm; or touching structure / a raised non-walkable object (> obj_h) / an already kept sliver
      with >= sliver_min_m2 cells, mostly observed (reliability 2 share >= rel2_min); or a freestanding post (see below).
    - Everything else is lowered to the opening."""
    k = max(3, int(round(sliver_w / res)) | 1)
    r_ = k // 2
    disk = np.hypot(*np.mgrid[-r_:r_ + 1, -r_:r_ + 1]) <= r_ + 0.5
    k10 = max(3, int(round(0.10 / res)) | 1)
    Hn = H.copy()
    pit_raised = np.zeros(H.shape, bool)
    if pit:
        Hp = np.maximum(Hn, ndimage.grey_closing(Hn, size=(3, 3)))
        pit_raised = ~near_robot & (Hp > Hn + sliver_h)  # real pits only (> 5 cm deep): floors keep their grain
        Hn = np.where(pit_raised, Hp, Hn)
    _, (wy, wx) = ndimage.distance_transform_edt(~walk, return_indices=True) if walk.any() else (None, (np.zeros(H.shape, int), np.zeros(H.shape, int)))
    stc = ndimage.distance_transform_edt(~seeds) * res <= 0.06 if seeds.any() else np.zeros(H.shape, bool)
    if mesh is not None:
        mz, my, mx = mesh
    kept = np.zeros(H.shape, bool)
    rm_all = np.zeros(H.shape, bool)
    n = {"connected": 0, "pole": 0, "stance": 0, "removed": 0, "removed_cells": 0, "largest_removed_m2": 0.0}
    poles = []
    for _ in range(iters):
        Op = np.minimum(ndimage.grey_opening(Hn, footprint=disk), ndimage.grey_opening(Hn, size=(k10, k10)))
        sl = (Hn - Op > sliver_h) & ~kept
        lab, m = ndimage.label(sl, structure=np.ones((3, 3)))
        if not m:
            break
        ids = np.arange(1, m + 1)
        guard = structure | kept | (~walk & ~sl & (Hn - Hn[wy, wx] > obj_h)) if walk.any() else structure | kept
        size = ndimage.sum(sl, lab, ids)
        touch = ndimage.maximum(ndimage.binary_dilation(guard), lab, ids).astype(bool)
        r2 = ndimage.mean(rel == 2, lab, ids)
        on_st = ndimage.maximum(stc, lab, ids).astype(bool)
        conn = touch & (size * res * res >= sliver_min_m2) & (r2 >= rel2_min)
        # freestanding thin structures (posts / poles): a small sliver that is not part of a larger structure (the structure blob it
        # touches is <= post_max_m2), >= post_h tall above its base, and supported by the scene mesh when it lines up (else: mostly
        # observed cells); a spike the fused mesh doesn't have is a single-view artefact
        slab, sn = ndimage.label(structure & ~kept, structure=np.ones((3, 3)))
        sarea = np.bincount(slab.ravel(), minlength=sn + 1) * res * res
        sarea[0] = 0.0
        ring = ndimage.grey_dilation(lab, size=(3, 3))
        big_struct = np.zeros(m + 1, bool)
        hit = (ring > 0) & (slab > 0)
        np.logical_or.at(big_struct, ring[hit], sarea[slab[hit]] > post_max_m2)
        base = ndimage.median(Op, lab, ids)
        top = ndimage.maximum(Hn, lab, ids)
        pole = ~big_struct[1:] & ~conn & (top - base >= post_h)
        if mesh is not None and pole.any():
            dl = ring[my, mx]
            for i in np.flatnonzero(pole):
                z = mz[dl == i + 1]
                z = z[(z > base[i]) & (z < top[i] + 0.10)]
                cov = float(np.mean(np.histogram(z, bins=np.arange(base[i] + 0.02, top[i] + 0.04, 0.04))[0] > 0)) if len(z) else 0.0
                # the mesh reaches (within 10 cm) the top and runs down towards the base (an overhang's edge has its points only at
                # the top); a column the mesh fills densely counts as observed even where the reliability mask says otherwise
                pole[i] = bool(len(z)) and float(z.max()) >= top[i] - 0.10 and cov >= post_cov and (r2[i] >= rel2_min or cov >= 0.7)
        else:
            pole &= r2 >= rel2_min
        for i in np.flatnonzero(pole)[:max(0, 200 - len(poles))]:
            yy, xx = np.nonzero(lab == i + 1)
            poles.append([int(yy.mean()), int(xx.mean()), int(size[i]), round(float(base[i]), 3), round(float(top[i]), 3)])
        keep = on_st | conn | pole
        n["stance"] += int(on_st.sum())
        n["connected"] += int((conn & ~on_st).sum())
        n["pole"] += int((pole & ~conn & ~on_st).sum())
        n["removed"] += int((~keep).sum())
        n["removed_cells"] += int(size[~keep].sum())
        if (~keep).any():
            n["largest_removed_m2"] = max(n["largest_removed_m2"], round(float(size[~keep].max()) * res * res, 4))
        kept |= np.isin(lab, ids[keep])
        rm = np.isin(lab, ids[~keep])
        if not rm.any():
            break
        Hn = np.where(rm, Op, Hn)
        rm_all |= rm
    nd = Hn - ndimage.grey_opening(Hn, size=(k10, k10)) > 0.10
    rep = {"rule": "openings", "opening_cells": k, "components_kept_connected": n["connected"], "components_kept_pole": n["pole"],
           "components_kept_stance": n["stance"], "components_removed": n["removed"], "cells_removed": n["removed_cells"],
           "cells_kept": int(kept.sum()), "largest_removed_m2": n["largest_removed_m2"], "pit_raised_cells": int(pit_raised.sum()),
           "needle_10cm_share_outside_kept": float(np.mean(nd & ~kept)), "mesh": mesh is not None, "poles_yx_cells_base_top": poles}
    return Hn, rm_all, kept, rep


def one(job):
    from bench.scene_recon import simready as S

    clip, v, out_root, ref, slivers, sliver_w, sliver_h, sliver_min_m2, base, sliver_close, sliver_disk, sliver_obj_h, o3 = job
    d = Path(v) / "clips" / clip
    try:
        pj = json.loads((d / "package.json").read_text())
        T = dict(np.load(d / "terrain" / "terrain_hfield_2cm.npz"))
        H0 = np.asarray(T["height"], np.float64)
        res, x0, y0 = float(T["res"]), float(T["x0"]), float(T["y0"])
        ny, nx = H0.shape
        obs = observed_mask(d, pj, T, clip)
        obs_known = obs is not None
        if obs is None:
            obs = np.ones(H0.shape, bool)
        # reliability
        lab, n = ndimage.label(~obs)
        sizes = ndimage.sum(np.ones_like(H0), lab, np.arange(1, n + 1)) * res * res if n else np.zeros(0)
        border = set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]]).tolist()) - {0}
        hole = np.isin(lab, [i + 1 for i in range(n) if sizes[i] <= 0.25 and (i + 1) not in border])
        outside = ~obs & ~hole
        rel = np.where(obs, 2, np.where(hole, 1, 0)).astype(np.uint8)
        # the G1 feet (every frame) and the stance cells
        r = np.load(d / "refs" / f"{ref}.npz")
        names = [str(x) for x in r["body_names"]]
        F = np.asarray(r["body_pos_w"], np.float64)[:, [names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")]].reshape(-1, 3)
        fx = np.round((F[:, 0] - x0) / res).astype(int)
        fy = np.round((F[:, 1] - y0) / res).astype(int)
        inb = (fx >= 0) & (fx < nx) & (fy >= 0) & (fy < ny)
        footmask = np.zeros(H0.shape, bool)
        footmask[fy[inb], fx[inb]] = True
        near_feet = ndimage.distance_transform_edt(~footmask) * res <= 0.15 if footmask.any() else np.zeros(H0.shape, bool)
        # every other robot link (any height; their capsules reach ~0.3 m from the origin): the same no-rise cap within 30 cm of its
        # path (seat / bench tops snapped up by 6-21 mm otherwise fail the body-penetration check)
        B = np.asarray(r["body_pos_w"], np.float64)[:, [i for i, nm in enumerate(names) if nm != "world"]].reshape(-1, 3)
        bx, by = np.round((B[:, 0] - x0) / res).astype(int), np.round((B[:, 1] - y0) / res).astype(int)
        bin_ = (bx >= 0) & (bx < nx) & (by >= 0) & (by < ny)
        lowb = np.zeros(len(B), bool)
        lowb[bin_] = True  # every link of every frame (a 30 cm z gate still left one seated window over body_pen)
        bodymask = np.zeros(H0.shape, bool)
        bodymask[by[lowb], bx[lowb]] = True
        near_body = ndimage.distance_transform_edt(~bodymask) * res <= 0.30 if bodymask.any() else np.zeros(H0.shape, bool)
        near_robot = near_feet | near_body
        stance = inb.copy()
        stance[inb] = F[inb, 2] - H0[fy[inb], fx[inb]] < 0.08 + 0.06  # ankle-roll link ~4-6 cm above the sole
        seeds = np.zeros(H0.shape, bool)
        seeds[fy[stance], fx[stance]] = True
        sup_patch = np.zeros(H0.shape, bool)
        sup_rep = None
        if o3.get("support_keep", 0) > 0:  # --support_keep: the patches the G1 rests on (contact labels) are support, not noise
            sup_patch, sup_seeds, sup_used = support_patch(d, ref, r, names, H0, x0, y0, res, o3["support_keep"],
                                                           groups=[g for g in o3.get("support_groups", "").split(",") if g])
            sup_rep = {"seed_cells": int(sup_seeds.sum()), "patch_cells": int(sup_patch.sum()), "frames_by_group": sup_used}
        # 1 despike
        med = ndimage.median_filter(H0, size=5, mode="nearest")
        op = ndimage.grey_closing(ndimage.grey_opening(H0, size=(3, 3)), size=(3, 3))
        spike = obs & (np.abs(H0 - med) > 0.03) & (np.abs(H0 - op) > 0.03)
        H = np.where(spike, med, H0)
        # 2 holes from observed cells; border-extrapolated cells keep the export's heights (flagged only: a floor fill would cut
        # cliffs / pits where the observed region ends high, e.g. the top of the lab stairs)
        if obs.any() and hole.any():
            _, (yy, xx) = ndimage.distance_transform_edt(~obs, return_indices=True)
            H = np.where(hole, H[yy, xx], H)
        sup = H[seeds] if seeds.any() else H[obs]
        low = sup[sup <= np.percentile(sup, 20)]
        hist, edges = np.histogram(low, bins=max(5, int(np.ptp(low) / 0.01) + 1))
        floor = float(0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1]))
        # 3 planar snapping
        path_xy = F[inb][:, :2]
        hp = S.planar_variant({"x0": x0, "y0": y0, "res": res, "height": H.astype(np.float32), "valid": obs}, PA, path_xy)
        Hc = np.asarray(hp["height"], np.float64)
        lp = np.asarray(hp["plane_label"])
        for k in np.unique(lp[lp > 0]):
            m = lp == k
            mo = m & obs
            if mo.sum() >= 4:
                Hc[m] += float(np.median(H[mo] - Hc[mo]))
        Hc = np.where(outside, H0, Hc)  # border-extrapolated cells unchanged
        # 4 the rise cap near the feet
        capped = near_robot & (Hc > H0 + 0.003)
        Hc = np.where(capped, H0 + 0.003, Hc)
        # 5 structure: not walkable from the stance cells and > 0.5 m above the nearest walkable cell
        walk, structure = structure_mask(Hc, seeds, res)
        Hb = None
        if base == "pkg":  # --base pkg: step 6 on top of the package's shipped clean file (its refs were pinned on it; steps 1-5
            # depend on the refs' paths, so re-running them on re-pinned refs moves other cells too). Reliability / structure as shipped.
            B = np.load(d / "terrain" / "terrain_hfield_2cm_clean.npz")
            Hb = np.asarray(B["height"], np.float64)
            rel, structure = np.asarray(B["reliability"], np.uint8), np.asarray(B["structure"], bool)
            stance = inb.copy()
            stance[inb] = F[inb, 2] - Hb[fy[inb], fx[inb]] < 0.08 + 0.06
            seeds = np.zeros(H0.shape, bool)
            seeds[fy[stance], fx[stance]] = True
            Hc = Hb.copy()
            spike[:] = False
            capped[:] = False
            walk = structure_mask(Hc, seeds, res)[0]
        # 6 (--slivers) thin upward slivers: cells > sliver_h above a grey opening of sliver_w (narrower than ~14 cm,
        # taller than 5 cm). A component is lowered to the opening only if it is isolated: no cell in (or next to) the structure
        # mask, or smaller than sliver_min_m2 (a needle; a post / railing stays), and no stance foot stands on it (within 6 cm).
        # Border-extrapolated cells are left as they are (--slivers 2: they get the same step; lowering only). The opening is taken
        # on the surface with 1-cell pits closed (--sliver_close 3): a pitted seat / table top is not a sliver.
        sl_rep = None
        if slivers >= 3:
            mesh, mesh_off = mesh_points(pj, clip, H0, x0, y0, res) if o3["mesh"] else (None, None)
            Hc, rm, kept_sl, sl_rep = spikes_v3(Hc, rel, structure, walk, seeds | sup_patch, near_robot, res, mesh, sliver_w, sliver_h, sliver_min_m2,
                                                sliver_obj_h, o3["rel2"], o3["pit"], o3["post_h"])
            sl_rep["mesh_offset_mm"] = None if mesh_off is None else round(1000 * mesh_off, 1)
            if Hb is None:
                walk, structure = structure_mask(Hc, seeds, res)
            else:
                walk, structure = structure_mask(Hc, seeds, res)[0], structure & ~rm
        elif slivers:
            pit_raised = np.zeros(H0.shape, bool)
            if o3["pit"]:  # (b): 1-2 cell pits deeper than sliver_h closed, raise only, outside the robot's no-rise zone
                Hp = np.maximum(Hc, ndimage.grey_closing(Hc, size=(3, 3)))
                pit_raised = ~near_robot & (Hp > Hc + sliver_h)
                Hc = np.where(pit_raised, Hp, Hc)
            k = max(3, int(round(sliver_w / res)) | 1)
            r_ = k // 2
            disk = np.hypot(*np.mgrid[-r_:r_ + 1, -r_:r_ + 1]) <= r_ + 0.5 if sliver_disk else np.ones((k, k), bool)  # width in any direction
            Op = ndimage.grey_opening(ndimage.grey_closing(Hc, size=(sliver_close, sliver_close)) if sliver_close else Hc, footprint=disk)
            sl = (Hc - Op > sliver_h) & ((rel >= 1) | (slivers >= 2))
            lab_s, ns = ndimage.label(sl, structure=np.ones((3, 3)))
            guard = structure
            if sliver_obj_h > 0 and walk.any():  # also keep slivers attached to a raised object (a seat's back / arm, a low wall)
                _, (wy, wx) = ndimage.distance_transform_edt(~walk, return_indices=True)
                guard = structure | (~walk & ~sl & (Hc - Hc[wy, wx] > sliver_obj_h))
            near_struct = ndimage.binary_dilation(guard, iterations=1)
            nmin = max(1, int(round(sliver_min_m2 / (res * res))))
            removed = kept = removed_cells = kept_cells = kept_stance = big_rm = 0
            rm = np.zeros(H0.shape, bool)
            stance_cells = ndimage.distance_transform_edt(~seeds) * res <= 0.06 if seeds.any() else np.zeros(H0.shape, bool)
            if ns:
                sizes = ndimage.sum(sl, lab_s, np.arange(1, ns + 1)).astype(int)
                touch = ndimage.maximum(near_struct, lab_s, np.arange(1, ns + 1)).astype(bool)
                # a sliver a stance foot stands on is support, not noise (lowering it would move the stance support)
                on_stance = ndimage.maximum(stance_cells, lab_s, np.arange(1, ns + 1)).astype(bool)
                go = ((~touch) | (sizes < nmin)) & ~on_stance
                kept_stance = int((((~touch) | (sizes < nmin)) & on_stance).sum())
                rm = np.isin(lab_s, 1 + np.flatnonzero(go))
                removed, kept = int(go.sum()), int((~go).sum())
                removed_cells, kept_cells = int(sizes[go].sum()), int(sizes[~go].sum())
                big_rm = int(sizes[go].max()) if go.any() else 0
            Hc = np.where(rm, Op, Hc)  # lowering only
            if Hb is None:
                walk, structure = structure_mask(Hc, seeds, res)
            else:
                walk, structure = structure_mask(Hc, seeds, res)[0], structure & ~rm
            sl_rep = {"opening_cells": k, "min_cells": nmin, "components_removed": removed, "components_kept_structure": kept - kept_stance,
                      "components_kept_stance": kept_stance,
                      "cells_removed": removed_cells, "cells_kept": kept_cells, "largest_removed_m2": round(big_rm * res * res, 4),
                      "pit_raised_cells": int(pit_raised.sum())}
        # 7 stance supports: within 6 cm of the cells the G1 feet stand on, the clean surface stays within 1 cm below the
        # package terrain (the footprint-height gate; despike / planar snapping lowered a few narrow supports by up to 10 cm)
        if slivers and seeds.any() and Hb is None:
            near_st = ndimage.distance_transform_edt(~seeds) * res <= 0.06
            low_clamped = near_st & (Hc < H0 - 0.0099)
            Hc = np.where(low_clamped, H0 - 0.0099, Hc)
            walk, structure = structure_mask(Hc, seeds, res)
            if sl_rep is not None:
                sl_rep["stance_lower_clamped_cells"] = int(low_clamped.sum())
        if sup_patch.any() and Hb is None:  # --support_keep: inside a support patch the clean stays within 1 cm below the raw surface
            low_p = sup_patch & (Hc < H0 - 0.0099)
            sup_rep["restored_cells"] = int(low_p.sum())
            sup_rep["restored_gt2cm_cells"] = int((sup_patch & (Hc < H0 - 0.02)).sum())
            sup_rep["max_restored_mm"] = round(float(1000 * (H0 - Hc)[low_p].max()), 1) if low_p.any() else 0.0
            Hc = np.where(low_p, H0 - 0.0099, Hc)
            walk, structure = structure_mask(Hc, seeds, res)
        o = Path(out_root) / clip
        o.mkdir(parents=True, exist_ok=True)
        np.savez(o / "terrain_hfield_2cm_clean.npz", x0=x0, y0=y0, res=res, height=Hc.astype(np.float32), reliability=rel, structure=structure)
        gy, gx = np.gradient(ndimage.gaussian_filter(H0, 0.06 / res), res)
        flat = ndimage.binary_erosion((rel >= 1) & (np.hypot(gx, gy) < 0.1) & ~structure, iterations=3)  # away from edges (6 cm scale)
        dH = Hc - H0
        rep = {"clip": clip, "source": clip.split("_")[0], "observed_mask": obs_known, "cells": int(H0.size), "res_m": res,
               "support_keep": sup_rep,
               "changed_gt2cm_share": float(np.mean(np.abs(dH) > 0.02)), "changed_gt2cm_share_observed": float(np.mean(np.abs(dH)[obs] > 0.02)),
               "raised_gt2cm_share": float(np.mean(dH > 0.02)), "lowered_gt2cm_share": float(np.mean(dH < -0.02)),
               "max_rise_near_feet_mm": float(1000 * dH[near_feet].max()) if near_feet.any() else None, "capped_cells": int(capped.sum()),
               "max_rise_near_robot_mm": float(1000 * dH[near_robot].max()) if near_robot.any() else None,
               "spike_cells": int(spike.sum()), "flat_share": float(flat.mean()),
               "flat_bumpiness_rms_mm": [None if bumpiness(H0, flat, res) is None else round(1000 * bumpiness(H0, flat, res), 2),
                                         None if bumpiness(Hc, flat, res) is None else round(1000 * bumpiness(Hc, flat, res), 2)],
               "structure_share": float(structure.mean()), "walkable_share": float(walk.mean()),
               "reliability_share": {"observed": float(np.mean(rel == 2)), "filled_inside": float(np.mean(rel == 1)), "border_extrapolated": float(np.mean(rel == 0))},
               "floor_m": floor, "slivers": sl_rep,
               "needle_10cm_share": [needle_share(H0, res), needle_share(Hc, res)]}
        if Hb is not None:
            rep["base"] = "package terrain_hfield_2cm_clean.npz"
            rep["needle_10cm_share_base"] = needle_share(Hb, res)
            rep["max_rise_vs_base_mm"] = float(1000 * (Hc - Hb).max())
            rep["max_rise_vs_base_near_robot_mm"] = float(1000 * (Hc - Hb)[near_robot].max()) if near_robot.any() else None
            rep["max_height_m"] = [float(Hb.max()), float(Hc.max())]
        # stance support (G1 feet within 8 cm of the terrain): the median terrain within 4 cm of the ankle-roll link xy, clean - raw
        if stance.any():
            offs = [(dy, dx) for dy in range(-3, 4) for dx in range(-3, 4) if (dy * dy + dx * dx) * res * res <= 0.04 ** 2 + 1e-9]
            jy = np.clip(fy[stance][:, None] + np.array([o_[0] for o_ in offs])[None], 0, ny - 1)
            jx = np.clip(fx[stance][:, None] + np.array([o_[1] for o_ in offs])[None], 0, nx - 1)
            dsup = np.abs(np.median(Hc[jy, jx], 1) - np.median(H0[jy, jx], 1))
            rep["stance_support"] = {"points": int(len(dsup)), "change_p95_mm": float(1000 * np.percentile(dsup, 95)),
                                     "change_max_mm": float(1000 * dsup.max()), "within_1cm_share": float(np.mean(dsup <= 0.01))}
            if Hb is not None:
                db = np.abs(np.median(Hc[jy, jx], 1) - np.median(Hb[jy, jx], 1))
                rep["stance_support_vs_base"] = {"change_max_mm": float(1000 * db.max()), "within_1cm_share": float(np.mean(db <= 0.01))}
        (o / "clean_report.json").write_text(json.dumps(rep, indent=1))
        return clip, {"status": "ok", **rep}
    except Exception as e:  # noqa: BLE001
        return clip, {"status": "fail", "error": f"{type(e).__name__}: {str(e)[:300]}"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v", required=True, help="the packages' root: <v>/clips/<clip>/package.json (+ optional <v>/index.json)")
    p.add_argument("--out", required=True)
    p.add_argument("--ref", default="p", help="comma list: the first refs/<ref>.npz that exists per clip")
    p.add_argument("--only", default="")
    p.add_argument("--skip_sources", default="", help="comma list of index.json sources / tiers to skip (e.g. flat: identity clean files)")
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--slivers", type=int, default=0, help="remove isolated thin upward slivers (step 6; 3: + posts / pits rules); 2: also on border-extrapolated cells")
    p.add_argument("--sliver_w", type=float, default=0.14)
    p.add_argument("--sliver_h", type=float, default=0.05)
    p.add_argument("--sliver_min_m2", type=float, default=0.005, help="structure-connected slivers smaller than this are needles too")
    p.add_argument("--sliver_close", type=int, default=3, help="grey closing (cells) before the sliver opening; 0: off")
    p.add_argument("--sliver_disk", type=int, default=1, help="1: disk opening (sliver_w wide in any direction); 0: square")
    p.add_argument("--sliver_obj_h", type=float, default=0.0, help=">0: slivers touching a non-walkable object this high above the walkable surface are kept too")
    p.add_argument("--sliver_rel2", type=float, default=0.5, help="--slivers 3: kept slivers need this share of observed (reliability 2) cells")
    p.add_argument("--sliver_mesh", type=int, default=1, help="--slivers 3: a post is kept only if the scene mesh reaches its top")
    p.add_argument("--sliver_post_h", type=float, default=0.3, help="--slivers 3: posts / poles stand at least this high above their base")
    p.add_argument("--pit_close", type=int, default=0, help="close 1-2 cell pits > sliver_h deep before step 6 (raise only, outside the no-rise zone)")
    p.add_argument("--support_keep", type=float, default=0.0, help="off by default: "
                   "raw surface patches under a contact-labelled G1 support point within this of the surface (m, e.g. 0.04) stay "
                   "within 1 cm below the raw surface and keep their slivers (seat / box tops the body rests on)")
    p.add_argument("--support_groups", default="", help="--support_keep: comma list of the contact groups that seed (rule 2: "
                   "pelvis,lhand,rhand,lknee,rknee; empty: all, incl. the feet = rule 1, dropped)")
    p.add_argument("--base", default="", help="pkg: run step 6 on the package's terrain_hfield_2cm_clean.npz instead of re-running steps 1-5")
    a = p.parse_args()
    if Path(a.v, "index.json").exists():
        idx = json.loads(Path(a.v, "index.json").read_text())
    else:  # a staging dir without an index: every clip dir
        idx = {"clips": [{"clip": c.name, "source": c.name.split("_")[0]} for c in sorted(Path(a.v, "clips").iterdir()) if c.is_dir()]}
    skip = set(x for x in a.skip_sources.split(",") if x)
    refs = [x for x in a.ref.split(",") if x]  # the first that exists per clip

    def ref_of(c):
        return next((r for r in refs if (Path(a.v) / "clips" / c / "refs" / f"{r}.npz").exists()), None)

    clips = [c for c in a.only.split(",") if c] or [c["clip"] for c in idx["clips"] if ref_of(c["clip"])
                                                    and c.get("source") not in skip and c.get("tier") not in skip]
    tasks = [(c, a.v, a.out, ref_of(c), a.slivers, a.sliver_w, a.sliver_h, a.sliver_min_m2, a.base, a.sliver_close, a.sliver_disk, a.sliver_obj_h,
                                {"rel2": a.sliver_rel2, "mesh": a.sliver_mesh, "pit": a.pit_close, "post_h": a.sliver_post_h,
                                 "support_keep": a.support_keep, "support_groups": a.support_groups}) for c in clips]
    if a.workers == 1:  # one worker: in this process (the same function per clip)
        res = dict(map(one, tasks))
    else:
        with Pool(a.workers) as pool:
            res = dict(pool.map(one, tasks, chunksize=2))
    ok = [v for v in res.values() if v["status"] == "ok"]
    by = {}
    for v in ok:
        by.setdefault(v["source"], []).append(v)
    summ = {"clips": len(ok), "failed": {k: v["error"] for k, v in res.items() if v["status"] != "ok"}}
    for src, L in sorted(by.items()):
        def m_(f):
            x = [f(v) for v in L if f(v) is not None]
            return round(float(np.median(x)), 4) if x else None
        summ[src] = {"clips": len(L), "observed_mask_found": int(sum(v["observed_mask"] for v in L)),
                     "changed_gt2cm_share_med": m_(lambda v: v["changed_gt2cm_share"]), "raised_gt2cm_share_med": m_(lambda v: v["raised_gt2cm_share"]),
                     "max_rise_near_feet_mm_max": round(float(max(v["max_rise_near_feet_mm"] for v in L if v["max_rise_near_feet_mm"] is not None)), 2),
                     "flat_bumpiness_rms_mm_med_before_after": [m_(lambda v: v["flat_bumpiness_rms_mm"][0]), m_(lambda v: v["flat_bumpiness_rms_mm"][1])],
                     "structure_share_med_p90": [m_(lambda v: v["structure_share"]), round(float(np.percentile([v["structure_share"] for v in L], 90)), 4)],
                     "border_extrapolated_share_med": m_(lambda v: v["reliability_share"]["border_extrapolated"]),
                     "filled_inside_share_med": m_(lambda v: v["reliability_share"]["filled_inside"]),
                     "needle_10cm_share_med_before_after": [m_(lambda v: v["needle_10cm_share"][0]), m_(lambda v: v["needle_10cm_share"][1])],
                     "needle_10cm_share_max_after": round(float(max(v["needle_10cm_share"][1] for v in L)), 4),
                     "sliver_components_removed_kept": [int(sum((v["slivers"] or {}).get("components_removed", 0) for v in L)),
                                                        int(sum((v["slivers"] or {}).get("components_kept_structure", 0) for v in L))],
                     "stance_change_p95_mm_med_max": [m_(lambda v: (v.get("stance_support") or {}).get("change_p95_mm")),
                                                      round(float(max([(v.get("stance_support") or {}).get("change_p95_mm", 0) for v in L])), 2)],
                     "stance_within_1cm_share_min": round(float(min([(v.get("stance_support") or {}).get("within_1cm_share", 1) for v in L])), 4),
                     "stance_change_max_mm_max": round(float(max([(v.get("stance_support") or {}).get("change_max_mm", 0) for v in L])), 2),
                     "stance_change_vs_base_max_mm": round(float(max([(v.get("stance_support_vs_base") or {}).get("change_max_mm", 0) for v in L])), 2),
                     "max_rise_vs_base_mm_max": round(float(max([v.get("max_rise_vs_base_mm", 0) for v in L])), 2),
                     "max_rise_vs_base_near_robot_mm_max": round(float(max([v.get("max_rise_vs_base_near_robot_mm") or 0 for v in L])), 2),
                     "max_height_drop_m_max": round(float(max([(v["max_height_m"][0] - v["max_height_m"][1]) if v.get("max_height_m") else 0 for v in L])), 3),
                     "needle_outside_kept_med": m_(lambda v: (v["slivers"] or {}).get("needle_10cm_share_outside_kept")),
                     "sliver_v3_components": {k_: int(sum((v["slivers"] or {}).get(k_, 0) for v in L)) for k_ in
                                              ("components_kept_connected", "components_kept_pole", "components_kept_stance", "components_removed", "pit_raised_cells")},
                     "mesh_used": int(sum(bool((v["slivers"] or {}).get("mesh")) for v in L)),
                     "needle_10cm_share_base_med": m_(lambda v: v.get("needle_10cm_share_base")),
                     "max_rise_near_robot_mm_max": round(float(max([v["max_rise_near_robot_mm"] or 0 for v in L])), 2),
                     "sliver_largest_removed_m2_max": round(float(max([(v["slivers"] or {}).get("largest_removed_m2", 0) for v in L])), 4),
                     "sliver_components_kept_stance": int(sum((v["slivers"] or {}).get("components_kept_stance", 0) for v in L)),
                     "sliver_cells_removed_kept": [int(sum((v["slivers"] or {}).get("cells_removed", 0) for v in L)),
                                                   int(sum((v["slivers"] or {}).get("cells_kept", 0) for v in L))]}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "summary.json").write_text(json.dumps({"summary": summ, "clips": res}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
