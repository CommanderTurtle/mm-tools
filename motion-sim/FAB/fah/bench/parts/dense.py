"""Finger-part correctness of the dense surface points (scripts/dense_part_eval.py).

Per hand, the raw keypoint-head dense points (pt_uv, crop pixels) with their MANO part (the per-point part of
train.part_masks.mano_parts) and the head's visibility (pt_vis > 0), against the crop's reference part labels
(0 background, 1 palm, 2-16 finger segments, 255 ignore):
  part_acc      visible points of part k inside the reference part-k region (points on ignore pixels skipped, only parts
                the label has; pooled over hands per part): mean over the 16 parts, fingers (2-16), distal segments, palm
  finger_acc    visible finger points inside any segment of their own finger
  cover         reference part-k pixels within 0.02 x the view side of a visible part-k point (per hand and part, then means)
  wrong_finger  visible finger points on another finger's region / visible finger points on labelled pixels
  sil_iou       union of discs (0.02 x the view side) around all points vs the labelled hand (1-16); ignore skipped
  ign_frac      visible points on ignore pixels; absent_ign_frac: the same for visible points of parts the label lacks
  *_strict      part_acc_fingers / wrong_finger with points on ignore pixels kept in the denominator
  bg_frac       visible finger points on background (incl. outside the crop) / all visible finger points
  confusion     6 x 8 counts of visible points: rows palm / thumb .. pinky (the point's own), columns the landing
                (palm / thumb .. pinky / background / ignore)
With GT meshes (records with verts_file, gt_geometry): breakdowns of the visible finger points by crossings (points
whose GT position lies where two fingers' projections overlap; hands by overlap fraction < 0.10, 0.10-0.22, >= 0.22),
occlusion (the point's finger by its GT-occluded fraction: < 0.6, 0.6-0.8, 0.8-0.95, >= 0.95) and hand size (label-box
long side in original pixels: < 128, 128-192, 192-256, >= 256); bins fixed before any model was scored.
The paper's cells (supplementary "dense-point finger assignment") are the "3d" group means of wrong_finger, part_acc_fingers_strict,
cover_fingers and sil_iou."""
import os

import cv2
import numpy as np

FINGER = np.array([-1, -1] + [f for f in range(5) for _ in range(3)])  # part id -> finger (thumb .. pinky), -1 bg / palm
TIPS = [4, 7, 10, 13, 16]
COARSE = np.array([-1, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])  # part id -> row: palm, thumb .. pinky
CROSS_BINS, OCC_BINS, SIZE_BINS = (0.10, 0.22), (0.6, 0.8, 0.95), (128, 192, 256)
GROUPS = {"3d": ("hot3d_lab_mesh", "dexycb_lab", "oakink2_ego_lab"), "2d": ("coco_wb_kin", "cmu_lab_kin", "hint_lab_kin")}
KEYS = ("wrong_finger", "part_acc_fingers", "part_acc_tips", "finger_acc", "cover_fingers", "sil_iou", "ign_frac", "part_acc_fingers_strict",
        "wrong_finger_strict", "bg_frac", "vis_frac")
GROUP_KEYS = ("part_acc", "part_acc_fingers", "part_acc_tips", "part_acc_palm", "finger_acc", "cover", "cover_fingers", "cover_tips",
              "wrong_finger", "part_acc_fingers_strict", "wrong_finger_strict", "sil_iou", "sil_iou_vis", "vis_frac", "ign_frac",
              "absent_frac", "absent_ign_frac", "bg_frac", *("ref_" + k for k in KEYS))


def score_hand(lab, uv, vis, pid, side, frac=0.02, geo=None):
    """Per-hand counts: per part (17) acc num / den, own-finger num, cover (nan: part absent), finger landing counts
    (own segment, own finger, other finger, palm, background), silhouette IoU (all points, visible points)."""
    H, W = lab.shape
    thr = frac * side
    xi, yi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    inb = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    L = np.zeros(len(uv), np.int64)  # outside the crop = background
    L[inb] = lab[yi[inb], xi[inb]]
    ok = vis & (L != 255)
    present = np.bincount(lab[lab != 255].ravel(), minlength=17)[:17] > 0
    Lc = np.minimum(L, 16)
    own = ok & (L == pid)
    same_f = ok & (L >= 2) & (L <= 16) & (FINGER[Lc] == FINGER[pid])
    use = present[pid]
    num = np.bincount(pid[own & use], minlength=17)
    den = np.bincount(pid[ok & use], minlength=17)
    fnum = np.bincount(pid[(same_f | own) & use], minlength=17)
    den_all = np.bincount(pid[vis & use], minlength=17)  # incl. points on ignore pixels (strict: those count as not inside)
    cover = np.full(17, np.nan)
    for k in range(1, 17):
        if not present[k]:
            continue
        reg = lab == k
        p = vis & inb & (pid == k)
        if not p.any():
            cover[k] = 0.0
            continue
        z = np.full((H, W), 255, np.uint8)
        z[yi[p], xi[p]] = 0
        cover[k] = float((cv2.distanceTransform(z, cv2.DIST_L2, 5)[reg] <= thr).mean())
    fp = ok & (pid >= 2)
    lf = np.where((L >= 2) & (L <= 16), FINGER[Lc], -1)
    land = np.array([(fp & (L == pid)).sum(), (fp & (lf == FINGER[pid]) & (L != pid)).sum(), (fp & (lf >= 0) & (lf != FINGER[pid])).sum(),
                     (fp & (L == 1)).sum(), (fp & (L == 0)).sum()])
    valid, gt = lab != 255, (lab >= 1) & (lab <= 16)
    sil = []
    for sel in (inb, inb & vis):
        fpm = np.zeros((H, W), np.uint8)
        for x, y in zip(xi[sel], yi[sel]):
            cv2.circle(fpm, (int(x), int(y)), max(int(round(thr)), 1), 1, -1)
        fpm = fpm.astype(bool)
        sil.append(float((fpm & gt & valid).sum() / max(((fpm | gt) & valid).sum(), 1)))
    absent = vis & ~present[pid]  # visible points of parts the label does not have
    ign = np.array([(vis & (L == 255)).sum(), (absent & (L == 255)).sum(), absent.sum()])  # on ignore; absent-part on ignore / all
    conf = np.zeros((6, 8), np.int64)  # own coarse class x landing (palm, 5 fingers, background, ignore)
    col = np.where(L == 255, 7, np.where(L == 0, 6, COARSE[Lc]))
    np.add.at(conf, (COARSE[pid[vis & (pid >= 1)]], col[vis & (pid >= 1)]), 1)
    fv = vis & (pid >= 2)
    wrong_all = fv & (lf >= 0) & (lf != FINGER[pid])
    cnt = lambda m: np.array([(fv & m).sum(), (wrong_all & m).sum(), (fv & m & (L == 0)).sum(), (fv & m & (L == 255)).sum(), ((pid >= 2) & m).sum()])
    out = dict(num=num, den=den, den_all=den_all, nfin_all=int(fv.sum()), fnum=fnum, cover=cover, land=land, sil=np.array(sil), ign=ign,
               nbg=int((fv & (L == 0)).sum()), conf=conf, nvis=int(vis.sum()), n=len(vis))
    if geo is not None:  # [visible finger pts, wrong finger, background, ignore, all finger pts] per crossing / occlusion bin
        gx, gy = np.clip(np.round(geo["uv"][:, 0]).astype(int), 0, W - 1), np.clip(np.round(geo["uv"][:, 1]).astype(int), 0, H - 1)
        inx = geo["overlap"][gy, gx] & geo["inb"]
        out["cross_pts"] = np.stack([cnt(inx), cnt(~inx)])
        ob = np.digitize(geo["occ"], OCC_BINS)[np.maximum(FINGER[pid], 0)]  # each point's finger's occlusion bin
        out["occ_pts"] = np.stack([cnt(ob == k) for k in range(len(OCC_BINS) + 1)])
        out["cross_bin"], out["size_bin"] = int(np.digitize(geo["overlap_frac"], CROSS_BINS)), int(np.digitize(geo["size"], SIZE_BINS))
    return out


def gt_geometry(r, crop_dir, lab, vidx, pid, faces, fpart):
    """Model-independent GT geometry of a crop (GT-mesh pools): the dense points' GT positions (crop px), the fingers'
    overlap mask and fraction, per-finger GT-occluded fraction of the dense points, label-box size (original px)."""
    import torch

    from train.finetune import default_K, vis_labels

    v = np.load(os.path.join(crop_dir, os.path.basename(r["verts_file"]))).astype(np.float64)
    K = np.asarray(r["K"], np.float64) if "K" in r else default_K(*r["hw"], r["is_right"])
    M = np.asarray(r["M"], np.float64)
    q = v @ K.T
    uv = (q[:, :2] / np.maximum(q[:, 2:], 1e-6)) @ M[:, :2].T + M[:, 2]
    H, W = lab.shape
    fm = np.zeros((5, H, W), np.uint8)
    tri = np.round(uv[faces]).astype(np.int32)
    ff = FINGER[fpart]
    for f in range(5):
        for t in tri[ff == f]:
            cv2.fillConvexPoly(fm[f], t, 1)
    cover = fm.sum(0)
    overlap = cover >= 2
    sil = ((lab >= 1) & (lab <= 16)).astype(np.uint8) * 255
    vis = vis_labels(torch.as_tensor(v, dtype=torch.float32)[None], torch.as_tensor(faces), torch.as_tensor(K, dtype=torch.float32)[None],
                     torch.as_tensor(M, dtype=torch.float32)[None], float(W), torch.as_tensor(sil)[None], torch.ones(1, dtype=torch.bool))[0].numpy()[vidx] > 0
    fin = FINGER[pid]
    occ = np.array([1 - vis[fin == f].mean() if (fin == f).any() else 0.0 for f in range(5)])
    b = np.asarray(r["box"], np.float64)
    du = uv[vidx]
    inb = (du[:, 0] >= 0) & (du[:, 0] < W) & (du[:, 1] >= 0) & (du[:, 1] < H)
    gx, gy = np.clip(np.round(du[:, 0]).astype(int), 0, W - 1), np.clip(np.round(du[:, 1]).astype(int), 0, H - 1)
    on = inb & (lab[gy, gx] >= 1) & (lab[gy, gx] <= 16)  # alignment check: GT-visible GT points on the labelled hand
    return dict(uv=du, inb=inb, on_label=float(on[vis].mean()) if vis.any() else None, overlap=overlap,
                overlap_frac=float(overlap.sum() / max((cover > 0).sum(), 1)), occ=occ, size=float(max(b[2] - b[0], b[3] - b[1])), vis=vis)


def metrics(h):
    """Pool-level metrics from stacked per-hand counts (dict of arrays over hands)."""
    num, den, fnum = h["num"].sum(0), h["den"].sum(0), h["fnum"].sum(0)
    acc = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    da = h["den_all"].sum(0)
    acc_s = np.where(da > 0, num / np.maximum(da, 1), np.nan)
    facc = np.where(den > 0, fnum / np.maximum(den, 1), np.nan)
    cnt = np.isfinite(h["cover"]).sum(0)
    cov = np.where(cnt > 0, np.nansum(h["cover"], 0) / np.maximum(cnt, 1), np.nan)
    land = h["land"].sum(0)
    m = lambda x, ks: float(np.nanmean(x[ks])) if np.isfinite(x[ks]).any() else None
    F, P = list(range(2, 17)), list(range(1, 17))
    return dict(n=int(len(h["num"])), part_acc=m(acc, P), part_acc_fingers=m(acc, F), part_acc_tips=m(acc, TIPS), part_acc_palm=m(acc, [1]),
                part_acc_fingers_strict=m(acc_s, F),
                wrong_finger_strict=float(land[2] / h["nfin_all"].sum()) if h["nfin_all"].sum() else None,
                finger_acc=m(facc, F), cover=m(cov, P), cover_fingers=m(cov, F), cover_tips=m(cov, TIPS),
                wrong_finger=float(land[2] / land.sum()) if land.sum() else None,
                land=dict(zip(["own_segment", "own_finger_other_segment", "other_finger", "palm", "background"],
                              (land / land.sum()).round(4).tolist() if land.sum() else [None] * 5)),
                sil_iou=float(h["sil"][:, 0].mean()) if len(h["sil"]) else None,
                sil_iou_vis=float(h["sil"][:, 1].mean()) if len(h["sil"]) else None,
                vis_frac=float(h["nvis"].sum() / h["n"].sum()) if h["n"].sum() else None,
                ign_frac=float(h["ign"][:, 0].sum() / h["nvis"].sum()) if h["nvis"].sum() else None,
                absent_frac=float(h["ign"][:, 2].sum() / h["nvis"].sum()) if h["nvis"].sum() else None,
                absent_ign_frac=float(h["ign"][:, 1].sum() / h["ign"][:, 2].sum()) if h["ign"][:, 2].sum() else None,
                bg_frac=float(h["nbg"].sum() / h["nfin_all"].sum()) if h["nfin_all"].sum() else None,
                part_acc_per_part=[None if np.isnan(x) else round(float(x), 4) for x in acc[1:]],
                cover_per_part=[None if np.isnan(x) else round(float(x), 4) for x in cov[1:]],
                confusion=rates(h["conf"].sum(0)), **breakdowns(h))


def rates(c):
    """Row-normalised counts (rows without points stay None)."""
    return [[round(float(x / r.sum()), 4) for x in r] if r.sum() else None for r in c]


def breakdowns(h):
    """Point-level crossing / occlusion bins from the summed [visible finger, wrong, background, ignore, all] counts."""
    out = {}
    for key, names in (("cross_pts", ("in_overlap", "outside_overlap")), ("occ_pts", ("occ<0.6", "occ0.6-0.8", "occ0.8-0.95", "occ>=0.95"))):
        if key in h and len(h[key]):
            c = h[key].sum(0)
            out[key[:-4]] = {nm: dict(n_vis=int(v[0]), wrong_finger=round(float(v[1] / v[0]), 4) if v[0] else None,
                                      bg_frac=round(float(v[2] / v[0]), 4) if v[0] else None, ign_frac=round(float(v[3] / v[0]), 4) if v[0] else None,
                                      vis_frac=round(float(v[0] / v[4]), 4) if v[4] else None) for nm, v in zip(names, c)}
    return out


def stack(hs):
    if not hs:
        shapes = dict(num=(17,), den=(17,), den_all=(17,), nfin_all=(), fnum=(17,), cover=(17,), land=(5,), sil=(2,), ign=(3,), nbg=(),
                      conf=(6, 8), nvis=(), n=())
        return {k: np.empty((0, *shape)) for k, shape in shapes.items()}
    return {k: np.stack([h[k] for h in hs]) for k in hs[0]}


def bootstrap(ha, hb, pools, B=1000, seed=0, clips=None):
    """Paired bootstrap over hands (stratified by pool) of the pool-mean difference b - a for KEYS; ha / hb: pool ->
    stacked per-hand counts of the same hands. clips (pool -> clip id per hand): resample whole clips instead (crops of
    one video are correlated). Each metric uses a fixed set of pools where both full-sample statistics
    exist. Resamples on which a metric is undefined in any of those pools add no interval sample; their count is reported."""
    if B < 0:
        raise ValueError("bootstrap resamples must be non-negative")
    for p in pools:
        if len(ha[p]["n"]) != len(hb[p]["n"]):
            raise ValueError(f"Unpaired hand counts in pool {p}")
    pools = [p for p in pools if len(ha[p]["n"])]
    rng = np.random.default_rng(seed)
    pick = lambda h, idx: {k: v[idx] for k, v in h.items()}
    members = {}
    if clips is not None:
        for p in pools:
            c = np.asarray(clips[p])
            if len(c) != len(ha[p]["n"]):
                raise ValueError(f"Clip ids do not match the hands of pool {p}")
            members[p] = [np.nonzero(c == u)[0] for u in np.unique(c)]
    base = {p: (metrics(ha[p]), metrics(hb[p])) for p in pools}
    included = {k: [p for p in pools if all(m[k] is not None for m in base[p])] for k in KEYS}
    full = {k: float(np.mean([base[p][1][k] - base[p][0][k] for p in included[k]])) if included[k] else None for k in KEYS}
    samp = {k: [] for k in KEYS}
    for _ in range(B):
        draw = {}
        for p in pools:
            if clips is None:
                idx = rng.integers(0, len(ha[p]["n"]), len(ha[p]["n"]))
            else:
                groups = members[p]
                idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            draw[p] = metrics(pick(ha[p], idx)), metrics(pick(hb[p], idx))
        for k in KEYS:
            ps = included[k]
            if ps and all(m[k] is not None for p in ps for m in draw[p]):
                samp[k].append(float(np.mean([draw[p][1][k] - draw[p][0][k] for p in ps])))
    return {k: dict(diff=round(full[k], 4) if full[k] is not None else None,
                    lo=round(float(np.percentile(samp[k], 2.5)), 4) if len(samp[k]) >= 2 else None,
                    hi=round(float(np.percentile(samp[k], 97.5)), 4) if len(samp[k]) >= 2 else None,
                    n_valid=len(samp[k]), n_resamples=B, n_pools=len(included[k]), unit="hand" if clips is None else "clip") for k in KEYS}


def mean_defined(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def group_report(rep, hands, pools):
    """rep[pool] metrics -> the "3d" / "2d" / "all" group entries: unweighted pool means of GROUP_KEYS, plus the
    group's hands pooled (micro) for the confusion, crossing / occlusion and size-bin breakdowns."""
    out = {}
    for g, ps in [*GROUPS.items(), ("all", tuple(pools))]:
        ps = [p for p in ps if p in pools]
        if not ps:
            continue
        e = {k: mean_defined(rep[p][k] for p in ps) for k in GROUP_KEYS}
        e["n"] = int(sum(rep[p]["n"] for p in ps))
        hs = [h for p in ps for h in hands[p] if h is not None]
        st = stack(hs)
        e["confusion"] = rates(st["conf"].sum(0))
        e.update(breakdowns(st))
        for key, nb in (("cross_bin", len(CROSS_BINS) + 1), ("size_bin", len(SIZE_BINS) + 1)):
            if all(key in h for h in hs):
                e["by_" + key[:-4]] = {}
                for k in range(nb):
                    sub = [h for h in hs if h[key] == k]
                    mm = metrics(stack(sub)) if sub else {}
                    e["by_" + key[:-4]][str(k)] = {"n": len(sub), **{x: mm.get(x) for x in KEYS}}
        out[g] = e
    return out
