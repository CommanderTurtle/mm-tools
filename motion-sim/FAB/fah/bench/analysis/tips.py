"""Fingertip extent against human 2D labels (bench2d-format runs; the supplementary fingertip table).

For each matched hand (bench.bench2d.evaluate.match) and finger with a visible DIP and tip: the signed error of the predicted
tip along the labelled distal phalanx (DIP -> tip), in units of that phalanx length (< 0: the prediction stops short of the
labelled tip); with all four finger joints visible, the predicted / labelled finger length (MCP -> PIP -> DIP -> tip
polyline). Medians per subset of the set and over all ("all").

Keys: a 2D joints key of the predictions (joints2d, joints2d_crop, ...); "mesh" = the MANO-topology mesh (verts3d) projected,
tips at the MANO tip vertices; "meshmax" = the mesh's furthest vertex along the labelled distal direction among that finger's
tip-region vertices (MANO template vertices within 12 mm of the tip vertex: the visible end of the rendered finger); "surf2d"
= the same on the 2D head's dense surface points (surface2d_head, last 778 rows, MANO topology). Mesh keys need the camera:
a frame's own K, else the intrinsics the method assumed on the 2D set (`intrinsics`: "fah" = focal at the image diagonal,
"wilor" = WiLoR's 5000 / 256 x the longer side; principal point at the centre).
"""
import glob
import os
import pickle
from collections import defaultdict

import numpy as np

from bench.bench2d.evaluate import match

FINGERS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
TIP_V = {"thumb": 744, "index": 320, "middle": 443, "ring": 555, "pinky": 672}  # MANO tip vertices
MESH_KEYS = ("mesh", "meshmax", "surf2d")


def project(v, K):
    return v[:, :2] / np.clip(v[:, 2:], 1e-6, None) * [K[0, 0], K[1, 1]] + K[:2, 2]


def default_K(h, w, intrinsics="fah"):
    f = (h ** 2 + w ** 2) ** 0.5 if intrinsics == "fah" else 5000 / 256 * max(h, w)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def tip_regions(mano_dir, radius=0.012):
    """Per finger the MANO template vertices within `radius` (m) of its tip vertex (left / right share the topology)."""
    path = os.path.join(mano_dir, "MANO_RIGHT.pkl")
    with open(path, "rb") as f:  # a single tip vertex measures a different quantity: never substitute it silently
        v = np.asarray(pickle.load(f, encoding="latin1")["v_template"])
    if v.shape != (778, 3) or not np.isfinite(v).all():
        raise ValueError(f"invalid MANO template in {path}: expected finite (778, 3) vertices")
    return {f: np.where(np.linalg.norm(v - v[i], axis=1) < radius)[0] for f, i in TIP_V.items()}


def _furthest(uv, j, g, region):
    j = j.copy()
    for f, (_, _, b, t) in FINGERS.items():
        d = g["joints2d"][t] - g["joints2d"][b]
        u = d / max(np.linalg.norm(d), 1e-6)
        idx = region[f]
        j[t] = uv[idx[np.argmax((uv[idx] - g["joints2d"][b]) @ u)]]
    return j


def tips_of(p, key, K, g, region=None, require_fitted=False):
    """The prediction's 21 2D joints with the tips taken per `key` (None: the key is missing on this prediction)."""
    if key == "surf2d":
        if p.get("surface2d_head") is None or len(p["surface2d_head"]) < 778:
            return None
        return _furthest(np.asarray(p["surface2d_head"], np.float64)[-778:], np.asarray(p["joints2d"], np.float64), g, region)
    if key in ("mesh", "meshmax"):
        if p.get("verts3d") is None or len(p["verts3d"]) != 778:
            return None
        if require_fitted and p.get("verts3d_dense") is None:
            raise ValueError("expected a fitted MANO output, but the prediction lacks verts3d_dense; use a run with the mesh fit")
        uv = project(np.asarray(p["verts3d"], np.float64), K)
        j = np.asarray(p["joints2d"], np.float64)
        if key == "meshmax":
            return _furthest(uv, j, g, region)
        j = j.copy()
        for f, (_, _, _, t) in FINGERS.items():
            j[t] = uv[TIP_V[f]]
        return j
    return None if p.get(key) is None else np.asarray(p[key], np.float64)


def score_frames(frames, key="joints2d", intrinsics="fah", region=None, require_fitted=False):
    """bench2d frames -> {subset: {"along": {finger: [signed errors]}, "length": [ratios]}} (subset "all" included)."""
    along, ratio = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    for fr in frames:
        if fr.get("K") is not None:
            K = np.asarray(fr["K"], np.float64)
        else:
            K = default_K(*fr["image_size"][:2], intrinsics)
        sub = fr.get("subset") or fr["key"].split("/")[0]
        gts = [dict(g, joints2d=np.asarray(g["joints2d"], np.float64), vis=np.asarray(g["vis"]), bbox=np.asarray(g["bbox"])) for g in fr["gt"]]
        m, _ = match(fr["pred"], gts)
        for gi, pi in m.items():
            g = gts[gi]
            j = tips_of(fr["pred"][pi], key, K, g, region, require_fitted)
            if j is None:
                continue
            for f, (a, b, c, t) in FINGERS.items():
                if g["vis"][c] and g["vis"][t]:
                    dd = g["joints2d"][t] - g["joints2d"][c]
                    L = np.linalg.norm(dd)
                    if L > 2:
                        for s in (sub, "all"):
                            along[s][f].append(float((j[t] - g["joints2d"][t]) @ dd / L / L))
                if all(g["vis"][q] for q in (a, b, c, t)):
                    gl = sum(np.linalg.norm(g["joints2d"][q2] - g["joints2d"][q1]) for q1, q2 in ((a, b), (b, c), (c, t)))
                    pl = sum(np.linalg.norm(j[q2] - j[q1]) for q1, q2 in ((a, b), (b, c), (c, t)))
                    if gl > 5:
                        ratio[sub].append(pl / gl)
                        ratio["all"].append(pl / gl)
    return {s: dict(along=dict(along[s]), length=ratio[s]) for s in along}


def summarize(res):
    """{subset: {"tip": median signed error, "short": share < 0, "n", "fingers": {finger: median}, "length": median ratio}}."""
    out = {}
    for s, r in res.items():
        allv = np.concatenate([np.asarray(v) for v in r["along"].values()]) if r["along"] else np.zeros(0)
        out[s] = dict(tip=float(np.median(allv)) if len(allv) else None, short=float(np.mean(allv < 0)) if len(allv) else None, n=int(len(allv)),
                      fingers={f: float(np.median(v)) for f, v in r["along"].items() if v},
                      length=float(np.median(r["length"])) if r["length"] else None)
    return out


def score_run(run_dir, key="joints2d", intrinsics="fah", region=None, require_fitted=False):
    """A bench2d run directory (*.pkl with "frames") -> summarize(score_frames(...))."""
    frames = (fr for path in sorted(glob.glob(os.path.join(run_dir, "*.pkl"))) for fr in pickle.load(open(path, "rb"))["frames"])
    if key in ("meshmax", "surf2d") and region is None:
        raise ValueError(f"key {key} needs the MANO tip regions (tip_regions(mano_dir))")
    return summarize(score_frames(frames, key, intrinsics, region, require_fitted))
