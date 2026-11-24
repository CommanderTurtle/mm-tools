"""Supplementary comparison figure (fig:mcompare): SAM 3D Body's hand branch vs FAH-L on bench2d v2 hands, one column per
stratum, each hand scored by its PCK@0.1 against the 2D ground truth (two hands: mean of both).

Inputs: bench2d predictions of both methods (`python -m bench.run --dataset bench2d --split none`) on bench2d v2 and on the
low-light set (bench2d v2's synthetic dark copies, subsets syn_*; docs/evaluation.md), e.g.
  --method sam3db_hand --method_args '{"checkpoint": "<sam-3d-body-dinov3>/model.ckpt"}'      (SAM 3D Body hand)
  --method fah --method_args '{"model": "fah", "mode": "image", "weights": "<weights>", "mesh": "none"}'
                                                    (FAH-L without the mesh fit, as in the paper figure)

Three steps:
  candidates  every GT hand both methods found with the GT handedness, SAM 3D Body's wrist within 0.2 GT box sides of the
              GT wrist, >= 12 visible joints, box side >= 48 px; per hand: subset, size relative to the image, overlap with
              the other hands, palm / back of the hand, curled fingers, skin tone (ITA of the hand's joint hull), image
              brightness, both methods' PCK@0.1 / 0.05, source sequence.
  select      per stratum (below), the stratum's eligible hands in a fixed random order (numpy default_rng(0)); the pick
              is the first one that passes the stratum's visual check and comes from a source sequence no other column
              uses; outdoor and motion blur also need FAH-L >= SAM 3D Body + 0.1 (PCK@0.1). The visual checks were done
              by eye on input-only crops (no predictions shown) and are recorded in data/compare_figure_checks.json: the
              inspected prefix of each order with pass / fail and the reason, so the selection is deterministic.
  render      input | SAM 3D Body hand | FAH-L (mesh + 2D skeleton; FAH-L's mesh is its dense MANO-topology surface,
              verts3d_dense when run with mesh "none"), labelled with PCK@0.1.

Strata: ego close-up (HOT3D, hand >= 0.25 of the short image side), small hand in an egocentric scene (EPIC via HInt,
< 0.12), third-person hand with an object (New Days via HInt, >= 15 visible joints), outdoor scene (COCO, < 0.15), darker
skin (COCO / New Days / CMU, ITA < 10, image L > 35), hard pose without an object (1-3 curled fingers, >= 15 visible joints;
HOT3D / HInt / COCO with a figure-safe licence), two hands (box IoU >= 0.1 with a second eligible hand), back of the hand
with self-occlusion, low light (synthetic copies), motion blur (lowest decile of the hand box's Laplacian variance at 128 px,
box side >= 80 px, figure-safe licence).

    python scripts/compare_figure.py candidates --v2 <bench2d_v2> --dark <bench2d_v2_dark> \
        --sam_v2 <run> --fah_v2 <run> --sam_dark <run> --fah_dark <run> --out cands.json
    python scripts/compare_figure.py select --cands cands.json --coco_wholebody <coco_wholebody_val_v1.0.json> --out picks.json
    python scripts/compare_figure.py render --picks picks.json --v2 ... --dark ... --sam_v2 ... --fah_v2 ... --sam_dark ... \
        --fah_dark ... --mano_dir <mano> --out compare.jpg

Image credits of the paper figure (bench2d v2 test images), left to right: 1 and 9 frames of the held-out HOT3D benchmark
clips (CC BY-SA 4.0, hand annotations CC BY-NC-SA 4.0; 9 with our synthetic low light); 2, 8 and 10 EPIC-KITCHENS-100 frames
via HInt (CC BY-NC 4.0); 3, 5 and 6 New Days of Hands frames via HInt (public test split); 4 "Luggage and all" by Sally Taylor
(Flickr, CC BY-NC 2.0) and 7 Flickr photo 7173995615 (CC BY-NC 2.0), both via COCO val2017 and COCO-WholeBody.
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.bench2d.evaluate import _norm_err, match  # noqa: E402

CHECKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "compare_figure_checks.json")
EGO = {"hot3d", "hint_epick"}
SAFE_COCO = {2, 4, 5, 7, 8}  # COCO image licences: CC BY-NC 2 / BY 4 / BY-SA 5 / no known restriction 7 / US Gov 8
SAFE_SUBSETS = ("hot3d", "hint_nd", "hint_epick")


def load(run):
    out = {}
    for f in sorted(glob.glob(os.path.join(run, "chunk*.pkl"))):
        for fr in pickle.load(open(f, "rb"))["frames"]:
            out[fr["key"]] = fr
    return out


def pck(fr, gi, thr):
    """(PCK of the prediction matched to GT hand gi, that prediction); (0, None) when missed."""
    pm, _ = match(fr["pred"], fr["gt"])
    if gi not in pm:
        return 0.0, None
    e, _ = _norm_err(np.asarray(fr["pred"][pm[gi]]["joints2d"]), fr["gt"][gi])
    return float((e < thr).mean()), fr["pred"][pm[gi]]


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / max(u, 1e-6)


def palm_side(j, is_right):
    """> 0: palm towards the camera, < 0: back of the hand (2D GT: wrist, index MCP 5, pinky MCP 17)."""
    a, b = j[5] - j[0], j[17] - j[0]
    c = (a[0] * b[1] - a[1] * b[0]) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-6)
    return float(-c if is_right else c)


def n_curled(j):
    """Fingers whose 2D tip is closer to the wrist than their PIP joint (curled / folded)."""
    out = 0
    for f in range(1, 5):
        pip, tip = 2 + 4 * f, 4 + 4 * f
        out += np.linalg.norm(j[tip] - j[0]) < np.linalg.norm(j[pip] - j[0])
    return int(out)


def ita(img, j, vis):
    """Individual typology angle (deg) and L* of the median colour inside the visible joints' convex hull."""
    pts = j[vis].astype(np.float32)
    if len(pts) < 3:
        return np.nan, np.nan
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(pts).astype(np.int32), 1)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    px = lab[m > 0]
    if len(px) < 30:
        return np.nan, np.nan
    L, bb = np.median(px[:, 0]) * 100 / 255, np.median(px[:, 2]) - 128
    return float(np.degrees(np.arctan2(L - 50, bb))), float(L)


def seq_of(k):
    k = re.sub(r"^syn_", "", k)
    return re.sub(r"(?<=[^/])_(frame_?)?\d+$", "", k)


def sets(a):
    return [("bench2d_v2", a.v2, a.sam_v2, a.fah_v2), ("dark_v1", a.dark, a.sam_dark, a.fah_dark)]


def candidates(a):
    rows = []
    for data, root, ra, rb in sets(a):
        A, B = load(ra), load(rb)
        recs = {r["id"]: r for r in json.load(open(os.path.join(root, "annotations.json")))["images"]}
        for key, fb in B.items():
            fa, rec = A.get(key), recs.get(key)
            if fa is None or rec is None:
                continue
            sub = rec.get("subset") or key.split("/")[0]
            if data == "dark_v1" and not sub.startswith("syn_"):
                continue  # real dark photos: not figure-safe
            img, (H, W), gts = None, (rec["height"], rec["width"]), fb["gt"]
            for gi, g in enumerate(gts):
                x0, y0, x1, y1 = g["bbox"]
                S = max(x1 - x0, y1 - y0, 1.0)
                vis = np.asarray(g["vis"]) > 0
                if vis.sum() < a.min_vis or S < a.min_side:
                    continue
                sa, pa = pck(fa, gi, 0.1)
                sb, pb = pck(fb, gi, 0.1)
                if pa is None or pb is None or bool(pa["is_right"]) != bool(g["is_right"]) or bool(pb["is_right"]) != bool(g["is_right"]):
                    continue
                if np.linalg.norm(np.asarray(pa["joints2d"])[0, :2] - g["joints2d"][0, :2]) / S > a.max_wrist:
                    continue
                ov = max([iou(g["bbox"], h["bbox"]) for hi, h in enumerate(gts) if hi != gi] or [0.0])
                partner = int(np.argmax([iou(g["bbox"], h["bbox"]) if hi != gi else -1 for hi, h in enumerate(gts)])) if len(gts) > 1 else -1
                if img is None:
                    img = cv2.imread(os.path.join(root, rec["file"]))
                j = np.asarray(g["joints2d"])[:, :2]
                t, L = ita(img, j, vis)
                rows.append(dict(data=data, key=key, gi=gi, subset=sub, ego=sub.replace("syn_", "") in EGO, side=float(S),
                                 rel=float(S / min(H, W)), nvis=int(vis.sum()), ov=float(ov), partner=partner,
                                 palm=palm_side(j, bool(g["is_right"])), curled=n_curled(j), ita=t, L=L,
                                 img_L=float(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].mean() * 100 / 255),
                                 pa=sa, pb=sb, pa05=pck(fa, gi, 0.05)[0], pb05=pck(fb, gi, 0.05)[0], seq=seq_of(key),
                                 file=rec["file"], bbox=[float(v) for v in g["bbox"]], W=W, H=H))
        print(data, len(rows), flush=True)
    json.dump(rows, open(a.out, "w"))
    print("->", a.out, len(rows))


def blur_score(img_gray, bbox):
    """Laplacian variance of the hand box resized to a 128 px long side (low: motion blur)."""
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    H, W = img_gray.shape
    c = img_gray[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    s = 128.0 / max(c.shape)
    c = cv2.resize(c, (max(8, int(c.shape[1] * s)), max(8, int(c.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(c, cv2.CV_64F).var())


def seeded(rows):
    perm = np.random.default_rng(0).permutation(len(rows))
    return [rows[i] for i in perm]


def strata(C, have):
    dark = lambda r: r["data"] == "dark_v1"
    return {
        "s1_ego_closeup": lambda r: r["subset"] == "hot3d" and r["rel"] >= 0.25,
        "s2_ego_scene_small": lambda r: r["subset"] == "hint_epick" and r["rel"] < 0.12,
        "s3_third_object": lambda r: r["subset"] == "hint_nd" and r["nvis"] >= 15,
        "s4_outdoor_scene": lambda r: r["subset"] == "coco" and r["rel"] < 0.15 and r["nvis"] >= 15,
        "s5_darker_skin": lambda r: r["subset"] in ("coco", "hint_nd", "cmu") and r["ita"] == r["ita"] and r["ita"] < 10 and r["img_L"] > 35,
        "s6_hard_pose_noobj": lambda r: r["data"] == "bench2d_v2" and r["subset"] in ("hot3d", "coco", "hint_nd", "hint_epick")
        and 1 <= r["curled"] <= 3 and r["nvis"] >= 15,
        "s7_two_hands": lambda r: not dark(r) and r["ov"] >= 0.1 and (r["key"], r["partner"]) in have and r["gi"] < r["partner"],
        "s8_back_selfocc": lambda r: not dark(r) and r["palm"] < -0.3 and (r["curled"] >= 2 or r["nvis"] < 21),
        "s9_low_light": dark,
        "s10_motion_blur": lambda r: r["data"] == "bench2d_v2" and r["subset"] in ("hot3d", "coco", "hint_nd", "hint_epick") and r["side"] >= 80,
    }


def walk(order, checks, rule):
    """First candidate of `order` that passed its recorded visual check and passes `rule` -> (index, candidate)."""
    for i, r in enumerate(order):
        if i >= len(checks):
            raise RuntimeError(f"candidate #{i} ({r['key']} hand {r['gi']}) has no recorded visual check")
        c = checks[i]
        if (c["key"], c["gi"]) != (r["key"], r["gi"]):
            raise RuntimeError(f"order mismatch at #{i}: {r['key']} {r['gi']} vs recorded {c['key']} {c['gi']}")
        if c["visual"] and rule(r):
            return i, r
    raise RuntimeError("no candidate passes")


def stratum_order(name, spec, C, have, safe, img_root):
    """The stratum's seeded order. mode order: all eligible hands; filter_then_order: the licence / excluded-source filters
    before the draw; order_then_filter: the same filters on the drawn order; blur: the lowest decile of blur_score among the
    filtered eligible hands, then the draw."""
    el = [r for r in C if strata(C, have)[name](r)]
    ok = lambda r: (safe(r) or not spec.get("licence")) and r["seq"] not in set(spec.get("exclude_seqs", []))
    mode = spec.get("mode", "order")
    if mode == "filter_then_order":
        return seeded([r for r in el if ok(r)])
    if mode == "order_then_filter":
        return [r for r in seeded(el) if ok(r)]
    if mode == "blur":
        pool = [r for r in el if ok(r)]
        for r in pool:
            r["blur"] = blur_score(cv2.imread(os.path.join(img_root(r), r["file"]), cv2.IMREAD_GRAYSCALE), r["bbox"])
        thr = float(np.percentile([r["blur"] for r in pool], 10))
        return seeded([r for r in pool if r["blur"] <= thr])
    return seeded(el)


def select(a):
    C = json.load(open(a.cands))
    K = json.load(open(a.checks))
    lic = {im["id"]: im["license"] for im in json.load(open(a.coco_wholebody))["images"]} if a.coco_wholebody else {}
    safe = lambda r: lic.get(int(r["key"].split("/")[1])) in SAFE_COCO if r["subset"] == "coco" else r["subset"] in SAFE_SUBSETS
    have = {(r["key"], r["gi"]) for r in C}
    img_root = lambda r: a.v2 if r["data"] == "bench2d_v2" else a.dark
    gain = lambda r: r["pb"] - r["pa"] >= 0.1 - 1e-9
    picks = {}
    for name in K["columns"]:
        spec = K["strata"][name]
        if spec.get("licence") and not lic:
            raise SystemExit(f"{name} needs --coco_wholebody (image licences)")
        order = stratum_order(name, spec, C, have, safe, img_root)
        i, r = walk(order, spec["checks"], gain if spec.get("gain") else (lambda r: True))
        gi = [r["gi"], r["partner"]] if name == "s7_two_hands" else r["gi"]
        picks[name] = dict(scale=spec["scale"], stratum=name, data=r["data"], key=r["key"], gi=gi, seq=r["seq"], rel=round(r["rel"], 3),
                           side=round(r["side"]), pa=round(r["pa"], 3), pb=round(r["pb"], 3), order_index=i)
        print(name, "#%d" % i, r["key"], gi, f"SAM 3D Body {r['pa']:.3f} FAH-L {r['pb']:.3f}", flush=True)
    json.dump([picks[n] for n in K["columns"]], open(a.out, "w"), indent=1)
    print("->", a.out)


def crop_img(img, ps, K, faces, box, tile):
    from fah.vis import LEFT_COLOR, RIGHT_COLOR, draw_hand_skeleton, render_meshes

    out = img.copy()
    if ps:
        mesh = lambda p: p["verts3d"] if p.get("verts3d") is not None else p.get("verts3d_dense")  # mesh "none": the dense surface
        out = render_meshes(out, [(np.asarray(mesh(p), np.float64), faces[bool(p["is_right"])], RIGHT_COLOR if p["is_right"] else LEFT_COLOR)
                                  for p in ps if mesh(p) is not None], np.asarray(K, np.float64))
    X0, Y0, side = box
    out = cv2.resize(out[Y0:Y0 + side, X0:X0 + side], (tile, tile), interpolation=cv2.INTER_AREA)
    for p in ps:
        out = draw_hand_skeleton(out, (np.asarray(p["joints2d"])[:, :2] - [X0, Y0]) * tile / side, thickness=2, radius=3)
    return out


def render(a):
    from bench.common import ManoLayer
    from fah.estimator import default_cam_int

    picks = json.load(open(a.picks))
    faces = ManoLayer(a.mano_dir).faces
    runs, cols, report = {}, [], []
    roots = {s[0]: s[1:] for s in sets(a)}
    T = a.tile
    for pk in picks:
        data = pk["data"]
        if data not in runs:
            root, ra, rb = roots[data]
            runs[data] = (load(ra), load(rb), {r["id"]: r for r in json.load(open(os.path.join(root, "annotations.json")))["images"]})
        A, B, recs = runs[data]
        key, gis = pk["key"], pk["gi"] if isinstance(pk["gi"], list) else [pk["gi"]]
        fa, fb, rec = A[key], B[key], recs[key]
        img = cv2.imread(os.path.join(roots[data][0], rec["file"]))
        H, W = img.shape[:2]
        K = fb["K"] if fb.get("K") is not None else default_cam_int(H, W)
        pas, pbs, sas, sbs = [], [], [], []
        x0 = y0 = 1e9
        x1 = y1 = -1e9
        for gi in gis:
            sa, pa = pck(fa, gi, 0.1)
            sb, pb = pck(fb, gi, 0.1)
            pas.append(pa), pbs.append(pb), sas.append(sa), sbs.append(sb)
            bx = fb["gt"][gi]["bbox"]
            x0, y0, x1, y1 = min(x0, bx[0]), min(y0, bx[1]), max(x1, bx[2]), max(y1, bx[3])
        side = int(min(max(x1 - x0, y1 - y0) * pk.get("scale", 1.5), H, W))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        X0, Y0 = int(np.clip(cx - side / 2, 0, W - side)), int(np.clip(cy - side / 2, 0, H - side))
        box = (X0, Y0, side)
        tiles = [cv2.resize(img[Y0:Y0 + side, X0:X0 + side], (T, T), interpolation=cv2.INTER_AREA),
                 crop_img(img, pas, K, faces, box, T), crop_img(img, pbs, K, faces, box, T)]
        sa, sb = float(np.mean(sas)), float(np.mean(sbs))
        for t, lab in zip(tiles[1:3], [f"{a.names[0]}: {sa:.2f}", f"{a.names[1]}: {sb:.2f}"]):
            cv2.rectangle(t, (0, T - 22), (T, T), (255, 255, 255), -1)
            cv2.putText(t, lab, (5, T - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cols.append(np.concatenate([np.pad(t, ((2, 2), (2, 2), (0, 0)), constant_values=255) for t in tiles], 0))
        report.append(dict(stratum=pk["stratum"], data=data, key=key, gi=gis, file=rec["file"], pck10_a=sas, pck10_b=sbs,
                           crop=[X0, Y0, side], image_size=[W, H]))
    grid = np.concatenate(cols, 1)
    head = np.full((grid.shape[0], 26, 3), 255, np.uint8)
    for k, lab in enumerate(["input"] + a.names):
        ty = k * (T + 4) + T // 2
        t = np.full((22, T, 3), 255, np.uint8)
        cv2.putText(t, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        r = cv2.rotate(t, cv2.ROTATE_90_COUNTERCLOCKWISE)
        y = max(0, ty - T // 2)
        head[y:y + r.shape[0], 2:2 + r.shape[1]] = r[: head.shape[0] - y]
    cv2.imwrite(a.out, np.concatenate([head, grid], 1), [cv2.IMWRITE_JPEG_QUALITY, 92])
    json.dump(report, open(a.out.rsplit(".", 1)[0] + ".json", "w"), indent=1)
    print("->", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["candidates", "select", "render"])
    for k in ("v2", "dark", "sam_v2", "fah_v2", "sam_dark", "fah_dark"):
        p.add_argument("--" + k, default="", help="bench2d v2 / low-light set roots; their SAM 3D Body / FAH-L run dirs")
    p.add_argument("--min_vis", type=int, default=12)
    p.add_argument("--min_side", type=float, default=48)
    p.add_argument("--max_wrist", type=float, default=0.2)
    p.add_argument("--cands", default="")
    p.add_argument("--checks", default=CHECKS, help="recorded visual checks + strata settings")
    p.add_argument("--coco_wholebody", default="", help="COCO-WholeBody val annotations (image licences)")
    p.add_argument("--picks", default="")
    p.add_argument("--names", default="SAM 3D Body hand,FAH-L", type=lambda s: s.split(","))
    p.add_argument("--tile", type=int, default=240)
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dict(candidates=candidates, select=select, render=render)[a.cmd](a)
