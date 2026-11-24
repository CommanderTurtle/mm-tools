"""Main-text contact figure (the contact table's GT-box protocol, docs/contact.md): per hand input crop | GT contact | HACO |
FAH-L, the three per-vertex maps on the same GT MANO mesh from its GT contact side (view search on the GT labels), each
method's probability mapped so 0.5 = its validation threshold (HACO 0.56; the released FAH-L head is calibrated, 0.5),
grey -> violet linear 0-1. Hands chosen by a fixed rule: pool = valid hot3d_bench + dexycb_s0test hands with >= --min_n10
GT contact vertices and crop scale <= --max_scale; one fixed random order (seed --seed); per slot the first hand of the
slot's set / grasp type (from the GT labels' palm / finger groups) with an object not used yet; last slot = the first hand
where FAH-L's vertex F1 < --fail_f1.

Inputs: <test_root>/<set>/ of train.contact_labels --test with the predictions of bench.contact_eval (pred_<tag>.npz):
--fah tag@threshold (default fah@0.5 = `predict --tag fah` with the released FAH-L head), --haco tag@threshold (default
haco_haco_hand2x@0.56 = `bench.contact_eval haco`).

    python scripts/contact_main_fig.py select --out sel.json
    python scripts/contact_main_fig.py render --sel sel.json --cap_px 45 --out out_dir
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fah.demo_render import TIPS_MANO, contact_colors_linear, render_contact_smooth  # noqa: E402

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
TEST = os.path.join(os.environ.get("CONTACT_ROOT", os.path.join(DATA_ROOT, "contact")), "test")  # --test_root
MANO_DIR = os.environ.get("MANO_DIR", "data/mano")
SETS = {"hot3d_bench": "HOT3D", "dexycb_s0test": "DexYCB"}
METHODS = {}  # haco / fah -> (prediction file, threshold): --haco / --fah
SLOTS = [("hot3d_bench", "pinch"), ("hot3d_bench", "power"), ("hot3d_bench", "fingers"), ("dexycb_s0test", "pinch"),
         ("dexycb_s0test", "power"), (None, "fail")]
GROUP = {0: 0, 13: 1, 14: 1, 15: 1, 1: 2, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 10: 4, 11: 4, 12: 4, 7: 5, 8: 5, 9: 5}  # palm, thumb..pinky


def mano_groups(ml):
    w = ml.layers[True].lbs_weights.numpy().argmax(1)
    return np.array([GROUP[k] for k in w])


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cal(p, thr):
    """Per-vertex probability mapped so that the method's threshold is 0.5."""
    return 1 / (1 + np.exp(-(logit(p) - logit(thr))))


def vf1(p, thr, lab):
    pb = np.asarray(p) >= thr
    tp, fp, fn = (pb & lab).sum(), (pb & ~lab).sum(), (~pb & lab).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def load_set(s):
    root = os.path.join(TEST, s)
    idx = [json.loads(l) for l in open(os.path.join(root, "index.jsonl"))]
    con = {json.loads(l)["file"]: json.loads(l) for l in open(os.path.join(root, "contact.jsonl"))}
    sdf = np.load(os.path.join(root, "sdf.npy")).astype(np.float32)
    pred = {}
    for m, (f, _) in METHODS.items():
        z = np.load(os.path.join(root, f))
        assert list(z["files"]) == [r["file"] for r in idx], (s, m)
        pred[m] = z["prob"].astype(np.float32)
    return idx, con, sdf, pred


def grasp(lab, groups):
    n = np.array([lab[groups == g].sum() for g in range(6)])
    fingers = [g for g in range(1, 6) if n[g] >= 5]
    if n[0] >= 10 and len(fingers) >= 4:
        return "power"
    if n[0] < 10 and 1 in fingers and len(fingers) <= 3:
        return "pinch"
    if n[0] < 10 and len(fingers) >= 4:
        return "fingers"
    return "other"


def select(a):
    from bench.common import ManoLayer

    groups = mano_groups(ManoLayer(MANO_DIR))
    pool = []
    for s in SETS:
        idx, con, sdf, pred = load_set(s)
        for i, r in enumerate(idx):
            c = con.get(r["file"], {})
            lab = sdf[i] < 10
            if not c.get("valid") or lab.sum() < a.min_n10 or r["M"][0][0] > a.max_scale:
                continue
            pool.append(dict(set=s, row=i, file=r["file"], obj=c.get("obj"), is_right=bool(r["is_right"]), n10=int(lab.sum()),
                             grasp=grasp(lab, groups), f1_fah=vf1(pred["fah"][i], METHODS["fah"][1], lab),
                             f1_haco=vf1(pred["haco"][i], METHODS["haco"][1], lab), scale=float(r["M"][0][0])))
    order = np.random.default_rng(a.seed).permutation(len(pool))
    picks, used = [], set()
    for s, kind in SLOTS:
        for k in order:
            h = pool[k]
            if h["obj"] in used or any(p["file"] == h["file"] and p["set"] == h["set"] for p in picks):
                continue
            if kind == "fail":
                ok = h["f1_fah"] < a.fail_f1
            else:
                ok = h["set"] == s and h["grasp"] == kind
            if ok:
                picks.append(dict(h, slot=kind, order_rank=int(np.nonzero(order == k)[0][0])))
                used.add(h["obj"])
                break
    rule = (f"pool: valid hands of hot3d_bench + dexycb_s0test with >= {a.min_n10} GT contact vertices (10 mm) and crop scale "
            f"<= {a.max_scale}; fixed random order (numpy default_rng({a.seed}).permutation); per slot the first hand of the "
            f"slot's set and GT grasp type (palm >= 10 contact vertices + >= 4 fingers = power; no palm, thumb + <= 3 fingers = "
            f"pinch; no palm, >= 4 fingers = fingers; a finger counts with >= 5 contact vertices) whose object is not used yet; "
            f"last slot: the first hand (either set) with FAH-L vertex F1 < {a.fail_f1}")
    json.dump(dict(rule=rule, pool=len(pool), methods=METHODS, fail_f1=a.fail_f1, picks=picks), open(a.out, "w"), indent=1)
    print(len(pool), "eligible hands")
    for p in picks:
        print(p["slot"], p["set"], p["file"], p["obj"], "n10", p["n10"], "F1 HACO %.2f FAH-L %.2f" % (p["f1_haco"], p["f1_fah"]))


def view_R(x, w, faces):
    """Yaw / pitch maximising the front-facing GT contact weight x the smallest fingertip gap."""
    fn = np.cross(x[faces[:, 1]] - x[faces[:, 0]], x[faces[:, 2]] - x[faces[:, 0]])
    vn = np.zeros_like(x)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    best = None
    for yaw in np.radians(np.arange(0, 360, 15)):
        for pitch in np.radians(np.arange(-60, 61, 15)):
            cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
            R = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            xr, nr = x @ R.T, vn @ R.T
            front = (nr[:, 2] < 0).astype(np.float64)
            t2 = xr[list(TIPS_MANO), :2]
            gap = min(np.linalg.norm(t2[i] - t2[j]) for i in range(5) for j in range(i + 1, 5))
            score = (front * w).sum() / max(w.sum(), 1e-9) * gap
            if best is None or score > best[0]:
                best = (score, R)
    return best[1]


def render_view(x, c, faces, R, size):
    xr = x @ R.T
    span = np.ptp(xr[:, :2], 0).max()
    z0 = 0.6
    f = 0.85 * size * z0 / span
    vv = xr + np.array([0, 0, z0])
    K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]])
    uv = vv[:, :2] / vv[:, 2:] * f
    K[0, 2], K[1, 2] = size / 2 - (uv[:, 0].min() + uv[:, 0].max()) / 2, size / 2 - (uv[:, 1].min() + uv[:, 1].max()) / 2
    return render_contact_smooth(np.full((size, size, 3), 255, np.uint8), [(vv, c)], faces, K, alpha=1.0, s=3)


def label(img, text, org, scale, color=(40, 40, 40), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick + 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render(a):
    from bench.common import ManoLayer

    sel = json.load(open(a.sel))
    METHODS.update({m: tuple(v) for m, v in sel.get("methods", {}).items()})  # the predictions the selection was made on
    ml = ManoLayer(MANO_DIR)
    data = {s: load_set(s) for s in {p["set"] for p in sel["picks"]}}
    P, cells, facts = a.panel, [], []
    th = max(1, int(round(2 * P / 300)))
    fs = a.font * P / 300
    if a.cap_px:  # labels sized by cap height (px at this panel size), e.g. 45 px = ~6.5 pt with 12 tiles across linewidth
        cap = a.cap_px * P / 300
        th = max(2, int(round(0.1 * cap)))
        probe = np.zeros((200, 200), np.uint8)  # measured ink height of "H" at scale 2 (getTextSize overstates it)
        cv2.putText(probe, "H", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 255, th, cv2.LINE_AA)
        ys = np.nonzero((probe > 127).any(1))[0]
        fs = 2.0 * cap / (ys.max() - ys.min() + 1)
    os.makedirs(a.out, exist_ok=True)
    for n, p in enumerate(sel["picks"]):
        idx, con, sdf, pred = data[p["set"]]
        r = idx[p["row"]]
        v = np.load(os.path.join(TEST, p["set"], r["verts_file"])).astype(np.float64)
        faces = ml.faces[bool(r["is_right"])]
        x = v - v.mean(0)
        if np.einsum("ij,ij->i", x[faces[:, 0]], np.cross(x[faces[:, 1]], x[faces[:, 2]])).sum() < 0:
            faces = faces[:, ::-1]  # outward normals for the view search (the GT left meshes come with the other winding)
        lab = sdf[p["row"]] < 10
        R = view_R(x, lab.astype(np.float64), faces)  # one view for all three maps: the GT contact side
        crop = cv2.imread(os.path.join(TEST, p["set"], r["file"]))
        tiles = [cv2.resize(crop, (P, P), interpolation=cv2.INTER_AREA)]
        maps = [("GT", lab.astype(np.float64), None)] + [(m, cal(pred[m][p["row"]], METHODS[m][1]), m) for m in ("haco", "fah")]
        for nm, pc, m in maps:
            t = render_view(x, contact_colors_linear(pc), faces, R, 2 * P)
            tiles.append(cv2.resize(t, (P, P), interpolation=cv2.INTER_AREA))
        band = int(0.17 * P) if not a.cap_px else int(round(1.75 * cap))  # text band: dataset under the input, F1 under predictions
        texts = [SETS[p["set"]], "", "F1 %.2f" % p["f1_haco"], "F1 %.2f" % p["f1_fah"]]
        tiles = [np.vstack([t, np.full((band, P, 3), 255, np.uint8)]) for t in tiles]
        for t, tx in zip(tiles, texts):
            if tx:
                label(t, tx, (int(0.04 * P), P + (int(0.13 * P) if not a.cap_px else int(round(1.3 * cap)))), fs, thick=th)
        pad = lambda t: cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cell = np.hstack([pad(t) for t in tiles])
        cv2.imwrite(os.path.join(a.out, f"hand{n + 1}_{p['slot']}.png"), cell)
        cells.append(cell)
        facts.append(dict(hand=n + 1, slot=p["slot"], dataset=SETS[p["set"]], file=p["file"], object=p["obj"], is_right=p["is_right"],
                          gt_contact_vertices=p["n10"], vF1_HACO=round(p["f1_haco"], 2), vF1_FAH_L=round(p["f1_fah"], 2),
                          fah_l_failure=p["f1_fah"] < sel.get("fail_f1", 0.3)))
    per_row = (len(cells) + a.rows - 1) // a.rows
    W = per_row * cells[0].shape[1] + (per_row - 1) * 10
    names = ["input", "ground truth", "HACO", "FAH-L"]
    lines = []  # a header that does not fit one tile wraps at its spaces
    for nm in names:
        w = cv2.getTextSize(nm, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][0]
        lines.append(nm.split(" ") if w > P - int(0.06 * P) else [nm])
    lh = int(round(1.65 * cap)) if a.cap_px else int(0.15 * P)
    head = np.full((int(0.04 * P) + lh * max(len(l) for l in lines) + int(0.04 * P), W, 3), 255, np.uint8)
    for j in range(per_row):
        for k, ls in enumerate(lines):
            for q, tx in enumerate(ls):
                label(head, tx, (j * (cells[0].shape[1] + 10) + k * (P + 4) + int(0.04 * P), int(0.02 * P) + lh * (q + 1) - int(0.1 * lh)), fs,
                      color=(0, 0, 0), thick=th)
    rows = []
    for i in range(a.rows):
        rc = cells[i * per_row:(i + 1) * per_row]
        if not rc:
            break
        gap = np.full((rc[0].shape[0], 10, 3), 255, np.uint8)
        row = np.hstack(sum([[c, gap] for c in rc], [])[:-1])
        rows.append(cv2.copyMakeBorder(row, 0, 8, 0, W - row.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    fig = np.vstack([head] + rows)
    cv2.imwrite(os.path.join(a.out, "contact_main_fig.png"), fig)
    cv2.imwrite(os.path.join(a.out, "contact_main_fig.jpg"), fig, [cv2.IMWRITE_JPEG_QUALITY, 95])
    bar = np.full((int(0.6 * P), int(0.13 * P), 3), 255, np.uint8)  # colour bar: calibrated probability 0 (bottom) -> 1 (top)
    h0, h1 = int(0.06 * P), int(0.54 * P)
    for y in range(h0, h1):
        c = contact_colors_linear(np.array([1 - (y - h0) / (h1 - h0)]))[0]
        bar[y, int(0.02 * P):int(0.06 * P)] = (np.clip(c, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(a.out, "colour_bar.png"), bar)
    json.dump(dict(rule=sel["rule"], hands=facts, thresholds={m: t for m, (_, t) in METHODS.items()}), open(os.path.join(a.out, "facts.json"), "w"), indent=1)
    print(json.dumps(facts, indent=1))
    print("->", os.path.join(a.out, "contact_main_fig.png"), fig.shape)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["select", "render"])
    p.add_argument("--test_root", default=TEST, help="held-out contact sets with their predictions (bench.contact_eval)")
    p.add_argument("--fah", default="fah@0.5", help="FAH-L prediction tag@threshold (pred_<tag>.npz; 0.5 for the calibrated head)")
    p.add_argument("--haco", default="haco_haco_hand2x@0.56", help="HACO prediction tag@threshold")
    p.add_argument("--min_n10", type=int, default=30)
    p.add_argument("--max_scale", type=float, default=5.0, help="crop scale (crop px per frame px) upper bound")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fail_f1", type=float, default=0.3)
    p.add_argument("--sel", default="")
    p.add_argument("--panel", type=int, default=300)
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--font", type=float, default=1.3, help="label scale at a 300 px panel (1.3: ~6 pt in a 2-row figure* at linewidth)")
    p.add_argument("--cap_px", type=float, default=0, help="label cap height in px at a 300 px panel (overrides --font)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    TEST = a.test_root
    for m, spec in (("haco", a.haco), ("fah", a.fah)):
        tag, thr = spec.rsplit("@", 1)
        METHODS[m] = (f"pred_{tag}.npz", float(thr))
    dict(select=select, render=render)[a.cmd](a)
