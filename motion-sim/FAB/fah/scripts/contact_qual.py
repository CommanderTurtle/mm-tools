"""Qualitative contact panels (the supplement's contact figure, docs/contact.md): the FAH image head's per-vertex contact on
complete fitted MANO meshes (fah.demo_render: smooth shading, nothing clipped), a palm-side view and, on HOT3D, the GT contact.

    # held-out HOT3D: candidates per category (pinch / power / two-hand / failures) from the test predictions (bench.contact_eval)
    python scripts/contact_qual.py select_hot3d --pred fah --thr 0.5 --out hot3d_cands.json
    # open-licence COCO images (CC BY / BY-SA / no known restriction / US Gov) with a large hand: run FAH, rank by contact
    python scripts/contact_qual.py select_coco --coco_root $DATA_ROOT/coco --out coco_cands.json
    # panels for a list of picks ({"hot3d": [...], "coco": [...]}) -> one PNG per pick + a contact sheet; the paper's picks:
    python scripts/contact_qual.py render --picks scripts/contact_qual_picks.json --hot3d $DATA_ROOT/hot3d/train_aria \
        --coco_root $DATA_ROOT/coco --out panels

The model: fah.FAH(--model, "image", contact=True) with the released weights in --weights (FAH-L by default). HOT3D picks need the
held-out set's labels (<test_root>/hot3d_bench: train.contact_labels --test / --frames).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fah.demo_render import best_view, contact_colors_linear, kabsch, render_contact_smooth, render_view  # noqa: E402

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
TEST = os.path.join(os.environ.get("CONTACT_ROOT", os.path.join(DATA_ROOT, "contact")), "test")  # --test_root
MANO_DIR = os.environ.get("MANO_DIR", "data/mano")
FINGER = {0: 0, 13: 1, 14: 1, 15: 1, 1: 2, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 10: 4, 11: 4, 12: 4, 7: 5, 8: 5, 9: 5}  # MANO joint -> palm, thumb .. pinky


def method(a):
    from fah.api import FAH

    return FAH(a.model, "image", contact=True, trt_dir=a.trt_dir, **({"weights": a.weights} if a.weights else {}))


def header(img, text, color=(255, 255, 255), bg=(40, 40, 40), h=40):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], h), bg, -1)
    cv2.putText(out, text, (12, int(h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return out


def mano_bits():
    from bench.common import ManoLayer

    mano = ManoLayer(MANO_DIR)
    part = {s: np.vectorize(FINGER.get)(mano.layers[s].lbs_weights.numpy().argmax(1)) for s in (True, False)}
    rest = {s: mano(s, np.zeros(10), np.zeros(45), np.zeros(3), np.zeros(3))[1] for s in (True, False)}
    return mano, part, rest


def select_hot3d(a):
    """Per category up to --per candidates (distinct clips), scored from the test predictions vs GT labels."""
    _, part, _ = mano_bits()
    idx = [json.loads(l) for l in open(os.path.join(TEST, "hot3d_bench", "index.jsonl"))]
    meta = [json.loads(l) for l in open(os.path.join(TEST, "hot3d_bench", "contact.jsonl"))]
    sdf = np.load(os.path.join(TEST, "hot3d_bench", "sdf.npy")).astype(np.float32)
    z = np.load(os.path.join(TEST, "hot3d_bench", f"pred_{a.pred}.npz"))
    prob, hand = z["prob"].astype(np.float32), z["hand"]
    vis = {}
    for r in idx:
        clip, key = r["src"].split(":")
        if clip not in vis:
            g = np.load(os.path.join(TEST, "hot3d_bench_frames", clip + ".npz"))
            vis[clip] = {(str(k), bool(s)): v for k, s, v in zip(g["key"], g["is_right"], g["vis"])}
    rows = []
    for i, (r, m) in enumerate(zip(idx, meta)):
        clip, key = r["src"].split(":")
        lab, pb = sdf[i] < 10, prob[i] >= a.thr
        tp, fp, fn = (pb & lab).sum(), (pb & ~lab).sum(), (~pb & lab).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        pt = part[bool(r["is_right"])]
        fing = [bool(lab[pt == k].sum() >= 5) for k in range(6)]
        v = vis[clip].get((key, bool(r["is_right"])))
        vfrac = float(v[lab].mean()) if v is not None and lab.any() else 1.0
        side = max(r["box"][2] - r["box"][0], r["box"][3] - r["box"][1])
        rows.append(dict(i=i, file=r["file"], clip=clip, key=key, is_right=bool(r["is_right"]), box=r["box"], obj=m.get("obj"),
                         min_mm=float(m["min_mm"]), n10=int(lab.sum()), npred=int(pb.sum()), f1=float(f1), flag=float(hand[i]),
                         fingers=fing, vis_contact=vfrac, side=float(side)))
    big = [r for r in rows if r["side"] >= a.min_side]
    frames = {}
    for r in rows:
        frames.setdefault((r["clip"], r["key"]), []).append(r)
    cats = {
        "pinch": [r for r in big if r["fingers"][1] and r["fingers"][2] and not r["fingers"][0] and sum(r["fingers"][3:]) <= 1 and r["f1"] >= 0.6],
        "power": [r for r in big if r["fingers"][0] and sum(r["fingers"][1:]) >= 4 and r["f1"] >= 0.7],
        "two_hand": [r for r in big if len(frames[(r["clip"], r["key"])]) == 2 and all(q["n10"] >= 40 and q["f1"] >= 0.65 for q in frames[(r["clip"], r["key"])]) and r["is_right"]],
        "fail_hover": [r for r in big if 10 <= r["min_mm"] < 30 and r["flag"] > 0.5 and r["npred"] >= 40],
        "fail_occluded": [r for r in big if r["n10"] >= 60 and r["vis_contact"] < 0.3 and r["f1"] < 0.35],
        "fail_depth": [r for r in big if r["min_mm"] >= 30 and r["npred"] >= 60],
    }
    key_fn = dict(pinch=lambda r: -r["f1"], power=lambda r: -r["f1"], two_hand=lambda r: -r["f1"], fail_hover=lambda r: -r["npred"],
                  fail_occluded=lambda r: -r["n10"], fail_depth=lambda r: -r["npred"])
    out = {}
    for c, lst in cats.items():
        seen, pick = set(), []
        for r in sorted(lst, key=key_fn[c]):
            if r["clip"] in seen:
                continue
            seen.add(r["clip"])
            pick.append(r)
            if len(pick) >= a.per:
                break
        out[c] = pick
        print(c, len(lst), "->", [(p["file"], p["obj"], round(p["f1"], 2), p["min_mm"]) for p in pick], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


def coco_images(root, min_side, splits=("val2017",)):
    """COCO images with an open licence (CC BY 4, CC BY-SA 5, no known restriction 7, US Gov 8) and a valid hand box of at
    least min_side px (COCO-WholeBody annotations of each split: <root>/coco_wholebody_<train|val>_v1.0.json)."""
    out = {}
    for split in splits:
        d = json.load(open(os.path.join(root, f"coco_wholebody_{split[:-4]}_v1.0.json")))
        ok = {im["id"]: dict(im, split=split) for im in d["images"] if im["license"] in (4, 5, 7, 8)}
        lic = {l["id"]: l["name"] for l in d["licenses"]}
        coco_hands(d, ok, lic, min_side, out)
    return out


def coco_hands(d, ok, lic, min_side, out):
    for an in d["annotations"]:
        im = ok.get(an["image_id"])
        if im is None:
            continue
        for k in ("lefthand", "righthand"):
            b = an[k + "_box"]
            if an[k + "_valid"] and max(b[2], b[3]) >= min_side:
                out.setdefault(im["id"], dict(file=os.path.join(im["split"], im["file_name"]), license=lic[im["license"]], license_id=im["license"],
                                              flickr_url=im["flickr_url"], coco_url=im["coco_url"], hands=[]))["hands"].append(
                    dict(box=[b[0], b[1], b[0] + b[2], b[1] + b[3]], is_right=k == "righthand"))


def intrinsics(img):
    f = max(img.shape[:2])
    return np.array([[f, 0, img.shape[1] / 2], [0, f, img.shape[0] / 2], [0, 0, 1]], np.float32)


def box_iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / max(u, 1e-9)


def select_coco(a):
    """Run FAH on the open-licence candidates; per hand (matched to a COCO-WholeBody hand box): contact score and the number of
    vertices >= 0.5 -> candidates sorted by contact."""
    m = method(a)
    cands = coco_images(a.coco_root, a.min_side, tuple(a.coco_splits.split(",")))
    rows = []
    for n, (iid, im) in enumerate(sorted(cands.items())):
        img = cv2.imread(os.path.join(a.coco_root, im["file"]))
        if img is None:
            continue
        m.reset()
        preds = [p for p in m(img, intrinsics(img)) if "contact" in p]
        for h in im["hands"]:
            best = max(preds, key=lambda p: box_iou(p["bbox"], h["box"]), default=None)
            if best is None or box_iou(best["bbox"], h["box"]) < 0.3:
                continue
            rows.append(dict(image_id=iid, file=im["file"], license=im["license"], flickr_url=im["flickr_url"], box=[float(x) for x in best["bbox"]],
                             is_right=bool(best["is_right"]), score=float(best["contact_score"]), ncontact=int((np.asarray(best["contact"]) >= 0.5).sum())))
        if n % 100 == 0:
            print(n, len(cands), len(rows), flush=True)
    rows.sort(key=lambda r: -(r["score"] * r["ncontact"]))
    json.dump(rows, open(a.out, "w"), indent=1)
    print(len(rows), "hands;", sum(r["score"] > 0.5 for r in rows), "flagged contact", flush=True)


def crop_K(K, cx, cy, L, P):
    s = P / L
    M = np.float32([[s, 0, P / 2 - s * cx], [0, s, P / 2 - s * cy]])
    Kz = np.array([[K[0, 0] * s, 0, K[0, 2] * s + M[0, 2]], [0, K[1, 1] * s, K[1, 2] * s + M[1, 2]], [0, 0, 1.0]])
    return M, Kz


def panel(img, K, hands, box, title, P, gt=None):
    """[image crop + meshes coloured by contact] [palm-side view of the first hand] (+ [GT contact on the GT meshes, same crop]).
    hands: [(verts, prob, faces, rest verts, palm vertex ids)]; gt: [(verts, binary labels, faces)]; box: crop around (x0, y0, x1, y1)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    L = 1.35 * max(box[2] - box[0], box[3] - box[1])
    M, Kz = crop_K(K, cx, cy, L, P)
    crop = cv2.warpAffine(img, M, (P, P), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(235, 235, 235))
    out = crop
    for v, prob, faces, _, _ in hands:
        out = render_contact_smooth(out, [(v, contact_colors_linear(prob))], faces, Kz)
    tiles = [header(out, title, color=(255, 255, 255), bg=(60, 60, 60), h=34)]
    v, prob, faces, rest, palm_idx = hands[0]
    x = (v - v[palm_idx].mean(0)) @ kabsch(v[palm_idx], rest[palm_idx]).T
    tiles.append(header(render_view(x, contact_colors_linear(prob), faces, best_view(x, prob, faces), np.ptp(rest, 0).max() * 1.2, P), "palm side",
                        color=(60, 60, 60), bg=(235, 235, 235), h=34))
    if gt is not None:
        out = crop
        for gv, glab, faces in gt:
            out = render_contact_smooth(out, [(gv, contact_colors_linear(glab.astype(np.float64)))], faces, Kz)
        tiles.append(header(out, "ground truth", color=(60, 60, 60), bg=(235, 235, 235), h=34))
    return np.hstack(tiles)


def union(boxes):
    b = np.asarray(boxes, np.float64)
    return [b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()]


def render(a):
    from bench.common import ManoLayer
    from bench.datasets.hot3d import Hot3DClip

    mano, _, rest = mano_bits()
    palm = {s: np.nonzero(mano.layers[s].lbs_weights.numpy().argmax(1) == 0)[0] for s in (True, False)}
    picks = json.load(open(a.picks))
    m = method(a)
    os.makedirs(a.out, exist_ok=True)
    sdf = np.load(os.path.join(TEST, "hot3d_bench", "sdf.npy")).astype(np.float32) if picks.get("hot3d") else None
    tiles, info = [], []
    rowidx = {}
    if picks.get("hot3d"):
        for i, l in enumerate(open(os.path.join(TEST, "hot3d_bench", "index.jsonl"))):
            q = json.loads(l)
            rowidx[(*q["src"].split(":"), bool(q["is_right"]))] = (i, q["box"])
    for r in picks.get("hot3d", []):
        clip = Hot3DClip(os.path.join(a.hot3d, r["clip"] + ".tar"), MANO_DIR, mano=ManoLayer(MANO_DIR, num_pca_comps=15))
        fr = clip.frame(clip.frame_keys.index(r["key"]))
        img, K = fr["image"], fr["K"]
        m.reset()
        preds = [p for p in m(img, K) if "contact" in p and p.get("verts3d") is not None]
        sides = [r["is_right"], not r["is_right"]] if r.get("both") else [r["is_right"]]
        hands, gts, boxes = [], [], []
        for sd in sides:
            row = rowidx.get((r["clip"], r["key"], sd))
            ps = [p for p in preds if bool(p["is_right"]) == sd]
            g = [h for h in fr["hands"] + fr["ignore"] if bool(h["is_right"]) == sd]
            if row is None or not ps or not g:
                continue
            p = max(ps, key=lambda q: box_iou(q["bbox"], row[1]))
            gh = max(g, key=lambda h: box_iou(h["bbox"], row[1]))
            hands.append((np.asarray(p["verts3d"], np.float64), np.asarray(p["contact"], np.float64), mano.faces[sd], rest[sd], palm[sd]))
            gts.append((np.asarray(gh["verts3d"], np.float64), sdf[row[0]] < 10, mano.faces[sd]))
            boxes += [p["bbox"], row[1]]
        if len(hands) != len(sides):
            print("skip (no matching prediction / GT)", r["file"], flush=True)
            continue
        title = r.get("title") or f"{r.get('label', '')} {r['obj']}".strip()
        im = panel(img, K, hands, union(boxes), title, a.panel, gt=gts)
        name = f"hot3d_{r['clip']}_{r['key']}_{'both' if r.get('both') else ('r' if r['is_right'] else 'l')}.png"
        cv2.imwrite(os.path.join(a.out, name), im)
        tiles.append(im)
        info.append(dict(name=name, **{k: r[k] for k in ("file", "obj", "min_mm", "n10", "f1") if k in r}, label=r.get("label", ""), title=title))
        print(name, flush=True)
    for r in picks.get("coco", []):
        img = cv2.imread(os.path.join(a.coco_root, r["file"]))
        K = intrinsics(img)
        m.reset()
        preds = [p for p in m(img, K) if "contact" in p and p.get("verts3d") is not None]
        p = max(preds, key=lambda q: box_iou(q["bbox"], r["box"]), default=None)
        if p is None or box_iou(p["bbox"], r["box"]) < 0.3:
            print("skip (no matching prediction)", r["file"], flush=True)
            continue
        s_ = bool(p["is_right"])
        im = panel(img, K, [(np.asarray(p["verts3d"], np.float64), np.asarray(p["contact"], np.float64), mano.faces[s_], rest[s_], palm[s_])],
                   p["bbox"], r.get("title") or r.get("label", ""), a.panel)
        name = f"coco_{os.path.splitext(os.path.basename(r['file']))[0]}_{'r' if s_ else 'l'}.png"
        cv2.imwrite(os.path.join(a.out, name), im)
        tiles.append(np.hstack([im, np.full((im.shape[0], a.panel, 3), 255, np.uint8)]) if picks.get("hot3d") else im)
        info.append(dict(name=name, file=r["file"], license=r.get("license"), flickr_url=r.get("flickr_url"), label=r.get("label", ""),
                         contact_score=float(p["contact_score"])))
        print(name, flush=True)
    if tiles:
        w = max(t_.shape[1] for t_ in tiles)
        cv2.imwrite(os.path.join(a.out, "sheet.jpg"), np.vstack([np.hstack([t_, np.full((t_.shape[0], w - t_.shape[1], 3), 255, np.uint8)]) for t_ in tiles]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    json.dump(info, open(os.path.join(a.out, "panels.json"), "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["select_hot3d", "select_coco", "render"])
    p.add_argument("--test_root", default=TEST, help="held-out contact sets (hot3d_bench, hot3d_bench_frames) with their predictions")
    p.add_argument("--pred", default="fah", help="select_hot3d: test prediction tag (bench.contact_eval predict)")
    p.add_argument("--thr", type=float, default=0.5, help="select_hot3d: its vertex threshold (0.5 for a calibrated head)")
    p.add_argument("--per", type=int, default=6)
    p.add_argument("--min_side", type=float, default=150, help="hand box side (px) for candidates")
    p.add_argument("--coco_root", default=os.path.join(DATA_ROOT, "coco"), help="COCO images (<split>/<file>) + COCO-WholeBody annotations")
    p.add_argument("--coco_splits", default="val2017,train2017")
    p.add_argument("--hot3d", default=os.path.join(DATA_ROOT, "hot3d", "train_aria"), help="render: the HOT3D clip tars")
    p.add_argument("--model", default="fah", help="fah-s | fah-b | fah (FAH-L) | fah-h")
    p.add_argument("--weights", default="", help="the release weights (default: fah.api's, $FAH_WEIGHTS)")
    p.add_argument("--trt_dir", default="")
    p.add_argument("--picks", default="")
    p.add_argument("--panel", type=int, default=400, help="panel side (px; the paper: 400)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    TEST = a.test_root
    dict(select_hot3d=select_hot3d, select_coco=select_coco, render=render)[a.cmd](a)
