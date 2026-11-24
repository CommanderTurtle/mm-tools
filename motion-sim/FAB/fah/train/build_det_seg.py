"""YOLO-seg version of the YOLO-pose hand detection dataset (train.build_det_visor_boxes output; 2 classes 0 = left /
1 = right kept): SAM 2.1 pseudo instance masks per labelled hand, prompted with its box + up to 8 visible GT
keypoints (positives) + the other hands' keypoints inside the box (negatives), clipped to the box (+3%).

Ultralytics derives segment boxes from the polygons, so the detector learns visible-hand boxes; masks failing QC
(empty / tiny vs the box / visible keypoints outside) are replaced by the ellipse inscribed in the GT box (keeps the
GT box). Low-light copies (train.build_det_dark --src <dst>) reuse the labels of their originals. Images missing from
the SAM results (shards not finished) get no image link.

  1. SAM (resumable; jsonl per shard):
     python -m train.build_det_seg sam --src $DATA_ROOT/det_boxes --sam_dir $DATA_ROOT/det_sam2 --shard 0 --num_shards 4
  2. labels + QC (summary json, review sheets; --kp_join 0.04: masks unioned with the visible keypoints' bones so the
     box covers every labelled joint):
     python -m train.build_det_seg write --src $DATA_ROOT/det_boxes --sam_dir $DATA_ROOT/det_sam2 --dst $DATA_ROOT/det_seg
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

# keypoints used as point prompts (wrist, MCPs, finger tips), OpenPose order (train.sam_masks)
PROMPT_JOINTS = [0, 5, 9, 13, 17, 4, 8, 12, 16, 20]


def read_labels(path, W, H):
    """YOLO-pose label file -> list of (cls, xyxy pixels, kp (21, 3) pixels / visibility)."""
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        v = np.asarray(line.split(), np.float64)
        if len(v) < 5:
            continue
        cx, cy, bw, bh = v[1:5] * [W, H, W, H]
        kp = v[5:].reshape(-1, 3) if len(v) > 5 else np.zeros((21, 3))
        kp = kp * [W, H, 1]
        out.append((int(v[0]), np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]), kp))
    return out


def prompts(hands, max_pos=8, max_neg=4):
    """Per hand: box + padded points (label -1 = no point) for SAM2ImagePredictor.predict_batch."""
    pos = [[k[j, :2] for j in PROMPT_JOINTS if k[j, 2] > 0][:max_pos] for _, _, k in hands]
    pts = []
    for i, (_, b, _) in enumerate(hands):
        neg = [p for j, ps in enumerate(pos) if j != i for p in ps
               if b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]][:max_neg]
        pts.append([(p, 1) for p in pos[i]] + [(p, 0) for p in neg])
    P = max(1, max(len(p) for p in pts))
    xy = np.zeros((len(hands), P, 2), np.float32)
    lab = -np.ones((len(hands), P), np.int32)
    for i, p in enumerate(pts):
        for k, (q, l) in enumerate(p):
            xy[i, k], lab[i, k] = q, l
    return np.stack([b for _, b, _ in hands]).astype(np.float32), xy, lab


def mask_to_polygon(m, min_frac=0.01):
    """Binary mask -> one polygon (N, 2) int pixels: external contours of the components (>= 1% of the mask),
    simplified, joined by zero-width bridges (ultralytics merge_multi_segment) when there are several."""
    from ultralytics.data.converter import merge_multi_segment

    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in cnts]
    tot = sum(areas)
    segs = [cv2.approxPolyDP(c, 0.8, True).reshape(-1, 2) for c, a in zip(cnts, areas)
            if len(c) >= 3 and a >= max(2.0, min_frac * tot)]
    if not segs:
        return None
    if len(segs) == 1:
        return segs[0]
    return np.round(np.concatenate(merge_multi_segment([s.reshape(-1).tolist() for s in segs]), 0)).astype(int)


def qc(m, box, kp):
    """kp_in: visible GT keypoints inside the 3 px dilated mask (>= 3 visible); fill: mask area / box area;
    box_iou: mask bounding box vs GT box."""
    x1, y1, x2, y2 = box
    barea = max((x2 - x1) * (y2 - y1), 1.0)
    vis = kp[:, 2] > 0
    kp_in = None
    if vis.sum() >= 3:
        md = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        xy = np.clip(kp[vis, :2].round().astype(int), 0, [m.shape[1] - 1, m.shape[0] - 1])
        kp_in = float(md[xy[:, 1], xy[:, 0]].mean())
    ys, xs = np.nonzero(m)
    box_iou = 0.0
    if len(xs):
        mx = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        iw = max(0.0, min(x2, mx[2]) - max(x1, mx[0]))
        ih = max(0.0, min(y2, mx[3]) - max(y1, mx[1]))
        inter = iw * ih
        box_iou = float(inter / (barea + (mx[2] - mx[0]) * (mx[3] - mx[1]) - inter))
    return dict(kp_in=kp_in, fill=float(len(xs) / barea), box_iou=box_iou)


def sam_stage(args):
    import torch
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from torch.utils.data import DataLoader, Dataset

    torch.backends.cuda.matmul.allow_tf32 = True
    items = sorted(f"{s}/{f}" for s in ("train", "val") for f in os.listdir(os.path.join(args.src, "images", s)))
    items = items[args.shard :: args.num_shards]
    os.makedirs(args.sam_dir, exist_ok=True)
    out_path = os.path.join(args.sam_dir, f"sam.{args.shard}.jsonl")
    done = set()
    for f in glob.glob(os.path.join(args.sam_dir, "sam.*.jsonl")):  # all shards: a rerun may shard differently
        for line in open(f):
            try:
                done.add(json.loads(line)["img"])
            except json.JSONDecodeError:  # torn last line after a preemption
                pass
    items = [i for i in items if i not in done]
    print(f"shard {args.shard}: {len(items)} images to label ({len(done)} done)", flush=True)

    class Images(Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            split, f = items[i].split("/")
            img = cv2.imread(os.path.join(args.src, "images", split, f))
            H, W = img.shape[:2]
            hands = read_labels(os.path.join(args.src, "labels", split, os.path.splitext(f)[0] + ".txt"), W, H)
            return items[i], img[:, :, ::-1].copy(), hands

    predictor = SAM2ImagePredictor(build_sam2_hf(args.model, device="cuda"))
    loader = DataLoader(Images(), batch_size=args.batch, num_workers=args.workers, collate_fn=list)
    fout = open(out_path, "a")
    for bi, chunk in enumerate(loader):
        chunk = [c for c in chunk if c[2]]
        if not chunk:
            continue
        P = [prompts(c[2]) for c in chunk]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image_batch([c[1] for c in chunk])
            masks, scores, _ = predictor.predict_batch(point_coords_batch=[p[1] for p in P],
                                                        point_labels_batch=[p[2] for p in P],
                                                        box_batch=[p[0] for p in P], multimask_output=False)
        for (name, img, hands), ms, sc in zip(chunk, masks, scores):
            H, W = img.shape[:2]
            ms = np.asarray(ms).reshape(len(hands), -1, H, W)[:, 0] > 0
            sc = np.asarray(sc).reshape(len(hands), -1)[:, 0]
            rec = []
            for (_, box, kp), m, s in zip(hands, ms, sc):
                x1, y1, x2, y2 = box
                e = 0.03 * max(x2 - x1, y2 - y1) + 1
                c = np.zeros_like(m)
                c[max(int(y1 - e), 0):int(np.ceil(y2 + e)), max(int(x1 - e), 0):int(np.ceil(x2 + e))] = True
                out_frac = float((m & ~c).sum() / max(m.sum(), 1))
                m &= c
                poly = mask_to_polygon(m)
                rec.append(dict(poly=None if poly is None else poly.tolist(), score=float(s), out_frac=out_frac, **qc(m, box, kp)))
            fout.write(json.dumps(dict(img=name, hw=[H, W], hands=rec)) + "\n")
        if bi % 20 == 0:
            fout.flush()
            print(f"{(bi + 1) * args.batch}/{len(items)}", flush=True)
    fout.close()


def ellipse(box, n=24):
    x1, y1, x2, y2 = box
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([(x1 + x2) / 2 + (x2 - x1) / 2 * np.cos(t), (y1 + y2) / 2 + (y2 - y1) / 2 * np.sin(t)], 1)


HAND_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]


def join_keypoints(poly, kp, box, frac):
    """Polygon unioned with the visible GT keypoints' bones (width frac x box side) and joint discs, so that its
    bounding box -- the YOLO-seg box -- covers every labelled joint (SAM often drops thin extended fingers). Joints more
    than 0.25 box sides outside the GT box are ignored; no join when they are the majority."""
    side = max(box[2] - box[0], box[3] - box[1])
    near = ((kp[:, :2] >= np.asarray(box[:2]) - 0.25 * side) & (kp[:, :2] <= np.asarray(box[2:]) + 0.25 * side)).all(1)
    vis = (kp[:, 2] > 0) & near  # joints far outside the box: label noise (another hand / misaligned), not fingers
    if not vis.any() or vis.sum() < 0.5 * (kp[:, 2] > 0).sum():
        return poly
    t = max(2, int(round(frac * side)))
    pts = np.concatenate([poly, kp[vis, :2]])
    o = np.floor(pts.min(0)).astype(int) - t - 2
    size = np.ceil(pts.max(0)).astype(int) - o + t + 3
    m = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillPoly(m, [np.round(poly - o).astype(np.int32)], 1)
    q = np.round(kp[:, :2] - o).astype(int)
    for c, pa in enumerate(HAND_PARENTS):
        if pa >= 0 and vis[c] and vis[pa]:
            cv2.line(m, tuple(q[pa]), tuple(q[c]), 1, t)
    for c in np.nonzero(vis)[0]:
        cv2.circle(m, tuple(q[c]), max(1, t // 2), 1, -1)
    out = mask_to_polygon(m, min_frac=0.0)
    return poly if out is None else out + o


def bad_mask(r, args):
    return (r["poly"] is None or r["fill"] < args.min_fill or (r["kp_in"] is not None and r["kp_in"] < args.min_kp_in)
            or r["box_iou"] < args.min_box_iou)


def write_stage(args):
    recs = {}
    for f in sorted(glob.glob(os.path.join(args.sam_dir, "sam.*.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            recs[r["img"]] = r
    stats = dict(images=0, hands=0, fallback=0, missing_images=0)
    rows = []
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            os.makedirs(os.path.join(args.dst, kind, split), exist_ok=True)
        for f in sorted(os.listdir(os.path.join(args.src, "images", split))):
            stem = os.path.splitext(f)[0]
            r = recs.get(f"{split}/{f}")
            if r is None:
                stats["missing_images"] += 1
                continue
            # per-file image links: ultralytics resolves the split directory, a linked directory would pair the
            # images with the source (pose) labels
            link = os.path.join(args.dst, "images", split, f)
            if not os.path.lexists(link):
                os.symlink(os.path.abspath(os.path.join(args.src, "images", split, f)), link)
            H, W = r["hw"]
            hands = read_labels(os.path.join(args.src, "labels", split, stem + ".txt"), W, H)
            assert len(hands) == len(r["hands"]), f
            lines = []
            for (cls, box, kp), h in zip(hands, r["hands"]):
                bad = bad_mask(h, args)
                poly = ellipse(box) if bad else np.asarray(h["poly"], np.float64)
                if args.kp_join > 0:
                    poly = join_keypoints(poly, kp, box, args.kp_join)
                vis = kp[:, 2] > 0  # visible GT joints inside the polygon's box (= the YOLO-seg training box)
                lo, hi = poly.min(0) - 1, poly.max(0) + 1
                kp_box = float(((kp[vis, :2] >= lo) & (kp[vis, :2] <= hi)).all(1).mean()) if vis.sum() >= 3 else None
                poly = np.clip(poly / [W, H], 0, 1)
                lines.append(" ".join([str(cls)] + [f"{v:.5f}" for v in poly.ravel()]))
                stats["fallback"] += bad
                rows.append(dict(img=f"{split}/{f}", src=f.split("_")[0], bad=bool(bad), kp_box=kp_box,
                                 **{k: h[k] for k in ("score", "kp_in", "fill", "box_iou", "out_frac")}))
            stats["images"] += 1
            stats["hands"] += len(lines)
            open(os.path.join(args.dst, "labels", split, stem + ".txt"), "w").write("\n".join(lines) + "\n")
    names = "names:\n  0: left\n  1: right\n"
    open(os.path.join(args.dst, "hands.yaml"), "w").write(f"path: {args.dst}\ntrain: images/train\nval: images/val\n{names}")
    summ = dict(stats, thresholds=dict(min_fill=args.min_fill, min_kp_in=args.min_kp_in, min_box_iou=args.min_box_iou))
    for src in sorted({r["src"] for r in rows}) + ["all"]:
        sel = [r for r in rows if src in ("all", r["src"])]
        d = dict(n=len(sel), fallback=float(np.mean([r["bad"] for r in sel])))
        for k in ("score", "kp_in", "fill", "box_iou", "out_frac", "kp_box"):
            v = np.array([r[k] for r in sel if r[k] is not None])
            if len(v):
                d[k] = dict(median=float(np.median(v)), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)))
        summ[src] = d
    json.dump(summ, open(os.path.join(args.dst, "qc_summary.json"), "w"), indent=1)
    with open(os.path.join(args.dst, "qc.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summ, indent=1))
    if args.sheets:
        review_sheets(args, recs, rows)


def review_sheets(args, recs, rows, n=16):
    """Overlay sheets (4 x 4 tiles around one hand each): random hands per source, lowest kp_in, fallbacks."""
    os.makedirs(args.sheets, exist_ok=True)
    rng = np.random.default_rng(0)
    by_img = {}
    for r in rows:
        by_img.setdefault(r["img"], []).append(r)
    groups = {f"random_{s}": [r for r in rows if r["src"] == s] for s in sorted({r["src"] for r in rows})}
    groups["low_kp_in"] = sorted([r for r in rows if r["kp_in"] is not None], key=lambda r: r["kp_in"])[:200]
    groups["fallback"] = [r for r in rows if r["bad"]]
    colors = [(40, 120, 255), (255, 160, 40)]  # left / right (BGR)
    for name, sel in groups.items():
        if not sel:
            continue
        pick = [sel[i] for i in rng.choice(len(sel), min(n, len(sel)), replace=False)]
        tiles = []
        for p in pick:
            split, f = p["img"].split("/")
            img = cv2.imread(os.path.join(args.src, "images", split, f))
            H, W = img.shape[:2]
            hands = read_labels(os.path.join(args.src, "labels", split, os.path.splitext(f)[0] + ".txt"), W, H)
            lab = open(os.path.join(args.dst, "labels", split, os.path.splitext(f)[0] + ".txt")).read().split("\n")
            over = img.copy()
            for (cls, box, kp), line in zip(hands, lab):
                poly = (np.asarray(line.split()[1:], np.float64).reshape(-1, 2) * [W, H]).round().astype(np.int32)
                cv2.fillPoly(over, [poly], colors[cls])
            img = cv2.addWeighted(img, 0.55, over, 0.45, 0)
            hi = by_img[p["img"]].index(p)
            x1, y1, x2, y2 = hands[hi][1]
            s = int(max(x2 - x1, y2 - y1) * 1.6 + 16)
            t = max(1, int(round(s / 256)))  # line width / dot radius ~1 px in the 256 tile
            for (cls, box, kp), line in zip(hands, lab):
                cv2.rectangle(img, tuple(int(v) for v in box[:2]), tuple(int(v) for v in box[2:]), (0, 255, 0), t)
                for x, y, v in kp:
                    if v > 0:
                        cv2.circle(img, (int(x), int(y)), t, (0, 255, 255), -1)
            c0, r0 = int((x1 + x2 - s) / 2), int((y1 + y2 - s) / 2)
            img = cv2.copyMakeBorder(img, s, s, s, s, cv2.BORDER_CONSTANT)  # square crops at the image border
            tile = cv2.resize(img[r0 + s:r0 + 2 * s, c0 + s:c0 + 2 * s], (256, 256))
            kin = "-" if p["kp_in"] is None else f"{p['kp_in']:.2f}"
            cv2.putText(tile, f"{p['src']} kp {kin} s {p['score']:.2f}{' FB' if p['bad'] else ''}", (4, 16), 0, 0.45,
                        (0, 255, 255), 1)
            tiles.append(tile)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
        cv2.imwrite(os.path.join(args.sheets, f"masks_{name}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("sheets ->", args.sheets)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["sam", "write"])
    p.add_argument("--src", required=True, help="YOLO-pose dataset (images/, labels/)")
    p.add_argument("--sam_dir", required=True, help="SAM results (sam.<shard>.jsonl)")
    p.add_argument("--dst", default="", help="write: YOLO-seg dataset")
    p.add_argument("--model", default="facebook/sam2.1-hiera-large")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--min_fill", type=float, default=0.08, help="fallback if mask area / box area is below")
    p.add_argument("--min_kp_in", type=float, default=0.5, help="fallback if fewer visible keypoints are inside")
    p.add_argument("--min_box_iou", type=float, default=0.2, help="fallback if mask box vs GT box IoU is below")
    p.add_argument("--kp_join", type=float, default=0.0,
                   help="write: union masks with the visible keypoints' bones of this width (x box side), 0 = off")
    p.add_argument("--sheets", default="", help="write: review sheets directory")
    a = p.parse_args()
    sam_stage(a) if a.stage == "sam" else write_stage(a)
