"""Background (no-hand) crops for the presence head, from exhaustively labelled detection frames (VISOR, HOT3D
in train.build_det_dataset output). Half random windows, half "near-hand" windows (shifted 1-2 hand sizes
from a hand: arms, sleeves, held objects -- typical detector false positives), all with IoU < max_iou
to every hand box. Crops follow the train.build_crops convention (record field negative=True).

    python -m train.build_neg_crops --det $DATA_ROOT/det --out $DATA_ROOT/crops --per_image 2
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

from train.build_crops import render_crop


def iou(a, b):
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(x2 - x1, 0) * max(y2 - y1, 0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def _work(arg):
    img_path, out, per_image, max_iou, seed = arg
    rng = np.random.RandomState(seed)
    lab = img_path.replace("/images/", "/labels/")[:-4] + ".txt"
    img = cv2.imread(img_path)
    if img is None or not os.path.exists(lab):
        return []
    H, W = img.shape[:2]
    boxes = []
    for line in open(lab):
        v = [float(x) for x in line.split()[:5]]
        cx, cy, w, h = v[1] * W, v[2] * H, v[3] * W, v[4] * H
        boxes.append(np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
    recs = []
    name = os.path.basename(img_path)[:-4]
    for k in range(per_image * 10):
        if len(recs) >= per_image:
            break
        if k % 2 and boxes:  # near a hand
            b = boxes[rng.randint(len(boxes))]
            s = max(b[2] - b[0], b[3] - b[1]) * rng.uniform(0.8, 1.5)
            ang = rng.uniform(0, 2 * np.pi)
            c = (b[:2] + b[2:]) / 2 + rng.uniform(1.0, 2.0) * s * np.array([np.cos(ang), np.sin(ang)])
        else:
            s = rng.uniform(0.06, 0.35) * min(H, W)
            c = rng.uniform([s / 2, s / 2], [W - s / 2, H - s / 2])
        box = np.array([c[0] - s / 2, c[1] - s / 2, c[0] + s / 2, c[1] + s / 2])
        if box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H or any(iou(box, b) > max_iou for b in boxes):
            continue
        is_right = bool(rng.rand() < 0.5)
        crop, M = render_crop(img, box.astype(np.float32), is_right)
        rel = f"{name[:24]}/{name}_{len(recs)}.jpg"
        os.makedirs(os.path.join(out, os.path.dirname(rel)), exist_ok=True)
        cv2.imwrite(os.path.join(out, rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        recs.append(dict(file=rel, src=f"{name}:neg{len(recs)}", is_right=is_right, box=box.tolist(),
                         M=M.round(5).tolist(), hw=[H, W], negative=True))
    return recs


def main(args):
    imgs = sorted(p for p in glob.glob(os.path.join(args.det, "images", "train", "*.jpg"))
                  if os.path.basename(p).startswith(tuple(args.sources.split(","))))
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)
    with Pool(args.workers) as pool:
        recs = [r for rs in pool.imap_unordered(_work, [(p, out, args.per_image, args.max_iou, i) for i, p in enumerate(imgs)],
                                                 chunksize=32) for r in rs]
    with open(os.path.join(out, "index.jsonl"), "w") as f:
        for r in sorted(recs, key=lambda r: r["file"]):
            f.write(json.dumps(r) + "\n")
    print(f"{args.name}: {len(recs)} negative crops from {len(imgs)} frames -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--det", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="neg")
    p.add_argument("--sources", default="visor_,hot3d_", help="exhaustively labelled frame prefixes")
    p.add_argument("--per_image", type=int, default=2)
    p.add_argument("--max_iou", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=10)
    main(p.parse_args())
