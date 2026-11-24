"""Build a pool of right-hand-oriented hand crops for distillation (labels not required).

Each crop is CONTEXT x the teacher's standard crop region (GetBBoxCenterScale(0.9) + TopdownAffine
framing used by HandEstimator), rendered at SIZE px; left hands are mirrored (as the model expects).
Training samples random affine views inside it, so augmentation never shows black borders.

    python -m train.build_crops --source freihand --root $DATA_ROOT/freihand_train --out $DATA_ROOT/crops
Sources: freihand | coco_wb | cmu | hot3d | egodex | images (any folder; WiLoR detector boxes)
Output: <out>/<source>/<shard>/<idx>.jpg + <out>/<source>/index.jsonl
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

SIZE = 640
CONTEXT = 1.25
PADDING = 0.9  # GetBBoxCenterScale padding used by the hand estimator


def teacher_region(box):
    """Center + square side of the teacher's crop for an xyxy box (mirrors bbox_xyxy2cs(0.9) +
    fix_aspect_ratio(0.75) + fix_aspect_ratio(1.0) in TopdownAffine)."""
    x1, y1, x2, y2 = box
    c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    w, h = (x2 - x1) * PADDING, (y2 - y1) * PADDING
    if w > 0.75 * h:
        h = w / 0.75
    else:
        w = h * 0.75
    return c, max(w, h)


def render_crop(img, box, is_right, size=SIZE, context=CONTEXT):
    h, w = img.shape[:2]
    if not is_right:  # mirror into a right hand
        img = img[:, ::-1]
        box = np.array([w - box[2] - 1, box[1], w - box[0] - 1, box[3]], dtype=np.float32)
    c, side = teacher_region(box)
    side *= context
    s = size / side
    M = np.array([[s, 0, size / 2 - s * c[0]], [0, s, size / 2 - s * c[1]]], dtype=np.float32)
    crop = cv2.warpAffine(np.ascontiguousarray(img), M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return crop, M


def crop_keypoints(kp, M, is_right, width):
    """(21, 3) image keypoints [u, v, flag] -> crop pixels (mirrored like the image for left hands)."""
    kp = np.asarray(kp, np.float64).copy()
    if not is_right:
        kp[:, 0] = width - 1 - kp[:, 0]
    kp[:, :2] = kp[:, :2] @ np.asarray(M, np.float64)[:, :2].T + np.asarray(M, np.float64)[:, 2]
    return kp


def crop_box(record):
    """Original-frame xyxy box in a stored, right-hand-oriented crop's pixels.

    Left-hand crops are already mirrored; their affine M maps mirrored original
    pixels, while the record's box stays in the original frame.
    """
    x0, y0, x1, y1 = np.asarray(record["box"], np.float64)
    if not record["is_right"]:
        width = record["hw"][1]
        x0, x1 = width - 1 - x1, width - 1 - x0
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    M = np.asarray(record["M"], np.float64)
    corners = corners @ M[:, :2].T + M[:, 2]
    return np.r_[corners.min(0), corners.max(0)].astype(np.float32)


# ---------------------------------------------------------------------------- source adapters
def items_freihand(root, stride=1):
    K = json.load(open(os.path.join(root, "training_K.json")))
    xyz = json.load(open(os.path.join(root, "training_xyz.json")))
    files = sorted(glob.glob(os.path.join(root, "training", "rgb", "*.jpg")))
    for f in files[::stride]:
        i = int(os.path.basename(f)[:8]) % len(K)
        k, j = np.array(K[i]), np.array(xyz[i])
        uv = (j[:, :2] / j[:, 2:]) * k[[0, 1], [0, 1]] + k[[0, 1], [2, 2]]
        yield f, [dict(box=[*uv.min(0), *uv.max(0)], is_right=True, kp=np.concatenate([uv, np.ones((21, 1))], 1),
                       j3d=j, K=k)]


def items_coco_wb(root, ann_file, min_box=24, stride=1):
    ann = json.load(open(ann_file))
    imgs = {im["id"]: im for im in ann["images"]}
    by = {}
    for a in ann["annotations"]:
        for side, is_r in (("lefthand", False), ("righthand", True)):
            if not a.get(f"{side}_valid"):
                continue
            x, y, bw, bh = a[f"{side}_box"]
            if max(bw, bh) < min_box:
                continue
            kp = np.asarray(a[f"{side}_kpts"], np.float64).reshape(21, 3)
            kp[:, 2] = kp[:, 2] > 0  # WholeBody hand scores -> labelled flag
            by.setdefault(a["image_id"], []).append(dict(box=[x, y, x + bw, y + bh], is_right=is_r, kp=kp))
    for iid in sorted(by)[::stride]:
        yield os.path.join(root, imgs[iid]["file_name"]), by[iid]


def items_cmu(root, split="manual_train"):
    for jf in sorted(glob.glob(os.path.join(root, "hand_labels", split, "*.json"))):
        a = json.load(open(jf))
        pts = np.asarray(a["hand_pts"])
        v = pts[pts[:, 2] > 0, :2]
        if len(v) < 8:
            continue
        img = jf[:-5] + ".jpg"
        if os.path.exists(img):
            yield img, [dict(box=[*v.min(0), *v.max(0)], is_right=not a["is_left"], kp=pts)]


def items_hint(root, splits=("TRAIN",)):
    for split in splits:
        for src in ("newdays", "epick"):
            for jf in sorted(glob.glob(os.path.join(root, f"{split}_{src}_img", "*.json"))):
                img = jf[:-5] + ".jpg"
                stem = os.path.splitext(os.path.basename(jf))[0]
                hands = []
                for a in json.load(open(jf)):
                    kp = np.concatenate([np.asarray(a["keypoints"], np.float64), (np.asarray(a["existence"]) > 0)[:, None]], 1)
                    v = kp[kp[:, 2] > 0, :2]
                    if len(v) >= 8:
                        hands.append(dict(box=[*v.min(0), *v.max(0)], is_right=stem.endswith("_r"), kp=kp))
                if hands and os.path.exists(img):
                    yield img, hands


def main(args):
    if args.source == "hint":
        items = items_hint(args.root)
    elif args.source == "freihand":
        items = items_freihand(args.root, args.stride)
    elif args.source == "coco_wb":
        items = items_coco_wb(args.root, args.ann, stride=args.stride)
    elif args.source == "cmu":
        items = items_cmu(args.root)
    else:
        raise ValueError(f"source {args.source} is built by its own script (hot3d/egodex use the dataset loaders)")
    out = os.path.join(args.out, args.source)
    os.makedirs(out, exist_ok=True)
    jobs = [(n, p, hands) for n, (p, hands) in enumerate(items)]
    with Pool(args.workers) as pool:
        recs = [r for rs in pool.imap(_work, [(j, out) for j in jobs], chunksize=32) for r in rs]
    with open(os.path.join(out, "index.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"{args.source}: {len(recs)} crops from {len(jobs)} images -> {out}")


def _work(arg):
    (n, path, hands), out = arg
    img = cv2.imread(path)
    if img is None:
        return []
    recs = []
    for k, h in enumerate(hands):
        crop, M = render_crop(img, np.asarray(h["box"], np.float32), h["is_right"])
        rel = f"{n // 1000:04d}/{n:07d}_{k}.jpg"
        os.makedirs(os.path.join(out, os.path.dirname(rel)), exist_ok=True)
        cv2.imwrite(os.path.join(out, rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        rec = dict(file=rel, src=path, is_right=bool(h["is_right"]), box=[float(v) for v in h["box"]],
                   M=M.round(5).tolist(), hw=list(img.shape[:2]))
        if h.get("kp") is not None:  # GT 2D keypoints (OpenPose order) in right-hand crop pixels + labelled flag
            rec["kp2d"] = crop_keypoints(h["kp"], M, h["is_right"], img.shape[1]).round(2).tolist()
        if h.get("j3d") is not None:  # camera-frame 3D (m) + intrinsics (right hands only: no mirroring)
            assert h["is_right"]
            rec["joints3d"] = np.asarray(h["j3d"], np.float64).round(5).tolist()
            rec["K"] = np.asarray(h["K"], np.float64).round(4).tolist()
        recs.append(rec)
    return recs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--ann", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--workers", type=int, default=16)
    main(p.parse_args())
