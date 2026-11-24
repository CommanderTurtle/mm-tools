"""Full-frame hand detection dataset (YOLO pose format, 2 classes: 0 = left, 1 = right, 21 keypoints) from
our training sources, for retraining the hand detector (same output format as WiLoR's yolov8m-pose).

Sources (none overlap the benchmarks): COCO-WholeBody train2017 (annotated persons' hands), VISOR train
(exhaustive hand masks -> boxes, no keypoints), HOT3D-Clips train (non-benchmark sequences; rectified +
upright like the benchmark; exhaustive), HInt train (New Days + EPIC), CMU hand_labels manual_train.
Images are re-encoded with long side <= --max_side. 2% of images (by name hash) go to val.

    python -m train.build_det_dataset --out $DATA_ROOT/det --coco $DATA_ROOT/coco/train2017 \
        --coco_ann $DATA_ROOT/coco/coco_wholebody_train_v1.0.json --visor .../GroundTruth-SparseAnnotations \
        --hot3d $DATA_ROOT/hot3d_train --hint $DATA_ROOT/hint/HInt_annotation_partial --cmu $DATA_ROOT/cmu_hand
"""
import argparse
import glob
import json
import os
import zipfile
import zlib
from multiprocessing import Pool

import cv2
import numpy as np

NK = 21


def _write(out, name, img, hands, max_side):
    """hands: list of (is_right, box xyxy, kp (21, 3) or None) in image pixels."""
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    split = "val" if zlib.crc32(name.encode()) % 50 == 0 else "train"
    lines = []
    for is_right, box, kp in hands:
        x1, y1, x2, y2 = np.clip(np.asarray(box, np.float64) * s, 0, [W, H, W, H])
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        k = np.zeros((NK, 3))
        if kp is not None:
            k = np.asarray(kp, np.float64).copy()
            k[:, :2] *= s
            inside = (k[:, 0] >= 0) & (k[:, 0] < W) & (k[:, 1] >= 0) & (k[:, 1] < H) & (k[:, 2] > 0)
            k[:, 2] = np.where(inside, 2, 0)
            k[~inside, :2] = 0
        k[:, 0] /= W
        k[:, 1] /= H
        vals = [int(bool(is_right)), (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H, *k.ravel()]
        lines.append(" ".join(f"{v:.6g}" for v in vals))
    if not lines:
        return 0
    os.makedirs(os.path.join(out, "images", split), exist_ok=True)
    os.makedirs(os.path.join(out, "labels", split), exist_ok=True)
    cv2.imwrite(os.path.join(out, "images", split, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    open(os.path.join(out, "labels", split, name + ".txt"), "w").write("\n".join(lines) + "\n")
    return len(lines)


# ---------------------------------------------------------------------------- sources
def coco_items(root, ann_file):
    ann = json.load(open(ann_file))
    imgs = {im["id"]: im for im in ann["images"]}
    by = {}
    for a in ann["annotations"]:
        for side, is_r in (("lefthand", False), ("righthand", True)):
            if a.get(f"{side}_valid"):
                x, y, bw, bh = a[f"{side}_box"]
                kp = np.asarray(a[f"{side}_kpts"], np.float64).reshape(NK, 3)
                kp[:, 2] = kp[:, 2] > 0
                by.setdefault(a["image_id"], []).append((is_r, [x, y, x + bw, y + bh], kp))
    for iid, hands in by.items():
        yield ("coco", os.path.join(root, imgs[iid]["file_name"]), f"coco_{iid}", hands)


def _coco_work(arg):
    _, path, name, hands = arg
    img = cv2.imread(path)
    return 0 if img is None else _write(ARGS.out, name, img, hands, ARGS.max_side)


def _visor_work(jf):
    from bench.seg.build import HAND, polygons_to_mask

    video = os.path.splitext(os.path.basename(jf))[0]
    zpath = os.path.join(ARGS.visor, "rgb_frames", "train", video.split("_")[0], video + ".zip")
    if not os.path.exists(zpath):
        return 0
    zf = zipfile.ZipFile(zpath)
    names = {os.path.basename(n): n for n in zf.namelist()}
    n = 0
    for k, fr in enumerate(json.load(open(jf))["video_annotations"]):
        if k % ARGS.visor_stride or fr["image"]["name"] not in names:
            continue
        img = cv2.imdecode(np.frombuffer(zf.read(names[fr["image"]["name"]]), np.uint8), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        hands = []
        for a in fr["annotations"]:
            if HAND.match(a["name"].strip().lower()):
                m = polygons_to_mask(a["segments"], (H, W), (W / 1920, H / 1080))
                if m.sum() >= 100:
                    ys, xs = np.nonzero(m)
                    hands.append((a["name"].lower().startswith("right"), [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], None))
        n += _write(ARGS.out, f"visor_{fr['image']['name'][:-4]}", img, hands, ARGS.max_side)  # frames w/o hands skipped
    return n


def _hot3d_work(tar):
    from bench.datasets.hot3d import Hot3DClip

    try:
        clip = Hot3DClip(tar, ARGS.mano_dir, min_visible=0.2)
    except Exception:
        return 0
    n = 0
    name = os.path.basename(tar)[:-4]
    for i in range(0, len(clip), ARGS.hot3d_stride):
        fr = clip.frame(i)
        hands = [(h["is_right"], h["bbox"], np.concatenate([h["joints2d"], np.ones((NK, 1))], 1)) for h in fr["hands"] + fr["ignore"]]
        n += _write(ARGS.out, f"hot3d_{name}_{i:04d}", fr["image"], hands, ARGS.max_side)
    return n


def _hint_work(jf):
    stem = os.path.splitext(os.path.basename(jf))[0]
    img = cv2.imread(jf[:-5] + ".jpg")
    if img is None:
        return 0
    hands = []
    for a in json.load(open(jf)):
        kp = np.concatenate([np.asarray(a["keypoints"], np.float64), (np.asarray(a["existence"]) > 0)[:, None]], 1)
        b = a["bbox"][0] if a.get("bbox") else [*kp[kp[:, 2] > 0, :2].min(0), *kp[kp[:, 2] > 0, :2].max(0)]
        hands.append((stem.endswith("_r"), b, kp))
    return _write(ARGS.out, f"hint_{stem}", img, hands, ARGS.max_side)


def _cmu_work(jf):
    a = json.load(open(jf))
    pts = np.asarray(a["hand_pts"], np.float64)
    v = pts[pts[:, 2] > 0, :2]
    img = cv2.imread(jf[:-5] + ".jpg")
    if img is None or len(v) < 8:
        return 0
    return _write(ARGS.out, f"cmu_{os.path.basename(jf)[:-5]}", img, [(not a["is_left"], [*v.min(0), *v.max(0)], pts)], ARGS.max_side)


def main(args):
    global ARGS
    ARGS = args
    total = {}
    with Pool(args.workers) as pool:
        if args.coco:
            total["coco"] = sum(pool.imap_unordered(_coco_work, coco_items(args.coco, args.coco_ann), chunksize=64))
            print("coco", total["coco"], flush=True)
        if args.visor:
            total["visor"] = sum(pool.imap_unordered(_visor_work, sorted(glob.glob(os.path.join(args.visor, "annotations", "train", "*.json")))))
            print("visor", total["visor"], flush=True)
        if args.hot3d:
            split = json.load(open("bench/splits/hot3d_bench_full.json"))
            val_seq = set(split["hawor_val_sequences"])
            from train.build_hot3d_crops import train_clips  # noqa: F401  (documents the selection)
            tars = sorted(glob.glob(os.path.join(args.hot3d, "train_aria", "*.tar")))
            total["hot3d"] = sum(pool.imap_unordered(_hot3d_work, tars))
            print("hot3d", total["hot3d"], "(benchmark sequences excluded at download:", len(val_seq), ")", flush=True)
        if args.hint:
            jfs = sorted(glob.glob(os.path.join(args.hint, "TRAIN_*_img", "*.json")))
            total["hint"] = sum(pool.imap_unordered(_hint_work, jfs, chunksize=32))
            print("hint", total["hint"], flush=True)
        if args.cmu:
            jfs = sorted(glob.glob(os.path.join(args.cmu, "hand_labels", "manual_train", "*.json")))
            total["cmu"] = sum(pool.imap_unordered(_cmu_work, jfs, chunksize=32))
            print("cmu", total["cmu"], flush=True)
    yaml = os.path.join(args.out, "hands.yaml")
    open(yaml, "w").write(f"path: {args.out}\ntrain: images/train\nval: images/val\nkpt_shape: [21, 3]\n"
                          "flip_idx: [" + ", ".join(str(i) for i in range(NK)) + "]\nnames:\n  0: left\n  1: right\n")
    print("hand boxes per source:", total, "->", yaml)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--coco", default="")
    p.add_argument("--coco_ann", default="")
    p.add_argument("--visor", default="")
    p.add_argument("--visor_stride", type=int, default=1)
    p.add_argument("--hot3d", default="")
    p.add_argument("--hot3d_stride", type=int, default=15)
    p.add_argument("--hint", default="")
    p.add_argument("--cmu", default="")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--max_side", type=int, default=1024)
    p.add_argument("--workers", type=int, default=10)
    main(p.parse_args())
