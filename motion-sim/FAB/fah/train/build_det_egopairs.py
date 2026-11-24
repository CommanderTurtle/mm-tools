"""Egocentric overlapping two-hand frames with GT for both hands, from training sources only (bench/splits/crosspair_dev.json:
HOT3D Aria train sequences with crc32 % 8 == 0 held out of detector training; no benchmark sequence anywhere), kept when two GT
hand boxes overlap (IoU >= --min_iou):

  --mode train -> YOLO-pose frames (train.build_det_dataset layout: GT boxes, GT 2D joints, every visible / ignore hand
                  labelled as the detection set's HOT3D frames) for train.build_det_seg_sam3 (SAM 3 masks, --split_pairs): the split's
                  train_clips (hot3d_train600 minus the held-out sequences; every --hot3d_stride-th frame, the detection set's stride-15
                  frames skipped) + the OakInk2-ego training windows (oakink2_ego_train.json);
  --mode dev   -> a bench2d-format crossed-pair dev set: the split's dev_clips (held-out sequences) + the OakInk2-ego dev
                  windows (oakink2_ego_dev.json, sequences crc32 % 20 == 0): frames with exactly two hands (both >= 50 %
                  visible, no partial hand) whose GT boxes overlap; images/ + annotations.json.

    python -m train.build_det_egopairs --mode train --out $DATA_ROOT/det_ego
    python -m train.build_det_egopairs --mode dev --out $DATA_ROOT/crosspair_dev
"""
import argparse
import glob
import json
import os
import zlib
from multiprocessing import Pool

import cv2
import numpy as np

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
ARGS = SPLIT = None


def box_iou(a, b):
    w, h = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h, 1e-9)


def overlapping(hands):
    return any(box_iou(hands[i]["bbox"], hands[j]["bbox"]) >= ARGS.min_iou for i in range(len(hands)) for j in range(i + 1, len(hands)))


def _hot3d(tar):
    from bench.datasets.hot3d import Hot3DClip
    from train.build_det_dataset import _write

    try:
        clip = Hot3DClip(tar, ARGS.mano_dir, min_visible=0.5 if ARGS.mode == "dev" else 0.2)
    except Exception:
        return []
    name, out = os.path.basename(tar)[:-4], []
    for i in range(0, len(clip), ARGS.hot3d_stride):
        if i % 15 == 0 and ARGS.mode == "train":  # the detection set already has these frames
            continue
        fr = clip.frame(i)
        hs = fr["hands"] + fr["ignore"]
        if ARGS.mode == "dev" and (len(fr["hands"]) != 2 or fr["ignore"]):
            continue
        if len(hs) < 2 or not overlapping(hs):
            continue
        hands = [(h["is_right"], h["bbox"], np.concatenate([h["joints2d"], np.ones((21, 1))], 1)) for h in hs]
        n = f"hot3dx_{name}_{i:04d}"
        out.append(("hot3d", n, fr["image"], hands, fr["K"]) if ARGS.mode == "dev" else
                   ("hot3d", n, _write(ARGS.stage, n, fr["image"], hands, ARGS.max_side)))
    return out


def _oakink2(entry):
    from bench.datasets.oakink2 import OakInk2Clip

    try:
        clip = OakInk2Clip(os.path.join(DATA_ROOT, "oakink2_raw"), entry, ARGS.mano_dir,
                           min_visible=0.5 if ARGS.mode == "dev" else 0.2)
    except Exception:
        return []
    out = []
    for i in range(0, len(clip), ARGS.oakink2_stride):
        fr = clip.frame(i, image=False)
        hs = fr["hands"] + fr["ignore"]
        if ARGS.mode == "dev" and (len(fr["hands"]) != 2 or fr["ignore"]):
            continue
        if len(hs) < 2 or not overlapping(hs):
            continue
        fr = clip.frame(i)
        n = f"oak2x_{entry.replace('@', '_')}_{i:04d}"
        hands = [(h["is_right"], h["bbox"], np.concatenate([h["joints2d"], np.ones((21, 1))], 1)) for h in hs]
        if ARGS.mode == "dev":
            out.append(("oakink2", n, fr["image"], hands, fr["K"]))
        else:
            from train.build_det_dataset import _write

            out.append(("oakink2", n, _write(ARGS.stage, n, fr["image"], hands, ARGS.max_side)))
    return out


def _cap(items, n):
    return sorted(items, key=lambda x: zlib.crc32(x[1].encode()))[:n]


def main(args):
    global ARGS, SPLIT
    ARGS = args
    SPLIT = json.load(open(args.split))
    os.makedirs(args.out, exist_ok=True)
    stats = {}
    if args.mode == "train":
        args.stage = args.stage or args.out.rstrip("/") + "_all"
        bench = {os.path.basename(c) for c in json.load(open("bench/splits/hot3d_bench_full.json"))["clips"]["Aria"]}
        tars = [os.path.join(DATA_ROOT, "hot3d_train600", "train_aria", c + ".tar") for c in SPLIT["train_clips"]]
        tars = [t for t in tars if os.path.basename(t) not in bench]
        entries = json.load(open("bench/splits/oakink2_ego_train.json"))["clips"]
        dev = set(json.load(open("bench/splits/oakink2_ego_dev.json"))["clips"])
        entries = [e for e in entries if e not in dev and e.rsplit("@", 1)[0] not in {d.rsplit("@", 1)[0] for d in dev}]
        for src, fn, jobs, cap in (("hot3d", _hot3d, tars, args.max_hot3d), ("oakink2", _oakink2, entries, args.max_oakink2)):
            with Pool(args.workers) as p:  # every candidate written to the staging dir, the capped subset linked into --out
                items = [x for r in p.imap_unordered(fn, jobs) for x in r if x[2] > 0]
            stats[src] = dict(candidates=len(items))
            items = _cap(items, cap)
            for _, name, _ in items:
                for split in ("train", "val"):
                    for kind, ext in (("images", ".jpg"), ("labels", ".txt")):
                        f = os.path.join(args.stage, kind, split, name + ext)
                        if os.path.exists(f):
                            os.makedirs(os.path.join(args.out, kind, split), exist_ok=True)
                            t = os.path.join(args.out, kind, split, name + ext)
                            if not os.path.lexists(t):
                                os.symlink(f, t)
            stats[src].update(selected=len(items), hands=sum(n for *_, n in items))
            print(src, stats[src], flush=True)
        y = f"path: {args.out}\ntrain: images/train\nval: images/val\nkpt_shape: [21, 3]\nflip_idx: {list(range(21))}\nnames:\n  0: left\n  1: right\n"
        open(os.path.join(args.out, "hands.yaml"), "w").write(y)
    else:
        entries = json.load(open("bench/splits/oakink2_ego_dev.json"))["clips"]
        tars = [os.path.join(DATA_ROOT, "hot3d_train600", "train_aria", c + ".tar") for c in SPLIT["dev_clips"]]
        with Pool(args.workers) as p:
            items = [x for r in p.imap_unordered(_hot3d, tars) for x in r] + [x for r in p.imap_unordered(_oakink2, entries) for x in r]
        os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
        recs = []
        for src, name, img, hands, K in sorted(items, key=lambda x: x[1]):
            cv2.imwrite(os.path.join(args.out, "images", name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            H, W = img.shape[:2]
            hs = []
            for is_right, b, kp in hands:
                vis = ((kp[:, 0] >= 0) & (kp[:, 0] < W) & (kp[:, 1] >= 0) & (kp[:, 1] < H)).astype(int)
                hs.append(dict(is_right=bool(is_right), kp2d=np.round(kp[:, :2], 2).tolist(), vis=vis.tolist(),
                               bbox=[round(float(v), 1) for v in b]))
            recs.append(dict(id=f"crosspair/{name}", file=f"images/{name}.jpg", subset="hot3d" if src == "hot3d" else "oakink2_ego",
                             width=W, height=H, K=np.asarray(K, np.float64).round(4).tolist(), fully_annotated=True,
                             sequence=name.rsplit("_", 1)[0], hands=hs))
        meta = dict(version="crosspair_dev", note=__doc__.split("\n\n")[0],
                    rule=f"HOT3D held-out train sequences (bench/splits/crosspair_dev.json, every {args.hot3d_stride}-th frame) + "
                         f"OakInk2-ego dev windows (every {args.oakink2_stride}-th frame): frames with exactly two >= 50 % visible hands "
                         f"whose GT boxes overlap (IoU >= {args.min_iou}); dev only, never train on it",
                    license="HOT3D research license (Meta); OakInk2 CC BY-SA 4.0")
        json.dump(dict(meta=meta, images=recs), open(os.path.join(args.out, "annotations.json"), "w"))
        stats = dict(frames=len(recs), hands=sum(len(r["hands"]) for r in recs), clips=len({r["sequence"] for r in recs}),
                     per_subset={k: sum(r["subset"] == k for r in recs) for k in ("hot3d", "oakink2_ego")})
    json.dump(stats, open(os.path.join(args.out, "build_stats.json"), "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["train", "dev"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="bench/splits/crosspair_dev.json")
    p.add_argument("--min_iou", type=float, default=0.1)
    p.add_argument("--hot3d_stride", type=int, default=3)
    p.add_argument("--oakink2_stride", type=int, default=4)
    p.add_argument("--max_hot3d", type=int, default=4000)
    p.add_argument("--max_oakink2", type=int, default=4000)
    p.add_argument("--max_side", type=int, default=1024)
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--stage", default="", help="train: every candidate frame (default <out>_all); --out links the capped subset")
    main(p.parse_args())
