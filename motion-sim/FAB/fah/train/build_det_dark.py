"""Detector dataset with low-light copies: symlinks every image / label of a YOLO-pose dataset and adds a
darkened copy (bench/bench2d/make_dark.py transform, both levels mixed) of a random fraction of the training
images with the same labels.

    python -m train.build_det_dark --src $DATA_ROOT/det_boxes --dst $DATA_ROOT/det_boxes_dark --frac 0.35
"""
import argparse
import os
from multiprocessing import Pool

import cv2
import numpy as np

from bench.bench2d.make_dark import LEVELS, darken


def work(a):
    src, dst, name, seed = a
    rng = np.random.default_rng(seed)
    img = cv2.imread(os.path.join(src, "images", "train", name))
    if img is None:
        return 0
    out = darken(img, rng, *LEVELS["vdark" if rng.random() < 0.35 else "dark"])
    stem = os.path.splitext(name)[0] + "_dark"
    cv2.imwrite(os.path.join(dst, "images", "train", stem + ".jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    lab = os.path.join(src, "labels", "train", os.path.splitext(name)[0] + ".txt")
    if os.path.exists(lab):
        os.symlink(lab, os.path.join(dst, "labels", "train", stem + ".txt"))
    return 1


def main(args):
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            os.makedirs(os.path.join(args.dst, kind, split), exist_ok=True)
            for f in os.listdir(os.path.join(args.src, kind, split)):
                t = os.path.join(args.dst, kind, split, f)
                if not os.path.lexists(t):
                    os.symlink(os.path.join(args.src, kind, split, f), t)
    names = sorted(os.listdir(os.path.join(args.src, "images", "train")))
    rng = np.random.default_rng(0)
    pick = [n for n in names if rng.random() < args.frac]
    with Pool(args.workers) as p:
        n = sum(p.map(work, [(args.src, args.dst, nm, i) for i, nm in enumerate(pick)], chunksize=64))
    yaml = open(os.path.join(args.src, "hands.yaml")).read().replace(f"path: {args.src}", f"path: {args.dst}")
    open(os.path.join(args.dst, "hands.yaml"), "w").write(yaml)
    print(f"{n} dark copies added to {len(names)} training images -> {args.dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--frac", type=float, default=0.35)
    p.add_argument("--workers", type=int, default=6)
    main(p.parse_args())
