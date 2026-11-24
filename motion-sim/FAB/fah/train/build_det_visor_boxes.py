"""The detection set of train.build_det_dataset with VISOR re-labelled consistently (hand-only boxes). VISOR GT masks
include the arm, so VISOR frames are kept only if *every* hand instance (>= 400 px) was matched by a detector hand box
(train.build_visor_crops --detector -> visor_lab/index.jsonl); those boxes are used as labels.

    python -m train.build_det_visor_boxes --det $DATA_ROOT/det --out $DATA_ROOT/det_boxes --visor_root ... --matched $DATA_ROOT/crops/visor_lab/index.jsonl
"""
import argparse
import glob
import json
import os
import shutil
import zipfile

import cv2
import numpy as np

from bench.seg.build import HAND, polygons_to_mask
from train.build_det_dataset import _write


def main(args):
    for d in ("images", "labels"):
        for split in ("train", "val"):
            os.makedirs(os.path.join(args.out, d, split), exist_ok=True)
            for f in os.listdir(os.path.join(args.det, d, split)):
                if not f.startswith("visor_"):
                    dst = os.path.join(args.out, d, split, f)
                    if not os.path.exists(dst):
                        os.link(os.path.join(args.det, d, split, f), dst)
    matched = {}
    for line in open(args.matched):
        r = json.loads(line)
        video, frame = r["src"].split(":")
        matched.setdefault((video, frame), []).append((r["is_right"], r["box"]))
    kept = n_frames = boxes = 0
    for jf in sorted(glob.glob(os.path.join(args.visor_root, "annotations", "train", "*.json"))):
        video = os.path.splitext(os.path.basename(jf))[0]
        zpath = os.path.join(args.visor_root, "rgb_frames", "train", video.split("_")[0], video + ".zip")
        if not os.path.exists(zpath):
            continue
        zf = zipfile.ZipFile(zpath)
        names = {os.path.basename(n): n for n in zf.namelist()}
        for fr in json.load(open(jf))["video_annotations"]:
            name = fr["image"]["name"]
            hands = [a for a in fr["annotations"] if HAND.match(a["name"].strip().lower())]
            if not hands or name not in names:
                continue
            n_frames += 1
            m_list = matched.get((video, name), [])
            img = None
            n_big = 0
            for a in hands:
                if img is None:
                    img = cv2.imdecode(np.frombuffer(zf.read(names[name]), np.uint8), cv2.IMREAD_COLOR)
                H, W = img.shape[:2]
                n_big += polygons_to_mask(a["segments"], (H, W), (W / 1920, H / 1080)).sum() >= 400
            if n_big == 0 or len(m_list) != n_big:
                continue
            kept += 1
            boxes += _write(args.out, f"visor_{name[:-4]}", img, [(r, b, None) for r, b in m_list], args.max_side)
    shutil.copy(os.path.join(args.det, "hands.yaml"), os.path.join(args.out, "hands.yaml"))
    y = open(os.path.join(args.out, "hands.yaml")).read().replace(f"path: {args.det}", f"path: {args.out}")
    open(os.path.join(args.out, "hands.yaml"), "w").write(y)
    print(f"VISOR: kept {kept}/{n_frames} fully matched frames, {boxes} hand boxes -> {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--det", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--visor_root", required=True)
    p.add_argument("--matched", required=True)
    p.add_argument("--max_side", type=int, default=1024)
    main(p.parse_args())
