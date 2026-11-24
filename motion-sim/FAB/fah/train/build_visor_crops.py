"""VISOR (EPIC-KITCHENS) train hand-instance crops with GT masks for the mask head (no keypoints).

    python -m train.build_visor_crops --visor_root .../GroundTruth-SparseAnnotations --out $DATA_ROOT/crops --stride 1
-> <out>/visor_lab/<video>/<frame>_<k>.jpg + masks/<same>.png + masks_index.0.jsonl (score 1.0 = GT)
"""
import argparse
import glob
import json
import os
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np

from bench.seg.build import HAND, polygons_to_mask
from train.build_crops import render_crop


DET = None


def _det_box(img, m, is_right, min_cover=0.5):
    """Hand box (detector convention) for a VISOR hand+arm instance: the same-side detection whose box is
    covered most by the instance mask (>= min_cover), else None."""
    boxes, scores, right = DET(img)
    best, best_c = None, min_cover
    for b, r in zip(boxes, right):
        if bool(r) != is_right:
            continue
        x1, y1, x2, y2 = np.round(b).astype(int).clip(0, [m.shape[1], m.shape[0], m.shape[1], m.shape[0]])
        if x2 <= x1 or y2 <= y1:
            continue
        c = m[y1:y2, x1:x2].mean()
        if c > best_c:
            best, best_c = np.asarray(b, np.float32), c
    return best


def _work(arg):
    jf, root, out, stride = arg
    video = os.path.splitext(os.path.basename(jf))[0]
    part = video.split("_")[0]
    zpath = os.path.join(root, "rgb_frames", "train", part, video + ".zip")
    if not os.path.exists(zpath):
        return []
    zf = zipfile.ZipFile(zpath)
    names = {os.path.basename(n): n for n in zf.namelist()}
    recs = []
    for k, fr in enumerate(json.load(open(jf))["video_annotations"]):
        if k % stride:
            continue
        hands = [a for a in fr["annotations"] if HAND.match(a["name"].strip().lower())]
        if not hands or fr["image"]["name"] not in names:
            continue
        img = cv2.imdecode(np.frombuffer(zf.read(names[fr["image"]["name"]]), np.uint8), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        for j, a in enumerate(hands):
            m = polygons_to_mask(a["segments"], (H, W), (W / 1920, H / 1080))
            if m.sum() < 400:
                continue
            is_right = a["name"].lower().startswith("right")
            if DET is not None:  # crop around the hand (detector box), the target mask stays hand + arm
                box = _det_box(img, m, is_right)
                if box is None:
                    continue
            else:
                ys, xs = np.nonzero(m)
                box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
            crop, M = render_crop(img, box, is_right)
            mc, _ = render_crop(m.astype(np.uint8)[..., None].repeat(3, 2) * 255, box, is_right)
            rel = f"{video}/{fr['image']['name'][:-4]}_{j}.jpg"
            os.makedirs(os.path.join(out, os.path.dirname(rel)), exist_ok=True)
            os.makedirs(os.path.join(out, "masks", os.path.dirname(rel)), exist_ok=True)
            cv2.imwrite(os.path.join(out, rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(os.path.join(out, "masks", rel[:-4] + ".png"), (mc[..., 0] > 127).astype(np.uint8) * 255)
            recs.append(dict(file=rel, src=f"{video}:{fr['image']['name']}", is_right=is_right, box=box.tolist(),
                             M=M.round(5).tolist(), hw=[H, W]))
    return recs


def main(args):
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)
    jfs = sorted(glob.glob(os.path.join(args.visor_root, "annotations", "train", "*.json")))
    jobs = [(j, args.visor_root, out, args.stride) for j in jfs]
    if args.detector:  # GPU detector: single process
        global DET
        from fah.detector import build_hand_detector

        DET = build_hand_detector("wilor", device="cuda", conf=0.2, fast=True)
        recs = [r for j in jobs for r in _work(j)]
    else:
        with Pool(args.workers) as pool:
            recs = [r for rs in pool.imap_unordered(_work, jobs) for r in rs]
    with open(os.path.join(out, "index.jsonl"), "w") as f, open(os.path.join(out, "masks_index.0.jsonl"), "w") as g:
        for r in sorted(recs, key=lambda r: r["file"]):
            f.write(json.dumps(r) + "\n")
            g.write(json.dumps(dict(file=r["file"], score=1.0)) + "\n")
    print(f"{args.name}: {len(recs)} hand crops with GT masks from {len(jfs)} videos -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--visor_root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="visor_lab")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--detector", action="store_true", help="crop around matched detector hand boxes (VISOR masks include the arm)")
    main(p.parse_args())
