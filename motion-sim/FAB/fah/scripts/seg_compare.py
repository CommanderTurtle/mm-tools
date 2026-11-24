"""Instance masks of the segmentation baselines on bench_seg frames (VISOR val / FreiHAND eval), from the same frames and
the same hand boxes as an FAH run, with per-frame timing.

Input: a `bench.run --dataset seg` run of FAH with "det_masks": true (per hand: bbox, is_right, joints2d, mask_det); its
boxes prompt the *_ours arms (deployable) and its joints2d cut the GT for the hand-only score. Each arm writes
bench-format pickles to <out_root>/seg/<run>-<arm>/; every predicted hand carries the joints2d of the FAH hand with the
best box IoU (>= 0.3). Arms: bench/seg/baselines.py (setup: scripts/setup_seg_baselines.sh), plus
  det   the FAH detector alone (--detector_weights, --det_conf, cross-class NMS 0.5, masks of every kept box).

Timing (the "latency" of every frame): the arm's call on the decoded frame between two CUDA synchronisations -- image
encoding, all prompts and the full-resolution binary masks (thresholding / upsampling, copy to the host); JPEG decode,
model loading, RLE encoding and scoring are outside. The first frame of the run is called three times first (warm-up,
not stored). Per chunk the median frame time is printed; the paper's fps = 1000 / median over the chunks of these
medians, slower of two runs, on shard 0 of 4 of VISOR val (prompt boxes: the FAH run's, so the detector is excluded).

    python scripts/seg_compare.py --run $RESULTS/seg/<fah run> --data_root $DATA_ROOT/bench_seg --out_root $RESULTS \
        --arms sam2b_ours,sam2l_ours,sam3_ours,sam3_text,hos,hands23 [--shard 0 --num_shards 4]
    python -m bench.seg.evaluate $RESULTS/seg/<fah run>-<arm>                        # hand + arm GT (VISOR's convention)
    python -m bench.seg.evaluate_hand $RESULTS/seg/<fah run>-<arm> --key mask_rle --margin 0 \
        --cut_ref $RESULTS/seg/<fah run> --ref_key mask_det                            # hand-only GT, one shared cut
"""
import argparse
import glob
import os
import pickle
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.common import box_iou  # noqa: E402
from bench.seg.baselines import BOX_ARMS, build_arms, mask_box  # noqa: E402
from bench.seg.rle import decode, encode  # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def with_joints(preds, ours):
    """joints2d of the FAH hand with the highest box IoU (>= 0.3), for the hand-only cut."""
    for p in preds:
        best = max(ours, key=lambda h: box_iou(p["bbox"], h["bbox"]), default=None)
        p["joints2d"] = best["joints2d"] if best is not None and box_iou(p["bbox"], best["bbox"]) >= 0.3 else None
    return preds


def nms_keep(boxes, scores, iou=0.5):
    keep = []
    for i in np.argsort(-scores):
        if all(box_iou(boxes[i], boxes[j]) < iou for j in keep):
            keep.append(i)
    return keep


def run_arm(a, arm, img, fr, ours, gt_boxes):
    """One timed call of arm `a` on a frame -> predictions (bbox, score, is_right, m)."""
    if a == "det":
        boxes, scores, is_right = arm(img)
        keep = nms_keep(boxes, scores)
        return [dict(bbox=boxes[i], score=float(scores[i]), is_right=bool(is_right[i]), m=m) for i, m in zip(keep, arm.masks(keep))]
    if a in BOX_ARMS:
        src = gt_boxes if a.endswith("_gt") else [h["bbox"] for h in ours]
        ms = arm(img, src)
        if a.endswith("_gt"):
            return [dict(bbox=b, score=1.0, is_right=g["is_right"], m=m) for b, g, m in zip(gt_boxes, fr["gt"], ms)]
        return [dict(bbox=h["bbox"], score=h["score"], is_right=h["is_right"], m=m) for h, m in zip(ours, ms)]
    return arm(img)


def main(args):
    files = sorted(glob.glob(os.path.join(args.run, "*.pkl")))[args.shard :: args.num_shards]
    names = args.arms.split(",")
    arms = build_arms(names, args.thirdparty, args.sam3, args.text, args.sam3_thresh)
    if "det" in names:
        from fah.detector import build_hand_detector

        arms["det"] = build_hand_detector("wilor", device="cuda", weights=args.detector_weights, conf=args.det_conf,
                                          fast=True, imgsz=args.det_size)
    run = os.path.basename(os.path.normpath(args.run))
    out_dir = {a: os.path.join(args.out_root, "seg", f"{run}-{a}") for a in names}
    todo = [p for p in files if not all(os.path.exists(os.path.join(out_dir[a], os.path.basename(p))) for a in names)]
    print(f"{len(files) - len(todo)} / {len(files)} chunks done", flush=True)  # resumable
    chunk_ms = {a: [] for a in names}
    for path in todo:
        clip = pickle.load(open(path, "rb"))
        frames = clip["frames"][: args.max_frames or None]
        outs = {a: [] for a in names}
        for fi, fr in enumerate(frames):
            img = cv2.imread(os.path.join(args.data_root, "images", *fr["key"].split("/")[1:]) + ".jpg")  # <set>/<video>/<frame>
            assert img is not None, fr["key"]
            ours = fr["pred"]
            gt_boxes = [mask_box(decode(g["mask_rle"])) for g in fr["gt"]]
            for a in names:
                for _ in range(3 if path == todo[0] and fi == 0 else 1):  # warm-up calls (graphs, autotune) on the first frame
                    sync()
                    t = time.perf_counter()
                    preds = run_arm(a, arms[a], img, fr, ours, gt_boxes)
                    sync()
                    dt = time.perf_counter() - t
                for p in preds:
                    p["mask_rle"] = encode(p.pop("m"))
                    if p["is_right"] is None:  # no side output: the matched FAH hand's side
                        best = max(ours, key=lambda h: box_iou(p["bbox"], h["bbox"]), default=None)
                        p["is_right"] = bool(best["is_right"]) if best is not None else True
                outs[a].append(dict(fr, pred=with_joints(preds, ours), latency=dt))
        for a in names:
            os.makedirs(out_dir[a], exist_ok=True)
            tmp = os.path.join(out_dir[a], os.path.basename(path) + ".tmp")
            pickle.dump(dict(clip, method=f"{clip['method']}-{a}", frames=outs[a]), open(tmp, "wb"))
            os.replace(tmp, os.path.join(out_dir[a], os.path.basename(path)))
            chunk_ms[a].append(1000 * float(np.median([f["latency"] for f in outs[a]])))
        print(path, {a: f"{chunk_ms[a][-1]:.1f} ms" for a in names}, flush=True)
    if todo:
        print("fps (1000 / median of the per-chunk median frame times):",
              {a: round(1000 / float(np.median(v)), 2) for a, v in chunk_ms.items()}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="bench seg run of FAH (det_masks): its boxes / joints prompt the arms")
    p.add_argument("--data_root", default=os.path.join(os.environ.get("DATA_ROOT", "data"), "bench_seg"))
    p.add_argument("--out_root", default=os.environ.get("RESULTS", "output/bench"))
    p.add_argument("--arms", default="sam2l_ours,sam2b_ours")
    p.add_argument("--thirdparty", default=os.environ.get("THIRDPARTY", "thirdparty"), help="VISOR-HOS / hands23_detector checkouts")
    p.add_argument("--detector_weights", default="", help="det arm: YOLO-seg weights (e.g. <weights>/shared/detector.pt)")
    p.add_argument("--det_conf", type=float, default=0.3)
    p.add_argument("--det_size", type=int, default=800)
    p.add_argument("--sam3", default="facebook/sam3")
    p.add_argument("--text", default="hand")
    p.add_argument("--sam3_thresh", type=float, default=0.5)
    p.add_argument("--max_frames", type=int, default=0, help="first N frames of each chunk only (smoke tests)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    main(p.parse_args())
