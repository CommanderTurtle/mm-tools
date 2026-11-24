"""Hand-detector latency (fah.detector fast path, or the ultralytics call for networks it does not implement, e.g.
WiLoR's YOLOv8-pose detector; TensorRT / fp16 optional): boxes only vs boxes + full-frame instance
masks of the top-k detections (YOLO-seg weights), one line per weights file, each the median of --iters CUDA-synchronised
calls after --warmup calls on the same decoded image.

Paper protocol (detector table, mask-cost table, detector speeds in the text): 50 warm-up + 500 timed calls, weights
listed in ABBA order within one process (each file appears twice), two rounds in separate processes, fps = 1000 / median
ms of the slower round. "boxes" = the detector alone; "+ masks of N hands" adds the instance masks of the top-N boxes
(the mask cost = the difference).

    python scripts/benchmark_detector.py --image hot3d_frame.jpg --det_size 800 --warmup 50 --iters 500 \
        --weights a.pt b.pt b.pt a.pt
    python scripts/benchmark_detector.py --image frame.jpg --det_size 640 --half --warmup 50 --iters 500 \
        --weights det.pt det.pt --trt '' <weights>/trt/detector-640-fp16.engine     # fp16 vs TensorRT
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fah.detector import build_hand_detector  # noqa: E402
from fah.utils import pick_device  # noqa: E402


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def timed(fn, device, iters):
    ts = []
    for _ in range(iters):
        sync(device)
        t = time.perf_counter()
        fn()
        sync(device)
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000)


def label(w):
    """<run>/weights/best.pt or <repo>/pretrained_models/<file> -> <run> / <repo>; anything else: the file name."""
    parts = os.path.normpath(w).split(os.sep)
    return parts[-3] if len(parts) >= 3 and parts[-2] in ("weights", "pretrained_models") else os.path.basename(w)


def main(args):
    device = pick_device(args.device)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    img = cv2.imread(args.image)
    assert img is not None, args.image
    print(f"device={device} image={img.shape[1]}x{img.shape[0]} det_size={args.det_size} conf={args.conf}")
    for i, w in enumerate(args.weights):
        trt = args.trt[i] if i < len(args.trt) else ""
        kw = dict(device=device, weights=w, conf=args.conf, imgsz=args.det_size, **({"half": True} if args.half else {}),
                  **({"trt": trt} if trt else {}))
        path = "fast path"
        try:
            det = build_hand_detector("wilor", fast=not args.ultralytics, **kw)
        except (ValueError, AttributeError) as e:  # a network fah.yolo does not implement (e.g. WiLoR's YOLOv8-pose)
            print(f"  {label(w)}: not on the fast path ({e}); timing the ultralytics call", flush=True)
            det, path = build_hand_detector("wilor", fast=False, **kw), "ultralytics call"
        if args.ultralytics:
            path = "ultralytics call"
        boxes, scores, _ = det(img)
        top = np.argsort(-scores)[: args.num_hands]

        def with_masks():
            det(img)
            det.masks(top)

        for _ in range(args.warmup):
            with_masks() if det.seg else det(img)
        t_box = timed(lambda: det(img), device, args.iters)
        line = f"  {label(w):28s} boxes {t_box:6.2f} ms"
        if det.seg:
            t_mask = timed(with_masks, device, args.iters)
            line += f" | + masks of {len(top)} hands {t_mask:6.2f} ms (masks {t_mask - t_box:.2f} ms)"
        print(line + f" | {len(boxes)} detections | {'TensorRT' if trt else path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True)
    p.add_argument("--weights", nargs="+", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--det_size", type=int, default=800)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--half", action="store_true", help="fp16 detector (det_half)")
    p.add_argument("--trt", nargs="*", default=[], help="TensorRT engine per --weights entry (det_trt; '' = none)")
    p.add_argument("--ultralytics", action="store_true", help="time the ultralytics call instead of the fast path")
    p.add_argument("--num_hands", type=int, default=2)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    main(p.parse_args())
