"""Per-frame timing of a bench.methods adapter on local JPEG frames (laptops / Apple silicon, where the CUDA timing
harness scripts/benchmark_latency.py does not run): frames decoded into memory first, the device synchronised around
every call (CUDA or MPS), the first --warmup frames of every stream dropped; median / mean / p90 ms and fps per stream
prefix and overall.

    python scripts/benchmark_frames.py --frames frames/ --method fah_pipeline \
        --method_args '{"device": "mps", ...}' --focal 610 --tag fah-m3pro --out fah_m3pro.json
"""
import argparse
import glob
import json
import os
import platform
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def main(args):
    from bench.run import load_method

    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.frames), "*.jpg")))
    streams = defaultdict(list)
    for f in files:
        streams[os.path.basename(f).rsplit("_", 1)[0]].append(cv2.imread(f))
    t = time.perf_counter()
    m = load_method(args.method, json.loads(os.path.expandvars(args.method_args)))
    load_s = time.perf_counter() - t
    lat = defaultdict(list)
    for name, imgs in streams.items():
        h, w = imgs[0].shape[:2]
        K = np.array([[args.focal, 0, w / 2], [0, args.focal, h / 2], [0, 0, 1]], np.float32)
        if hasattr(m, "reset"):
            m.reset()
        for i, img in enumerate(imgs):
            sync()
            t = time.perf_counter()
            m(img, K)
            sync()
            if i >= args.warmup:
                lat[name].append(1000 * (time.perf_counter() - t))
    summ = {}
    for name, v in list(lat.items()) + [("all", sum(lat.values(), []))]:
        v = np.asarray(v)
        summ[name] = dict(frames=len(v), median_ms=float(np.median(v)), mean_ms=float(v.mean()), p90_ms=float(np.percentile(v, 90)),
                          fps=float(1000 / np.median(v)))
        print(f"{name:16s} {summ[name]['median_ms']:7.1f} ms median ({summ[name]['fps']:.1f} fps), mean {summ[name]['mean_ms']:.1f}, p90 {summ[name]['p90_ms']:.1f}")
    rec = dict(tag=args.tag, method=args.method, method_args=json.loads(os.path.expandvars(args.method_args)), load_s=load_s,
               machine=platform.machine(), processor=platform.processor(), torch=torch.__version__, summary=summ)
    if args.out:
        json.dump(rec, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", required=True, help="dir of <stream>_<index>.jpg")
    p.add_argument("--method", default="fah")
    p.add_argument("--method_args", default="{}")
    p.add_argument("--focal", type=float, default=610.0)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="")
    main(p.parse_args())
