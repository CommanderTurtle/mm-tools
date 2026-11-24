"""FAH hands on a prepared real-to-sim clip (prep_clip.py): per-frame MANO meshes / joints in the camera frame (metres) +
detector masks -> <clip>/hands.pkl (frames[i] = list of hand dicts as fah.FAH returns them, the keys in KEEP). K from
clip.json, else from MoGe-2 (depth.py runs first for videos without intrinsics). Video / world mode: reset() once, then
frame by frame (causal). The study used FAH-L (--model fah) in video mode, with the detector's instance masks
(det_masks) and the TensorRT engines of fah-build-trt (--trt_dir; PyTorch otherwise: slightly different numerics).

    python scripts/real2sim/hands.py --clip $R2S/mustard --model fah --mode video --trt_dir $TRT
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fah.api import DEFAULT_WEIGHTS, FAH  # noqa: E402

KEEP = ("is_right", "score", "bbox", "joints3d", "joints2d", "verts3d", "mask_rle", "tip_ext", "track_id")


def clip_K(clip_dir):
    c = json.load(open(os.path.join(clip_dir, "clip.json")))
    if c.get("K") is not None:
        return np.asarray(c["K"], np.float64)
    return np.asarray(np.load(os.path.join(clip_dir, "depth.npz"))["K"], np.float64)


def main(args):
    K = clip_K(args.clip)
    method = FAH(args.model, args.mode, weights=args.weights, device=args.device, trt_dir=args.trt_dir, det_masks=True)
    method.reset()
    frames, lat = [], []
    for p in sorted(glob.glob(os.path.join(args.clip, "frames", "*.jpg"))):
        img = cv2.imread(p)
        t0 = time.perf_counter()
        hands = method(img, K)
        lat.append(time.perf_counter() - t0)
        frames.append([{k: h[k] for k in KEEP if k in h} for h in hands])
    pickle.dump(dict(method=method.name, method_args=json.dumps(method.args, default=str), K=K, frames=frames),
                open(os.path.join(args.clip, args.out), "wb"))
    print(args.clip, len(frames), "frames, hands/frame %.2f, median %.1f ms" % (np.mean([len(f) for f in frames]),
          np.median(lat) * 1000))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True)
    p.add_argument("--model", default="fah", help="fah-s / fah-b / fah (FAH-L) / fah-h")
    p.add_argument("--mode", default="video", choices=["image", "video"])
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--trt_dir", default="", help="fah-build-trt engines (CUDA)")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="hands.pkl")
    main(p.parse_args())
