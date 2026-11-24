"""MoGe-2 metric depth for every frame of a real-to-sim clip -> <clip>/depth.npz (depth (N, H, W) float16 m, 0 = invalid;
K). With K in clip.json the horizontal FOV is given to MoGe-2; otherwise MoGe-2 estimates it on every --fov_stride-th
frame, the median FOV fixes K and the depth pass reruns with it. Needs MoGe-2 (github.com/microsoft/MoGe; the FAH
environment with `pip install -e ".[world]"`, or its own); weights: MOGE_CKPT (a local model.pt) or the hub's
Ruicheng/moge-2-vitl-normal.

    python scripts/real2sim/depth.py --clip $R2S/mustard
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch

MOGE = os.environ.get("MOGE_CKPT", "Ruicheng/moge-2-vitl-normal")


@torch.no_grad()
def main(args):
    from moge.model.v2 import MoGeModel

    net = MoGeModel.from_pretrained(MOGE).cuda().eval()
    clip = json.load(open(os.path.join(args.clip, "clip.json")))
    paths = sorted(glob.glob(os.path.join(args.clip, "frames", "*.jpg")))
    load = lambda p: torch.from_numpy(np.ascontiguousarray(cv2.imread(p)[..., ::-1])).cuda().permute(2, 0, 1).float() / 255
    H, W = cv2.imread(paths[0]).shape[:2]
    if clip.get("K") is not None:
        K = np.asarray(clip["K"], np.float64)
    else:
        fx = []
        for p in paths[::args.fov_stride]:
            k = net.infer(load(p), use_fp16=True)["intrinsics"].float().cpu().numpy()  # normalised by the image size
            fx.append(k[0, 0] * W)
        f = float(np.median(fx))
        K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]])
        print("MoGe-2 focal %.1f px (spread %.1f..%.1f)" % (f, min(fx), max(fx)))
    fov = float(np.degrees(2 * np.arctan(W / (2 * K[0, 0]))))
    D = np.zeros((len(paths), H, W), np.float16)
    for i, p in enumerate(paths):
        d = net.infer(load(p), fov_x=fov, use_fp16=True)["depth"].float().cpu().numpy()
        D[i] = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    np.savez(os.path.join(args.clip, "depth.npz"), depth=D, K=K)
    print(args.clip, D.shape, "median depth %.3f m" % np.median(D[D > 0]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True)
    p.add_argument("--fov_stride", type=int, default=10)
    main(p.parse_args())
