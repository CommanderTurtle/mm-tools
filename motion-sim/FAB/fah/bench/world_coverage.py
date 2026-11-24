"""Static coverage per fed frame as bench/world_dpvo.py stores it (sig_cov: the share of DPVO's 1/4-resolution patch grid
left unmasked by the hand masks of frame t - --mask_delay), for DPVO runs stored without it (bench/world_fuse.py --cov).
Use the DPVO run's --size / --stride / --mask_delay / --mask_box and hand pickles.
    python -m bench.world_coverage --frames <dir> --mask_run <hands> --out <dir> [--size 384 --stride 2 --mask_delay 1]
-> <out>/<clip>.npz: cov (fed), fed (the frame index of each fed frame)
"""
import argparse
import glob
import os
import pickle

import cv2
import numpy as np

from bench.world_dpvo import box_mask, rasterize


def main(args):
    faces = None
    if not args.mask_box:
        faces = pickle.load(open(os.path.join(args.mano_dir, "MANO_RIGHT.pkl"), "rb"))["f"].astype(np.int64)
    os.makedirs(args.out, exist_ok=True)
    clips = sorted(d for d in os.listdir(args.frames) if os.path.exists(os.path.join(args.frames, d, "done")))
    for name in clips[args.shard::args.num_shards]:
        out = os.path.join(args.out, f"{name}.npz")
        hp = os.path.join(args.mask_run, f"{name}.pkl")
        if os.path.exists(out) or not os.path.exists(hp):
            continue
        files = sorted(glob.glob(os.path.join(args.frames, name, "*.jpg")))
        n = len(files)
        H0, W0 = cv2.imread(files[0]).shape[:2]
        s = args.size / max(H0, W0)
        H1, W1 = int(round(H0 * s)) // 16 * 16, int(round(W0 * s)) // 16 * 16
        K = np.load(os.path.join(args.frames, name, "K.npy")).copy()
        K[0] *= W1 / W0
        K[1] *= H1 / H0
        frames = {f["index"]: f for f in pickle.load(open(hp, "rb"))["frames"]}
        fed = list(range(0, n, args.stride))
        cov = []
        for i in fed:  # the masks of bench/world_dpvo.py main()
            src = i - args.mask_delay
            if src not in frames:
                m = np.zeros((H1, W1), np.uint8)
            elif args.mask_box:
                m = box_mask(frames[src]["pred"], W1 / frames[src]["image_size"][1], (H1, W1))
            else:
                m = rasterize(frames[src]["pred"], K, (H1, W1), faces)
            m4 = cv2.resize(m.astype(np.float32), (W1 // 4, H1 // 4), interpolation=cv2.INTER_AREA) > 0
            cov.append(1.0 - float(m4.mean()))
        np.savez(out, cov=np.array(cov), fed=np.array(fed))
        print(f"[coverage] {name}: {len(fed)} fed, coverage median {np.median(cov):.3f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", required=True)
    p.add_argument("--mask_run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--mask_delay", type=int, default=1)
    p.add_argument("--mesh_masks", dest="mask_box", action="store_false",
                   help="the hand meshes instead of the boxes (a world_dpvo run without --mask_box)")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    main(p.parse_args())
