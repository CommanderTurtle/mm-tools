"""Offline camera for a set of clips (docs/world.md "Offline camera"): fah.world.fuse_offline of a masked DROID-SLAM
run (bench/world_slam.py) and a real-time camera run (bench/world_fuse.py), clip by clip; the batch form of
fah.world.estimate_cameras_offline's last step for benchmark runs whose two cameras already exist.
    python -m bench.world_offline --slam <slam dir> --ours <camera dir> --out <dir>
-> <out>/<clip>.npz: index, T_world_cam, k_ours_over_slam, fallback (clips present in both dirs).
"""
import argparse
import glob
import os

import numpy as np

from fah.world import OFFLINE_ALPHA, OFFLINE_FALLBACK, fuse_offline


def poses(path):
    d = np.load(path)
    T = np.asarray(d["T_world_cam"], np.float64)
    idx = np.asarray(d["index"]) if "index" in d else np.arange(len(T))
    assert (idx == np.arange(len(T))).all(), f"{path}: poses must cover frames 0..N-1 in order"
    return T


def main(args):
    os.makedirs(args.out, exist_ok=True)
    names = sorted(set(map(os.path.basename, glob.glob(os.path.join(args.slam, "*.npz"))))
                   & set(map(os.path.basename, glob.glob(os.path.join(args.ours, "*.npz")))))
    n_fb = 0
    for name in names:
        T, k, fb = fuse_offline(poses(os.path.join(args.slam, name)), poses(os.path.join(args.ours, name)), args.alpha,
                                args.fallback)
        np.savez(os.path.join(args.out, name), index=np.arange(len(T)), T_world_cam=T, k_ours_over_slam=k, fallback=fb)
        n_fb += int(fb)
    print(f"{len(names)} clips -> {args.out} ({n_fb} on the real-time camera)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slam", required=True, help="bench/world_slam.py output (<clip>.npz)")
    p.add_argument("--ours", required=True, help="real-time camera (<clip>.npz, bench/world_fuse.py)")
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, default=OFFLINE_ALPHA)
    p.add_argument("--fallback", type=float, default=OFFLINE_FALLBACK)
    main(p.parse_args())
