"""Cache bench clip frames at reduced resolution for the camera estimators (DPVO, DROID-SLAM).

Frames come from the bench loaders (bench.run.iter_clips: rectified upright pinhole for HOT3D), are resized by
--scale (INTER_AREA) and stored as q95 JPEGs with the matching intrinsics; decoding them is not counted as camera cost
(the hand pipeline decodes the same frames).
    python -m bench.world_frames --dataset hot3d --split bench/splits/hot3d_bench.json --data_root $DATA_ROOT/hot3d \
        --out $RESULTS/world/frames704 --scale 0.5 [--shard i --num_shards n]
-> <out>/<clip>/{%05d.jpg, K.npy, done}
"""
import argparse
import os

import cv2
import numpy as np


def main(args):
    from bench.run import iter_clips

    cv2.setNumThreads(2)
    for name, make_clip in iter_clips(args):
        d = os.path.join(args.out, name)
        if os.path.exists(os.path.join(d, "done")):
            continue
        os.makedirs(d, exist_ok=True)
        clip = make_clip()
        for i in range(len(clip)):
            fr = clip.frame(i)
            img = fr["image"]
            if args.scale != 1:
                img = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(d, f"{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        s = img.shape[1] / fr["image"].shape[1]
        K = fr["K"].astype(np.float64).copy()
        K[:2, :2] *= s
        K[:2, 2] = (K[:2, 2] + 0.5) * s - 0.5
        np.save(os.path.join(d, "K.npy"), K)
        open(os.path.join(d, "done"), "w").close()
        print(f"[world_frames] {name}: {len(clip)} frames {img.shape[1]}x{img.shape[0]}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="hot3d", choices=["hot3d", "egostandard"])
    p.add_argument("--split", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--out", required=True)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    main(p.parse_args())
