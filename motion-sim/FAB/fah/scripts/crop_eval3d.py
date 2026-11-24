"""3D accuracy on a held-out training-format crop pool (train.build_*_crops: index.jsonl with the original-frame box,
crop affine M, and K / joints3d already mirrored for left hands), e.g. EgoDex part 2 (Vision Pro hand tracking, ARKit labels
converted to the MANO convention; never in a pose training mix) as an egocentric tracking-GT validation set. Each crop is run as an
image with its own intrinsics (M o K) and the GT box (no detector; left hands are stored mirrored into right hands, so
they run as right hands with the mirrored box): PA-MPJPE / rr-MPJPE (root-relative, mm) of `joints3d`,
per episode-stratified sample. Aggregates only.

    python scripts/crop_eval3d.py --pool $DATA_ROOT/crops/egodex_part2_mano --n 4000 --method_args '<json>' --out x.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.common import procrustes  # noqa: E402
from train.build_crops import crop_box  # noqa: E402


def sample(pool, n, seed=0):
    """Preserve the episode-stratified sample without retaining the whole index."""
    def rows():
        with open(os.path.join(pool, "index.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                if r.get("joints3d") is not None:
                    yield r.get("episode", r["src"].split(":")[0]), r

    counts = defaultdict(int)
    for e, _ in rows():
        counts[e] += 1
    rng = np.random.default_rng(seed)
    eps = sorted(counts)
    per = max(1, n // max(len(eps), 1))
    selected = []
    for e in eps:  # a few frames from every episode, spread over the episode
        selected += [(e, int(i)) for i in sorted(rng.choice(counts[e], min(per, counts[e]), replace=False))]
    rng.shuffle(selected)
    slots = {key: i for i, key in enumerate(selected[:n])}
    out = [None] * len(slots)
    counts.clear()
    # A second streaming pass materializes only the selected records, in the
    # same shuffled order as the original eager sampler.
    for e, r in rows():
        key = (e, counts[e])
        counts[e] += 1
        if key in slots:
            out[slots[key]] = r
    if any(r is None for r in out):
        raise RuntimeError("Crop index changed while sampling")
    return out


def main(args):
    from bench.run import load_method

    m = load_method("fah_pipeline", json.loads(os.path.expandvars(args.method_args)))
    base = open(os.path.join(args.pool, "BASE_POOL")).read().strip() if os.path.exists(os.path.join(args.pool, "BASE_POOL")) else ""
    img_root = os.path.join(os.path.dirname(args.pool.rstrip("/")), base) if base else args.pool
    pa, rr, miss = [], [], 0
    for r in sample(args.pool, args.n, args.seed):
        img = cv2.imread(os.path.join(img_root, r["file"]))
        if img is None:
            img = cv2.imread(os.path.join(args.pool, r["file"]))
        M = np.vstack([np.asarray(r["M"], np.float64), [0, 0, 1]])
        K = M @ np.asarray(r["K"], np.float64)  # crop pixels = M o original pixels
        box = crop_box(r)[None]
        if getattr(m, "reset", None):
            m.reset()
        hands = m.est.process_image(img, boxes=box, is_right=np.ones(1, bool), scores=np.ones(1, np.float32), cam_int=K.astype(np.float32))
        if not hands:
            miss += 1
            continue
        j = np.asarray(hands[0].get("keypoints_3d_mano", hands[0]["keypoints_3d"]), np.float64)
        g = np.asarray(r["joints3d"], np.float64)
        pa.append(np.linalg.norm(procrustes(j, g) - g, axis=1).mean() * 1000)
        rr.append(np.linalg.norm((j - j[:1]) - (g - g[:1]), axis=1).mean() * 1000)
    res = dict(pa_mpjpe=float(np.mean(pa)), pa_median=float(np.median(pa)), rr_mpjpe=float(np.mean(rr)), hands=len(pa), missed=miss)
    print(json.dumps(res))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", required=True, help="extracted crop pool directory (index.jsonl)")
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method_args", required=True)
    p.add_argument("--out", default="")
    main(p.parse_args())
