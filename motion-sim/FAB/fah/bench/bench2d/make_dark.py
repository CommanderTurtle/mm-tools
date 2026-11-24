"""Low-light variant of a bench2d folder (same annotations, darkened images) -> "dark hands" benchmark with GT.

Per image (deterministic per id): linear-light exposure drop, Poisson-Gaussian sensor noise at the reduced
signal, weak gain (x0.8-1.5) with a slight colour cast, sRGB gamma, slight blur, JPEG q 60-85.
Levels (exposure): "dark" 0.05-0.15, "vdark" 0.015-0.05.

    python -m bench.bench2d.make_dark --src $DATA_ROOT/bench2d_v2 --dst $DATA_ROOT/bench2d_v2_dark --level dark
"""
import argparse
import hashlib
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

LEVELS = {"dark": (0.05, 0.15), "vdark": (0.015, 0.05)}


def darken(img, rng, lo, hi):
    lin = (img.astype(np.float32) / 255.0) ** 2.2  # sRGB-ish -> linear
    expo = rng.uniform(lo, hi)
    full_well = rng.uniform(800, 3000)  # photons at white: shot noise grows as the exposure drops
    sig = lin * expo * full_well
    noisy = rng.poisson(np.clip(sig, 0, None)).astype(np.float32) + rng.normal(0, rng.uniform(1.0, 4.0), sig.shape)
    lin = np.clip(noisy / full_well, 0, 1)
    gain = rng.uniform(0.8, 1.5) * rng.uniform(0.9, 1.1, 3)  # weak auto gain + slight white-balance cast
    out = np.clip(lin * gain, 0, 1) ** (1 / 2.2)
    out = (out * 255).astype(np.uint8)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 1.2))
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 86))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def work(a):
    src, dst, rec, level = a
    rng = np.random.default_rng(int(hashlib.md5((rec["id"] + level).encode()).hexdigest()[:8], 16))
    img = cv2.imread(os.path.join(src, rec["file"]))
    out = darken(img, rng, *LEVELS[level])
    path = os.path.join(dst, os.path.splitext(rec["file"])[0] + ".jpg")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return dict(rec, file=os.path.relpath(path, dst))


def main(args):
    ann = json.load(open(os.path.join(args.src, "annotations.json")))
    with Pool(args.workers) as p:
        recs = p.map(work, [(args.src, args.dst, r, args.level) for r in ann["images"]], chunksize=16)
    ann["images"] = recs
    ann["meta"] = dict(ann["meta"], name=os.path.basename(args.dst.rstrip("/")), darkened_from=args.src, level=args.level,
                       note="synthetic low light (bench/bench2d/make_dark.py): exposure drop + shot/read noise + gamma + blur + JPEG")
    json.dump(ann, open(os.path.join(args.dst, "annotations.json"), "w"))
    print("wrote", len(recs), "images to", args.dst)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--level", default="dark", choices=list(LEVELS))
    p.add_argument("--workers", type=int, default=8)
    main(p.parse_args())
