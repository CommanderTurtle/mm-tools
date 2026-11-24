"""EgoDex hand crops (label-free for distillation; ARKit 3D + K for stage 2, no 2D labels).

    python -m train.build_egodex_crops --root $DATA_ROOT/egodex/test --out $DATA_ROOT/crops --stride 15
Boxes come from the projected ARKit joints (padded); a hand is kept if its wrist confidence passes and
enough joints project into the image. Left hands are mirrored (3D x -> -x, cx -> W-1-cx) like every pool.
"""
import argparse
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

from train.build_crops import render_crop
from train.egodex import EgoDexEpisode, list_episodes


def _work(arg):
    path, out, stride, min_box = arg
    try:
        ep = EgoDexEpisode(path)
    except Exception as e:
        print("skip", path, e, flush=True)
        return []
    name = f"{ep.task_dir}_{ep.episode_id}"
    os.makedirs(os.path.join(out, name), exist_ok=True)
    recs = []
    for fr in ep.frames(step=stride):
        img = fr["image"]
        W = img.shape[1]
        for h in fr["hands"]:
            b = h["bbox"]
            if not h["hand_valid"] or b is None or max(b[2] - b[0], b[3] - b[1]) < min_box:
                continue
            crop, M = render_crop(img, np.asarray(b, np.float32), h["is_right"])
            rel = f"{name}/{fr['frame_idx']:05d}_{'r' if h['is_right'] else 'l'}.jpg"
            cv2.imwrite(os.path.join(out, rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            j3, K = h["joints_cam"].astype(np.float64).copy(), fr["K"].astype(np.float64).copy()
            if not h["is_right"]:
                j3[:, 0] *= -1
                K[0, 2] = W - 1 - K[0, 2]
            recs.append(dict(file=rel, src=f"{name}:{fr['frame_idx']}", is_right=bool(h["is_right"]),
                             box=[float(v) for v in b], M=M.round(5).tolist(), hw=list(img.shape[:2]),
                             joints3d=j3.round(5).tolist(), K=K.round(4).tolist(), episode=name))
    return recs


def main(args):
    eps = list_episodes(args.root)
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)
    with Pool(args.workers) as pool:
        recs = [r for rs in pool.imap_unordered(_work, [(p, out, args.stride, args.min_box) for p in eps]) for r in rs]
    with open(os.path.join(out, "index.jsonl"), "w") as f:
        for r in sorted(recs, key=lambda r: r["file"]):
            f.write(json.dumps(r) + "\n")
    print(f"{args.name}: {len(recs)} crops from {len(eps)} episodes -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="egodex")
    p.add_argument("--stride", type=int, default=15)
    p.add_argument("--min_box", type=int, default=48)
    p.add_argument("--workers", type=int, default=8)
    main(p.parse_args())
