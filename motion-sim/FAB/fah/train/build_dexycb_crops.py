"""DexYCB s0-train crop pool with GT labels + MANO vertices (bench/splits/dexycb_s0_train.json: all subjects,
sequences i % 5 != 4, all 8 cameras; the s0 test sequences of the benchmark are asserted absent).

Frames come from the benchmark loader (bench.datasets.dexycb), boxes from the projected GT mesh, crops / labels /
mirroring exactly as train.build_hot3d_crops (--labels --verts). Every stride-th frame per clip, phase-shifted by
camera index so the 8 views of a sequence sample different instants. src = "<sequence>:<serial>/<frame>", so the
LabeledCrops train/val hash keeps all views of a sequence on one side.

    python -m train.build_dexycb_crops --root $DATA_ROOT/dexycb --out $DATA_ROOT/crops --stride 8 --labels --verts
"""
import argparse
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

from bench.datasets.dexycb import SERIALS
from train.build_crops import crop_keypoints, render_crop

SPLIT = "bench/splits/dexycb_s0_train.json"
TEST = "bench/splits/dexycb_s0_test_full.json"


_MANO = {}


def _work(arg):
    path, out, stride, mano_dir, labels, verts = arg
    from bench.datasets.dexycb import DexYCBClip, DexYCBMano

    name = os.path.basename(path)[:-4]
    done = os.path.join(out, name, ".done.json")
    if os.path.exists(done):  # resumable per clip
        return json.load(open(done))
    if mano_dir not in _MANO:  # one MANO layer per worker process
        _MANO[mano_dir] = DexYCBMano(mano_dir)
    clip = DexYCBClip(path, min_visible=0.3, mano=_MANO[mano_dir])
    serial = clip.info["serial"]
    os.makedirs(os.path.join(out, name), exist_ok=True)
    recs = []
    for i in range(SERIALS.index(serial) % stride, len(clip), stride):
        fr = clip.frame(i)
        for h in fr["hands"]:
            b = h["bbox"]
            if max(b[2] - b[0], b[3] - b[1]) < 32:
                continue
            crop, M = render_crop(fr["image"], np.asarray(b, np.float32), h["is_right"])
            rel = f"{name}/{int(fr['key']):04d}_{'r' if h['is_right'] else 'l'}.jpg"  # original frame number
            cv2.imwrite(os.path.join(out, rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rec = dict(file=rel, src=f"{clip.sequence_id}:{serial}/{fr['key']}", is_right=bool(h["is_right"]),
                       box=[float(v) for v in b], M=M.round(5).tolist(), hw=list(fr["image"].shape[:2]),
                       sequence_id=clip.sequence_id)
            if labels:  # GT in the right-hand (mirrored for left) frame: crop-pixel 2D, camera 3D (m), K
                W = fr["image"].shape[1]
                j2 = h["joints2d"]
                inside = (j2[:, 0] >= 0) & (j2[:, 0] < W) & (j2[:, 1] >= 0) & (j2[:, 1] < fr["image"].shape[0])
                rec["kp2d"] = crop_keypoints(np.concatenate([j2, inside[:, None]], 1), M, h["is_right"], W).round(2).tolist()
                j3, K = np.asarray(h["joints3d"], np.float64).copy(), np.asarray(fr["K"], np.float64).copy()
                if not h["is_right"]:
                    j3[:, 0] *= -1
                    K[0, 2] = W - 1 - K[0, 2]
                rec["joints3d"] = j3.round(5).tolist()
                rec["K"] = K.round(4).tolist()
                if verts:  # GT MANO mesh (camera frame, mirrored like the joints), float16 side file
                    v3 = np.asarray(h["verts3d"], np.float64).copy()
                    if not h["is_right"]:
                        v3[:, 0] *= -1
                    vf = rel[:-4] + "_v.npy"
                    np.save(os.path.join(out, vf), v3.astype(np.float16))
                    rec["verts_file"] = vf
            recs.append(rec)
    json.dump(recs, open(done, "w"))
    return recs


def main(args):
    clips = json.load(open(SPLIT))["clips"]
    test = set(json.load(open(TEST))["test_sequences"])
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)
    jobs = [(os.path.join(args.root, c), out, args.stride, args.mano_dir, args.labels, args.verts) for c in clips]
    recs = []
    with Pool(args.workers) as pool:
        for k, rs in enumerate(pool.imap_unordered(_work, jobs, chunksize=4)):
            recs += rs
            if k % 500 == 0:
                print(f"{k}/{len(jobs)} clips, {len(recs)} crops", flush=True)
    assert not any(r["sequence_id"] in test for r in recs), "s0 test sequence leaked into training crops"
    with open(os.path.join(out, "index.jsonl"), "w") as f:
        for r in sorted(recs, key=lambda r: r["file"]):
            f.write(json.dumps(r) + "\n")
    print(f"{args.name}: {len(recs)} crops from {len(clips)} clips -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="converted DexYCB (bench.datasets.build_dexycb --out)")
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--name", default="dexycb_lab", help="output subdir / source name")
    p.add_argument("--labels", action="store_true", help="store GT 2D/3D/K per crop")
    p.add_argument("--verts", action="store_true", help="also store the GT MANO vertices (778 x 3) per crop")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--workers", type=int, default=16)
    main(p.parse_args())
