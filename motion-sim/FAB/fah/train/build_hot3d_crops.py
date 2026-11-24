"""HOT3D-Clips (Aria) training crops for distillation: every clip whose sequence is NOT one of HaWoR's
27 validation sequences (= our benchmark, bench/splits/hot3d_bench_full.json).

Frames are rectified/rotated exactly as the benchmark loader does (bench.datasets.hot3d), boxes come
from the projected MANO GT vertices (visible part), crops as in train.build_crops.

    python -m train.build_hot3d_crops --download $DATA_ROOT/hot3d_train --out $DATA_ROOT/crops --stride 5
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

from train.build_crops import crop_keypoints, render_crop

SPLIT = "bench/splits/hot3d_bench_full.json"


def train_clips():
    from huggingface_hub import hf_hub_download, list_repo_files

    val = set(json.load(open(SPLIT))["hawor_val_sequences"])
    defs = json.load(open(hf_hub_download("bop-benchmark/hot3d", "clip_definitions.json", repo_type="dataset")))
    files = [f for f in list_repo_files("bop-benchmark/hot3d", repo_type="dataset")
             if f.startswith("train_aria/") and f.endswith(".tar")]
    keep = [f for f in files if defs[str(int(os.path.basename(f)[5:11]))]["sequence_id"] not in val]
    return sorted(keep), len(files)


def _work(arg):
    tar, out, stride, mano_dir, labels, verts = arg
    from bench.datasets.hot3d import Hot3DClip

    try:
        clip = Hot3DClip(tar, mano_dir, min_visible=0.3)
    except Exception as e:  # corrupt / partial download
        print("skip", tar, e, flush=True)
        return []
    name = os.path.basename(tar)[:-4]
    os.makedirs(os.path.join(out, name), exist_ok=True)
    recs = []
    for i in range(0, len(clip), stride):
        fr = clip.frame(i)
        for h in fr["hands"]:
            b = h["bbox"]
            if max(b[2] - b[0], b[3] - b[1]) < 32:
                continue
            crop, M = render_crop(fr["image"], np.asarray(b, np.float32), h["is_right"])
            rel = f"{name}/{i:04d}_{'r' if h['is_right'] else 'l'}.jpg"
            cv2.imwrite(os.path.join(out, rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rec = dict(file=rel, src=f"{name}:{fr['key']}", is_right=bool(h["is_right"]),
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
    return recs


def main(args):
    if args.download:
        from huggingface_hub import snapshot_download

        keep, total = train_clips()
        print(f"{len(keep)} / {total} train_aria clips outside the HaWoR validation sequences", flush=True)
        snapshot_download("bop-benchmark/hot3d", repo_type="dataset", allow_patterns=keep, local_dir=args.download,
                          max_workers=args.dl_workers)
    tars = sorted(glob.glob(os.path.join(args.download or args.clips, "train_aria", "*.tar")))
    val = set(json.load(open(SPLIT))["hawor_val_sequences"])
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)
    with Pool(args.workers) as pool:
        recs = [r for rs in pool.imap_unordered(_work, [(t, out, args.stride, args.mano_dir, args.labels, args.verts) for t in tars]) for r in rs]
    assert not any(r["sequence_id"] in val for r in recs), "validation sequence leaked into training crops"
    with open(os.path.join(out, "index.jsonl"), "w") as f:
        for r in sorted(recs, key=lambda r: r["file"]):
            f.write(json.dumps(r) + "\n")
    print(f"{args.name}: {len(recs)} crops from {len(tars)} clips -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download", default="", help="download non-validation train_aria clips here first")
    p.add_argument("--clips", default="", help="already-downloaded HOT3D root (with train_aria/)")
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--name", default="hot3d", help="output subdir / source name")
    p.add_argument("--labels", action="store_true", help="store GT 2D/3D/K per crop")
    p.add_argument("--verts", action="store_true", help="also store the GT MANO vertices (778 x 3) per crop")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dl_workers", type=int, default=16)
    main(p.parse_args())
