"""Halpe-FullBody (HICO-DET train images, in-the-wild human-object interaction) labelled hand crops.

Annotations: halpe_train_v1.json (136 keypoints per person: hands at 94-114 left, 115-135 right, OpenPose
order) from the HF repo ayh015/HICO-Det_Halpe_HAKE; images: HICO-DET train parquet shards (zhimeng/hico_det,
MIT). Crops follow train.build_crops (kp2d in right-hand crop pixels).

    python -m train.build_halpe_crops --work $DATA_ROOT/halpe --out $DATA_ROOT/crops --name halpe_lab
"""
import argparse
import glob
import io
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

from train.build_crops import crop_keypoints, render_crop

HANDS = ((94, False), (115, True))


def _work(arg):
    parquet, by_file, out, min_kps, min_box = arg
    import pyarrow.parquet as pq

    recs = []
    t = pq.ParquetFile(parquet)
    for rg in range(t.num_row_groups):
        for row in t.read_row_group(rg, columns=["image"]).to_pylist():
            name = os.path.basename(row["image"]["path"])
            anns = by_file.get(name)
            if not anns:
                continue
            img = cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            for pi, kps in enumerate(anns):
                for start, is_right in HANDS:
                    kp = kps[start:start + 21].copy()
                    vis = kp[:, 2] > 0
                    if vis.sum() < min_kps:
                        continue
                    v = kp[vis, :2]
                    box = np.array([*v.min(0), *v.max(0)], np.float32)
                    if max(box[2] - box[0], box[3] - box[1]) < min_box:
                        continue
                    kp[:, 2] = vis
                    crop, M = render_crop(img, box, is_right)
                    rel = f"{name[:-4]}_{pi}_{'r' if is_right else 'l'}.jpg"
                    rel = f"{rel[15:19]}/{rel}"
                    os.makedirs(os.path.join(out, os.path.dirname(rel)), exist_ok=True)
                    cv2.imwrite(os.path.join(out, rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    recs.append(dict(file=rel, src=name, is_right=is_right, box=box.tolist(), M=M.round(5).tolist(),
                                     hw=list(img.shape[:2]), kp2d=crop_keypoints(kp, M, is_right, img.shape[1]).round(2).tolist()))
    return recs


def main(args):
    from huggingface_hub import hf_hub_download, snapshot_download

    ann = json.load(open(hf_hub_download("ayh015/HICO-Det_Halpe_HAKE", "Annotation/hico-fullbody-pose/halpe_train_v1.json",
                                         repo_type="dataset", local_dir=args.work)))
    files = {im["id"]: im["file_name"] for im in ann["images"]}
    by_file = {}
    for a in ann["annotations"]:
        by_file.setdefault(files[a["image_id"]], []).append(np.asarray(a["keypoints"], np.float64).reshape(-1, 3))
    snapshot_download("zhimeng/hico_det", repo_type="dataset", local_dir=args.work, allow_patterns=["data/train-*.parquet"],
                      max_workers=8)
    parquets = sorted(glob.glob(os.path.join(args.work, "data", "train-*.parquet")))
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)
    with Pool(args.workers) as pool:
        recs = [r for rs in pool.imap_unordered(_work, [(p, by_file, out, args.min_kps, args.min_box) for p in parquets]) for r in rs]
    with open(os.path.join(out, "index.jsonl"), "w") as f:
        for r in sorted(recs, key=lambda r: r["file"]):
            f.write(json.dumps(r) + "\n")
    print(f"{args.name}: {len(recs)} hand crops from {len(by_file)} annotated images -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default="$DATA_ROOT/halpe")
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="halpe_lab")
    p.add_argument("--min_kps", type=int, default=8)
    p.add_argument("--min_box", type=int, default=24)
    p.add_argument("--workers", type=int, default=6)
    main(p.parse_args())
