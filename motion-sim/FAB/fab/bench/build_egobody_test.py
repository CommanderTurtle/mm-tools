"""EgoBody test bench set, egocentric (the EgoBody / EgoHMR benchmark view): HoloLens PV frames of the interactee with the
released gendered SMPL fits in the PV camera frame (annotation_egocentric_smpl_npz/egocapture_test_smpl.npz), every
--stride-th annotated frame, full images, oracle box = the projected GT mesh clipped to the image (people are often
truncated: --min_visible 0.3, as the ego training pool). Per-recording convention check: GT limb joints vs the released
OpenPose keypoints. Writes <out>/images/*.jpg + <out>/index.pkl (bench/datasets.wds_test). Reads the release zips
(annotation_egocentric_smpl_npz.zip, egocentric_color.zip) from --egobody, the local EgoBody download.

    python -m bench.build_egobody_test --egobody $FAB_DATA/egobody --out $FAB_DATA/egobody_test --stride 10
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from bench.body_models import SMPL
from bench.testset_utils import GENDER, OP_LIMBS, SMPL_LIMBS, box_of, egobody_zip, limb_err, to_camera
from fab import paths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--egobody", required=True, help="the local EgoBody release folder (its zips)")
    p.add_argument("--out", required=True)
    p.add_argument("--bm_dir", default=str(paths.body_models_dir()))
    p.add_argument("--split", default="test")
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--min_visible", type=float, default=0.3)
    p.add_argument("--min_height", type=float, default=96.0)
    p.add_argument("--max_check_px", type=float, default=30.0)
    a = p.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    bodies = {g: SMPL(Path(a.bm_dir) / "smpl" / f"SMPL_{n}.pkl").cuda().eval() for g, n in ((1, "MALE"), (2, "FEMALE"))}
    ann = np.load(io.BytesIO(egobody_zip(a.egobody, "annotation_egocentric_smpl_npz.zip").read(
        f"annotation_egocentric_smpl_npz/egocapture_{a.split}_smpl.npz")), allow_pickle=True)
    print("npz keys", ann.files, flush=True)
    color = egobody_zip(a.egobody, "egocentric_color.zip")
    img_names = {n.split("egocentric_color/", 1)[-1]: n for n in color.namelist() if n.endswith(".jpg")}
    names = [str(x) for x in ann["imgname"]]
    by_rec = defaultdict(list)
    for i, n in enumerate(names):
        m = re.search(r"(recording_[^/]+)/", n)
        if m:
            by_rec[m[1]].append(i)
    index, stats = [], defaultdict(int)
    for rec in sorted(by_rec):
        rows, checks = [], []
        for i in by_rec[rec][:: a.stride]:
            key = names[i].split("egocentric_color/", 1)[-1]
            if key not in img_names:
                stats["no_image"] += 1
                continue
            g = str(ann["gender"][i]).lower()
            gender = GENDER.get(g, GENDER.get({"m": "male", "f": "female"}.get(g[:1], ""), 0))
            if not gender:
                stats["no_gender"] += 1
                continue
            K = np.array([[ann["fx"][i], 0, ann["cx"][i]], [0, ann["fy"][i], ann["cy"][i]], [0, 0, 1]], np.float64)
            pose, tr, v, jts = to_camera(bodies[gender], np.eye(3), np.zeros(3), ann["global_orient_pv"][i],
                                         ann["body_pose"][i], ann["betas"][i][:10], ann["transl_pv"][i])
            if v[:, 2].min() < 0.05:
                stats["behind"] += 1
                continue
            uv = v @ K.T
            uv = uv[:, :2] / uv[:, 2:3]
            uj = jts @ K.T
            uj = uj[:, :2] / uj[:, 2:3]
            kp = np.asarray(ann["valid_keypoints"][i], np.float64)
            checks.append(limb_err(uj, kp, SMPL_LIMBS, OP_LIMBS, kp[:, 2]))
            rows.append((i, key, K, pose, tr, gender, uv))
        med = float(np.nanmedian(checks)) if np.isfinite(checks).any() else float("nan")
        if np.isfinite(med) and med > a.max_check_px:
            print(f"{rec}: convention check failed ({med:.1f} px): skipped", flush=True)
            stats["rec_failed"] += 1
            continue
        n_rec = 0
        for i, key, K, pose, tr, gender, uv in rows:
            img = cv2.imdecode(np.frombuffer(color.read(img_names[key]), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                stats["bad_image"] += 1
                continue
            H, W = img.shape[:2]
            b = box_of(uv, W, H, a.min_visible, a.min_height)
            if b is None:
                stats["out_of_frame_or_small"] += 1
                continue
            path = out / "images" / f"{key.replace('/', '_')}"
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            full = np.concatenate([uv.min(0), uv.max(0)])
            vis = float(np.prod(np.clip(b[2:] - b[:2], 0, None)) / max(np.prod(full[2:] - full[:2]), 1e-6))
            m = re.search(r"frame_(\d+)", key)
            index.append({"id": f"{rec}/{int(m[1]) if m else i:05d}", "image": str(path), "bbox": np.asarray(b, np.float32),
                          "K": K.astype(np.float32), "gender": {1: "male", 2: "female"}[gender],
                          "global_orient": pose[:3].astype(np.float32), "body_pose": pose[3:72].astype(np.float32),
                          "betas": np.asarray(ann["betas"][i][:10], np.float32), "transl": np.asarray(tr, np.float32),
                          "extrinsic": np.eye(4, dtype=np.float32), "visible": vis, "check_px": med})
            n_rec += 1
        stats["kept"] += n_rec
        print(f"{rec}: {n_rec} samples, check {med:.1f} px", flush=True)
    with open(out / "index.pkl", "wb") as f:
        pickle.dump(index, f)
    vis = np.array([s["visible"] for s in index])
    stats.update(recordings=len({s["id"].split("/")[0] for s in index}), visible_median=float(np.median(vis)),
                 truncated_lt_08=int((vis < 0.8).sum()))
    (out / "build_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(dict(stats), flush=True)


if __name__ == "__main__":
    main()
