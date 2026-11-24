"""Harmony4D test bench set: the person instances of SAM 3D Body's harmony4d_test annotations (facebook/sam-3d-body-dataset,
39 test sequences, 11,190 rows) with the released Harmony4D test SMPL fits (neutral) as ground truth, in the camera frame.

Per row: the exo image of the release (test zip, or an extracted sequence directory) undistorted as SAM 3D Body's data
script does (OpenCV fisheye, balance 0.8, at the scale whose new K matches the annotation's cam_int), the annotation's box,
the fit person whose projected mesh box overlaps that box most (IoU stored), the fit posed in that camera
(the Harmony4D release conventions). Writes <out>/images/*.jpg, <out>/index.pkl (SMPL GT, bench/datasets.wds_test)
and <out>/index_mhr.pkl (same samples with the annotation's MHR parameters as GT instead, for comparison).

    python -m bench.build_harmony4d_test --ann $FAB_DATA/sam3db_annotations/data/harmony4d_test \
        --zips "$FAB_DATA/harmony4d/zips/test/*.zip" --out $FAB_DATA/harmony4d_test
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch

from bench.body_models import SMPL, _rodrigues
from bench.testset_utils import DirSource, colmap_from_world, read_colmap, rotmat_to_aa
from fab import paths


# SMPL joints vs the annotation's MHR70 2D keypoints (shoulders, elbows, wrists, hips, knees, ankles; left first)
SMPL_J = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
MHR70_J = [5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]


def iou(a, b):
    lo, hi = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    inter = np.prod(np.clip(hi - lo, 0, None))
    return inter / (np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - inter + 1e-9)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ann", required=True, help="harmony4d_test parquet directory")
    p.add_argument("--zips", required=True)
    p.add_argument("--dirs", default="", help="comma-separated extracted <activity>/<sequence> directories")
    p.add_argument("--out", required=True)
    p.add_argument("--bm_dir", default=str(paths.body_models_dir()))
    p.add_argument("--max_rows", type=int, default=0)
    args = p.parse_args()
    import pandas as pd

    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{args.ann}/*.parquet"))], ignore_index=True)
    if args.max_rows:
        df = df.iloc[: args.max_rows]
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    sources = {}  # "<activity>/<sequence>" -> (source, prefix inside it)
    for z in sorted(glob.glob(args.zips)):
        import zipfile

        zf = zipfile.ZipFile(z)
        for n in zf.namelist():
            parts = n.split("/")
            if len(parts) > 2 and parts[1]:
                sources.setdefault("/".join(parts[:2]), (zf, "/".join(parts[:2])))
    for d in [x for x in args.dirs.split(",") if x]:
        sources["/".join(Path(d).parts[-2:])] = (DirSource(d), "")
    body = SMPL(Path(args.bm_dir) / "smpl" / "SMPL_NEUTRAL.pkl").eval()
    t_ = lambda a: torch.from_numpy(np.asarray(a, np.float32))[None]  # noqa: E731
    cams, members, index, index_mhr, stats = {}, {}, [], [], {"rows": len(df), "kept": 0, "no_source": 0, "no_fit": 0}
    written = set()
    for _, r in df.iterrows():
        _, act, seq, cam, fname = r["image"].split("/")
        key = f"{act}/{seq}"
        if key not in sources:
            stats["no_source"] += 1
            continue
        src, pre = sources[key]
        join = lambda *x: "/".join([q for q in (pre, *x) if q])  # noqa: E731
        if key not in members:
            members[key] = set(src.namelist())
        ck = (key, cam)
        K_ann = np.stack(r["cam_int"]).astype(np.float64)
        if ck not in cams:
            work = join("colmap", "workplace")
            exo = read_colmap(src.read(f"{work}/cameras.txt").decode(), src.read(f"{work}/images.txt").decode())
            T = colmap_from_world(src, work, members[key])
            s = float(np.cbrt(np.linalg.det(T[:3, :3])))
            R_i, t_i, (_, W0, H0, prm) = exo[cam]
            K0 = np.array([[prm[0], 0, prm[2]], [0, prm[1], prm[3]], [0, 0, 1]], np.float64)
            D = prm[4:8].astype(np.float64)
            best = None
            for scale in (1.0, 0.5, 0.25):  # the scale SAM 3D Body's undistortion used: its new K = the annotation's
                W, H = int(W0 * scale), int(H0 * scale)
                K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K0, D, (W0, H0), np.eye(3),
                                                                           new_size=(W, H), balance=0.8)
                err = np.abs(K - K_ann).max() / K_ann[0, 0]
                if best is None or err < best[0]:
                    best = (err, scale, W, H, K)
            err, scale, W, H, K = best
            maps = cv2.fisheye.initUndistortRectifyMap(K0, D, np.eye(3), K, (W, H), cv2.CV_16SC2)
            cams[ck] = dict(Rc=R_i @ (T[:3, :3] / s), tc=(R_i @ T[:3, 3] + t_i) / s, K=K, maps=maps, scale=scale,
                            k_err=float(err))
        c = cams[ck]
        fr = int(Path(fname).stem)
        fit_name = join("processed_data", "smpl", f"{fr:05d}.npy")
        if fit_name not in members[key]:
            stats["no_fit"] += 1
            continue
        fits = np.load(io.BytesIO(src.read(fit_name)), allow_pickle=True).item()
        box = np.asarray(r["bbox"], np.float64)
        kp2d = np.stack(r["keypoints_2d"]).astype(np.float64)
        cand = []
        for person, d in sorted(fits.items()):
            betas = np.asarray(d["betas"], np.float64)[:10]
            v_shaped = body.v_template + torch.einsum("bl,vcl->bvc", t_(betas), body.shapedirs)
            J0 = (body.J_regressor[0] @ v_shaped[0]).numpy().astype(np.float64)
            R_w = _rodrigues(torch.from_numpy(np.asarray(d["global_orient"], np.float64)[None])).numpy()[0]
            go = rotmat_to_aa(c["Rc"] @ R_w)
            tr = c["Rc"] @ (J0 + np.asarray(d["transl"], np.float64)) + c["tc"] - J0
            pose = np.concatenate([go, np.asarray(d["body_pose"], np.float64)])
            with torch.no_grad():
                v, j = body(t_(pose[:3]), t_(pose[3:72]), t_(betas), t_(tr))
            uv = v[0].numpy().astype(np.float64) @ c["K"].T
            if (uv[:, 2] < 0.1).any():
                continue
            uv = uv[:, :2] / uv[:, 2:3]
            uj = j[0, SMPL_J].numpy().astype(np.float64) @ c["K"].T
            uj = uj[:, :2] / uj[:, 2:3]
            ref = kp2d[MHR70_J]
            ok = ref[:, 2] > 0 if kp2d.shape[1] > 2 else np.ones(len(ref), bool)
            d2 = np.median(np.linalg.norm(uj[ok] - ref[ok, :2], axis=1)) / max(box[2] - box[0], box[3] - box[1]) \
                if ok.sum() >= 4 else np.nan
            cand.append((iou(np.concatenate([uv.min(0), uv.max(0)]), box), d2, person, pose, betas, tr))
        if not cand:
            stats["no_fit"] += 1
            continue
        # the fit person whose limb joints land on the annotation's 2D keypoints (box IoU when they are missing)
        m_iou, m_kp, person, pose, betas, tr = min(cand, key=lambda x: (x[1] if np.isfinite(x[1]) else 9.0, -x[0]))
        img_key = f"{act}_{seq}_{cam}_{fr:05d}"
        img_path = out / "images" / f"{img_key}.jpg"
        if img_key not in written:
            if not img_path.exists():
                raw = src.read(join("exo", cam, "images", fname))
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.remap(img, *c["maps"], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written.add(img_key)
        s = {"id": f"{act}_{seq}/{cam}/{fr:05d}/{int(r['person_id'])}", "image": str(img_path),
             "bbox": box.astype(np.float32), "K": K_ann.astype(np.float32), "gender": "neutral",
             "global_orient": pose[:3].astype(np.float32), "body_pose": pose[3:72].astype(np.float32),
             "betas": betas.astype(np.float32), "transl": tr.astype(np.float32), "extrinsic": np.eye(4, dtype=np.float32),
             "match_iou": float(m_iou), "match_kp": float(m_kp), "fit_person": str(person), "k_err": c["k_err"],
             "kp2d": kp2d.astype(np.float32)}
        index.append(s)
        index_mhr.append(dict({k: v for k, v in s.items() if k not in ("global_orient", "body_pose", "betas", "transl")},
                              model_params=np.asarray(r["model_params"], np.float32),
                              shape_params=np.asarray(r["shape_params"], np.float32)))
        stats["kept"] += 1
        if stats["kept"] % 500 == 0:
            print(stats, flush=True)
    for name, idx in (("index.pkl", index), ("index_mhr.pkl", index_mhr)):
        with open(out / name, "wb") as f:
            pickle.dump(idx, f)
    ious = np.array([s["match_iou"] for s in index])
    kps = np.array([s["match_kp"] for s in index])
    stats.update(images=len(written), iou_median=float(np.median(ious)), iou_lt_05=int((ious < 0.5).sum()),
                 kp_median=float(np.nanmedian(kps)), kp_p95=float(np.nanpercentile(kps, 95)),
                 kp_gt_01=int((kps > 0.1).sum()), kp_nan=int(np.isnan(kps).sum()),
                 k_err_max=float(max(c["k_err"] for c in cams.values())),
                 scales=sorted({c["scale"] for c in cams.values()}), sequences=len({s["id"].split("/")[0] for s in index}))
    (out / "build_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(stats)


if __name__ == "__main__":
    main()
