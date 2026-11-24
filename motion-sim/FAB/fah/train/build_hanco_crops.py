"""HanCo (https://lmb.informatik.uni-freiburg.de/projects/contra-hand/; research only, no commercial use) crop pool from
the manually validated MANO fits: 32 subjects (the FreiHAND capture), 8 calibrated cameras at 5 Hz, 224x224 images,
right hands; only time steps with meta.json is_valid (9,055 of 107,538; the other fits are network predictions).

Read straight from the zips: HanCo_shape (MANO poses 48 / shapes 10 / global_t, cam0 = world frame; smplx with
flat_hand_mean=False and transl = global_t reproduces the xyz fingertips = FreiHAND tip vertices exactly),
HanCo_calib_meta (K, M = world -> camera per camera and time step, meta.json), HanCo_rgb_merged (green screen
replaced by random backgrounds). Labels / crops as train.build_hot3d_crops --labels --verts (joints3d = kinematic MANO
joints + HOT3D fingertip vertices, OpenPose order). A hand is kept iff >= 30% of its mesh projects inside the image and
the box is >= 32 px. src = "<sequence>:cam<k>/<frame>": LabeledCrops holds out whole sequences.

    python -m train.build_hanco_crops --raw $DATA_ROOT/hanco_raw --out $DATA_ROOT/crops --verts
"""
import argparse
import json
import os
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np

from train.build_crops import crop_keypoints, render_crop

_STATE = {}


def _zips(raw, rgb):
    if raw not in _STATE:  # one handle per zip and worker
        _STATE[raw] = {n: zipfile.ZipFile(os.path.join(raw, f"HanCo_{n}.zip")) for n in ("shape", "calib_meta", rgb)}
    return _STATE[raw]


def _work(arg):
    sid, fids, subject, raw, rgb, out, mano_dir, verts = arg
    from bench.common import ManoLayer

    done = os.path.join(out, f"{sid:04d}", ".done.json")
    if os.path.exists(done):  # resumable per sequence
        return json.load(open(done))
    if mano_dir not in _STATE:
        _STATE[mano_dir] = ManoLayer(mano_dir)
    ml, z = _STATE[mano_dir], _zips(raw, rgb)
    os.makedirs(os.path.join(out, f"{sid:04d}"), exist_ok=True)
    recs = []
    for fid in fids:
        s = json.loads(z["shape"].read(f"shape/{sid:04d}/{fid:08d}.json"))
        c = json.loads(z["calib_meta"].read(f"calib/{sid:04d}/{fid:08d}.json"))
        p = np.asarray(s["poses"][0], np.float32)
        jw, vw = ml(True, s["shapes"][0], p[3:], p[:3], np.asarray(s["global_t"]).reshape(3))
        for cam in range(len(c["K"])):
            name = f"{rgb}/{sid:04d}/cam{cam}/{fid:08d}.jpg"
            try:
                buf = z[rgb].read(name)
            except KeyError:
                continue
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            H, W = img.shape[:2]
            K, Mw = np.asarray(c["K"][cam], np.float64), np.asarray(c["M"][cam], np.float64)
            j, v = (Mw[:3, :3] @ jw.T + Mw[:3, 3:]).T, (Mw[:3, :3] @ vw.T + Mw[:3, 3:]).T
            proj = lambda X: (X[:, :2] / np.clip(X[:, 2:], 1e-6, None)) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
            uv = proj(v)
            inside = (v[:, 2] > 0.01) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if inside.mean() < 0.3:
                continue
            b = np.array([*uv[inside].min(0), *uv[inside].max(0)], dtype=np.float32)
            if max(b[2] - b[0], b[3] - b[1]) < 32:
                continue
            crop, M = render_crop(img, b, True)
            rel = f"{sid:04d}/cam{cam}_{fid:08d}_r.jpg"
            cv2.imwrite(os.path.join(out, rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            j2 = proj(j)
            jin = (j2[:, 0] >= 0) & (j2[:, 0] < W) & (j2[:, 1] >= 0) & (j2[:, 1] < H)
            rec = dict(file=rel, src=f"{sid:04d}:cam{cam}/{fid:08d}", is_right=True, box=[float(x) for x in b],
                       M=M.round(5).tolist(), hw=[H, W], sequence_id=f"{sid:04d}", subject=int(subject),
                       kp2d=crop_keypoints(np.concatenate([j2, jin[:, None]], 1), M, True, W).round(2).tolist(),
                       joints3d=j.round(5).tolist(), K=K.round(4).tolist())
            if verts:
                rec["verts_file"] = rel[:-4] + "_v.npy"
                np.save(os.path.join(out, rec["verts_file"]), v.astype(np.float16))
            recs.append(rec)
    json.dump(recs, open(done, "w"))
    return recs


def main(args):
    meta = json.loads(zipfile.ZipFile(os.path.join(args.raw, "HanCo_calib_meta.zip")).read("meta.json"))
    jobs = []
    for sid, (valid, subj) in enumerate(zip(meta["is_valid"], meta["subject_id"])):
        fids = [f for f, ok in enumerate(valid) if ok]
        if fids:
            jobs.append((sid, fids, subj[fids[0]], args.raw, args.rgb, os.path.join(args.out, args.name), args.mano_dir, args.verts))
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)
    recs = []
    with Pool(args.workers) as pool:
        for rs in pool.imap_unordered(_work, jobs, chunksize=4):
            recs += rs
    with open(os.path.join(out, "index.jsonl"), "w") as f:
        for r in sorted(recs, key=lambda r: r["file"]):
            f.write(json.dumps(r) + "\n")
    print(f"{args.name}: {len(recs)} crops from {len(jobs)} sequences ({len({r['subject'] for r in recs})} subjects) -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True, help="HanCo download (HanCo_{shape,calib_meta,rgb_merged}.zip)")
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="hanco_lab", help="output subdir / source name")
    p.add_argument("--rgb", default="rgb_merged", help="image variant (rgb = green screen, rgb_merged = backgrounds)")
    p.add_argument("--verts", action="store_true", help="also store the MANO vertices (778 x 3) per crop")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--workers", type=int, default=8)
    main(p.parse_args())
