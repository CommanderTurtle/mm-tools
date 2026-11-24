"""InterHand2.6M (5 fps, v1.0; https://mks0601.github.io/InterHand2.6M/, CC BY-NC 4.0) crop pool with NeuralAnnot MANO
labels: 27 subjects, ~80 calibrated cameras, single and interacting hands.

Labels follow the HOT3D pools (train.build_hot3d_crops --labels --verts): joints3d = kinematic MANO joints + HOT3D
fingertip vertices (bench.common.ManoLayer, OpenPose order) in the camera frame, kp2d = their projection, box = projected
visible mesh. NeuralAnnot params are in world coordinates (m), 48-d pose relative to the MANO mean pose (smplx
flat_hand_mean=False, left shapedirs fix: as InterHand2.6M's MANO_render/render.py); camera: x_cam = R (x_world - campos).
All three official splits are used (more subjects; never evaluate on InterHand2.6M with models trained on this pool).
Subjects are balanced: of a subject's n candidate images, those with crc32(<split>/<file_name>) % max(n // 4000, 1) == 0
are kept (--images_per_subject; spread over cameras / frames; the raw counts differ 30x between captures). A hand is
kept iff the image annotation covers it (hand_type, hand_type_valid, >= 10 valid joints), >= 30% of its mesh projects
inside the image, the box is >= 32 px and the fit agrees with the annotated 3D joints (mean non-tip error < --max_fit_mm).
src = "<split>/Capture<k>/<sequence>:<camera>/<frame>": LabeledCrops holds out whole subject-sequences.

    python -m train.build_interhand_crops --raw $DATA_ROOT/interhand_raw --out $DATA_ROOT/crops --verts
Images are streamed from the downloaded tar parts (<raw>/images/*.part??, cat-able into one tar): no extraction.
"""
import argparse
import glob
import json
import os
import subprocess
import tarfile
import zlib
from multiprocessing import Pool

import cv2
import numpy as np

from train.build_crops import crop_keypoints, render_crop

SPLITS = ("train", "val", "test")
# InterHand joint index (per hand: thumb4..1, index4..1, middle4..1, ring4..1, pinky4..1, wrist) -> OpenPose order
IH_TO_OPENPOSE = [20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16]
_MANO = {}


def load_split(raw, split):
    """-> {<split>/<file_name>: image info + annotated sides} for images with an annotated, NeuralAnnot-fitted hand."""
    a = os.path.join(raw, "annotations", split, f"InterHand2.6M_{split}")
    data = json.load(open(f"{a}_data.json"))
    mano = json.load(open(f"{a}_MANO_NeuralAnnot.json"))
    imgs = {im["id"]: im for im in data["images"]}
    want = {}
    for an in data["annotations"]:
        im = imgs[an["image_id"]]
        fit = mano.get(str(im["capture"]), {}).get(str(im["frame_idx"]))
        if not an["hand_type_valid"] or fit is None:
            continue
        jv = np.asarray(an["joint_valid"]).reshape(42)
        sides = {"right": [True], "left": [False], "interacting": [True, False]}[an["hand_type"]]
        # joint_valid: 0-20 right, 21-41 left
        sides = [s for s in sides if (jv[:21] if s else jv[21:]).sum() >= 10 and fit["right" if s else "left"] is not None]
        if sides:
            want[f"{split}/{im['file_name']}"] = dict(im, split=split, sides=sides)
    return want


def attach(raw, split, want):
    """Camera, NeuralAnnot fit and annotated 3D joints of the selected images of one split."""
    a = os.path.join(raw, "annotations", split, f"InterHand2.6M_{split}")
    cams = json.load(open(f"{a}_camera.json"))
    mano = json.load(open(f"{a}_MANO_NeuralAnnot.json"))
    j3d = json.load(open(f"{a}_joint_3d.json"))
    for w in want.values():
        c, f, cam = str(w["capture"]), str(w["frame_idx"]), w["camera"]
        cp = cams[c]
        w["cam"] = dict(campos=cp["campos"][cam], camrot=cp["camrot"][cam], focal=cp["focal"][cam], princpt=cp["princpt"][cam])
        w["mano"] = mano[c][f]
        jj = j3d.get(c, {}).get(f)
        w["j3d"] = None if jj is None else dict(world_coord=jj["world_coord"], joint_valid=jj["joint_valid"])


def _work(arg):
    key, jpg, w, out, mano_dir, verts, max_fit_mm = arg
    from bench.common import ManoLayer

    if mano_dir not in _MANO:  # one MANO layer per worker process
        _MANO[mano_dir] = ManoLayer(mano_dir)
    ml = _MANO[mano_dir]
    if w["mano"] is None:
        return []
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return []
    H, W = img.shape[:2]
    R = np.asarray(w["cam"]["camrot"], np.float64)
    t = np.asarray(w["cam"]["campos"], np.float64) / 1000
    K = np.array([[w["cam"]["focal"][0], 0, w["cam"]["princpt"][0]], [0, w["cam"]["focal"][1], w["cam"]["princpt"][1]], [0, 0, 1]])
    proj = lambda X: (X[:, :2] / np.clip(X[:, 2:], 1e-6, None)) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
    split, rest = key.split("/", 1)
    cap, seq, cam, fname = rest.split("/")
    recs = []
    for is_right in w["sides"]:
        p = w["mano"]["right" if is_right else "left"]
        if p is None:
            continue
        pose = np.asarray(p["pose"], np.float32)
        jw, vw = ml(is_right, p["shape"], pose[3:], pose[:3], p["trans"])
        fit_err = None
        if w["j3d"] is not None:  # NeuralAnnot fit vs annotated 3D joints (non-tip, valid only)
            g = np.asarray(w["j3d"]["world_coord"], np.float64).reshape(42, 3)[(0 if is_right else 21) + np.array(IH_TO_OPENPOSE)] / 1000
            gv = np.asarray(w["j3d"]["joint_valid"]).reshape(42)[(0 if is_right else 21) + np.array(IH_TO_OPENPOSE)] > 0
            m = gv.copy()
            m[[4, 8, 12, 16, 20]] = False
            if m.sum() >= 8:
                fit_err = float(np.linalg.norm(jw[m] - g[m], axis=1).mean() * 1000)
        if fit_err is None or fit_err > max_fit_mm:
            continue
        j = (R @ (jw - t).T).T
        v = (R @ (vw - t).T).T
        uv = proj(v)
        inside = (v[:, 2] > 0.01) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if inside.mean() < 0.3:
            continue
        b = np.array([*uv[inside].min(0), *uv[inside].max(0)], dtype=np.float32)
        if max(b[2] - b[0], b[3] - b[1]) < 32:
            continue
        crop, M = render_crop(img, b, is_right)
        rel = f"{split}/{cap}/{seq}/{cam}_{fname[5:-4]}_{'r' if is_right else 'l'}.jpg"
        os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True)
        cv2.imwrite(os.path.join(out, rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        j2 = proj(j)
        jin = (j2[:, 0] >= 0) & (j2[:, 0] < W) & (j2[:, 1] >= 0) & (j2[:, 1] < H)
        j3, Km = j.copy(), K.copy()
        if not is_right:  # GT in the right-hand (mirrored) frame, as build_hot3d_crops
            j3[:, 0] *= -1
            Km[0, 2] = W - 1 - Km[0, 2]
        rec = dict(file=rel, src=f"{split}/{cap}/{seq}:{cam}/{fname[5:-4]}", is_right=bool(is_right), box=[float(x) for x in b],
                   M=M.round(5).tolist(), hw=[H, W], sequence_id=f"{split}/{cap}/{seq}", subject=int(w["subject"]),
                   kp2d=crop_keypoints(np.concatenate([j2, jin[:, None]], 1), M, is_right, W).round(2).tolist(),
                   joints3d=j3.round(5).tolist(), K=Km.round(4).tolist(), fit_err_mm=round(fit_err, 2))
        if verts:
            v3 = v.copy()
            if not is_right:
                v3[:, 0] *= -1
            rec["verts_file"] = rel[:-4] + "_v.npy"
            np.save(os.path.join(out, rec["verts_file"]), v3.astype(np.float16))
        recs.append(rec)
    return recs


def stream(raw, want, args, out):
    parts = sorted(glob.glob(os.path.join(raw, "images", "InterHand2.6M.images.5.fps.v1.0.tar.part??")))
    parts = parts[: args.max_parts or len(parts)]
    cat = subprocess.Popen(["cat"] + parts, stdout=subprocess.PIPE, bufsize=1 << 24)
    tar = tarfile.open(fileobj=cat.stdout, mode="r|")

    def jobs():
        try:
            for m in tar:
                if not m.isfile() or "/images/" not in m.name:
                    continue
                key = m.name.split("/images/", 1)[1]
                w = want.get(key)
                if w is not None:
                    yield key, tar.extractfile(m).read(), w, out, args.mano_dir, args.verts, args.max_fit_mm
        except tarfile.ReadError:
            if not args.max_parts:  # truncated stream is expected only for smoke tests
                raise

    recs, n = [], 0
    with Pool(args.workers) as pool:
        for rs in pool.imap_unordered(_work, jobs(), chunksize=16):
            recs += rs
            n += 1
            if n % 20000 == 0:
                print(f"{n} images, {len(recs)} crops", flush=True)
    cat.kill()
    return recs, n


def main(args):
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)
    cand = {}
    for s in args.splits.split(","):
        cand.update(load_split(args.raw, s))
    by_subj = {}
    for k, w in cand.items():
        by_subj.setdefault(w["subject"], []).append(k)
    want = {}
    for subj, keys in sorted(by_subj.items()):  # balance subjects: ~images_per_subject images each (all if fewer)
        keep = max(len(keys) // args.images_per_subject, 1)
        sel = [k for k in keys if zlib.crc32(k.encode()) % keep == 0]
        want.update({k: cand[k] for k in sel})
        print(f"subject {subj}: {len(keys)} candidate images, keep 1/{keep} -> {len(sel)}", flush=True)
    del cand
    for s in args.splits.split(","):
        attach(args.raw, s, {k: w for k, w in want.items() if w["split"] == s})
    recs, n = stream(args.raw, want, args, out)
    with open(os.path.join(out, "index.jsonl"), "w") as f:
        for r in sorted(recs, key=lambda r: r["file"]):
            f.write(json.dumps(r) + "\n")
    print(f"{args.name}: {len(recs)} crops from {n} images ({len({r['subject'] for r in recs})} subjects) -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True, help="InterHand2.6M download (images/*.part??, annotations/<split>/)")
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="interhand_lab", help="output subdir / source name")
    p.add_argument("--images_per_subject", type=int, default=4000, help="sample ~this many images per subject")
    p.add_argument("--max_fit_mm", type=float, default=15.0)
    p.add_argument("--verts", action="store_true", help="also store the MANO vertices (778 x 3) per crop")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--splits", default=",".join(SPLITS))
    p.add_argument("--max_parts", type=int, default=0, help="smoke test: stream only the first tar parts")
    main(p.parse_args())
