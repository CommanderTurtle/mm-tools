"""HO-3D v3 (https://github.com/shreyashampali/ho3d; academic non-commercial use, no redistribution) crop pool with MANO
labels: 55 train sequences, 10 subjects, right hands grasping YCB objects, 640x480, read straight from HO3D_v3.zip.

Annotated frames = train.txt (<sequence>/<id>); meta/<id>.pkl (numpy only; read with a restricted unpickler) holds
handPose (48, full axis-angle), handBeta, handTrans and handJoints3D (16 MANO joints + 5 tips) in an OpenGL camera
frame (y up, z backwards): x_cv = x_gl * (1, -1, -1). MANO convention checked against handJoints3D (CHECK_MM).
Labels / crops as train.build_hot3d_crops --labels --verts (joints3d = kinematic MANO joints + HOT3D fingertip
vertices, OpenPose order). Every stride-th annotated frame per sequence; >= 30% of the mesh inside the image, box >= 32
px. src = "<sequence>:<id>": LabeledCrops holds out whole sequences.

    python -m train.build_ho3d_crops --zip $DATA_ROOT/ho3d_raw/HO3D_v3.zip --out $DATA_ROOT/crops --stride 3 --verts
"""
import argparse
import io
import json
import os
import pickle
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np

from train.build_crops import crop_keypoints, render_crop

GL_TO_CV = np.array([1.0, -1.0, -1.0])
CHECK_MM = 1.0  # max |MANO kinematic joints - handJoints3D[:16]| accepted (mm)
_MANO = {}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        module = module.replace("numpy._core", "numpy.core")
        if (module, name) in {("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
                              ("numpy", "ndarray"), ("numpy", "dtype"), ("_codecs", "encode")}:  # py2 pickles
            return getattr(__import__(module, fromlist=[name]), name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def load_meta(z, root, seq, fid):
    return SafeUnpickler(io.BytesIO(z.read(f"{root}train/{seq}/meta/{fid}.pkl")), encoding="latin1").load()


def _work(arg):
    seq, fids, zpath, out, mano_dir, verts = arg
    from bench.common import ManoLayer

    done = os.path.join(out, seq, ".done.json")
    if os.path.exists(done):  # resumable per sequence
        return json.load(open(done))
    if mano_dir not in _MANO:
        _MANO[mano_dir] = ManoLayer(mano_dir, flat_hand_mean=True)
    ml = _MANO[mano_dir]
    z = zipfile.ZipFile(zpath)
    root = "" if "train.txt" in z.namelist() else z.namelist()[0].split("/")[0] + "/"
    os.makedirs(os.path.join(out, seq), exist_ok=True)
    recs = []
    for fid in fids:
        a = load_meta(z, root, seq, fid)
        if a.get("handPose") is None:
            continue
        pose = np.asarray(a["handPose"], np.float32).reshape(48)
        j, v = ml(True, np.asarray(a["handBeta"]).reshape(10), pose[3:], pose[:3], np.asarray(a["handTrans"]).reshape(3))
        err = np.abs(j[[0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]] - np.asarray(a["handJoints3D"])[:16]).max() * 1000
        if err > CHECK_MM:
            print(f"skip {seq}/{fid}: MANO vs handJoints3D {err:.2f} mm", flush=True)
            continue
        j, v = j * GL_TO_CV, v * GL_TO_CV
        K = np.asarray(a["camMat"], np.float64)
        img = cv2.imdecode(np.frombuffer(z.read(f"{root}train/{seq}/rgb/{fid}.jpg"), np.uint8), cv2.IMREAD_COLOR)
        H, W = img.shape[:2]
        proj = lambda X: (X[:, :2] / np.clip(X[:, 2:], 1e-6, None)) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
        uv = proj(v)
        inside = (v[:, 2] > 0.01) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if inside.mean() < 0.3:
            continue
        b = np.array([*uv[inside].min(0), *uv[inside].max(0)], dtype=np.float32)
        if max(b[2] - b[0], b[3] - b[1]) < 32:
            continue
        crop, M = render_crop(img, b, True)
        rel = f"{seq}/{fid}_r.jpg"
        cv2.imwrite(os.path.join(out, rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        j2 = proj(j)
        jin = (j2[:, 0] >= 0) & (j2[:, 0] < W) & (j2[:, 1] >= 0) & (j2[:, 1] < H)
        rec = dict(file=rel, src=f"{seq}:{fid}", is_right=True, box=[float(x) for x in b], M=M.round(5).tolist(), hw=[H, W],
                   sequence_id=seq, kp2d=crop_keypoints(np.concatenate([j2, jin[:, None]], 1), M, True, W).round(2).tolist(),
                   joints3d=j.round(5).tolist(), K=K.round(4).tolist())
        if verts:
            rec["verts_file"] = rel[:-4] + "_v.npy"
            np.save(os.path.join(out, rec["verts_file"]), v.astype(np.float16))
        recs.append(rec)
    json.dump(recs, open(done, "w"))
    return recs


def main(args):
    z = zipfile.ZipFile(args.zip)
    root = "" if "train.txt" in z.namelist() else z.namelist()[0].split("/")[0] + "/"
    by_seq = {}
    for line in z.read(f"{root}train.txt").decode().split():
        seq, fid = line.strip().split("/")
        by_seq.setdefault(seq, []).append(fid)
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)
    jobs = [(s, sorted(f)[:: args.stride], args.zip, out, args.mano_dir, args.verts) for s, f in sorted(by_seq.items())]
    recs = []
    with Pool(args.workers) as pool:
        for rs in pool.imap_unordered(_work, jobs):
            recs += rs
    with open(os.path.join(out, "index.jsonl"), "w") as f:
        for r in sorted(recs, key=lambda r: r["file"]):
            f.write(json.dumps(r) + "\n")
    print(f"{args.name}: {len(recs)} crops from {len(by_seq)} sequences -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", required=True, help="HO3D_v3.zip")
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="ho3d_lab", help="output subdir / source name")
    p.add_argument("--stride", type=int, default=3, help="every stride-th annotated frame per sequence")
    p.add_argument("--verts", action="store_true", help="also store the MANO vertices (778 x 3) per crop")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--workers", type=int, default=8)
    main(p.parse_args())
