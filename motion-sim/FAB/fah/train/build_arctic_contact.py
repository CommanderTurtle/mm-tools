"""ARCTIC (Fan et al., CVPR 2023, https://arctic.is.tue.mpg.de; MPI-IS non-commercial research licence) egocentric crop pool
with contact labels (train.contact_sdf: signed distance of each GT MANO vertex to the object surface, inside negative), for
the contact head (5 / 10 / 20 / 35 mm channels from the stored distances). ARCTIC is fetched by each user under its licence
(its repo's download + unpack scripts); nothing of it is redistributed.

Hands / object from ARCTIC's own layers (its repo: common.body_models MANO, axis-angle, flat_hand_mean False, left hand as
shipped; common.object_tensors with the per-frame articulation), world -> egocentric camera by its per-frame egocam
(R_k_cam, T_k_cam). Images: the released "cropped" egocentric frames (view 0 = the full 2800 x 2000 frame scaled by 0.3,
distorted), read from the per-sequence zips (image number = frame + ioi_offset), undistorted with the scaled intrinsics and
the 8 distortion coefficients -> an exact pinhole view; crops / mirroring / records as the other pools
(train.build_crops.render_crop; verts in the right-hand frame, float16). A hand is kept iff >= 30 % of its mesh projects
inside the image and the box is >= 32 px. Every stride-th frame. Splits: ARCTIC protocol p1 -- train sequences -> the pool,
val -> a held-out test set (test/arctic_val; ARCTIC's test split has no public GT).

--arctic: the ARCTIC repo checkout with its data unpacked as its scripts do (unpack/arctic_data/data/{raw_seqs,meta,
splits_json}, downloads/data/cropped_images_zips/<subject>/<sequence>.zip); --out: the contact data root.

    python -m train.build_arctic_contact --arctic $DATA_ROOT/arctic --out $DATA_ROOT/contact --split train --stride 8
    python -m train.build_arctic_contact --arctic $DATA_ROOT/arctic --out $DATA_ROOT/contact --split val --stride 8
"""
import argparse
import json
import os
import sys
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np

REPO = os.path.abspath(os.environ.get("ARCTIC_ROOT", os.path.join("data", "arctic")))  # --arctic (configure)
DATA = os.path.join(REPO, "unpack", "arctic_data", "data")
ZIPS = os.path.join(REPO, "downloads", "data", "cropped_images_zips")
C = os.environ.get("CONTACT_ROOT", os.path.join("data", "contact"))  # --out (configure)
EGO_SCALE = 0.3
_L = {}


def configure(arctic="", out=""):
    """The ARCTIC checkout / output roots (before any worker forks: workers inherit them)."""
    global REPO, DATA, ZIPS, C
    if arctic:
        REPO = os.path.abspath(arctic)
        DATA, ZIPS = os.path.join(REPO, "unpack", "arctic_data", "data"), os.path.join(REPO, "downloads", "data", "cropped_images_zips")
    C = out or C


def layers():
    if not _L:
        np.bool = bool  # ARCTIC's object loader predates numpy 1.24
        os.chdir(REPO)
        sys.path.insert(0, REPO)
        import torch

        from common.body_models import build_mano_aa
        from common.object_tensors import ObjectTensors

        torch.set_num_threads(1)
        _L.update(right=build_mano_aa(True, create_transl=True, flat_hand=False), left=build_mano_aa(False, create_transl=True, flat_hand=False),
                  obj=ObjectTensors())
    return _L


def mesh_sdf(V, F, pts):
    """Signed distance (mm) of pts to the triangle mesh (V, F): exact point-to-triangle, negative inside (winding number on
    every point in the mesh's box: train.contact_sdf)."""
    from train.contact_sdf import mesh_sdf as sdf

    return sdf(V, F, pts)


def mkdir(d):
    try:  # makedirs(exist_ok=True) can still raise EEXIST when another worker creates it at the same time
        os.makedirs(d, exist_ok=True)
    except FileExistsError:
        pass


def _work(arg):
    sid, seq, out, stride = arg
    import torch

    from train.build_crops import render_crop

    done = os.path.join(out, sid, seq, ".done.json")
    if os.path.exists(done):
        return json.load(open(done))
    L = layers()
    raw = os.path.join(DATA, "raw_seqs", sid)
    mano = np.load(os.path.join(raw, f"{seq}.mano.npy"), allow_pickle=True).item()
    objp = np.load(os.path.join(raw, f"{seq}.object.npy"), allow_pickle=True)
    ego = np.load(os.path.join(raw, f"{seq}.egocam.dist.npy"), allow_pickle=True).item()
    misc = json.load(open(os.path.join(DATA, "meta", "misc.json")))[sid]
    try:
        zf = zipfile.ZipFile(os.path.join(ZIPS, sid, f"{seq}.zip"))
    except (zipfile.BadZipFile, OSError) as e:  # still downloading: skipped now (no .done), labelled on a later pass
        print("skip", sid, seq, type(e).__name__, flush=True)
        return []
    names = set(zf.namelist())
    K = np.asarray(ego["intrinsics"], np.float64).copy()
    K[:2] *= EGO_SCALE
    dist = np.asarray(ego["dist8"], np.float64).reshape(-1)
    T = len(mano["right"]["rot"])
    fr = np.arange(0, T, stride)
    t = lambda x: torch.as_tensor(np.asarray(x, np.float32))
    with torch.no_grad():
        verts = {}
        for side in ("right", "left"):
            m = mano[side]
            o = L[side](global_orient=t(m["rot"])[fr], hand_pose=t(m["pose"])[fr], betas=t(m["shape"])[None].repeat(len(fr), 1),
                        transl=t(m["trans"])[fr])
            verts[side] = o.vertices.numpy().astype(np.float64)
        name = seq.split("_")[0]
        oo = L["obj"](angles=t(objp[fr, :1]).view(-1, 1), global_orient=t(objp[fr, 1:4]), transl=t(objp[fr, 4:7]) / 1000,
                      query_names=[name] * len(fr))
        vo, fo = oo["v"].numpy().astype(np.float64), oo["f"].numpy()
        vlen, flen = int(oo["v_len"][0]), int(oo["f_len"][0])
    R, Tc = np.asarray(ego["R_k_cam_np"], np.float64), np.asarray(ego["T_k_cam_np"], np.float64).reshape(-1, 3)
    mkdir(os.path.join(out, sid, seq))
    recs = []
    for n, f in enumerate(fr):
        img_name = f"0/{f + misc['ioi_offset']:05d}.jpg"
        if img_name not in names:
            continue
        cam = lambda X: X @ R[f].T + Tc[f]
        img = cv2.imdecode(np.frombuffer(zf.read(img_name), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.undistort(img, K, dist)
        H, W = img.shape[:2]
        Vo, Fo = cam(vo[n, :vlen]), fo[0, :flen] if fo.ndim == 3 else fo[:flen]
        for side in ("right", "left"):
            v = cam(verts[side][n])
            uv = v[:, :2] / np.clip(v[:, 2:], 1e-6, None) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
            inside = (v[:, 2] > 0.01) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if inside.mean() < 0.3:
                continue
            b = np.array([*uv[inside].min(0), *uv[inside].max(0)], np.float32)
            if max(b[2] - b[0], b[3] - b[1]) < 32:
                continue
            is_right = side == "right"
            crop, M = render_crop(img, b, is_right)
            rel = f"{sid}/{seq}/{f:05d}_{side[0]}.jpg"
            cv2.imwrite(os.path.join(out, rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            Km, v3 = K.copy(), v.copy()
            if not is_right:  # right-hand (mirrored) frame, as every pool
                Km[0, 2] = W - 1 - Km[0, 2]
                v3[:, 0] *= -1
            np.save(os.path.join(out, rel[:-4] + "_v.npy"), v3.astype(np.float16))
            sd = mesh_sdf(Vo, Fo, v)
            recs.append(dict(file=rel, src=f"{sid}/{seq}:{f:05d}", is_right=is_right, box=[float(x) for x in b], M=M.round(5).tolist(),
                             hw=[H, W], K=Km.round(4).tolist(), sequence_id=f"{sid}/{seq}", verts_file=rel[:-4] + "_v.npy",
                             sdf=sd.astype(np.float16).tolist(), visible=float(inside.mean())))
    json.dump(recs, open(done, "w"))
    return recs


def main(a):
    split = json.load(open(os.path.join(DATA, "splits_json", "protocol_p1.json")))[a.split]
    if a.sequences:  # only these (e.g. sample sequences for a parity check)
        split = [s for s in split if s in set(a.sequences.split(","))]
    jobs = [(s.split("/")[0], s.split("/")[1], a.out, a.stride) for s in split if os.path.exists(os.path.join(ZIPS, s + ".zip"))]
    print(f"{a.split}: {len(jobs)} / {len(split)} sequences with images", flush=True)
    mkdir(a.out)
    recs = []
    with Pool(a.workers) as p:
        for k, rs in enumerate(p.imap_unordered(_work, jobs)):
            recs += rs
            if k % 20 == 0:
                print(f"{k}/{len(jobs)} sequences, {len(recs)} crops", flush=True)
    recs.sort(key=lambda r: r["file"])
    sdf = np.asarray([r.pop("sdf") for r in recs], np.float16)
    with open(os.path.join(a.out, "index.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    lab = a.labels
    mkdir(lab)
    np.save(os.path.join(lab, "sdf.npy"), sdf)
    with open(os.path.join(lab, "contact.jsonl"), "w") as f:
        for r, d in zip(recs, sdf.astype(np.float32)):
            f.write(json.dumps(dict(file=r["file"], src=r["src"], is_right=r["is_right"], valid=True, min_mm=round(float(d.min()), 2),
                                    n10=int((d < 10).sum()), obj=r["sequence_id"].split("/")[1].split("_")[0])) + "\n")
    n = (sdf.astype(np.float32) < 10).sum(1)
    print(json.dumps(dict(split=a.split, crops=len(recs), hands_in_contact=float((n > 0).mean()) if len(n) else None,
                          contact_vertex_frac=float(n.mean() / 778) if len(n) else None)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="train", help="protocol p1 split: train | val")
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--arctic", default="", help="the ARCTIC repo checkout with its data (default $ARCTIC_ROOT or data/arctic)")
    p.add_argument("--out", default="", help="the contact data root (default $CONTACT_ROOT or data/contact)")
    p.add_argument("--pool", default="", help="crop pool dir (default <out>/pools/arctic_ego or <out>/test/arctic_val)")
    p.add_argument("--labels", default="", help="label dir (default <out>/arctic_ego or the test dir)")
    p.add_argument("--sequences", default="", help="only these split sequences (comma-separated <subject>/<sequence>)")
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    configure(a.arctic, a.out)
    a.out = a.pool or (os.path.join(C, "pools", "arctic_ego") if a.split == "train" else os.path.join(C, "test", "arctic_val"))
    if not a.labels:
        a.labels = os.path.join(C, "arctic_ego") if a.split == "train" else a.out
    main(a)
