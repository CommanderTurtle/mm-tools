"""ARCTIC (Fan et al., CVPR 2023, https://arctic.is.tue.mpg.de; MPI-IS non-commercial research licence) egocentric video
clips with per-frame contact GT, for the video contact model (train.contact_video) and its validation. ARCTIC is fetched by
each user under its licence; nothing of it is redistributed.

Hands / object / camera exactly as train.build_arctic_contact (ARCTIC's own MANO layers, as-shipped left hand; articulated
object per frame; world -> egocentric camera by the per-frame egocam; the released view-0 frames, 0.3 x 2800 x 2000,
undistorted with the scaled intrinsics + 8 distortion coefficients -> exact pinhole frames). Labels: train.contact_sdf (the
object mesh is rebuilt per frame, it articulates). One clip per sequence: every step-th frame (default 1 = 30 fps) over the
contiguous run of frames with an image, as bench.datasets.contactclip clips (frames/%06d.jpg, key = ARCTIC frame index;
image number = key + ioi_offset). Joints: MANO 16 + HOT3D fingertip vertices, OpenPose order (bench.common). Splits:
protocol p1 train -> train clips, val (s05) -> val clips (+ test/arctic_val_frames: per-frame sdf + z-buffer visibility).

Outputs (--out, the contact data root): video/arctic/clips/<sid>_<seq>/, video/arctic/split_{train,val}.json,
video_gt/arctic/<clip>.npz (per-frame GT through the loader), test/arctic_val_frames/<clip>.npz.

    python -m train.build_arctic_video --arctic $DATA_ROOT/arctic --out $DATA_ROOT/contact --split train
    python -m train.build_arctic_video --arctic $DATA_ROOT/arctic --out $DATA_ROOT/contact --split val
"""
import argparse
import json
import os
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np

CHUNK = 256


def _work(arg):
    sid, seq, root, step, frames_dir, C = arg
    import open3d as o3d
    import torch

    from bench.common import MANO21_TO_OPENPOSE, MANO_TIP_VERTS
    from bench.datasets.contactclip import ContactClip  # noqa: F401 (imported before layers() changes the working directory)
    from train.build_arctic_contact import DATA, EGO_SCALE, ZIPS, layers, mkdir
    from train.build_hoi4d_contact import video_gt
    from train.contact_sdf import MeshSDF

    torch.set_num_threads(1)
    name = f"{sid}_{seq}"
    d = os.path.join(root, "clips", name)
    gt_path = os.path.join(C, "video_gt", "arctic", name + ".npz")
    if os.path.exists(os.path.join(d, "meta.json")) and os.path.exists(gt_path):
        return name
    L = layers()
    raw = os.path.join(DATA, "raw_seqs", sid)
    mano = np.load(os.path.join(raw, f"{seq}.mano.npy"), allow_pickle=True).item()  # ARCTIC's own release files (as build_arctic_contact)
    objp = np.load(os.path.join(raw, f"{seq}.object.npy"), allow_pickle=True)
    ego = np.load(os.path.join(raw, f"{seq}.egocam.dist.npy"), allow_pickle=True).item()
    off = json.load(open(os.path.join(DATA, "meta", "misc.json")))[sid]["ioi_offset"]
    try:
        zf = zipfile.ZipFile(os.path.join(ZIPS, sid, f"{seq}.zip"))
    except (zipfile.BadZipFile, OSError) as e:
        print("skip", sid, seq, type(e).__name__, flush=True)
        return None
    names = set(zf.namelist())
    K = np.asarray(ego["intrinsics"], np.float64).copy()
    K[:2] *= EGO_SCALE
    dist = np.asarray(ego["dist8"], np.float64).reshape(-1)
    T = len(mano["right"]["rot"])
    have = [f for f in range(T) if f"0/{f + off:05d}.jpg" in names]
    if not have:
        return None
    f0 = have[0]
    f1 = f0
    while f1 + 1 < T and f"0/{f1 + 1 + off:05d}.jpg" in names:  # contiguous run from the first frame with an image
        f1 += 1
    fr = np.arange(f0, f1 + 1, step)
    R, Tc = np.asarray(ego["R_k_cam_np"], np.float64), np.asarray(ego["T_k_cam_np"], np.float64).reshape(-1, 3)
    t = lambda x: torch.as_tensor(np.asarray(x, np.float32))
    name_obj = seq.split("_")[0]
    mkdir(os.path.join(d, "frames"))
    rows = {k: [] for k in ("frame", "is_right", "verts", "joints", "sdf", "valid", "T_obj", "vis")}
    H = W = None
    for c0 in range(0, len(fr), CHUNK):
        ch = fr[c0:c0 + CHUNK]
        with torch.no_grad():
            hv, hj = {}, {}
            for side in ("right", "left"):
                m = mano[side]
                o = L[side](global_orient=t(m["rot"])[ch], hand_pose=t(m["pose"])[ch], betas=t(m["shape"])[None].repeat(len(ch), 1),
                            transl=t(m["trans"])[ch])
                hv[side] = o.vertices.numpy().astype(np.float64)
                hj[side] = o.joints[:, :16].numpy().astype(np.float64)
            oo = L["obj"](angles=t(objp[ch, :1]).view(-1, 1), global_orient=t(objp[ch, 1:4]), transl=t(objp[ch, 4:7]) / 1000,
                          query_names=[name_obj] * len(ch))
            vo, fo = oo["v"].numpy().astype(np.float64), oo["f"].numpy()
            vlen, flen = int(oo["v_len"][0]), int(oo["f_len"][0])
        for n, f in enumerate(ch):
            cam = lambda X: X @ R[f].T + Tc[f]
            img = cv2.imdecode(np.frombuffer(zf.read(f"0/{f + off:05d}.jpg"), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.undistort(img, K, dist)
            H, W = img.shape[:2]
            cv2.imwrite(os.path.join(d, "frames", f"{f:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            Fo = fo[0, :flen] if fo.ndim == 3 else fo[:flen]
            obj = MeshSDF(cam(vo[n, :vlen]), Fo)  # camera frame
            for side in ("right", "left"):
                v = cam(hv[side][n])
                j = np.concatenate([cam(hj[side][n]), v[MANO_TIP_VERTS]], 0)[MANO21_TO_OPENPOSE]
                vis = np.zeros(778, bool)
                if frames_dir:  # z-buffer over hand + object, 2 mm, inside the image
                    sc = o3d.t.geometry.RaycastingScene(nthreads=1)
                    sc.add_triangles(o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(L[side].faces.astype(np.uint32)))
                    dd = np.linalg.norm(v, axis=1)
                    rays = np.concatenate([np.zeros_like(v), v / dd[:, None]], 1).astype(np.float32)
                    th = np.minimum(sc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy(), obj.sc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy())
                    uv = v[:, :2] / np.clip(v[:, 2:], 1e-6, None) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
                    ins = (v[:, 2] > 0.01) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                    vis = (th >= dd - 0.002) & ins
                for k, x in (("frame", int(f)), ("is_right", side == "right"), ("verts", v.astype(np.float32)), ("joints", j.astype(np.float32)),
                             ("sdf", obj(v).astype(np.float16)), ("valid", True), ("T_obj", np.eye(4, dtype=np.float32)), ("vis", vis)):
                    rows[k].append(x)
    gt = {k: np.asarray(v) for k, v in rows.items()}
    np.savez(os.path.join(d, "gt.tmp.npz"), **gt)
    os.replace(os.path.join(d, "gt.tmp.npz"), os.path.join(d, "gt.npz"))
    meta = dict(dataset="arctic", sequence=f"{sid}/{seq}", K=K.round(5).tolist(), hw=[H, W], fps=30 / step, first=int(fr[0]), last=int(fr[-1]),
                step=step, frames="frames/%06d.jpg", obj=name_obj, ioi_offset=off, mano_frames=T, split="val" if frames_dir else "train",
                licence="ARCTIC non-commercial research (MPI-IS)")
    json.dump(meta, open(os.path.join(d, "meta.tmp.json"), "w"))
    os.replace(os.path.join(d, "meta.tmp.json"), os.path.join(d, "meta.json"))
    video_gt(d, gt_path, frames_dir and os.path.join(frames_dir, name + ".npz"))
    return name


def main(a):
    from train import build_arctic_contact as A
    from train.build_hoi4d_contact import video_stats

    A.configure(a.arctic, a.out)
    DATA, ZIPS, C, mkdir = A.DATA, A.ZIPS, A.C, A.mkdir
    split = json.load(open(os.path.join(DATA, "splits_json", "protocol_p1.json")))[a.split]
    if a.sequences:  # only these (e.g. sample sequences for a parity check)
        split = [s for s in split if s in set(a.sequences.split(","))]
    a.out = ""
    root = a.out or os.path.join(C, "video", "arctic")
    fdir = os.path.join(C, "test", "arctic_val_frames") if a.split == "val" else None
    for x in (os.path.join(root, "clips"), os.path.join(C, "video_gt", "arctic"), fdir):
        if x:
            mkdir(x)
    jobs = [(s.split("/")[0], s.split("/")[1], root, a.step, fdir, C) for s in split if os.path.exists(os.path.join(ZIPS, s + ".zip"))]
    print(f"{a.split}: {len(jobs)} / {len(split)} sequences with images", flush=True)
    done = []
    with Pool(a.workers) as p:
        for k, r in enumerate(p.imap_unordered(_work, jobs)):
            if r:
                done.append(r)
            if k % 20 == 0:
                print(f"{k}/{len(jobs)} clips", flush=True)
    json.dump(dict(dataset="arctic", split=a.split, clips=sorted(done)), open(os.path.join(root, f"split_{a.split}.json"), "w"), indent=1)
    st = video_stats(os.path.join(C, "video_gt", "arctic"), sorted(done))
    json.dump(st, open(os.path.join(root, f"stats_{a.split}.json"), "w"), indent=1)
    print(json.dumps(st), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="train", help="protocol p1 split: train | val")
    p.add_argument("--step", type=int, default=1, help="frame step inside a clip (1 = 30 fps)")
    p.add_argument("--arctic", default="", help="the ARCTIC repo checkout with its data (default $ARCTIC_ROOT or data/arctic)")
    p.add_argument("--out", default="", help="the contact data root (default $CONTACT_ROOT or data/contact)")
    p.add_argument("--sequences", default="", help="only these split sequences (comma-separated <subject>/<sequence>)")
    p.add_argument("--workers", type=int, default=64)
    main(p.parse_args())
