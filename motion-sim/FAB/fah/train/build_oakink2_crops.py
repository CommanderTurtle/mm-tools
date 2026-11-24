"""OakInk2 (CVPR 2024, https://oakink.net/v2, HF kelvin34501/OakInk-v2, CC BY-SA 4.0) crop pool with mocap MANO labels:
bimanual hand-object manipulation, 4 synchronized views (egocentric head camera + 3 allocentric), 848x480 PNG at 30 fps.

Per sequence: data/<seq>.tar (<serial>/%06d.png, image frame ids = every 4th mocap frame) + anno_preview/<seq>.pkl
(cam_def serial -> view, per-frame cam_intr / cam_extr (world -> camera), raw_mano per mocap frame: {rh,lh}__pose_coeffs
16 quaternions (w first), __tsl, __betas). MANO convention of the toolkit (manotorch, rot_mode quat, flat_hand_mean,
center_idx=0, left shapedirs as shipped): world = MANO(pose, betas) - wrist + tsl. The pkl holds torch tensors; it is read
with a restricted unpickler (numpy / OrderedDict / torch tensors, deserialized lazily with the weights-only loader).
Labels / mirroring as train.build_hot3d_crops --labels --verts (joints3d = kinematic MANO joints + HOT3D fingertip
vertices, OpenPose order, camera frame). Every stride-th image frame per view, phase-shifted per view so the 4 views
sample different instants; a hand is kept iff >= 30% of its mesh projects inside the image, the box is >= 32 px and
>= --min_joints_in of the 21 joints project inside the image (oakink2_ego_lab: 15; oakink2_lab: 0).
src = "<seq>:<view>/<frame>": LabeledCrops holds out whole sequences. Subject = the A0xx / O0xx code of the sequence name;
subjects are balanced to ~--per_subject crops (crc32 of the crop path; dropped crops are deleted before packing).
--views egocentric reads <raw>/egocentric/<seq>.tar (train.fetch_oakink2_view) where the full tar is not downloaded.
--objects (extracted object_repair.tar): per crop obj_dist_mm = min distance of the hand mesh to the vertices of the
sequence's object meshes (obj_transf: object -> world per frame), tip_contact = fingertip vertices within 10 mm,
obj = nearest object; the per-subject cap then keeps hand-object contact crops (obj_dist_mm < 10) first, up to
--contact_frac of the cap.

    python -m train.build_oakink2_crops --raw $DATA_ROOT/oakink2_raw --out $DATA_ROOT/crops --stride 8 --verts
    python -m train.build_oakink2_crops --raw $DATA_ROOT/oakink2_raw --out $DATA_ROOT/crops --name oakink2_ego_lab \
        --views egocentric --stride 4 --min_joints_in 15 --per_subject 20000 \
        --objects $DATA_ROOT/oakink2_objects/object_repair/align_ds --verts
"""
import argparse
import glob
import io
import json
import os
import pickle
import tarfile
import zlib
from multiprocessing import Pool

import cv2
import numpy as np

from train.build_crops import crop_keypoints, render_crop

VIEWS = ("egocentric", "allocentric_top", "allocentric_left", "allocentric_right")
_MANO = {}
_SAFE = {("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"), ("numpy", "ndarray"),
         ("numpy", "dtype"), ("collections", "OrderedDict")}


class LazyTensor:
    """A pickled torch tensor, deserialized (weights-only) on .numpy(): most of the pkl (raw_smplx, unsampled frames)
    is never needed, and ~20k-150k torch.load calls per sequence otherwise dominate the build time."""

    def __init__(self, storage_bytes, *args):
        self.b, self.args = storage_bytes, args

    def numpy(self):
        import torch

        storage = torch.load(io.BytesIO(self.b), weights_only=True)
        return torch._utils._rebuild_tensor_v2(storage, *self.args).numpy()


class SafeUnpickler(pickle.Unpickler):
    tensors = True  # False: torch tensors load as None (fast metadata-only read, e.g. cam_def)

    def find_class(self, module, name):
        module = module.replace("numpy._core", "numpy.core")
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return LazyTensor if self.tensors else (lambda *a: None)
        if (module, name) == ("torch.storage", "_load_from_bytes"):  # nested torch pickle: kept as bytes until used
            return (lambda b: b) if self.tensors else (lambda b: None)
        if (module, name) in _SAFE:
            return getattr(__import__(module, fromlist=[name]), name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def load_anno(path, tensors=True):
    u = SafeUnpickler(open(path, "rb"))
    u.tensors = tensors
    return u.load()


def quat_to_aa(q):
    """(..., 4) unit quaternions, real part first -> axis-angle (..., 3)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(q[..., :1] < 0, -q, q)
    s = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    ang = 2 * np.arctan2(s, q[..., :1])
    return q[..., 1:] * np.where(s > 1e-8, ang / np.maximum(s, 1e-8), 2.0)


def mano_layer(mano_dir):
    if mano_dir not in _MANO:  # one layer per worker; manotorch keeps the left shapedirs as shipped
        import torch

        from bench.common import ManoLayer

        ml = ManoLayer(mano_dir, flat_hand_mean=True)
        sd = pickle.load(open(f"{mano_dir}/MANO_LEFT.pkl", "rb"), encoding="latin1")["shapedirs"]
        ml.layers[False].shapedirs[:] = torch.as_tensor(np.asarray(sd, np.float32))
        _MANO[mano_dir] = ml
    return _MANO[mano_dir]


def hand_world(ml, m, is_right):
    pre = "rh" if is_right else "lh"
    aa = quat_to_aa(m[f"{pre}__pose_coeffs"].numpy().reshape(16, 4).astype(np.float64)).astype(np.float32)
    j, v = ml(is_right, m[f"{pre}__betas"].numpy().reshape(10), aa[1:].reshape(45), aa[0], np.zeros(3))
    shift = m[f"{pre}__tsl"].numpy().reshape(3).astype(np.float64) - j[0]
    return j + shift, v + shift


def seq_tar(raw, seq, views):
    """Full data/<seq>.tar, else the single-view tar of train.fetch_oakink2_view; None if neither is complete."""
    if os.path.exists(os.path.join(raw, "data", f"{seq}.tar.done")):
        return os.path.join(raw, "data", f"{seq}.tar")
    if len(views) == 1 and os.path.exists(os.path.join(raw, views[0], f"{seq}.tar.done")):
        return os.path.join(raw, views[0], f"{seq}.tar")
    return None


def object_points(obj_dir, ids):
    """{object id: (KD-tree of its mesh vertices (m, object frame), bbox diagonal mm)} for the meshes that exist."""
    from scipy.spatial import cKDTree

    out = {}
    for o in ids:
        p = os.path.join(obj_dir, o, "model.obj")
        if os.path.exists(p):
            v = np.array([l.split()[1:4] for l in open(p) if l.startswith("v ")], np.float64)
            out[o] = (cKDTree(v), float(np.linalg.norm(v.max(0) - v.min(0)) * 1000))
    return out


def contact(objs, transf, fid, vw):
    """(min hand-vertex distance to any object (mm), fingertip vertices within 10 mm, nearest object id)."""
    from bench.common import MANO_TIP_VERTS

    best = (1e9, 0, None)
    for o, (tree, _) in objs.items():
        if transf.get(o, {}).get(fid) is None:
            continue
        T = np.asarray(transf[o][fid], np.float64)
        local = (vw - T[:3, 3]) @ T[:3, :3]  # world -> object frame
        d = tree.query(local)[0] * 1000
        if d.min() < best[0]:
            best = (float(d.min()), int((d[MANO_TIP_VERTS] < 10).sum()), o)
    return best


def _work(arg):
    seq, raw, out, stride, mano_dir, verts, views, obj_dir, min_joints_in = arg
    done = os.path.join(out, seq, ".done.json")
    if os.path.exists(done):  # resumable per sequence
        return json.load(open(done))
    ml = mano_layer(mano_dir)
    anno = load_anno(os.path.join(raw, "anno_preview", f"{seq}.pkl"))
    tar = tarfile.open(seq_tar(raw, seq, views))
    objs = object_points(obj_dir, anno["obj_list"]) if obj_dir else {}
    names = {n.split("/", 1)[1]: n for n in tar.getnames() if n.endswith(".png")}
    serial = {v: k for k, v in anno["cam_def"].items()}
    fids = anno["frame_id_list"]
    subject = seq.split("__")[1].split("++")[0]
    os.makedirs(os.path.join(out, seq), exist_ok=True)
    recs = []
    for vi, view in enumerate(views):
        for fid in fids[vi * stride // len(views) :: stride]:
            name = names.get(f"{serial[view]}/{fid:06d}.png")
            if name is None or fid not in anno["raw_mano"]:
                continue
            img = cv2.imdecode(np.frombuffer(tar.extractfile(name).read(), np.uint8), cv2.IMREAD_COLOR)
            H, W = img.shape[:2]
            K = np.asarray(anno["cam_intr"][view][fid], np.float64)
            E = np.asarray(anno["cam_extr"][view][fid], np.float64)
            proj = lambda X: (X[:, :2] / np.clip(X[:, 2:], 1e-6, None)) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
            for is_right in (True, False):
                jw, vw = hand_world(ml, anno["raw_mano"][fid], is_right)
                j, v = (E[:3, :3] @ jw.T + E[:3, 3:]).T, (E[:3, :3] @ vw.T + E[:3, 3:]).T
                uv = proj(v)
                inside = (v[:, 2] > 0.01) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                if inside.mean() < 0.3:
                    continue
                b = np.array([*uv[inside].min(0), *uv[inside].max(0)], dtype=np.float32)
                if max(b[2] - b[0], b[3] - b[1]) < 32:
                    continue
                j2 = proj(j)
                jin = (j2[:, 0] >= 0) & (j2[:, 0] < W) & (j2[:, 1] >= 0) & (j2[:, 1] < H)
                if jin.sum() < min_joints_in:  # truncated at the image border
                    continue
                crop, M = render_crop(img, b, is_right)
                rel = f"{seq}/{view}_{fid:06d}_{'r' if is_right else 'l'}.jpg"
                cv2.imwrite(os.path.join(out, rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                j3, Km = j.copy(), K.copy()
                if not is_right:  # GT in the right-hand (mirrored) frame, as build_hot3d_crops
                    j3[:, 0] *= -1
                    Km[0, 2] = W - 1 - Km[0, 2]
                rec = dict(file=rel, src=f"{seq}:{view}/{fid:06d}", is_right=is_right, box=[float(x) for x in b],
                           M=M.round(5).tolist(), hw=[H, W], sequence_id=seq, subject=subject, view=view,
                           kp2d=crop_keypoints(np.concatenate([j2, jin[:, None]], 1), M, is_right, W).round(2).tolist(),
                           joints3d=j3.round(5).tolist(), K=Km.round(4).tolist())
                if objs:
                    d, tips, o = contact(objs, anno["obj_transf"], fid, vw)
                    rec.update(obj_dist_mm=round(d, 1), tip_contact=tips, obj=o, obj_size_mm=round(objs[o][1], 1) if o else None)
                if verts:
                    v3 = v.copy()
                    if not is_right:
                        v3[:, 0] *= -1
                    rec["verts_file"] = rel[:-4] + "_v.npy"
                    np.save(os.path.join(out, rec["verts_file"]), v3.astype(np.float16))
                recs.append(rec)
    json.dump(recs, open(done, "w"))
    return recs


def main(args):
    views = args.views.split(",")
    seqs = sorted(os.path.basename(p)[:-9] for p in glob.glob(os.path.join(args.raw, "anno_preview", "*.pkl.done")))
    seqs = [s for s in seqs if seq_tar(args.raw, s, views)]
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)
    recs = []
    jobs = [(s, args.raw, out, args.stride, args.mano_dir, args.verts, views, args.objects, args.min_joints_in) for s in seqs]
    with Pool(args.workers) as pool:
        for k, rs in enumerate(pool.imap_unordered(_work, jobs)):
            recs += rs
            if k % 20 == 0:
                print(f"{k + 1}/{len(seqs)} sequences, {len(recs)} crops", flush=True)
    n_all, by_subj = len(recs), {}
    for r in recs:
        by_subj.setdefault(r["subject"], []).append(r)
    recs = []
    for subj, rs in sorted(by_subj.items()):  # balance subjects (sequence counts / lengths differ 10x): ~per_subject each
        if args.objects:  # contact crops first (up to contact_frac of the cap), each group in crc32 order
            h = lambda r: zlib.crc32(r["file"].encode())
            con = sorted((r for r in rs if r["obj_dist_mm"] < 10), key=h)
            far = sorted((r for r in rs if r["obj_dist_mm"] >= 10), key=h)
            n = min(len(rs), args.per_subject)
            nc = min(len(con), max(int(n * args.contact_frac), n - len(far)))
            kept = con[:nc] + far[: n - nc]
            msg = f"{len(con)} in contact, kept {nc} + {len(kept) - nc}"
        else:
            keep = -(-len(rs) // args.per_subject)
            kept = [r for r in rs if zlib.crc32(r["file"].encode()) % keep == 0]
            msg = f"keep 1/{keep}"
        kept_files = {r["file"] for r in kept}
        for r in rs:
            if r["file"] not in kept_files:  # dropped crops are not packed
                for k in ("file", "verts_file"):
                    if k in r and os.path.exists(os.path.join(out, r[k])):
                        os.remove(os.path.join(out, r[k]))
        print(f"subject {subj}: {len(rs)} crops, {msg} -> {len(kept)}", flush=True)
        recs += kept
    with open(os.path.join(out, "index.jsonl"), "w") as f:
        for r in sorted(recs, key=lambda r: r["file"]):
            f.write(json.dumps(r) + "\n")
    print(f"{args.name}: {len(recs)} of {n_all} crops from {len(seqs)} sequences ({len(by_subj)} subjects) -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True, help="OakInk2 download (data/<seq>.tar, anno_preview/<seq>.pkl + .done markers)")
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="oakink2_lab", help="output subdir / source name")
    p.add_argument("--stride", type=int, default=8, help="every stride-th image frame per view")
    p.add_argument("--verts", action="store_true", help="also store the MANO vertices (778 x 3) per crop")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--per_subject", type=int, default=12000, help="keep ~this many crops per subject")
    p.add_argument("--views", default=",".join(VIEWS))
    p.add_argument("--objects", default="", help="object_repair/align_ds dir: hand-object distances + contact-first cap")
    p.add_argument("--contact_frac", type=float, default=0.75, help="share of the per-subject cap for contact crops")
    p.add_argument("--min_joints_in", type=int, default=0, help="also require >= this many of the 21 joints in the image")
    p.add_argument("--workers", type=int, default=8)
    main(p.parse_args())
