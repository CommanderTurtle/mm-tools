"""ARCTIC (Fan et al., CVPR 2023, https://arctic.is.tue.mpg.de; MPI non-commercial research licence, no redistribution)
crop pool with MoCap MANO labels: bimanual manipulation of 11 articulated objects, 10 subjects, 1 egocentric (head-mounted,
moving, lens distortion) + 8 static allocentric views.

Layout (arctic_data/data/): raw_seqs/<sid>/<seq>.{mano,egocam.dist}.npy (pickled dicts; read with a restricted unpickler),
meta/misc.json (per subject: world2cam / intris_mat of the 8 static views, image_size of all 9, ioi_offset), splits_json/
protocol_p{1,2}.json (train / val / test sequence lists; test GT is withheld), cropped_images/<sid>/<seq>/<view>/<frame>.jpg
(view 0 = ego downscaled x EGO_SCALE; views 1-8 = 1.5x object-centred box resized to 1000 px, box recomputed by allo_bbox).
Conventions (ARCTIC common/body_models.py, src/arctic/processing.py, scripts_data/crop_images.py):
- MANO: smplx, use_pca=False, flat_hand_mean=False, per-subject shape, params in world (m). Left hands use the left
  shapedirs as shipped (--left_fix 0): ARCTIC's own layers never apply the smplx left-shapedirs fix that
  bench.common.ManoLayer applies, and its fits only match the images that way (with the fix the left mesh moves by up to
  22 mm on s02; checked on image overlays).
- ego camera: X_cam = R_k_cam_np X_world + T_k_cam_np per MoCap frame, K = intrinsics (2800 x 2000 px), distortion dist8 =
  OpenCV rational (k1 k2 p1 p2 k3 k4 k5 k6) on the raw image. Crops are rectified to the pinhole camera K: every crop
  pixel is mapped through the inverse crop affine, K^-1, the distortion and K into the source image (cv2.remap), so the
  labels stay exact pinhole projections (joints3d / K / kp2d as every other pool).
- static cameras: X_cam = world2cam X_world, K = intris_mat (undistorted).
- image file <frame>.jpg <-> MoCap frame int(frame) - ioi_offset (negative / beyond the sequence = no GT).
Labels / mirroring as train.build_hot3d_crops --labels --verts (joints3d = kinematic MANO joints + HOT3D fingertip
vertices 744/320/443/554/671, OpenPose order; ARCTIC's own 21-joint tips are other vertices). A hand is kept iff >= 30% of
its mesh projects inside the image, its box is >= 32 px and >= --min_joints_in joints are inside. src = "<sid>/<seq>:<view>/
<frame>": LabeledCrops holds out whole sequences; --split picks protocol p2 (ego) / p1 (allocentric) sequence lists.

    python -m train.build_arctic_crops --root $DATA_ROOT/arctic/unpack/arctic_data/data --out $DATA_ROOT/crops --views 0 --split train --verts
    python -m train.build_arctic_crops --root ... --out <check dir> --check     (conventions vs ARCTIC's processed joints)
"""
import argparse
import glob
import json
import os
import pickle
import zipfile
import zlib
from multiprocessing import Pool

import cv2
import numpy as np

from train.build_crops import CONTEXT, SIZE, crop_keypoints, render_crop, teacher_region

EGO_SCALE = 0.3  # scripts_data/crop_images.py: ego images are downscaled, not cropped (K scale = stored / original width)
ALLO_CAP = 1000  # allocentric crops: 1.5x the processed bbox, resized to 1000 px
_MANO = {}
_SAFE = {("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"), ("numpy", "ndarray"), ("numpy", "dtype"),
         ("_codecs", "encode")}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        module = module.replace("numpy._core", "numpy.core")
        if (module, name) in _SAFE:
            return getattr(__import__(module, fromlist=[name]), name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def safe_npy(path):
    """np.load(path, allow_pickle=True).item() without running arbitrary pickle code (numpy arrays / dicts only)."""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read(f)
        if not dtype.hasobject:
            f.seek(0)
            return np.load(f, allow_pickle=False)
        arr = SafeUnpickler(f, encoding="latin1").load()
    return arr.item() if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == () else arr


def distort(xy, d):
    """Normalised pinhole coords (N, 2) -> distorted normalised coords, OpenCV rational model d = k1 k2 p1 p2 k3 k4 k5 k6."""
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    r4, r6 = r2 * r2, r2 * r2 * r2
    rad = (1 + d[0] * r2 + d[1] * r4 + d[4] * r6) / (1 + d[5] * r2 + d[6] * r4 + d[7] * r6)
    return np.stack([x * rad + 2 * d[2] * x * y + d[3] * (r2 + 2 * x * x), y * rad + 2 * d[3] * x * y + d[2] * (r2 + 2 * y * y)], 1)


def project(X, K):
    return X[:, :2] / np.clip(X[:, 2:], 1e-6, None) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]


def rectified_crop(img, box, is_right, K, dist, W):
    """render_crop of the pinhole image of a distorted camera: same affine M (pinhole px -> crop px, mirrored for left
    hands), pixels fetched from the distorted source image."""
    if not is_right:
        box = np.array([W - box[2] - 1, box[1], W - box[0] - 1, box[3]], dtype=np.float32)
    c, side = teacher_region(box)
    side *= CONTEXT
    s = SIZE / side
    M = np.array([[s, 0, SIZE / 2 - s * c[0]], [0, s, SIZE / 2 - s * c[1]]], dtype=np.float32)
    g = np.stack(np.meshgrid(np.arange(SIZE, dtype=np.float64), np.arange(SIZE, dtype=np.float64)), -1).reshape(-1, 2)
    p = (g - M[:, 2]) / s  # crop px -> (mirrored) pinhole px
    if not is_right:
        p[:, 0] = W - 1 - p[:, 0]
    q = distort((p - K[[0, 1], [2, 2]]) / K[[0, 1], [0, 1]], dist) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
    q = q.reshape(SIZE, SIZE, 2).astype(np.float32)
    crop = cv2.remap(img, q[..., 0], q[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return crop, M


def mano_layer(mano_dir, left_fix):
    from bench.common import ManoLayer

    ml = ManoLayer(mano_dir, flat_hand_mean=False)  # applies the smplx left shapedirs fix
    if not left_fix:  # ARCTIC's own smplx.MANO layers: left shapedirs as shipped
        import smplx

        raw = smplx.create(f"{mano_dir}/MANO_LEFT.pkl", "mano", is_rhand=False, use_pca=False, flat_hand_mean=False)
        ml.layers[False].shapedirs[:] = raw.shapedirs
    return ml


def mano_world(ml, hand, f):
    return ml(hand["side"] == "right", hand["shape"], hand["pose"][f], hand["rot"][f], hand["trans"][f])


def load_seq(root, sid, seq):
    m = safe_npy(os.path.join(root, "raw_seqs", sid, f"{seq}.mano.npy"))
    hands = []
    for side in ("right", "left"):
        h = m[side]
        T = len(h["rot"])
        hands.append(dict(side=side, rot=np.asarray(h["rot"], np.float32), pose=np.asarray(h["pose"], np.float32),
                          trans=np.asarray(h["trans"], np.float32), shape=np.asarray(h["shape"], np.float32).reshape(-1)[:10],
                          err=np.asarray(h.get("fitting_err", np.zeros(T)), np.float32)))
    ego = safe_npy(os.path.join(root, "raw_seqs", sid, f"{seq}.egocam.dist.npy"))
    return hands, ego


def cam_of(view, misc_s, ego, f, scale):
    """(R, T, K, dist or None) of a view at MoCap frame f; ego K for the stored image (scale = stored / original width)."""
    if view == 0:
        R, T = np.asarray(ego["R_k_cam_np"][f], np.float64), np.asarray(ego["T_k_cam_np"][f], np.float64).reshape(3)
        K = np.asarray(ego["intrinsics"], np.float64).copy()
        if scale != 1:
            K[:2, :2] *= scale
            K[:2, 2] = (K[:2, 2] + 0.5) * scale - 0.5  # PIL resize: pixel centres
        return R, T, K, np.asarray(ego["dist8"], np.float64).reshape(-1)
    E = np.asarray(misc_s["world2cam"][view - 1], np.float64)
    return E[:3, :3], E[:3, 3], np.asarray(misc_s["intris_mat"][view - 1], np.float64), None


def _work(arg):
    sid, seq, view, root, out, stride, mano_dir, verts, min_joints_in, left_fix, max_err, img_dir, zip_dir = arg
    name = f"{sid}/{seq}/{view}"
    done = os.path.join(out, name, ".done.json")
    if os.path.exists(done):  # resumable per sequence view
        return json.load(open(done))
    key = (mano_dir, left_fix)
    if key not in _MANO:
        _MANO[key] = mano_layer(mano_dir, left_fix)
    ml = _MANO[key]
    misc = json.load(open(os.path.join(root, "meta", "misc.json")))[sid]
    hands, ego = load_seq(root, sid, seq)
    nf = len(hands[0]["rot"])
    if zip_dir:  # read straight from the per-sequence download zip (<view>/<frame>.jpg members); incomplete zips are skipped
        try:
            z = zipfile.ZipFile(os.path.join(zip_dir, sid, f"{seq}.zip"))
        except (OSError, zipfile.BadZipFile) as e:
            print("skip", sid, seq, e, flush=True)
            return []
        files = sorted(n for n in z.namelist() if n.startswith(f"{view}/") and n.endswith(".jpg"))
        read = lambda n: cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
    else:
        files = sorted(glob.glob(os.path.join(root, img_dir, sid, seq, str(view), "*.jpg")))
        read = cv2.imread
    if view > 0 and img_dir == "cropped_images":
        bbox = allo_bbox(root, sid, seq, view, misc)
    os.makedirs(os.path.join(out, name), exist_ok=True)
    recs = []
    phase = zlib.crc32(name.encode()) % stride
    for fp in files[phase::stride]:
        stem = os.path.basename(fp)[:-4]
        f = int(stem) - misc["ioi_offset"]
        if f < 0 or f >= nf:
            continue
        img = read(fp)
        H, W = img.shape[:2]
        full = W == misc["image_size"][view][0]  # full-resolution images/ instead of cropped_images/
        R, T, K, dist = cam_of(view, misc, ego, f, W / misc["image_size"][0][0])
        if view > 0 and not full:  # allocentric crop (crop_images.py): PIL crop of the 1.5x box (rounded), resized to ALLO_CAP
            cx, cy, sc = bbox[f]
            dim = sc * 200 * 1.5
            x0, y0, x1, y1 = (round(t) for t in (cx - dim / 2, cy - dim / 2, cx + dim / 2, cy + dim / 2))
            kx, ky = W / (x1 - x0), H / (y1 - y0)
            K = K.copy()
            K[0] *= kx
            K[1] *= ky
            K[0, 2] = (K[0, 2] / kx - x0 + 0.5) * kx - 0.5  # pixel centres
            K[1, 2] = (K[1, 2] / ky - y0 + 0.5) * ky - 0.5
            W0, H0 = misc["image_size"][view]  # the crop's zero padding is outside the camera image
            lim = ((-x0 + 0.5) * kx - 0.5, (-y0 + 0.5) * ky - 0.5, (W0 - x0 + 0.5) * kx - 0.5, (H0 - y0 + 0.5) * ky - 0.5)
        else:
            lim = (0, 0, W, H)
        for h in hands:
            if max_err and h["err"][f] > max_err:
                continue
            is_right = h["side"] == "right"
            j, v = mano_world(ml, h, f)
            j, v = j @ R.T + T, v @ R.T + T
            uv = project(v, K)
            if dist is not None:  # in-image test on the distorted image
                d = distort((uv - K[[0, 1], [2, 2]]) / K[[0, 1], [0, 1]], dist) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
            else:
                d = uv
            inside = (v[:, 2] > 0.01) & (d[:, 0] >= max(lim[0], 0)) & (d[:, 0] < min(lim[2], W)) & (d[:, 1] >= max(lim[1], 0)) & (d[:, 1] < min(lim[3], H))
            if inside.mean() < 0.3:
                continue
            b = np.array([*uv[inside].min(0), *uv[inside].max(0)], dtype=np.float32)
            if max(b[2] - b[0], b[3] - b[1]) < 32:
                continue
            j2 = project(j, K)
            jd = j2 if dist is None else distort((j2 - K[[0, 1], [2, 2]]) / K[[0, 1], [0, 1]], dist) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
            jin = (j[:, 2] > 0.01) & (jd[:, 0] >= max(lim[0], 0)) & (jd[:, 0] < min(lim[2], W)) & (jd[:, 1] >= max(lim[1], 0)) & (jd[:, 1] < min(lim[3], H))
            if jin.sum() < min_joints_in:
                continue
            if dist is not None:
                crop, M = rectified_crop(img, b, is_right, K, dist, W)
            else:
                crop, M = render_crop(img, b, is_right)
            rel = f"{name}/{stem}_{'r' if is_right else 'l'}.jpg"
            cv2.imwrite(os.path.join(out, rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            j3, Kc = j.copy(), K.copy()
            if not is_right:  # right-hand frame (mirrored), as every pool
                j3[:, 0] *= -1
                Kc[0, 2] = W - 1 - Kc[0, 2]
            rec = dict(file=rel, src=f"{sid}/{seq}:{view}/{stem}", is_right=bool(is_right), box=[float(x) for x in b], M=M.round(5).tolist(),
                       hw=[H, W], sequence_id=f"{sid}/{seq}", subject=sid, view=int(view), frame=int(f),
                       kp2d=crop_keypoints(np.concatenate([j2, jin[:, None]], 1), M, is_right, W).round(2).tolist(),
                       joints3d=j3.round(5).tolist(), K=Kc.round(4).tolist(), fit_err=float(h["err"][f]))
            if verts:
                v3 = v.copy()
                if not is_right:
                    v3[:, 0] *= -1
                rec["verts_file"] = rel[:-4] + "_v.npy"
                np.save(os.path.join(out, rec["verts_file"]), v3.astype(np.float16))
            recs.append(rec)
    json.dump(recs, open(done, "w"))
    return recs


def load_obj_v(path):
    return np.array([[float(x) for x in l.split()[1:4]] for l in open(path) if l.startswith("v ")])


def allo_bbox(root, sid, seq, view, misc_s):
    """Per-MoCap-frame (cx, cy, scale) of allocentric view `view` (1-8), recomputed as ARCTIC's processing does
    (src/arctic/processing.py forward_define_bbox, common/object_tensors.py): the articulated object mesh (mesh.obj, mm;
    vertices with parts.json == 0 rotated by obj_arti about -z, then obj_rot, + obj_trans) projected into the view,
    square box of 1.1 x its larger extent + 0.6 x that extent, scale = side / 200 clamped to >= 3."""
    from scipy.spatial.transform import Rotation

    obj = seq.split("_")[0]
    v = load_obj_v(os.path.join(root, "meta", "object_vtemplates", obj, "mesh.obj")) / 1000
    parts = np.asarray(json.load(open(os.path.join(root, "meta", "object_vtemplates", obj, "parts.json"))), bool)
    art = ~parts  # object_tensors: parts_ids = parts + 1, the articulated rotation goes to parts_ids == 1
    o = np.asarray(np.load(os.path.join(root, "raw_seqs", sid, f"{seq}.object.npy"), allow_pickle=False), np.float64)
    E = np.asarray(misc_s["world2cam"][view - 1], np.float64)
    K = np.asarray(misc_s["intris_mat"][view - 1], np.float64)
    out = np.zeros((len(o), 3))
    for f, (a, rx, ry, rz, tx, ty, tz) in enumerate(o):
        w = v.copy()
        w[art] = Rotation.from_rotvec(np.array([0, 0, -1.0]) * a).apply(w[art])
        w = Rotation.from_rotvec([rx, ry, rz]).apply(w) + np.array([tx, ty, tz]) / 1000
        uv = project(w @ E[:3, :3].T + E[:3, 3], K)
        (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
        m = max(x1 - x0, y1 - y0)
        out[f] = ((x0 + x1) / 2, (y0 + y1) / 2, max((1.1 * m + 0.6 * m) / 200, 3.0))
    return out


def seq_list(root, protocol, split):
    js = json.load(open(os.path.join(root, "splits_json", f"protocol_{protocol}.json")))
    return [s for s in js[split]]


def check(args):
    """Our MANO joints (kinematic + MANO joints[:16]) vs ARCTIC's processed camera joints (same MoCap fit) for both left
    shapedirs readings, on the first frames of a few train sequences; plus an overlay sheet of rectified ego crops."""
    from bench.common import MANO21_TO_OPENPOSE

    inv = np.argsort(MANO21_TO_OPENPOSE)
    seqs = seq_list(args.root, "p2", "train")[: args.check_seqs]
    rows = {True: [], False: []}
    for fix in (True, False):
        ml = mano_layer(args.mano_dir, fix)
        for s in seqs:
            sid, seq = s.split("/")
            hands, ego = load_seq(args.root, sid, seq)
            pp = os.path.join(args.root, "processed_seqs", sid, f"{seq}.npy")
            if not os.path.exists(pp):
                print("no processed_seqs for the joint check:", pp, flush=True)
                break
            proc = safe_npy(pp)
            for f in range(0, len(hands[0]["rot"]), 97)[:5]:
                for h in hands:
                    j, v = mano_world(ml, h, f)
                    ref = np.asarray(proc["world_coord"][f"joints.{h['side']}"][f], np.float64)
                    # ARCTIC 21 joints = MANO joints[:16] + its own tips, MANO order -> compare the 16 MANO joints
                    rows[fix].append((h["side"], float(np.abs(j[inv][:16] - ref[:16]).max() * 1000)))
    for fix, r in rows.items():
        for side in ("right", "left"):
            e = [x for s, x in r if s == side]
            if e:
                print(f"left_fix={fix} {side}: max |our MANO joints - ARCTIC joints| median {np.median(e):.3f} mm, max {np.max(e):.3f} mm", flush=True)


def main(args):
    if args.check:
        return check(args)
    protocol = "p2" if args.views == "0" else "p1"
    seqs = seq_list(args.root, protocol, args.split)
    if args.max_seqs:
        seqs = seqs[: args.max_seqs]
    views = [int(v) for v in args.views.split(",")]
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)
    jobs = [(s.split("/")[0], s.split("/")[1], v, args.root, out, args.stride, args.mano_dir, args.verts, args.min_joints_in,
             args.left_fix, args.max_err, args.images, args.zips) for s in seqs for v in views]
    recs = []
    with Pool(args.workers) as pool:
        for k, rs in enumerate(pool.imap_unordered(_work, jobs)):
            recs += rs
            if k % 20 == 0:
                print(f"{k}/{len(jobs)} sequence views, {len(recs)} crops", flush=True)
    other = {s for sp in ("val", "test") if sp != args.split for s in seq_list(args.root, protocol, sp)}
    assert not any(r["sequence_id"] in other for r in recs), "held-out sequence leaked into the pool"
    with open(os.path.join(out, "index.jsonl"), "w") as f:
        for r in sorted(recs, key=lambda r: r["file"]):
            f.write(json.dumps(r) + "\n")
    print(f"{args.name}: {len(recs)} crops from {len(jobs)} sequence views ({protocol} {args.split}) -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="arctic_data/data")
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="arctic_ego_lab", help="output subdir / source name")
    p.add_argument("--views", default="0", help="0 = egocentric (protocol p2), 1-8 = allocentric (p1), comma list")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--stride", type=int, default=4, help="every stride-th image per sequence view (phase-shifted per view)")
    p.add_argument("--min_joints_in", type=int, default=15)
    p.add_argument("--max_err", type=float, default=0.0, help="skip hands whose MANO fitting_err exceeds this (0 = keep all)")
    p.add_argument("--left_fix", type=int, default=0, help="1 = smplx left shapedirs fix (wrong for ARCTIC's fits)")
    p.add_argument("--max_seqs", type=int, default=0)
    p.add_argument("--images", default="cropped_images", help="cropped_images | images (full resolution)")
    p.add_argument("--zips", default="", help="read images from <zips>/<sid>/<seq>.zip (cropped_images_zips) instead")
    p.add_argument("--verts", action="store_true", help="also store the MANO vertices (778 x 3) per crop")
    p.add_argument("--check", action="store_true")
    p.add_argument("--check_seqs", type=int, default=4)
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--workers", type=int, default=8)
    main(p.parse_args())
