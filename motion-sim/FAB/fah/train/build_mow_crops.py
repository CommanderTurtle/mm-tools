"""MOW (Cao et al., ICCV 2021, https://zhec.github.io/rhoi/; 512 in-the-wild 100DOH frames, one right MANO hand + one
object per image; MANO = an optimisation fit with the mean shape, not mocap GT) crop pool with MANO labels. Splits =
HACO's (data/mow/splits/{train,test}.json, 350 / 92); only --split ids are built, the other split is asserted absent.

Conventions (ZheC/MOW vis_anno.py + utils.PerspectiveRenderer, checked by --check against masks/both):
- MANO: smplx right hand, num_pca_comps=45 (= axis-angle: smplx turns PCA off at 45), flat_hand_mean=False; hand_pose =
  [global_orient(3), pose(45)], transl = trans, betas = 0; posed mesh_cam = hand_s * (mesh @ hand_R + hand_t) (row
  vectors, scale after translation).
- camera: neural_renderer K [[1, 0, .5], [0, 1, .5]] on the image zero-padded to L = max(H, W) at the bottom / right ->
  f = L px, principal point (L / 2, L / 2) of the original image, y down.
- hand_s (median 1.85) only moves the hand along the camera rays (projection invariant): the labels drop it
  (mesh @ hand_R + hand_t), so the hand keeps MANO's mean size and the depth is consistent with f = L.
Labels as train.build_hot3d_crops --labels --verts (joints3d = kinematic MANO joints + HOT3D fingertip vertices, OpenPose
order). Hands with hand_s at the fit's bounds (< 0.2 or > 9) are skipped. src = "<video>:<frame>" so LabeledCrops keeps a
video's frames on one side.

    python -m train.build_mow_crops --root $DATA_ROOT/mow --out $DATA_ROOT/crops --verts
    python -m train.build_mow_crops --root $DATA_ROOT/mow --out <check dir> --check   (convention check + sheet)
"""
import argparse
import json
import os
import re

import cv2
import numpy as np

from train.build_crops import crop_keypoints, render_crop


def video_of(image_id):
    return re.sub(r"_frame\d+$", "", image_id)


def hand_mesh(ml, a):
    """MOW hand -> (joints (21, 3), verts (778, 3)) in the camera frame without hand_s, and hand_s."""
    pose = np.asarray(a["hand_pose"], np.float32)
    j, v = ml(True, np.zeros(10), pose[3:], pose[:3], np.asarray(a["trans"], np.float32))
    R, t = np.asarray(a["hand_R"], np.float64).reshape(3, 3), np.asarray(a["hand_t"], np.float64).reshape(3)
    return j @ R + t, v @ R + t, float(a["hand_s"])


def intrinsics(H, W, centre="pad"):
    L = max(H, W)
    c = (L / 2, L / 2) if centre == "pad" else (W / 2, H / 2)
    return np.array([[L, 0, c[0]], [0, L, c[1]], [0, 0, 1]], np.float64)


def project(X, K):
    return X[:, :2] / np.clip(X[:, 2:], 1e-6, None) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]


def load(root, split):
    ids = json.load(open(os.path.join(root, "splits", f"{split}.json")))
    other = json.load(open(os.path.join(root, "splits", f"{'test' if split == 'train' else 'train'}.json")))
    assert not set(ids) & set(other), "MOW train / test overlap"
    poses = {a["image_id"]: a for a in json.load(open(os.path.join(root, "poses.json")))}
    return ids, poses, other


def check(args, ml):
    """Fraction of projected hand vertices inside the hand + object mask for the two principal-point readings, and a
    label overlay sheet (MOW frames: keep the sheet private, they are not redistributed)."""
    ids, poses, other = load(args.root, args.split)
    vids, ovids = {video_of(i) for i in ids}, {video_of(i) for i in other}
    print(f"{args.split}: {len(ids)} images, {len(vids)} videos, {len(vids & ovids)} videos also in the other split", flush=True)
    rows, tiles = [], []
    for n, i in enumerate(ids):
        a = poses[i]
        img = cv2.imread(os.path.join(args.root, "mow", "images", f"{i}.jpg"))
        m = cv2.imread(os.path.join(args.root, "masks", "both", f"{i}.jpg"))[:, :, 0] > 128
        H, W = img.shape[:2]
        j, v, s = hand_mesh(ml, a)
        r = dict(id=i, hand_s=s, H=H, W=W)
        for c in ("pad", "centre"):
            uv = np.round(project(v, intrinsics(H, W, c))).astype(int)
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            r[f"in_mask_{c}"] = float(m[uv[ok, 1], uv[ok, 0]].mean()) if ok.any() else 0.0
            r[f"in_img_{c}"] = float(ok.mean())
        rows.append(r)
        if n < args.sheet:
            K = intrinsics(H, W)
            uv, j2 = project(v, K), project(j, K)
            b = np.array([*uv.min(0), *uv.max(0)], np.float32)
            vis = img.copy()
            vis[m] = (0.6 * vis[m] + 0.4 * np.array([0, 160, 0])).astype(np.uint8)
            for p in uv[::3]:
                cv2.circle(vis, (int(p[0]), int(p[1])), 1, (255, 200, 0), -1)
            from bench.bench2d.gallery import draw_hand

            draw_hand(vis, j2, thick=max(1, int(max(b[2] - b[0], b[3] - b[1]) / 80)))
            crop, _ = render_crop(vis, b, True, size=256, context=1.6)
            cv2.putText(crop, f"{n} s{s:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            tiles.append(crop)
    os.makedirs(args.out, exist_ok=True)
    json.dump(rows, open(os.path.join(args.out, f"mow_check_{args.split}.json"), "w"))
    for c in ("pad", "centre"):
        x = np.array([r[f"in_mask_{c}"] for r in rows])
        print(f"principal point {c}: projected verts inside hand+object mask median {np.median(x):.3f}, p10 {np.percentile(x, 10):.3f}, "
              f"share > 0.8 {np.mean(x > 0.8):.3f}", flush=True)
    if tiles:
        cols = 8
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
        sheet = np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])
        cv2.imwrite(os.path.join(args.out, f"mow_sheet_{args.split}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def build(args, ml):
    ids, poses, other = load(args.root, args.split)
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)
    recs, skipped = [], 0
    for i in ids:
        a = poses[i]
        j, v, s = hand_mesh(ml, a)
        if not 0.2 <= s <= 9:  # fit hit its scale bounds
            skipped += 1
            continue
        img = cv2.imread(os.path.join(args.root, "mow", "images", f"{i}.jpg"))
        H, W = img.shape[:2]
        K = intrinsics(H, W)
        uv = project(v, K)
        inside = (v[:, 2] > 0.01) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if inside.mean() < 0.3:
            skipped += 1
            continue
        b = np.array([*uv[inside].min(0), *uv[inside].max(0)], dtype=np.float32)
        if max(b[2] - b[0], b[3] - b[1]) < 32:
            skipped += 1
            continue
        crop, M = render_crop(img, b, True)
        rel = f"{i}_r.jpg"
        cv2.imwrite(os.path.join(out, rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        j2 = project(j, K)
        jin = (j2[:, 0] >= 0) & (j2[:, 0] < W) & (j2[:, 1] >= 0) & (j2[:, 1] < H)
        vid = video_of(i)
        rec = dict(file=rel, src=f"{vid}:{i[len(vid) + 1:]}", is_right=True, box=[float(x) for x in b], M=M.round(5).tolist(),
                   hw=[H, W], sequence_id=vid, kp2d=crop_keypoints(np.concatenate([j2, jin[:, None]], 1), M, True, W).round(2).tolist(),
                   joints3d=j.round(5).tolist(), K=K.round(4).tolist(), hand_s=round(s, 4), obj=a.get("obj_name"))
        if args.verts:
            rec["verts_file"] = rel[:-4] + "_v.npy"
            np.save(os.path.join(out, rec["verts_file"]), v.astype(np.float16))
        recs.append(rec)
    assert not {r["file"][:-6] for r in recs} & set(other)
    with open(os.path.join(out, "index.jsonl"), "w") as f:
        for r in sorted(recs, key=lambda r: r["file"]):
            f.write(json.dumps(r) + "\n")
    print(f"{args.name}: {len(recs)} crops from {len(ids)} {args.split} images ({skipped} skipped) -> {out}")


def main(args):
    from bench.common import ManoLayer

    ml = ManoLayer(args.mano_dir, num_pca_comps=45, flat_hand_mean=False)
    (check if args.check else build)(args, ml)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="MOW dir (mow/images, masks/both, poses.json, splits/)")
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--name", default="mow_lab", help="output subdir / source name")
    p.add_argument("--verts", action="store_true", help="also store the MANO vertices (778 x 3) per crop")
    p.add_argument("--check", action="store_true", help="convention check against masks/both + overlay sheet, no pool")
    p.add_argument("--sheet", type=int, default=48, help="--check: images on the overlay sheet")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    main(p.parse_args())
