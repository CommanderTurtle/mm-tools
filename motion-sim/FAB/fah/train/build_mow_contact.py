"""MOW (Cao et al., ICCV 2021, https://zhec.github.io/rhoi/; 512 in-the-wild 100DOH frames, one right hand + one object per
image) contact data, with HACO's preprocessed 35 mm contact labels, masks and train / test splits (350 / 92 images; HACO's
release). MOW and HACO's files are fetched by each user under their terms; nothing of them is redistributed.

--mow: the MOW directory = mow/images/<id>.jpg, masks/both/<id>.jpg (hand + object mask), poses.json (MOW's MANO fits),
splits/{train,test}.json, preprocessed_data/{train,test}/contact_data/<id>.npy (778 values on MANO's right-hand template,
> 0.5 = contact within 35 mm). --out: the contact data root.

  pool   MOW train as a contact crop pool for the 35 mm output only: the FAH detection most inside the hand + object mask
         box (ranked by box IoA x (0.5 + score), kept if IoA >= 0.5), crops as every pool (train.build_crops.render_crop),
         labels = HACO's 35 mm contact as a signed distance of 0 / 100 mm (channels.json: 35 mm only)
         -> <out>/pools/mow_train_c35 (index.jsonl + crops), <out>/mow_train_c35 (sdf.npy, contact.jsonl, channels.json).
         The 92 test images are never used.
  preds  MOW test predictions for the evaluation (HACO's protocol, scripts/contact_mow_pairs.py), <out>/test/mow/pred_<tag>.npz
         (files, prob 92 x 778, hand): --model fah = the FAH pipeline (the detection most inside the mask box; the head's
         35 mm output when it has several tolerances; no hand found -> no contact), --model haco = bench.methods.haco on
         HACO's own MOW crop (the mask box with expansion 1.3).
  poses  MOW's own MANO fits as a pose crop pool (MANO labels, as train.build_hot3d_crops --labels --verts); --check:
         the fits' projection convention against masks/both + an overlay sheet (local only).

Camera: MOW has no intrinsics. The FAH pipeline runs with f = max(H, W) px and the principal point at the image centre
(W / 2, H / 2). MOW's MANO fits (ZheC/MOW vis_anno.py + utils.PerspectiveRenderer) follow neural_renderer's convention on
the image zero-padded to L = max(H, W) at the bottom / right: f = L px, principal point (L / 2, L / 2) of the original image;
smplx right hand, num_pca_comps=45 (= axis-angle), flat_hand_mean=False, betas 0; mesh_cam = hand_s * (mesh @ hand_R +
hand_t). hand_s (median 1.85) only moves the hand along the camera rays (projection invariant), so the pose labels drop it
(the hand keeps MANO's mean size and its depth stays consistent with f = L). The fits are optimisation fits (no mocap GT).

    python -m train.build_mow_contact pool --mow $DATA_ROOT/mow --out $DATA_ROOT/contact --finetune_ckpt <fah contact ckpt>
    python -m train.build_mow_contact preds --mow $DATA_ROOT/mow --out $DATA_ROOT/contact --model fah --finetune_ckpt <ckpt> --tag fahL-contact
    # a head trained with train.contact_head (contact.pt) on a released network:
    python -m train.build_mow_contact preds --mow $DATA_ROOT/mow --out $DATA_ROOT/contact --model fah --finetune_ckpt $WEIGHTS/fah/model.pt \
        --contact_ckpt $CKPT_ROOT/contact/<arm>/contact.pt --tag <arm>
    python -m train.build_mow_contact preds --mow $DATA_ROOT/mow --out $DATA_ROOT/contact --model haco --tag haco
    python -m train.build_mow_contact poses --mow $DATA_ROOT/mow --out $DATA_ROOT/crops --verts [--check]
"""
import argparse
import json
import os
import re

import cv2
import numpy as np

from train.build_crops import crop_keypoints, render_crop


def splits(root, split):
    ids = json.load(open(os.path.join(root, "splits", f"{split}.json")))
    other = json.load(open(os.path.join(root, "splits", f"{'test' if split == 'train' else 'train'}.json")))
    assert not set(ids) & set(other), "MOW train / test overlap"
    return ids, other


def mask_box(root, i, expand=1.3):
    """HACO's MOW crop: mask2bbox of the hand + object mask (masks/both) with expansion `expand` -> (image, (x, y, w, h), the
    tight mask box (x0, y0, x1, y1))."""
    from bench.methods.haco import mask_box as mb

    img = cv2.imread(os.path.join(root, "mow", "images", f"{i}.jpg"))
    m = cv2.imread(os.path.join(root, "masks", "both", f"{i}.jpg"))[:, :, 0] > 128
    ys, xs = np.nonzero(m)
    return img, mb(m, expand), (xs.min(), ys.min(), xs.max(), ys.max())


def fah_camera(H, W):
    return np.array([[max(H, W), 0, W / 2], [0, max(H, W), H / 2], [0, 0, 1]], np.float32)


def fah_method(a, **kw):
    from bench.methods.fah import FAHMethod

    return FAHMethod(checkpoint=a.checkpoint, device="cuda", finetune_ckpt=a.finetune_ckpt, verify=True, min_crop=1.1,
                     detector_weights=a.detector_weights, det_conf=0.05, pair_wrist=0.5, contact_ckpt=a.contact_ckpt, **kw)


def best_in_mask(preds, mb):
    from bench.methods.fah import box_ioa

    if not preds:
        return None
    ioa = lambda q: box_ioa(np.asarray(q["bbox"], np.float64), np.asarray(mb, np.float64))
    return max(preds, key=lambda q: ioa(q) * (0.5 + q["score"])), ioa


def cmd_pool(a):
    root = a.mow
    ids, _ = splits(root, "train")
    m = fah_method(a)
    out, lab_dir = os.path.join(a.out, "pools", "mow_train_c35"), os.path.join(a.out, "mow_train_c35")
    os.makedirs(out, exist_ok=True), os.makedirs(lab_dir, exist_ok=True)
    recs, sdf = [], []
    for i in ids[:a.limit] if a.limit else ids:
        img, _, mb = mask_box(root, i, 0)
        Hh, Ww = img.shape[:2]
        r = best_in_mask(m(img, fah_camera(Hh, Ww)), mb)
        if r is None:
            continue
        p, ioa = r
        if ioa(p) < 0.5:
            continue
        crop, M = render_crop(img, np.asarray(p["bbox"], np.float32), bool(p["is_right"]))
        cv2.imwrite(os.path.join(out, f"{i}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        recs.append(dict(file=f"{i}.jpg", src=f"{i}:0", is_right=bool(p["is_right"]), box=[float(x) for x in p["bbox"]], M=M.round(5).tolist(), hw=[Hh, Ww]))
        c = np.load(os.path.join(root, "preprocessed_data", "train", "contact_data", f"{i}.npy")) > 0.5
        sdf.append(np.where(c, 0.0, 100.0).astype(np.float16))
    with open(os.path.join(out, "index.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    np.save(os.path.join(lab_dir, "sdf.npy"), np.stack(sdf))
    with open(os.path.join(lab_dir, "contact.jsonl"), "w") as f:
        for r, d in zip(recs, sdf):
            f.write(json.dumps(dict(file=r["file"], src=r["src"], is_right=r["is_right"], valid=True, min_mm=float(d.min()), n10=0, obj=None)) + "\n")
    json.dump(dict(dists=[35.0], note="MOW train, HACO's preprocessed 35 mm labels"), open(os.path.join(lab_dir, "channels.json"), "w"))
    print("mow pool", len(recs), "of", len(ids), flush=True)


def cmd_preds(a):
    root = a.mow
    ids, _ = splits(root, "test")
    ids = ids[:a.limit] if a.limit else ids
    probs, hand = [], []
    if a.model == "haco":
        from bench.methods.haco import HACOMethod

        m = HACOMethod(repo=a.haco_repo or None, crop="mask13", thr=0.52, jpeg_quality=None)
        for i in ids:
            img, box, _ = mask_box(root, i)
            p = m.predict(img, [[0, 0, 1, 1]], [True], windows=[box])[0]
            probs.append(p["contact"]), hand.append(p["contact_score"])
    else:
        m = fah_method(a, dtype="bfloat16", recrop_border=0.03, gpu_frame=True, pair_keep=0.15, fast_batch=True, tip_extend=0.6)
        for i in ids:
            img, _, mb = mask_box(root, i, 0)
            Hh, Ww = img.shape[:2]
            r = best_in_mask([p for p in m(img, fah_camera(Hh, Ww)) if "contact" in p], mb)
            if r is not None:
                best = r[0]
                c35 = best["contact"].astype(np.float32)
                if "contact_k" in best:  # the head's 35 mm output = MOW's label tolerance
                    dk = list(m.est.model.keypoint_head.contact_dists.cpu().numpy())
                    c35 = best["contact_k"][:, dk.index(35.0)].astype(np.float32) if 35.0 in dk else c35
                probs.append(c35), hand.append(best["contact_score"])
            else:  # no hand found: no contact predicted
                probs.append(np.zeros(778, np.float32)), hand.append(0.0)
    d = os.path.join(a.out, "test", "mow")
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, f"pred_{a.tag}.npz"), files=np.asarray(ids), prob=np.stack(probs).astype(np.float16), hand=np.asarray(hand))
    print("mow", a.tag, len(ids), flush=True)


# ---------------------------------------------------------------------------- MOW's own MANO fits (pose labels)

def video_of(image_id):
    return re.sub(r"_frame\d+$", "", image_id)


def hand_mesh(ml, a):
    """MOW hand -> (joints (21, 3), verts (778, 3)) in the camera frame without hand_s, and hand_s."""
    pose = np.asarray(a["hand_pose"], np.float32)
    j, v = ml(True, np.zeros(10), pose[3:], pose[:3], np.asarray(a["trans"], np.float32))
    R, t = np.asarray(a["hand_R"], np.float64).reshape(3, 3), np.asarray(a["hand_t"], np.float64).reshape(3)
    return j @ R + t, v @ R + t, float(a["hand_s"])


def fit_camera(H, W, centre="pad"):
    L = max(H, W)
    c = (L / 2, L / 2) if centre == "pad" else (W / 2, H / 2)
    return np.array([[L, 0, c[0]], [0, L, c[1]], [0, 0, 1]], np.float64)


def project(X, K):
    return X[:, :2] / np.clip(X[:, 2:], 1e-6, None) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]


def check(a, ml):
    """Fraction of projected hand vertices inside the hand + object mask for the two principal-point readings, and a label
    overlay sheet (local only: MOW frames are not redistributed)."""
    ids, other = splits(a.mow, a.split)
    poses = {p["image_id"]: p for p in json.load(open(os.path.join(a.mow, "poses.json")))}
    rows, tiles = [], []
    for n, i in enumerate(ids):
        img = cv2.imread(os.path.join(a.mow, "mow", "images", f"{i}.jpg"))
        m = cv2.imread(os.path.join(a.mow, "masks", "both", f"{i}.jpg"))[:, :, 0] > 128
        H, W = img.shape[:2]
        j, v, s = hand_mesh(ml, poses[i])
        r = dict(id=i, hand_s=s, H=H, W=W)
        for c in ("pad", "centre"):
            uv = np.round(project(v, fit_camera(H, W, c))).astype(int)
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            r[f"in_mask_{c}"] = float(m[uv[ok, 1], uv[ok, 0]].mean()) if ok.any() else 0.0
        rows.append(r)
        if n < a.sheet:
            from bench.bench2d.gallery import draw_hand

            K = fit_camera(H, W)
            uv, j2 = project(v, K), project(j, K)
            b = np.array([*uv.min(0), *uv.max(0)], np.float32)
            vis = img.copy()
            vis[m] = (0.6 * vis[m] + 0.4 * np.array([0, 160, 0])).astype(np.uint8)
            for q in uv[::3]:
                cv2.circle(vis, (int(q[0]), int(q[1])), 1, (255, 200, 0), -1)
            draw_hand(vis, j2, thick=max(1, int(max(b[2] - b[0], b[3] - b[1]) / 80)))
            crop, _ = render_crop(vis, b, True, size=256, context=1.6)
            tiles.append(crop)
    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(os.path.join(a.out, f"mow_check_{a.split}.json"), "w"))
    for c in ("pad", "centre"):
        x = np.array([r[f"in_mask_{c}"] for r in rows])
        print(f"principal point {c}: projected verts inside the hand + object mask median {np.median(x):.3f}, share > 0.8 {np.mean(x > 0.8):.3f}", flush=True)
    if tiles:
        cols = 8
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
        cv2.imwrite(os.path.join(a.out, f"mow_sheet_{a.split}.jpg"), np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])


def cmd_poses(a):
    from bench.common import ManoLayer

    ml = ManoLayer(a.mano_dir, num_pca_comps=45, flat_hand_mean=False)
    if a.check:
        return check(a, ml)
    ids, other = splits(a.mow, a.split)
    poses = {p["image_id"]: p for p in json.load(open(os.path.join(a.mow, "poses.json")))}
    out = os.path.join(a.out, a.name)
    os.makedirs(out, exist_ok=True)
    recs, skipped = [], 0
    for i in ids:
        p = poses[i]
        j, v, s = hand_mesh(ml, p)
        if not 0.2 <= s <= 9:  # the fit hit its scale bounds
            skipped += 1
            continue
        img = cv2.imread(os.path.join(a.mow, "mow", "images", f"{i}.jpg"))
        H, W = img.shape[:2]
        K = fit_camera(H, W)
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
                   joints3d=j.round(5).tolist(), K=K.round(4).tolist(), hand_s=round(s, 4), obj=p.get("obj_name"))
        if a.verts:
            rec["verts_file"] = rel[:-4] + "_v.npy"
            np.save(os.path.join(out, rec["verts_file"]), v.astype(np.float16))
        recs.append(rec)
    assert not {r["file"][:-6] for r in recs} & set(other)
    with open(os.path.join(out, "index.jsonl"), "w") as f:
        for r in sorted(recs, key=lambda r: r["file"]):
            f.write(json.dumps(r) + "\n")
    print(f"{a.name}: {len(recs)} crops from {len(ids)} {a.split} images ({skipped} skipped) -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["pool", "preds", "poses"])
    p.add_argument("--mow", default=os.path.join(os.environ.get("DATA_ROOT", "data"), "mow"))
    p.add_argument("--out", default=os.environ.get("CONTACT_ROOT", os.path.join("data", "contact")), help="the contact data root (poses: the crop root)")
    p.add_argument("--checkpoint", default="checkpoints/sam-3d-body-dinov3/model.ckpt", help="pool / preds (fah): the base checkpoint")
    p.add_argument("--finetune_ckpt", default="", help="pool / preds (fah): the FAH network with its contact head")
    p.add_argument("--contact_ckpt", default="", help="preds (fah): a separate contact head (train.contact_head's contact.pt) for --finetune_ckpt")
    p.add_argument("--detector_weights", default="", help="pool / preds (fah): the hand detector")
    p.add_argument("--model", default="fah", choices=["fah", "haco"], help="preds: which method")
    p.add_argument("--haco_repo", default="", help="preds (haco): the HACO checkout (default $THIRDPARTY/HACO)")
    p.add_argument("--tag", default="fah", help="preds: output name pred_<tag>.npz")
    p.add_argument("--limit", type=int, default=0, help="pool / preds: only the first n images (e.g. a parity check)")
    p.add_argument("--split", default="train", choices=["train", "test"], help="poses")
    p.add_argument("--name", default="mow_lab", help="poses: output subdir / source name")
    p.add_argument("--verts", action="store_true", help="poses: also store the MANO vertices (778 x 3) per crop")
    p.add_argument("--check", action="store_true", help="poses: convention check against masks/both + overlay sheet, no pool")
    p.add_argument("--sheet", type=int, default=48, help="poses --check: images on the overlay sheet")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    a = p.parse_args()
    dict(pool=cmd_pool, preds=cmd_preds, poses=cmd_poses)[a.cmd](a)
