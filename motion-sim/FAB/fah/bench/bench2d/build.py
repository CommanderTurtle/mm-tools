"""Build the 2D hand-keypoint benchmark (bench2d) from its sources.

Subsets (21 keypoints/hand, OpenPose order: wrist, thumb1-4, index1-4, middle1-4, ring1-4, pinky1-4):
  hot3d  egocentric Aria RGB (HOT3D-Clips, HaWoR's 27 val sequences), fisheye rectified to an upright
         pinhole camera, GT = projected MANO joints; every hand annotated (fully_annotated=True).
  cmu    CMU Panoptic hand_labels manual_test (MPII + NZSL, human-labelled, in the wild); one hand
         labelled per image (fully_annotated=False).
  coco   COCO-WholeBody val2017 hands (human-labelled, in the wild); only hands of annotated persons
         with >= min_kps labelled joints and box >= min_box px (fully_annotated=False).
  hint_nd / hint_epick   (v2) HInt TEST (HaMeR, Pavlakos et al. 2024): New Days (YouTube, in the wild) and
         EPIC-KITCHENS VISOR (egocentric) frames; joints with existence=1 are scored, occlusion stored.

    python -m bench.bench2d.build --out $DATA_ROOT/bench2d_v2 --hot3d_root ... --cmu_root ... --coco_root ...
"""
import argparse
import glob
import json
import os
import shutil

import cv2
import numpy as np


def _box(kp, vis):
    p = kp[vis > 0]
    return [float(v) for v in (*p.min(0), *p.max(0))]


def build_hot3d(args, out, recs):
    from bench.common import ManoLayer
    from bench.datasets.hot3d import Hot3DClip

    mano = ManoLayer(args.mano_dir, num_pca_comps=15)
    split = json.load(open(args.hot3d_split))
    os.makedirs(os.path.join(out, "images", "hot3d"), exist_ok=True)
    for c in split["clips"]["Aria"]:
        clip = Hot3DClip(os.path.join(args.hot3d_root, c), args.mano_dir, mano=mano, min_visible=0.3)
        name = os.path.splitext(os.path.basename(c))[0]
        for i in range(0, len(clip), args.hot3d_stride):
            fr = clip.frame(i)
            h, w = fr["image"].shape[:2]
            hands = []
            for g in fr["hands"]:
                kp = g["joints2d"]
                vis = ((kp[:, 0] >= 0) & (kp[:, 0] < w) & (kp[:, 1] >= 0) & (kp[:, 1] < h) & (g["joints3d"][:, 2] > 0.01)).astype(int)
                if vis.sum() < args.min_kps:
                    continue
                hands.append(dict(is_right=bool(g["is_right"]), kp2d=kp.round(2).tolist(), vis=vis.tolist(), bbox=_box(kp, vis)))
            if not hands:
                continue
            rel = f"images/hot3d/{name}_{fr['key']}.jpg"
            cv2.imwrite(os.path.join(out, rel), fr["image"], [cv2.IMWRITE_JPEG_QUALITY, 92])
            recs.append(dict(id=f"hot3d/{name}_{fr['key']}", file=rel, subset="hot3d", width=w, height=h,
                             K=fr["K"].round(3).tolist(), fully_annotated=True, sequence=fr["sequence_id"], hands=hands))
        print(f"hot3d {name}: total records {sum(r['subset'] == 'hot3d' for r in recs)}", flush=True)


def build_cmu(args, out, recs):
    src = os.path.join(args.cmu_root, "hand_labels", "manual_test")
    os.makedirs(os.path.join(out, "images", "cmu"), exist_ok=True)
    by_image = {}
    for jf in sorted(glob.glob(os.path.join(src, "*.json"))):
        a = json.load(open(jf))
        pts = np.asarray(a["hand_pts"], dtype=np.float64)
        kp, vis = pts[:, :2], (pts[:, 2] > 0).astype(int)
        if vis.sum() < args.min_kps:
            continue
        stem = os.path.splitext(os.path.basename(jf))[0]
        img = stem[:-2] if stem.endswith(("_l", "_r")) else stem
        by_image.setdefault(img, []).append(dict(is_right=not bool(a["is_left"]), kp2d=kp.round(2).tolist(),
                                                 vis=vis.tolist(), bbox=_box(kp, vis)))
    for img, hands in by_image.items():
        cand = [os.path.join(src, img + ext) for ext in (".jpg", ".png")] + glob.glob(os.path.join(src, img + "_*.jpg"))
        path = next((p for p in cand if os.path.exists(p)), None)
        if path is None:
            continue
        im = cv2.imread(path)
        rel = f"images/cmu/{img}.jpg"
        cv2.imwrite(os.path.join(out, rel), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        recs.append(dict(id=f"cmu/{img}", file=rel, subset="cmu", width=im.shape[1], height=im.shape[0], K=None,
                         fully_annotated=False, hands=hands))
    print(f"cmu: {sum(r['subset'] == 'cmu' for r in recs)} images", flush=True)


def build_coco(args, out, recs):
    ann = json.load(open(os.path.join(args.coco_root, "coco_wholebody_val_v1.0.json")))
    imgs = {im["id"]: im for im in ann["images"]}
    by_image = {}
    for a in ann["annotations"]:
        for side, is_right in (("lefthand", False), ("righthand", True)):
            if not a.get(f"{side}_valid", False):
                continue
            pts = np.asarray(a[f"{side}_kpts"], dtype=np.float64).reshape(21, 3)
            kp, vis = pts[:, :2], (pts[:, 2] > 0).astype(int)
            if vis.sum() < args.min_kps:
                continue
            box = _box(kp, vis)
            if max(box[2] - box[0], box[3] - box[1]) < args.min_box:
                continue
            by_image.setdefault(a["image_id"], []).append(dict(is_right=is_right, kp2d=kp.round(2).tolist(),
                                                               vis=vis.tolist(), bbox=box))
    os.makedirs(os.path.join(out, "images", "coco"), exist_ok=True)
    for iid, hands in sorted(by_image.items()):
        im = imgs[iid]
        rel = f"images/coco/{im['file_name']}"
        shutil.copy(os.path.join(args.coco_root, "val2017", im["file_name"]), os.path.join(out, rel))
        recs.append(dict(id=f"coco/{iid}", file=rel, subset="coco", width=im["width"], height=im["height"], K=None,
                         fully_annotated=False, hands=hands))
    print(f"coco: {sum(r['subset'] == 'coco' for r in recs)} images", flush=True)


def build_hint(args, out, recs):
    for src, sub in (("newdays", "hint_nd"), ("epick", "hint_epick")):
        d = os.path.join(args.hint_root, f"TEST_{src}_img")
        os.makedirs(os.path.join(out, "images", sub), exist_ok=True)
        by_frame = {}
        for jf in sorted(glob.glob(os.path.join(d, "*.json"))):
            stem = os.path.splitext(os.path.basename(jf))[0]
            frame = stem[:-2] if stem.endswith(("_l", "_r")) else stem
            for a in json.load(open(jf)):
                kp = np.asarray(a["keypoints"], dtype=np.float64)
                vis = (np.asarray(a["existence"]) > 0).astype(int)
                if vis.sum() < args.min_kps:
                    continue
                by_frame.setdefault(frame, dict(img=jf[:-5] + ".jpg", hands=[]))["hands"].append(dict(
                    is_right=stem.endswith("_r"), kp2d=kp.round(2).tolist(), vis=vis.tolist(), bbox=_box(kp, vis),
                    occluded=[int(o > 0) for o in a["occlusion"]]))
        for frame, v in sorted(by_frame.items()):
            im = cv2.imread(v["img"])
            if im is None:
                continue
            rel = f"images/{sub}/{frame}.jpg"
            cv2.imwrite(os.path.join(out, rel), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            recs.append(dict(id=f"{sub}/{frame}", file=rel, subset=sub, width=im.shape[1], height=im.shape[0], K=None,
                             fully_annotated=False, hands=v["hands"]))
        print(f"{sub}: {sum(r['subset'] == sub for r in recs)} images", flush=True)


def main(args):
    os.makedirs(args.out, exist_ok=True)
    recs = []
    if args.hint_root:
        build_hint(args, args.out, recs)
    if args.cmu_root:
        build_cmu(args, args.out, recs)
    if args.coco_root:
        build_coco(args, args.out, recs)
    if args.hot3d_root:
        build_hot3d(args, args.out, recs)
    stats = {}
    for r in recs:
        s = stats.setdefault(r["subset"], dict(images=0, hands=0, right=0))
        s["images"] += 1
        s["hands"] += len(r["hands"])
        s["right"] += sum(h["is_right"] for h in r["hands"])
    meta = dict(version=args.version, keypoints="OpenPose 21 (wrist, thumb1-4, index1-4, middle1-4, ring1-4, pinky1-4)",
                stats=stats, params=dict(min_kps=args.min_kps, min_box=args.min_box, hot3d_stride=args.hot3d_stride),
                sources=dict(hot3d="bop-benchmark/hot3d HOT3D-Clips (Meta, research license), HaWoR val sequences",
                             cmu="CMU Panoptic hand_labels manual_test (Simon et al. 2017)",
                             coco="COCO-WholeBody v1.0 val (Jin et al. 2020) on COCO val2017 images",
                             hint="HInt TEST New Days + EPIC-KITCHENS VISOR (Pavlakos et al. 2024, MIT; Ego4D part excluded)"))
    json.dump(dict(meta=meta, images=recs), open(os.path.join(args.out, "annotations.json"), "w"))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--version", default="bench2d_v2")
    p.add_argument("--hot3d_root", default="")
    p.add_argument("--hot3d_split", default="bench/splits/hot3d_bench.json")
    p.add_argument("--hot3d_stride", type=int, default=10)
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--cmu_root", default="")
    p.add_argument("--coco_root", default="")
    p.add_argument("--hint_root", default="", help="HInt_annotation(_partial) dir (v2)")
    p.add_argument("--min_kps", type=int, default=8)
    p.add_argument("--min_box", type=float, default=32)
    main(p.parse_args())
