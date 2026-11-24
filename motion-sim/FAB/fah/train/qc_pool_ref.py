"""Label-convention QC of a new labelled crop pool against a reference model that never saw it (e.g. the FAH-L network
with the released detector), and the in-domain agreement of a trained network with a held-out labelled pool: the full
pipeline runs on each stored crop (right-hand oriented, so the expected side is always right), the detection with the
best IoU to the label box is matched. Per crop: predicted side, 2D error / box
side, PA-MPJPE (mm), per-joint offset of the PA-aligned prediction in the label's hand frame, pred / label bone ratios.
Summary = medians (+ the side-flip share); systematic offsets / ratios point at conventions, a wide spread at fit noise.

    python -m train.qc_pool_ref --pool $DATA_ROOT/crops/mow_test --ckpt $CKPT_ROOT/finetune/<run>/final.pt \
        --detector <weights>/shared/detector.pt --num 600 --device cpu --out <dir>/mow_test.json
"""
import argparse
import json
import os

import cv2
import numpy as np

from bench.common import procrustes
from train.build_crops import crop_box
from train.pool_proportions import BONES

NAMES = ["wrist", "t1", "t2", "t3", "t4", "i1", "i2", "i3", "i4", "m1", "m2", "m3", "m4", "r1", "r2", "r3", "r4", "p1", "p2", "p3", "p4"]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(x2 - x1, 0) * max(y2 - y1, 0)
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def hand_frame(j):
    """Rows = axes of the hand frame of 21 OpenPose joints: y wrist -> middle MCP, z palm normal (index x pinky MCP), x."""
    y = j[9] - j[0]
    y /= np.linalg.norm(y)
    z = np.cross(j[5] - j[0], j[17] - j[0])
    z -= y * (z @ y)
    z /= np.linalg.norm(z)
    return np.stack([np.cross(y, z), y, z])


def bones(j):
    return np.array([np.linalg.norm(j[b] - j[a]) for a, b in BONES.values()])


def main(args):
    from bench.methods.fah import FAHMethod

    recs = [json.loads(l) for l in open(os.path.join(args.pool, "index.jsonl"))]
    recs = [r for r in recs if "joints3d" in r]
    if args.num and len(recs) > args.num:
        recs = [recs[i] for i in sorted(np.random.default_rng(0).choice(len(recs), args.num, replace=False))]
    m = FAHMethod(checkpoint=args.checkpoint, device=args.device, finetune_ckpt=args.ckpt, verify=True, min_crop=1.1,
                      dtype=None if args.device == "cpu" else "bfloat16", detector_weights=args.detector, det_conf=0.05,
                      recrop_border=0.03, pair_keep=0.15, tip_extend=0.6, pair_wrist=0.5)
    rows = []
    for n, r in enumerate(recs):
        img = cv2.imread(os.path.join(args.pool, r["file"]))
        A = np.asarray(r["M"], np.float64)
        K = np.asarray(r["K"], np.float64).copy()  # crop intrinsics (the crop is a scale + shift of the right-oriented frame)
        K[:2] = A @ K
        box = crop_box(r)
        preds = m(img, K.astype(np.float32))
        row = dict(file=r["file"], found=False)
        if preds:
            p = max(preds, key=lambda q: iou(np.asarray(q["bbox"], np.float64), box))
            if iou(np.asarray(p["bbox"], np.float64), box) >= 0.3:
                gt3, gt2 = np.asarray(r["joints3d"], np.float64), np.asarray(r["kp2d"], np.float64)
                pr3, pr2 = np.asarray(p["joints3d"], np.float64), np.asarray(p["joints2d"], np.float64)
                side = max(box[2] - box[0], box[3] - box[1])
                ok = gt2[:, 2] > 0
                al = procrustes(pr3, gt3)
                off = (al - gt3) @ hand_frame(gt3).T * 1000
                sh = (gt2[ok, :2] - pr2[ok]).mean(0) if ok.any() else np.zeros(2)  # label - prediction, mean over joints
                row.update(found=True, right=bool(p["is_right"]), score=float(p["score"]), side=float(side), pred2d=pr2.round(2).tolist(),
                           err2d=float(np.linalg.norm(pr2[ok] - gt2[ok, :2], axis=1).mean() / side) if ok.any() else None,
                           shift2d=(sh / side).round(4).tolist(),
                           err2d_t=float(np.linalg.norm(pr2[ok] + sh - gt2[ok, :2], axis=1).mean() / side) if ok.any() else None,
                           pa=float(np.linalg.norm(al - gt3, axis=1).mean() * 1000), off=off.round(2).tolist(),
                           ratio=(bones(al) / np.maximum(bones(gt3), 1e-6)).round(4).tolist())
        rows.append(row)
        if n % 50 == 0:
            print(n, len(recs), row.get("pa"), flush=True)
    ok = [x for x in rows if x["found"]]
    off = np.array([x["off"] for x in ok]) if ok else np.zeros((0, 21, 3))
    ratio = np.array([x["ratio"] for x in ok]) if ok else np.zeros((0, len(BONES)))
    e2 = np.array([x["err2d"] for x in ok if x["err2d"] is not None])
    summ = dict(pool=os.path.basename(args.pool.rstrip("/")), crops=len(rows), matched=len(ok),
                side_flip=float(np.mean([not x["right"] for x in ok])) if ok else None,
                pa_median=float(np.median([x["pa"] for x in ok])) if ok else None,
                pa_p90=float(np.percentile([x["pa"] for x in ok], 90)) if ok else None,
                err2d_median=float(np.median(e2)) if len(e2) else None, pck2d_05=float(np.mean(e2 < 0.05)) if len(e2) else None,
                err2d_t_median=float(np.median([x["err2d_t"] for x in ok if x["err2d_t"] is not None])) if len(e2) else None,
                shift2d_median=np.median([x["shift2d"] for x in ok], 0).round(4).tolist() if ok else None,
                offset_median_mm={NAMES[j]: np.median(off[:, j], 0).round(2).tolist() for j in (0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20)},
                bone_ratio_median={k: round(float(v), 3) for k, v in zip(BONES, np.median(ratio, 0))} if ok else {})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(dict(summary=summ, rows=rows), open(args.out, "w"))
    print(json.dumps(summ, indent=1))
    if args.sheet and ok:  # label (green) vs reference (magenta) at evenly spaced err2d quantiles; dataset frames: keep private
        from bench.bench2d.gallery import draw_hand

        by = sorted([x for x in ok if x["err2d"] is not None], key=lambda x: x["err2d"])
        pick = [by[int(q)] for q in np.linspace(0, len(by) - 1, min(args.sheet, len(by)))]
        lab = {r["file"]: r for r in recs}
        tiles = []
        for x in pick:
            r = lab[x["file"]]
            b = crop_box(r)
            img = cv2.imread(os.path.join(args.pool, r["file"]))
            pa, pb = img.copy(), img.copy()
            draw_hand(pa, np.asarray(r["kp2d"])[:, :2], thick=2)
            draw_hand(pb, np.asarray(x["pred2d"]), thick=2)
            c = ((b[:2] + b[2:]) / 2).astype(int)
            s = int(max(b[2] - b[0], b[3] - b[1]) * 0.75) + 8
            win = lambda im: cv2.resize(im[max(c[1] - s, 0):c[1] + s, max(c[0] - s, 0):c[0] + s], (240, 240))
            t = np.hstack([win(pa), win(pb)])
            cv2.putText(t, f"{x['err2d']:.3f} pa{x['pa']:.1f} {'R' if x['right'] else 'L'}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            tiles.append(t)
        cols = 4
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
        sheet = np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])
        cv2.imwrite(args.out.replace(".json", "_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", required=True, help="crop pool dir with index.jsonl (joints3d, kp2d, K, M)")
    p.add_argument("--out", required=True)
    p.add_argument("--num", type=int, default=0, help="random subset (0 = all)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sheet", type=int, default=0, help="label vs reference sheet of this many crops (err2d quantiles)")
    p.add_argument("--checkpoint", default="checkpoints/sam-3d-body-dinov3/model.ckpt")
    p.add_argument("--ckpt", required=True, help="pose network (train.finetune checkpoint)")
    p.add_argument("--detector", required=True, help="detector weights, e.g. <weights>/shared/detector.pt")
    main(p.parse_args())
