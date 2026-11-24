"""Part-segmentation labels for 2D-labelled pools from a trained FAH model (train/part_masks.py layout, route
"model+mask"): the model's mesh on each crop (box mode: the crop's label box, right hand, no handedness flip) rendered
per part (MANO skinning parts), silhouette from the pool's SAM hand mask (label transfer, at most `max_fill` x the hand
size beyond the rendered hand, else ignore); a finger whose 2D joints miss the pool's labelled joints by more than
`max_err` x the hand size is ignore, a hand whose median joint error exceeds it gets no label.

    python -m train.part_labels_model --pool $DATA_ROOT/crops/coco_wb_kin --shard 0 --num_shards 4 --method_args '<json>'
-> <pool>/<out_dir>/<file>.png + <pool>/<out_dir>_index.<shard>.jsonl
"""
import argparse
import json
import os

import cv2
import numpy as np

from train.part_masks import COLORS, label_transfer, mano_parts, render_parts
from train.build_crops import crop_box

FINGER_J = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
FINGER_P = [[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16]]


class _BoxDet:  # the method's detector replaced by the crop's own box (right hand)
    def __init__(self):
        self.box = None

    def __call__(self, img):
        return self.box[None].astype(np.float32), np.ones(1, np.float32), np.ones(1, bool)


def main(args):
    from bench.run import load_method

    faces, fpart = mano_parts(args.mano_dir)
    m = load_method("fah_pipeline", dict(json.loads(os.path.expandvars(args.method_args)), verify=False))
    m.detector = m.est.detector = det = _BoxDet()  # verify off: the estimator calls its own detector reference
    m.schedule = None
    recs = [json.loads(l) for l in open(os.path.join(args.pool, "index.jsonl"))][args.shard::args.num_shards]
    if args.limit:
        recs = recs[: args.limit]
    os.makedirs(os.path.join(args.pool, args.out_dir), exist_ok=True)
    idx_f = open(os.path.join(args.pool, f"{args.out_dir}_index.{args.shard}.jsonl"), "w")
    K = np.array([[args.focal, 0, 320], [0, args.focal, 320], [0, 0, 1]], np.float32)
    n_ok, vis_tiles = 0, []
    for n, r in enumerate(recs):
        if r.get("negative") or r.get("kp2d") is None:
            continue
        mpath = os.path.join(args.pool, args.mask_dir, os.path.splitext(r["file"])[0] + ".png")
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE) if os.path.exists(mpath) else None
        img = cv2.imread(os.path.join(args.pool, r["file"]))
        if mask is None or img is None:
            continue
        det.box = crop_box(r)
        hs = m(img, K)
        if not hs or hs[0].get("verts3d") is None:
            continue
        h = hs[0]
        kp = np.asarray(r["kp2d"], np.float64)
        vis = kp[:, 2] > 0
        if vis.sum() < 6:
            continue
        j2 = np.asarray(h["joints2d"], np.float64)
        size_h = max(np.ptp(kp[vis, 0]), np.ptp(kp[vis, 1]), 1.0)
        err = np.linalg.norm(j2 - kp[:, :2], axis=1) / size_h
        if np.median(err[vis]) > args.max_err:
            continue
        v = np.asarray(h["verts3d"], np.float64)
        uv = (v @ K.T.astype(np.float64))[:, :2] / np.clip(v[:, 2:], 1e-4, None)
        lab = render_parts(uv, v[:, 2], faces, fpart, mask.shape[0])
        lab = label_transfer(lab, mask > 127, max_fill=args.max_fill * size_h)
        for js, ps in zip(FINGER_J, FINGER_P):
            jv = [j for j in js if vis[j]]
            if jv and err[jv].mean() > args.max_err:
                lab[np.isin(lab, ps)] = 255
        out = os.path.join(args.pool, args.out_dir, os.path.splitext(r["file"])[0] + ".png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, lab)
        idx_f.write(json.dumps(dict(file=r["file"], route="model+mask", err=round(float(np.median(err[vis])), 4),
                                    parts=np.unique(lab[(lab > 0) & (lab < 255)]).tolist())) + "\n")
        n_ok += 1
        if args.vis and len(vis_tiles) < 12 and n % 37 == 0:
            col = COLORS[np.minimum(lab, 16)].copy()
            col[lab == 255] = (255, 255, 255)
            bl = np.where((lab > 0)[..., None], (0.45 * img + 0.55 * col).astype(np.uint8), img)
            for (a, c) in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
                           (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]:
                if vis[a] and vis[c]:
                    cv2.line(bl, tuple(kp[a, :2].astype(int)), tuple(kp[c, :2].astype(int)), (255, 255, 255), 1)
            vis_tiles.append(cv2.resize(np.concatenate([img, bl], 1), (512, 256)))
    idx_f.close()
    if args.vis and vis_tiles:
        while len(vis_tiles) % 2:
            vis_tiles.append(np.zeros_like(vis_tiles[0]))
        cv2.imwrite(args.vis, np.concatenate([np.concatenate(vis_tiles[i:i + 2], 1) for i in range(0, len(vis_tiles), 2)], 0))
    print("done", args.pool, args.shard, len(recs), "labelled", n_ok, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--method_args", required=True)
    p.add_argument("--focal", type=float, default=640.0)
    p.add_argument("--max_err", type=float, default=0.1)
    p.add_argument("--max_fill", type=float, default=0.08)
    p.add_argument("--out_dir", default="masks_parts_model")
    p.add_argument("--mask_dir", default="masks", help="hand masks inside the pool")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--vis", default="")
    main(p.parse_args())
