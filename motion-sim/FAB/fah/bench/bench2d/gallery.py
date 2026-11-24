"""Side-by-side 2D keypoint gallery for bench2d runs: GT | method 1 | method 2 ... per sampled image.

    python -m bench.bench2d.gallery --root $DATA_ROOT/bench2d_v2 \
        --runs $RESULTS/bench2d/fah-image,$RESULTS/bench2d/wilor \
        --out output/bench2d_gallery --per_subset 16
"""
import argparse
import glob
import json
import os
import pickle

import cv2
import numpy as np

from bench.bench2d.evaluate import match

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
FINGER_COLORS = [(255, 255, 255), (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0), (255, 0, 255)]


def draw_hand(img, kp, vis=None, thick=2):
    for a, b in BONES:
        if vis is not None and not (vis[a] and vis[b]):
            continue
        c = FINGER_COLORS[1 + (b - 1) // 4]
        cv2.line(img, tuple(int(v) for v in kp[a]), tuple(int(v) for v in kp[b]), c, thick, cv2.LINE_AA)
    for j, p in enumerate(kp):
        if vis is None or vis[j]:
            cv2.circle(img, (int(p[0]), int(p[1])), thick + 1, (255, 255, 255), -1, cv2.LINE_AA)


def load_run(d):
    """d = run_dir[:key[:label]] (key = prediction field to draw, default joints2d)."""
    d, key, label = (d.split(":") + ["joints2d", None])[:3]
    frames = {}
    for path in sorted(glob.glob(os.path.join(d, "*.pkl"))):
        clip = pickle.load(open(path, "rb"))
        for fr in clip["frames"]:
            for p in fr["pred"]:
                if p.get(key) is not None:
                    p["joints2d"] = p[key]
            frames[fr["key"]] = fr
    return label or os.path.basename(d.rstrip("/")), frames


def main(args):
    runs = [load_run(d) for d in args.runs.split(",")]
    ann = json.load(open(args.ann or os.path.join(args.root, "annotations.json")))
    recs = {r["id"]: r for r in ann["images"]}
    keys = sorted(set.intersection(*[set(f) for _, f in runs]))
    rng = np.random.RandomState(args.seed)
    os.makedirs(args.out, exist_ok=True)
    picked = set(l.strip() for l in open(args.keys) if l.strip()) if args.keys else None  # --keys: these images only
    for sub in args.subsets.split(","):
        ks = [k for k in keys if (recs[k].get("subset") or k.split("/")[0]) == sub and (picked is None or k in picked)]
        for k in rng.choice(ks, min(args.per_subset, len(ks)), replace=False):
            rec = recs[k]
            img = cv2.imread(os.path.join(args.root, rec["file"]))
            gts = runs[0][1][k]["gt"] if not args.ann else []
            if not gts:  # evolving labels (hard-pose / real dark): take the current annotation file's hands
                gts = [dict(joints2d=np.asarray(h["kp2d"], np.float64), vis=np.asarray(h["vis"]), bbox=np.asarray(h["bbox"]),
                            is_right=h["is_right"]) for h in rec["hands"] if h.get("status") not in ("needs_label", "ignore")]
            if not gts:
                continue
            matches = {name: match(frames[k]["pred"], gts)[0] for name, frames in runs}
            # one panel row per image (default) or per GT hand (--per_hand: crop around that hand, its match only)
            views = [(gi, [gts[gi]]) for gi in range(len(gts))] if args.per_hand else [(None, gts)]
            for gsel, gv in views:
                boxes = np.array([g["bbox"] for g in gv])
                x1, y1 = boxes[:, :2].min(0)
                x2, y2 = boxes[:, 2:].max(0)
                pad = (0.5 if args.per_hand else 0.35) * max(x2 - x1, y2 - y1)
                x1, y1 = int(max(x1 - pad, 0)), int(max(y1 - pad, 0))
                x2, y2 = int(min(x2 + pad, img.shape[1])), int(min(y2 + pad, img.shape[0]))
                th = max(2, int(round(2 * max(x2 - x1, y2 - y1) / args.tile)))  # visible after downscaling
                panels = []
                gt_img = img.copy()
                for g in gv:
                    draw_hand(gt_img, g["joints2d"], g["vis"] > 0, thick=th)
                panels.append((args.ref_label, gt_img))
                for name, frames in runs:
                    fr = frames[k]
                    pi = img.copy()
                    m = matches[name]
                    errs = []
                    for gi, p in m.items():
                        if gsel is not None and gi != gsel:
                            continue
                        g = gts[gi]
                        S = max(g["bbox"][2] - g["bbox"][0], g["bbox"][3] - g["bbox"][1], 1.0)
                        v = g["vis"] > 0
                        errs.append(np.linalg.norm(np.asarray(fr["pred"][p]["joints2d"])[v] - g["joints2d"][v], axis=1).mean() / S)
                    shown = [fr["pred"][m[gsel]]] if gsel is not None and gsel in m else ([] if gsel is not None else fr["pred"])
                    if args.mesh and (not args.mesh_runs or name in args.mesh_runs.split(",")) and any(p.get("verts3d") is not None for p in shown):
                        from bench.common import ManoLayer
                        from fah.vis import LEFT_COLOR, RIGHT_COLOR, render_meshes

                        faces = getattr(args, "_faces", None) or ManoLayer(os.environ.get("MANO_DIR", "data/mano")).faces
                        args._faces = faces
                        from fah.estimator import default_cam_int

                        Kf = fr["K"] if fr.get("K") is not None else default_cam_int(*img.shape[:2])  # images: the estimator's default camera
                        pi = render_meshes(pi, [(np.asarray(p["verts3d"], np.float64), faces[bool(p["is_right"])],
                                                 RIGHT_COLOR if p["is_right"] else LEFT_COLOR) for p in shown if p.get("verts3d") is not None],
                                           np.asarray(Kf, np.float64))
                    else:
                        for p in shown:
                            if p.get("joints2d") is not None:
                                draw_hand(pi, np.asarray(p["joints2d"]), thick=th)
                    label = f"{name}  err {np.mean(errs):.3f}" if errs else f"{name}  (miss)"
                    panels.append((label, pi))
                tiles = []
                for label, p in panels:
                    t = p[y1:y2, x1:x2]
                    s = args.tile / max(t.shape[:2])
                    t = cv2.resize(t, (int(t.shape[1] * s), int(t.shape[0] * s)), interpolation=cv2.INTER_AREA)
                    canvas = np.zeros((args.tile + 28, args.tile, 3), np.uint8)
                    canvas[28 : 28 + t.shape[0], : t.shape[1]] = t
                    cv2.putText(canvas, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                    tiles.append(canvas)
                suffix = "" if gsel is None else f"_h{gsel:02d}"
                cv2.imwrite(os.path.join(args.out, f"{sub}_{k.split('/')[-1].replace(':', '_')}{suffix}.jpg"), np.concatenate(tiles, 1),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", len(os.listdir(args.out)), "panels to", args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per_subset", type=int, default=16)
    p.add_argument("--tile", type=int, default=420)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mesh", action="store_true", help="render the predicted meshes (MANO faces) instead of 2D joints")
    p.add_argument("--mesh_runs", default="", help="run labels to draw as meshes (others as joints; default all)")
    p.add_argument("--subsets", default="hot3d,cmu,coco,hint_nd,hint_epick")
    p.add_argument("--ann", default="", help="labels file overriding the GT stored in the prediction pickles")
    p.add_argument("--per_hand", action="store_true", help="one row per GT hand (crowded images, e.g. pose sheets)")
    p.add_argument("--ref_label", default="reference (GT)")
    p.add_argument("--keys", default="", help="file with one image key per line: sample only among these")
    main(p.parse_args())
