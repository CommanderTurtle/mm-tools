"""Score stored bench2d-format predictions (pickles of bench.run --dataset bench2d) against the annotations of a set:
the released custom sets in datasets/ (shadow_v1, hard_mined_v2) or a rebuilt bench2d v2 / its synthetic low-light copy.
The GT stored in the pickles at run time is replaced by the hands of --ann; frames of images that are not in --ann are
dropped (e.g. a run on a larger copy of the set); hands with status needs_label / ignore are neither scored nor counted
as false positives; precision only counts images marked fully_annotated. The prediction dirs are not written to.

Paper 2D-table cells: recall, PCK@0.1 on matched hands ("PCK"), PCK@0.1 with missed hands counted as wrong ("incl.");
also PCK@0.05, fingertip PCK@0.1, handedness, precision. --fit3d adds the 3D fit (the 3D joints projected at the
translation that best fits the GT 2D), --overlap the overlapping-hands subset (GT hands whose box overlaps another GT
hand's box, IoU >= 0.1; recall and PCK@0.1 incl. misses).

    python -m bench.custom_eval --ann datasets/shadow_v1/annotations.json --runs $RESULTS/bench2d/shadow-fah-l
    python -m bench.custom_eval --ann $DATA_ROOT/bench2d_v2/annotations.json --runs $RESULTS/bench2d/v2-fah-l --fit3d --overlap
"""
import argparse
import glob
import json
import os
import pickle
import tempfile

import numpy as np

from bench.bench2d.evaluate import _norm_err, evaluate_dir, match
from bench.common import box_iou

SKIP = ("needs_label", "ignore")


def load_ann(path):
    ann = json.load(open(path))
    gt, ignore, recs = {}, {}, {}
    for r in ann["images"]:
        gt[r["id"]] = [dict(is_right=h["is_right"], joints2d=np.asarray(h["kp2d"], np.float32), vis=np.asarray(h["vis"]),
                            bbox=np.asarray(h["bbox"], np.float32), joints3d=None, verts3d=None)
                       for h in r["hands"] if h.get("status") not in SKIP]
        ignore[r["id"]] = [np.asarray(h["bbox"], np.float32) for h in r["hands"] if h.get("status") in SKIP]
        recs[r["id"]] = r
    return ann, gt, ignore, recs


def rescore(run_dir, gt, ignore, recs, key, tmp):
    """Pickles of run_dir with the GT of --ann -> tmp; returns the number of frames kept / dropped."""
    kept = dropped = 0
    for path in sorted(glob.glob(os.path.join(run_dir, "*.pkl"))):
        clip = pickle.load(open(path, "rb"))
        frames = []
        for fr in clip["frames"]:
            r = recs.get(fr["key"])
            if r is None:
                dropped += 1
                continue
            fr["gt"], fr["subset"], fr["fully_annotated"] = gt[fr["key"]], r["subset"], bool(r.get("fully_annotated", True))
            ig = ignore[fr["key"]]
            if ig:  # scored hands are matched first; only the remaining predictions inside an ignored box are exempt
                kp = fr["pred"] if key == "joints3d_fit" else [dict(p, joints2d=p.get(key, p.get("joints2d"))) for p in fr["pred"]]
                hit = set(match(kp, fr["gt"])[0].values())
                fr["pred"] = [p for i, p in enumerate(fr["pred"]) if i in hit or p.get("bbox") is None
                              or all(box_iou(np.asarray(p["bbox"], np.float32), b) < 0.3 for b in ig)]
            frames.append(fr)
        kept += len(frames)
        clip["frames"] = frames
        pickle.dump(clip, open(os.path.join(tmp, os.path.basename(path)), "wb"))
    return kept, dropped


def _iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = w * h
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


def overlap(pred_dir, key="joints2d", min_iou=0.1):
    """Recall / handedness / PCK incl. misses on the GT hands whose box overlaps another GT hand's (IoU >= min_iou)."""
    n = hit = side = 0
    errs = []
    for f in sorted(glob.glob(os.path.join(pred_dir, "*.pkl"))):
        for fr in pickle.load(open(f, "rb"))["frames"]:
            gt = fr["gt"]
            over = [gi for gi, g in enumerate(gt) if any(_iou(g["bbox"], o["bbox"]) >= min_iou for oi, o in enumerate(gt) if oi != gi)]
            if not over:
                continue
            preds = [dict(p, joints2d=p.get(key, p.get("joints2d"))) for p in fr["pred"]]
            m, _ = match(preds, gt)
            for gi in over:
                n += 1
                if gi in m:
                    p = preds[m[gi]]
                    hit += 1
                    side += bool(p["is_right"]) == bool(gt[gi].get("is_right"))
                    e = np.asarray(_norm_err(np.asarray(p["joints2d"]), gt[gi])[0])
                    errs.append((np.mean(e < 0.05), np.mean(e < 0.1)))
                else:
                    errs.append((0.0, 0.0))
    if not n:
        return None
    e = np.asarray(errs)
    return dict(hands=n, recall=hit / n, side=side / max(hit, 1), pck05_all=float(e[:, 0].mean()), pck10_all=float(e[:, 1].mean()))


def main(args):
    ann, gt, ignore, recs = load_ann(args.ann)
    res = {"ann": args.ann, "labels": ann["meta"].get("version") or ann["meta"].get("name"), "runs": {}}
    for spec in args.runs.split(","):
        d, key = (spec.split(":") + ["joints2d"])[:2]
        name = os.path.basename(d.rstrip("/")) + ("" if key == "joints2d" else f":{key}")
        with tempfile.TemporaryDirectory() as tmp:
            kept, dropped = rescore(d, gt, ignore, recs, key, tmp)
            if not kept:
                print(f"{name}: none of its {dropped} frames is an image of --ann (another set, or other image ids)")
                continue
            m = evaluate_dir(tmp, key)
            if args.fit3d:
                m["fit3d"] = evaluate_dir(tmp, "joints3d_fit")
            if args.overlap:
                m["overlap"] = overlap(tmp, key)
        m["frames"] = dict(scored=kept, dropped=dropped, missing=len(recs) - kept)
        res["runs"][name] = m
        a = m["all"]
        line = (f"{name:40s} images {a['images']:5d} hands {a['gt_hands']:5d} recall {a['recall']:.4f} PCK@.1 {a['pck10'] or 0:.4f} "
                f"incl. {a['pck10_all']:.4f} PCK@.05 {a['pck05'] or 0:.4f} tip@.1 {a['tip_pck10'] or 0:.4f} side {a['handedness_acc']:.4f}")
        if args.fit3d:
            line += f" 3Dfit {m['fit3d']['all']['pck10'] or 0:.4f}"
        if args.overlap and m["overlap"]:
            line += f" overlap {m['overlap']['pck10_all']:.4f} ({m['overlap']['hands']} hands)"
        if kept < len(recs):
            line += f"  [{len(recs) - kept} annotated images without predictions]"
        print(line)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ann", required=True, help="annotations.json of the set")
    p.add_argument("--runs", required=True, help="comma-separated prediction dirs, each optionally :<prediction field>")
    p.add_argument("--fit3d", action="store_true")
    p.add_argument("--overlap", action="store_true")
    p.add_argument("--out", default="")
    main(p.parse_args())
