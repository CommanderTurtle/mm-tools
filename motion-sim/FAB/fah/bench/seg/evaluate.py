"""Hand instance-segmentation evaluation for bench_seg predictions (pickles from bench.run --dataset seg).

Per frame, predicted instance masks (pred["mask_rle"]) are matched to GT hand instances greedily by mask
IoU (>= 0.1 to count as a match; handedness is not used for matching, it is scored separately).
Reported:
  miou        mean IoU over GT hands (unmatched GT = 0)        miou_matched   over matched GT hands only
  recall50 / precision50 / f1_50   instance-level at IoU >= 0.5 (VISOR frames are exhaustively annotated)
  handedness_acc   on matched hands       boundary F (2 px)   on matched hands       latency
    python -m bench.seg.evaluate output/bench/seg/<run>
"""
import argparse
import glob
import json
import os
import pickle

import cv2
import numpy as np

from bench.seg.rle import decode


def boundary_f(a, b, tol=2):
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)

    def edge(m):
        m = m.astype(np.uint8)
        return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0

    ea, eb = edge(a), edge(b)
    if not ea.any() or not eb.any():
        return 0.0
    pa = (ea & (cv2.dilate(eb.astype(np.uint8), k) > 0)).sum() / ea.sum()
    pb = (eb & (cv2.dilate(ea.astype(np.uint8), k) > 0)).sum() / eb.sum()
    return 0.0 if pa + pb == 0 else float(2 * pa * pb / (pa + pb))


def evaluate_dir(pred_dir, key="mask_rle"):
    ious, ious_m, bfs, side, lat = [], [], [], [], []
    tp50 = n_gt = n_pred = 0
    method = None
    for path in sorted(glob.glob(os.path.join(pred_dir, "*.pkl"))):
        clip = pickle.load(open(path, "rb"))
        method = clip["method"]
        for fr in clip["frames"]:
            lat.append(fr["latency"])
            gts = [decode(g["mask_rle"]) for g in fr["gt"]]
            preds = [p for p in fr["pred"] if p.get(key) is not None]
            pm = [decode(p[key]) for p in preds]
            n_gt += len(gts)
            n_pred += len(pm)
            cand = []
            for gi, g in enumerate(gts):
                for pi, p in enumerate(pm):
                    inter = np.logical_and(g, p).sum()
                    if inter:
                        cand.append((inter / np.logical_or(g, p).sum(), gi, pi))
            cand.sort(reverse=True)
            used_g, used_p, match = set(), set(), {}
            for iou, gi, pi in cand:
                if iou < 0.1 or gi in used_g or pi in used_p:
                    continue
                used_g.add(gi)
                used_p.add(pi)
                match[gi] = (pi, iou)
            for gi, g in enumerate(gts):
                if gi in match:
                    pi, iou = match[gi]
                    ious.append(iou)
                    ious_m.append(iou)
                    tp50 += iou >= 0.5
                    side.append(bool(preds[pi]["is_right"]) == bool(fr["gt"][gi]["is_right"]))
                    bfs.append(boundary_f(g, pm[pi]))
                else:
                    ious.append(0.0)
    rec, prec = tp50 / max(n_gt, 1), tp50 / max(n_pred, 1)
    return dict(method=method, frames=len(lat), gt_hands=n_gt, pred_hands=n_pred,
                miou=float(np.mean(ious)) if ious else None, miou_matched=float(np.mean(ious_m)) if ious_m else None,
                recall50=rec, precision50=prec, f1_50=2 * rec * prec / max(rec + prec, 1e-9),
                handedness_acc=float(np.mean(side)) if side else None, boundary_f=float(np.mean(bfs)) if bfs else None,
                latency_ms_median=float(np.median(lat) * 1000) if lat else None)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pred_dir")
    p.add_argument("--key", default="mask_rle")
    a = p.parse_args()
    res = evaluate_dir(a.pred_dir, a.key)
    name = "metrics_seg.json" if a.key == "mask_rle" else f"metrics_seg_{a.key}.json"
    json.dump(res, open(os.path.join(a.pred_dir, name), "w"), indent=1)
    print(json.dumps(res, indent=1))
