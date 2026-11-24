"""Hand-only proxy of the bench_seg evaluation. VISOR GT instances are hand + visible arm; masks that stop at the
wrist (YOLO-seg detector masks, rendered meshes, the mask head's hand channel) are scored against GT cut at the wrist
line of the matched prediction's 2D keypoints (perpendicular to wrist -> middle MCP, `margin` hand lengths behind the
wrist, as train.qc_masks). Matching as bench.seg.evaluate (on the full GT); unmatched GT hands count as IoU 0.
--cut_ref <run>: the same cut GT for every method (keypoints of the reference run's prediction with the highest mask
IoU to each GT hand; GT hands it misses stay uncut), for methods without (reliably matched) keypoints.

    python -m bench.seg.evaluate_hand $RESULTS/seg/<run> --key mask_det [--cut_ref <ref run> --ref_key mask_det]
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np

from bench.seg.evaluate import boundary_f
from bench.seg.rle import decode


def cut_arm(mask, kp, margin=0.25):
    d = kp[9, :2] - kp[0, :2]
    L = np.linalg.norm(d)
    if L < 4:
        return mask
    d = d / L
    ys, xs = np.nonzero(mask)
    s = (np.stack([xs, ys], 1) - kp[0, :2]) @ d
    out = mask.copy()
    out[ys[s < -margin * L], xs[s < -margin * L]] = False
    return out


def reference_keypoints(ref_dir, ref_key="mask_det", kp_key="joints2d"):
    """frame key -> per GT hand: joints2d of the reference prediction with the highest mask IoU (>= 0.1) or None."""
    out = {}
    for path in sorted(glob.glob(os.path.join(ref_dir, "*.pkl"))):
        for fr in pickle.load(open(path, "rb"))["frames"]:
            preds = [p for p in fr["pred"] if p.get(ref_key) is not None]
            pm = [decode(p[ref_key]) for p in preds]
            kps = []
            for g in fr["gt"]:
                g = decode(g["mask_rle"])
                best, bi = None, 0.1
                for p, m in zip(preds, pm):
                    inter = np.logical_and(g, m).sum()
                    iou = inter / max(np.logical_or(g, m).sum(), 1) if inter else 0.0
                    if iou >= bi:
                        best, bi = p.get(kp_key), iou
                kps.append(best)
            out[fr["key"]] = kps
    return out


def evaluate_dir(pred_dir, key="mask_rle", kp_key="joints2d", margin=0.25, ref_kps=None):
    ious, bfs, cut_frac = [], [], []
    tp50 = n_gt = n_pred = n_nokp = 0
    for path in sorted(glob.glob(os.path.join(pred_dir, "*.pkl"))):
        clip = pickle.load(open(path, "rb"))
        for fr in clip["frames"]:
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
                match[gi] = pi
            for gi, g in enumerate(gts):
                if gi not in match:
                    ious.append(0.0)
                    continue
                pi = match[gi]
                kp = preds[pi].get(kp_key) if ref_kps is None else ref_kps[fr["key"]][gi]
                n_nokp += kp is None  # no keypoints for this prediction (e.g. a SAM arm without a matched hand): GT uncut
                gc = g if kp is None else cut_arm(g, np.asarray(kp, np.float64), margin)
                cut_frac.append(1 - gc.sum() / max(g.sum(), 1))
                iou = np.logical_and(gc, pm[pi]).sum() / max(np.logical_or(gc, pm[pi]).sum(), 1)
                ious.append(float(iou))
                tp50 += iou >= 0.5
                bfs.append(boundary_f(gc, pm[pi]))
    rec, prec = tp50 / max(n_gt, 1), tp50 / max(n_pred, 1)
    return dict(convention=f"hand only (GT cut {margin} hand lengths behind the predicted wrist)", key=key,
                gt_hands=n_gt, pred_hands=n_pred, miou=float(np.mean(ious)) if ious else None,
                recall50=rec, precision50=prec, f1_50=2 * rec * prec / max(rec + prec, 1e-9),
                boundary_f=float(np.mean(bfs)) if bfs else None,
                gt_cut_frac_median=float(np.median(cut_frac)) if cut_frac else None, matched_without_keypoints=n_nokp)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pred_dir")
    p.add_argument("--key", default="mask_rle")
    p.add_argument("--kp_key", default="joints2d")
    p.add_argument("--margin", type=float, default=0.25)
    p.add_argument("--cut_ref", default="", help="reference run whose keypoints cut the GT for every method")
    p.add_argument("--ref_key", default="mask_det")
    a = p.parse_args()
    ref = reference_keypoints(a.cut_ref, a.ref_key, a.kp_key) if a.cut_ref else None
    res = evaluate_dir(a.pred_dir, a.key, a.kp_key, a.margin, ref)
    if ref is not None:
        res["cut_ref"] = a.cut_ref
    tag = ("" if a.margin == 0.25 else f"_m{a.margin:g}") + ("_ref" if ref is not None else "")
    json.dump(res, open(os.path.join(a.pred_dir, f"metrics_seg_hand_{a.key}{tag}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
