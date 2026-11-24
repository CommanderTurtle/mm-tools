"""2D keypoint evaluation for bench2d predictions (pickles written by bench.run --dataset bench2d).

Per GT hand, error e_j = ||pred_j - gt_j|| / S over GT-visible joints, S = max side of the GT
keypoint box. Predictions are matched to GT hands per image by mean e (greedy, e < 0.5); handedness
is NOT used for matching (it is scored separately). Reported per subset and overall:
  pck05/pck10/pck20, auc (mean PCK over thresholds 0..0.2), mean_err (normalized), epe_px  -- on matched hands
  *_all variants count unmatched GT hands as failed (all joints wrong) -> penalises missed detections
  recall, handedness_acc, precision (fully annotated subsets only), latency
    python -m bench.bench2d.evaluate output/bench/bench2d/<run>
"""
import argparse
import glob
import json
import os
import pickle
from collections import defaultdict

import numpy as np

THRESH = np.linspace(0, 0.2, 21)[1:]
TIPS = np.array([4, 8, 12, 16, 20])


def _norm_err(pred_kp, gt):
    vis = gt["vis"] > 0
    S = max(gt["bbox"][2] - gt["bbox"][0], gt["bbox"][3] - gt["bbox"][1], 1.0)
    d = np.linalg.norm(pred_kp - gt["joints2d"], axis=1)
    return d[vis] / S, d[vis]


def match(preds, gts, max_err=0.5):
    cands = []
    for gi, g in enumerate(gts):
        for pi, p in enumerate(preds):
            if p.get("joints2d") is None:
                continue
            e, _ = _norm_err(np.asarray(p["joints2d"]), g)
            if not len(e) or not np.isfinite(e).all():  # unusable pose: never matched, stays an unmatched prediction
                continue
            cands.append((float(e.mean()), gi, pi))
    cands.sort()
    used_g, used_p, out = set(), set(), {}
    for e, gi, pi in cands:
        if e > max_err or gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out[gi] = pi
    return out, len(preds) - len(used_p)


def fit_project(j3, g, K):
    """Root-relative 3D joints placed at the camera translation (linear least squares) that best reprojects them onto
    the visible GT 2D joints, then projected."""
    X, vis = j3 - j3[:1], g["vis"] > 0
    d = g["joints2d"][vis] - K[:2, 2]
    n = int(vis.sum())
    M = np.zeros((2 * n, 3))
    M[:n, 0], M[n:, 1], M[:n, 2], M[n:, 2] = K[0, 0], K[1, 1], -d[:, 0], -d[:, 1]
    r = np.concatenate([d[:, 0] * X[vis, 2] - K[0, 0] * X[vis, 0], d[:, 1] * X[vis, 2] - K[1, 1] * X[vis, 1]])
    t = np.linalg.lstsq(M, r, rcond=None)[0]
    P = X + t
    return (P @ K.T)[:, :2] / np.clip(P[:, 2:], 1e-4, None)


def evaluate_dir(pred_dir, key="joints2d"):
    acc = defaultdict(lambda: dict(errs=[], px=[], tips=[], n_gt=0, n_match=0, side_ok=0, fp=0, n_pred=0, lat=[], fa_gt=0, fa_match=0))
    method = None
    for path in sorted(glob.glob(os.path.join(pred_dir, "*.pkl"))):
        clip = pickle.load(open(path, "rb"))
        method = clip["method"]
        for fr in clip["frames"]:
            sub = fr.get("subset") or fr["key"].split("/")[0]
            if key == "joints3d_fit":  # 3D articulation vs the GT 2D: matched on the method's 2D, then the root-relative
                preds = list(fr["pred"])  # 3D joints projected at the translation that best fits the GT 2D
                m, fp = match(preds, fr["gt"])
                h, w = fr["image_size"][:2] if fr.get("image_size") is not None else (1000, 1000)
                K = np.asarray(fr["K"], np.float64) if fr.get("K") is not None else \
                    np.array([[np.hypot(h, w), 0, w / 2], [0, np.hypot(h, w), h / 2], [0, 0, 1]])
                preds = [dict(p) for p in preds]
                for gi, pi in m.items():
                    j3 = preds[pi].get("joints3d")
                    uv = None
                    if j3 is not None:
                        j3 = np.asarray(j3, np.float64)
                        if j3.shape == (len(fr["gt"][gi]["joints2d"]), 3) and np.isfinite(j3).all():
                            try:
                                uv = fit_project(j3, fr["gt"][gi], K)
                            except np.linalg.LinAlgError:
                                pass
                    preds[pi]["joints2d"] = uv if uv is not None and np.isfinite(uv).all() else None
                m = {gi: pi for gi, pi in m.items() if preds[pi]["joints2d"] is not None}
                fp = len(preds) - len(m)  # unusable 3D must not disappear from precision accounting
            else:
                preds = [dict(p, joints2d=p.get(key, p.get("joints2d"))) for p in fr["pred"]]
                m, fp = match(preds, fr["gt"])
            for s in (sub, "all"):
                a = acc[s]
                a["lat"].append(fr["latency"])
                a["n_gt"] += len(fr["gt"])
                a["n_match"] += len(m)
                if fr.get("fully_annotated"):
                    a["fp"] += fp
                    a["n_pred"] += len(preds)
                    a["fa_gt"] += len(fr["gt"])
                    a["fa_match"] += len(m)
                for gi, g in enumerate(fr["gt"]):
                    if gi in m:
                        p = preds[m[gi]]
                        e, px = _norm_err(np.asarray(p["joints2d"]), g)
                        a["errs"].append(e)
                        a["px"].append(px)
                        tv = g["vis"][TIPS] > 0  # fingertips only (normalised like e)
                        S = max(g["bbox"][2] - g["bbox"][0], g["bbox"][3] - g["bbox"][1], 1.0)
                        a["tips"].append(np.linalg.norm(np.asarray(p["joints2d"])[TIPS] - g["joints2d"][TIPS], axis=1)[tv] / S)
                        a["side_ok"] += int(bool(p["is_right"]) == bool(g["is_right"]))
                    else:
                        a["errs"].append(int((g["vis"] > 0).sum()))  # missed: its visible joints all fail
    res = {"method": method}
    for s, a in acc.items():
        matched = [e for e in a["errs"] if not isinstance(e, int)]
        allj = np.concatenate(matched) if matched else np.zeros(0)
        n_vis_missed = sum(e for e in a["errs"] if isinstance(e, int))
        pck = lambda t: float((allj < t).mean()) if len(allj) else None
        pck_all = lambda t: float((allj < t).sum() / max(len(allj) + n_vis_missed, 1))
        res[s] = dict(
            images=len(a["lat"]), gt_hands=a["n_gt"], recall=a["n_match"] / max(a["n_gt"], 1),
            handedness_acc=a["side_ok"] / max(a["n_match"], 1),
            precision=(a["fa_match"] / max(a["fa_match"] + a["fp"], 1)) if a["fa_gt"] else None,
            pck05=pck(0.05), pck10=pck(0.1), pck20=pck(0.2),
            auc=float(np.mean([(allj < t).mean() for t in THRESH])) if len(allj) else None,
            mean_err=float(allj.mean()) if len(allj) else None,
            epe_px=float(np.concatenate(a["px"]).mean()) if a["px"] else None,
            pck10_all=pck_all(0.1), auc_all=float(np.mean([pck_all(t) for t in THRESH])),
            tip_pck05=float((np.concatenate(a["tips"]) < 0.05).mean()) if a["tips"] else None,
            tip_pck10=float((np.concatenate(a["tips"]) < 0.1).mean()) if a["tips"] else None,
            latency_ms_median=float(np.median(a["lat"]) * 1000),
        )
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pred_dir")
    p.add_argument("--key", default="joints2d", help="prediction field to score (e.g. joints2d_mhr)")
    a = p.parse_args()
    res = evaluate_dir(a.pred_dir, a.key)
    name = "metrics2d.json" if a.key == "joints2d" else f"metrics2d_{a.key}.json"
    json.dump(res, open(os.path.join(a.pred_dir, name), "w"), indent=1)
    for s in sorted(k for k in res if k != "method"):
        r = res[s]
        print(f"{s:6s} n={r['gt_hands']:5d} recall={r['recall']:.3f} side={r['handedness_acc']:.3f} "
              f"PCK@.05={r['pck05'] or 0:.3f} PCK@.1={r['pck10'] or 0:.3f} AUC={r['auc'] or 0:.3f} "
              f"AUC_all={r['auc_all']:.3f} err={r['mean_err'] or 0:.4f} {r['latency_ms_median']:.0f}ms")
