"""Aggregate metrics for one method run (a directory of per-clip pickles from bench.run).

Pose metrics are over matched hands; the coverage-aware "-p" variants (ViDiHand, arXiv 2606.30308, Sec. 4.1 /
App. A.1) average over matched + missed scored GT hands (false positives excluded), charging every missed hand
(or a prediction without finite joints) the error of a canonical MANO hand -- identity global orientation, zero
articulation, mean shape, zero translation -- so a miss costs the same for every method. The placeholder is
wrist-aligned and not Procrustes-aligned (it has no pose to align; about 130 mm per miss), and the same per-hand
charge enters pa_mpjpe_p, rr_mpjpe_p and pa_mpvpe_p. Needs --mano_dir.
Missing or nonfinite meshes receive the vertex miss charge without discarding valid joint predictions.
Coverage metrics remain defined when every prediction is missing or invalid, provided canonical assets are supplied.

    python -m bench.evaluate output/bench/hot3d/fah-image [--iou 0.3] [--mano_dir $MANO_DIR]
-> <dir>/metrics.json
"""
import argparse
import glob
import json
import os
import pickle
from collections import defaultdict

import numpy as np

from bench.common import VertexJoints, accel_error, box_iou, hand_metrics, match_hands, procrustes


def canonical_hands(mano_dir):
    """Canonical MANO per side (identity orientation, flat, mean shape, zero translation): joints, vertices."""
    from bench.common import ManoLayer

    ml = ManoLayer(mano_dir, flat_hand_mean=True)
    return {s: ml(s, np.zeros(10), np.zeros(45), np.zeros(3), np.zeros(3)) for s in (True, False)}


def miss_charge(canon, g):
    """Error (mm) charged to a missed GT hand g: wrist-aligned canonical joints / vertices vs GT."""
    cj, cv = canon[bool(g["is_right"])]
    gj = g["joints3d"]
    out = {"j": float(np.linalg.norm((cj - cj[:1]) - (gj - gj[:1]), axis=1).mean() * 1000)}
    if g.get("verts3d") is not None:
        out["v"] = float(np.linalg.norm((cv - cj[:1]) - (g["verts3d"] - gj[:1]), axis=1).mean() * 1000)
    return out


def evaluate_dir(pred_dir, iou_thresh=0.3, mano_dir=None, key="joints3d"):
    vj = VertexJoints(mano_dir) if mano_dir else None
    canon = canonical_hands(mano_dir) if mano_dir else None
    cov = defaultdict(list)  # coverage-aware per-GT-hand errors (TP: measured, FN: canonical charge)
    files = sorted(glob.glob(os.path.join(pred_dir, "*.pkl")))
    per_hand = defaultdict(list)
    det = dict(gt=0, matched=0, matched50=0, fp=0, side_ok=0, nonfinite=0)
    mrrpe, accel, jitter, latency, hands_per_frame = [], [], [], [], []
    method = None
    for path in files:
        with open(path, "rb") as f:
            clip = pickle.load(f)
        method = clip["method"]
        tracks = defaultdict(list)  # GT side -> [(frame index, pred joints, gt joints)]
        for fr in clip["frames"]:
            if key != "joints3d":  # evaluate another prediction field (e.g. joints3d_head)
                fr["pred"] = [dict(p, joints3d=p.get(key)) for p in fr["pred"]]
            gts, preds = fr["gt"], fr["pred"]
            latency.append(fr["latency"])
            hands_per_frame.append(len(preds))
            matches, fp = match_hands(preds, gts, iou_thresh)
            if fp and fr.get("ignore"):  # detections on unscored (mostly out-of-view) GT hands are not FPs
                used = {pi for _, pi in matches if pi is not None}
                rest = [p for k, p in enumerate(preds) if k not in used]
                _, fp = match_hands(rest, fr["ignore"], iou_thresh)
            det["fp"] += fp
            roots = {}
            for gi, pi in matches:
                g = gts[gi]
                det["gt"] += 1
                p = preds[pi] if pi is not None else {}
                joints = p.get("joints3d")
                verts = p.get("verts3d")
                bad_joints = joints is not None and not np.isfinite(joints).all()
                bad_verts = verts is not None and not np.isfinite(verts).all()
                if bad_joints or bad_verts:
                    det["nonfinite"] += 1
                if joints is None or bad_joints:
                    if canon is not None:
                        c = miss_charge(canon, g)
                        cov["pa_mpjpe"].append(c["j"])
                        cov["rr_mpjpe"].append(c["j"])
                        if "v" in c:
                            cov["pa_mpvpe"].append(c["v"])
                if pi is None:
                    continue
                det["matched"] += 1
                det["matched50"] += int(box_iou(p["bbox"], g["bbox"]) >= 0.5)
                side_ok = bool(p["is_right"]) == bool(g["is_right"])
                det["side_ok"] += int(side_ok)
                if joints is None or bad_joints:
                    continue
                # An invalid mesh must not discard valid joint metrics or disappear
                # from the coverage denominator. Charge it as a missing mesh below.
                if bad_verts:
                    verts = None
                m = hand_metrics(joints, g["joints3d"], verts, g.get("verts3d"))
                m["side_ok"] = side_ok
                if vj is not None and verts is not None and g.get("verts3d") is not None:
                    pj, gj = vj(verts), vj(g["verts3d"])  # same joint definition for all methods
                    m["pa_mpjpe_v"] = float(np.linalg.norm(procrustes(pj, gj) - gj, axis=1).mean() * 1000)
                for k, v in m.items():
                    per_hand[k].append(v)
                for k in ("pa_mpjpe", "rr_mpjpe", "pa_mpvpe"):
                    if k in m:
                        cov[k].append(m[k])
                    elif k == "pa_mpvpe" and canon is not None and g.get("verts3d") is not None:
                        cov[k].append(miss_charge(canon, g)["v"])  # no predicted mesh: charged like a miss
                roots[g["is_right"]] = (p["joints3d"][0], g["joints3d"][0])
                tracks[g["is_right"]].append((fr["index"], p["joints3d"], g["joints3d"]))
            if len(roots) == 2:
                (pr, gr), (pl, gl) = roots[True], roots[False]
                mrrpe.append(np.linalg.norm((pr - pl) - (gr - gl)) * 1000)
        if not clip.get("video", clip.get("dataset") == "hot3d"):
            continue  # image datasets: no temporal metrics
        for side, tr in tracks.items():
            for seg in _segments(tr, clip.get("stride", 1)):  # runs of consecutive sampled frames
                if len(seg) >= 3:
                    p = np.stack([s[1] for s in seg])
                    g = np.stack([s[2] for s in seg])
                    w = len(seg) - 2  # weight by number of acceleration samples
                    accel.append((accel_error(p, g), w))
                    pr = p - p[:, :1]
                    jitter.append((float(np.linalg.norm(pr[2:] - 2 * pr[1:-1] + pr[:-2], axis=-1).mean() * 1000), w))

    def mean(k, mask=None):
        v = np.asarray(per_hand.get(k, []), dtype=np.float64)
        if mask is not None:
            v = v[mask]
        return float(v.mean()) if len(v) else None

    side_ok = np.asarray(per_hand.get("side_ok", []), dtype=bool)
    res = dict(
        method=method, num_clips=len(files), num_frames=len(latency),
        det_recall=det["matched"] / max(det["gt"], 1), det_recall_iou50=det["matched50"] / max(det["gt"], 1),
        det_precision=det["matched"] / max(det["matched"] + det["fp"], 1),
        handedness_acc=det["side_ok"] / max(det["matched"], 1),
        num_gt_hands=det["gt"], num_matched=det["matched"], num_fp=det["fp"], num_nonfinite_pred=det["nonfinite"],
        pa_mpjpe=mean("pa_mpjpe"), pa_mpjpe_v=mean("pa_mpjpe_v"), pa_mpvpe=mean("pa_mpvpe"), f5=mean("f5"), f15=mean("f15"),
        rr_mpjpe=mean("rr_mpjpe"), cam_mpjpe=mean("cam_mpjpe"), root_err=mean("root_err"),
        pa_mpjpe_side_ok=mean("pa_mpjpe", side_ok) if len(side_ok) else None,
        pa_mpjpe_p=float(np.mean(cov["pa_mpjpe"])) if cov["pa_mpjpe"] else None,
        rr_mpjpe_p=float(np.mean(cov["rr_mpjpe"])) if cov["rr_mpjpe"] else None,
        pa_mpvpe_p=float(np.mean(cov["pa_mpvpe"])) if cov["pa_mpvpe"] else None,
        det_f1=2 * det["matched"] / max(2 * det["matched"] + det["fp"] + (det["gt"] - det["matched"]), 1),
        mrrpe=float(np.mean(mrrpe)) if mrrpe else None,
        accel_err=_wmean(accel), jitter=_wmean(jitter),
        latency_ms_median=float(np.median(latency) * 1000) if latency else None,
        latency_ms_mean=float(np.mean(latency) * 1000) if latency else None,
        hands_per_frame=float(np.mean(hands_per_frame)) if hands_per_frame else None,
    )
    return res


def _wmean(vals):
    if not vals:
        return None
    v, w = np.array([a for a, _ in vals]), np.array([b for _, b in vals], dtype=np.float64)
    return float((v * w).sum() / w.sum())


def _segments(track, stride=1):
    """Split [(frame_idx, ...)] into runs of consecutive sampled frames (index step == stride)."""
    if not track:
        return []
    segs, cur = [], [track[0]]
    for prev, item in zip(track[:-1], track[1:]):
        if item[0] - prev[0] == stride:
            cur.append(item)
        else:
            segs.append(cur)
            cur = [item]
    segs.append(cur)
    return segs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pred_dir")
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--mano_dir", default=None, help="enables pa_mpjpe_v (joints from MANO vertices, same for all)")
    p.add_argument("--key", default="joints3d", help="prediction field with the 21 joints")
    a = p.parse_args()
    res = evaluate_dir(a.pred_dir, a.iou, a.mano_dir, a.key)
    with open(os.path.join(a.pred_dir, "metrics.json" if a.key == "joints3d" else f"metrics_{a.key}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
