"""Metric-scale calibration of the fast camera (bench/world_dpvo.py) on HOT3D *train* clips.

Per calibration clip: s_gt = the clip's GT-optimal Sim(3) scale between the estimated camera centres (camera npz,
translation in SLAM units) and the GT camera centres (bench.run pickles' T_world_cam); clips whose GT camera path is
shorter than --min_path m are skipped (scale ill-defined). s_est = the scale estimate under test:
  --est first:<per-keyframe key>   first finite per-keyframe estimate (causal; e.g. first:m3d_kf for world_dpvo.py)
  --est <scalar key>               a stored clip scale (e.g. scale_m3dm_1)
  --est calls_median               median of world_dpvo.py --causal's Metric3D calls (its running-median poses *_rs)
Correction c = exp(median log(s_gt / s_est)) (robust; spread and a bootstrap 90 % interval reported).
    python -m bench.world_calib fit --cams <calib cams dir> --preds <pickle dir> --est first:m3d_kf --out c.json
    python -m bench.world_calib apply --cams <eval cams dir> --calib c.json --out <dir>
-> apply writes <dir>/<clip>.npz with index, T_world_cam (SLAM units), scale_raw, scale_cal = c x scale_raw
   (evaluate with bench/world.py --cams <dir>@scale_cal)
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np

from bench.world import sim_fit


def estimate(d, est):
    if est == "calls_median":  # world_dpvo.py --causal --scale_calls n: median of all n Metric3D calls
        c = d["scale_calls"]
        return float(np.median(c[:, 2])) if len(c) else float("nan")
    if est.startswith("first:"):
        kf = d[est.split(":", 1)[1]]
        v = kf[np.isfinite(kf) & (kf > 0)]
        return float(v[0]) if len(v) else float("nan")
    return float(d[est])


def fit(args):
    rows = []
    for f in sorted(glob.glob(os.path.join(args.cams, "clip-*.npz"))):
        name = os.path.basename(f)[:-4]
        p = os.path.join(args.preds, f"{name}.pkl")
        if not os.path.exists(p):
            continue
        d = np.load(f)
        gt = {fr["index"]: np.asarray(fr["T_world_cam"]) for fr in pickle.load(open(p, "rb"))["frames"]}
        idx = [int(i) for i in d["index"] if int(i) in gt]
        cg = np.stack([gt[i][:3, 3] for i in idx])
        path = np.linalg.norm(np.diff(cg, axis=0), axis=1).sum()
        s_est = estimate(d, args.est)
        failed = bool(d["init_failed"]) if "init_failed" in d else False
        if path < args.min_path or not np.isfinite(s_est) or s_est <= 0 or failed:
            continue
        s_gt, _, _ = sim_fit(d["T_world_cam"][idx, :3, 3], cg)
        if np.isfinite(s_gt) and s_gt > 0:
            rows.append((name, s_gt, s_est, path))
    lr = np.log([r[1] / r[2] for r in rows])
    rng = np.random.default_rng(0)
    boot = [np.median(rng.choice(lr, len(lr))) for _ in range(2000)]
    res = dict(est=args.est, cams=args.cams, num_clips=len(rows), correction=float(np.exp(np.median(lr))),
               ci90=[float(np.exp(np.percentile(boot, 5))), float(np.exp(np.percentile(boot, 95)))],
               log_ratio_mad=float(np.median(np.abs(lr - np.median(lr)))),
               err_pct_median_raw=float(np.median(np.abs(np.exp(lr) - 1)) * 100),
               err_pct_median_cal=float(np.median(np.abs(np.exp(lr - np.median(lr)) - 1)) * 100))
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res))


def apply(args):
    c = json.load(open(args.calib))
    os.makedirs(args.out, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(args.cams, "clip-*.npz"))):
        d = np.load(f)
        s = estimate(d, c["est"])
        np.savez(os.path.join(args.out, os.path.basename(f)), index=d["index"], T_world_cam=d["T_world_cam"],
                 scale_raw=s, scale_cal=c["correction"] * s)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fit")
    a.add_argument("--cams", required=True)
    a.add_argument("--preds", required=True)
    a.add_argument("--est", default="first:m3d_kf")
    a.add_argument("--min_path", type=float, default=0.1)
    a.add_argument("--out", required=True)
    b = sub.add_parser("apply")
    b.add_argument("--cams", required=True)
    b.add_argument("--calib", required=True)
    b.add_argument("--out", required=True)
    args = p.parse_args()
    fit(args) if args.cmd == "fit" else apply(args)
