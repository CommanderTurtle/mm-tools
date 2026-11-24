"""Paired sequence bootstrap of world-space differences between runs, under bench.world's --protocol wham.

Per clip, the sums behind bench.world.evaluate_world(protocol="wham") are collected for every run: the same own true
positives, tracks, contiguous runs (>= --min_run frames), chunks (--chunk; 0 = whole run) and WHAM alignments
(bench.world.wham_errors). Sequences are resampled with replacement, with the same resamples for every run (HOT3D: the clip
tar's sequence_id; EgoStandard: the episode, one clip each). W-MPJPE / WA-MPJPE (means over frames) and RTE (run means
weighted by run length, as evaluate_world's rte for wham) are recomputed from the summed numerators / denominators.
Full-sample values are checked against evaluate_world. --avg name=a+b+c adds the mean of the metrics of runs a, b, c
(e.g. 3 camera seeds or SLAM runs; predictions are never averaged) as one more run, recomputed per resample; --common d1,d2
scores every run only on the hand-frames that are true positives in all of d1, d2 (bench.world --common); per-run track
statistics (scored runs, frames, run lengths, share of the W error sum carried by the 10 largest runs) are stored under
"stats".
    python -m bench.world_bootstrap --dataset hot3d --runs fah=<dir>,hawor=<dir> --cams fah=<cams>,hawor=<cams> \
        --ref hawor --chunk 100 --out <json>
"""
import argparse
import glob
import json
import os
import pickle
import tarfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bench.world import (DATA_ROOT, RESULTS, common_support, evaluate_world, gt_cams_for_clip, load_pred_cams, tp_hands,
                         to_world, wham_errors)

KEYS = ("w_mpjpe", "wa_mpjpe", "rte")


def clip_sums(clip, cams, chunk, min_run, iou=0.3, data_root=None, gt_cache=None, min_frames=3, common=None):
    """WHAM-protocol sums of one clip: W / WA error sums and frame count, RTE numerator / denominator, and per scored run
    (length, W error sum)."""
    data_root = data_root or os.path.join(DATA_ROOT, "hot3d")
    gt_cache = gt_cache or os.path.join(RESULTS, "world", "gt_cams")
    frames = clip["frames"]
    if any(fr.get("T_world_cam") is None for fr in frames):
        gtc = gt_cams_for_clip(clip, data_root, gt_cache)
        gt_T = {fr["index"]: gtc[fr["key"]] for fr in frames}
    else:
        gt_T = {fr["index"]: np.asarray(fr["T_world_cam"]) for fr in frames}
    pc = load_pred_cams(cams, clip)
    pred_T = gt_T if pc is None else pc
    tracks, tidx = defaultdict(list), defaultdict(list)
    for fr in frames:
        for side, (pj, gj) in tp_hands(fr, iou).items():
            if common is not None and (clip["clip"], fr["index"], side) not in common:
                continue
            if fr["index"] not in pred_T:
                continue
            tracks[side].append((to_world(pred_T[fr["index"]], pj), to_world(gt_T[fr["index"]], gj)))
            tidx[side].append(fr["index"])
    s = dict(w=0.0, wa=0.0, n=0, rte_num=0.0, rte_den=0, runs=[])
    for side, tr in tracks.items():
        if len(tr) < min_frames:
            continue
        P, G = np.stack([a for a, _ in tr]), np.stack([b for _, b in tr])
        idx = np.asarray(tidx[side])
        cuts = np.flatnonzero(np.diff(idx) != 1) + 1
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(idx)]):
            if b - a < min_run:
                continue
            w, wa, rte = wham_errors(P[a:b], G[a:b], chunk)
            s["w"] += float(w.sum())
            s["wa"] += float(wa.sum())
            s["n"] += len(w)
            s["runs"].append((b - a, float(w.sum())))
            if rte is not None:
                s["rte_num"] += float(rte.mean()) * (b - a)
                s["rte_den"] += b - a
    return s


def sequence_of(path, dataset, data_root):
    name = os.path.splitext(os.path.basename(path))[0]
    if dataset != "hot3d":
        return name
    tars = glob.glob(os.path.join(data_root, "**", f"{name}.tar"), recursive=True)
    assert tars, f"no tar for {name} under {data_root} (needed for its sequence id)"
    tar = tarfile.open(tars[0])
    info = next(m for m in tar.getnames() if m.endswith(".info.json"))
    return json.load(tar.extractfile(info))["sequence_id"]


def metrics(t):
    return dict(w_mpjpe=t[0] / t[2], wa_mpjpe=t[1] / t[2], rte=t[3] / t[4] if t[4] else np.nan)


def track_stats(runs):
    L = np.array([n for n, _ in runs]); W = np.array([w for _, w in runs])
    return dict(runs=len(runs), frames=int(L.sum()), len_median=float(np.median(L)), len_p90=float(np.percentile(L, 90)),
                top10_w_share=float(np.sort(W)[::-1][:10].sum() / W.sum()))


_COMMON = None


def _init(common):
    global _COMMON
    _COMMON = common


def _run_sums(job):
    d, cams, names, seq_idx, n_seq, chunk, min_run, data_root, gt_cache = job
    S, rr = np.zeros((n_seq, 5)), []
    for n, i in zip(names, seq_idx):
        s = clip_sums(pickle.load(open(os.path.join(d, n), "rb")), cams, chunk, min_run, data_root=data_root,
                      gt_cache=gt_cache, common=_COMMON)
        S[i] += [s["w"], s["wa"], s["n"], s["rte_num"], s["rte_den"]]
        rr += s["runs"]
    if len(names) == len(glob.glob(os.path.join(d, "*.pkl"))):  # evaluate_world scores every clip of the run
        ref = evaluate_world(d, cams=cams, data_root=data_root, gt_cache=gt_cache, common=_COMMON, protocol="wham",
                             chunk=chunk, min_run=min_run)
        mine = metrics(S.sum(0))
        assert all(abs(mine[k] - ref[k]) < 1e-6 * max(1.0, abs(ref[k])) for k in KEYS), (d, cams, mine, ref)
    return S, track_stats(rr)


def bootstrap(runs, cams, ref, dataset="hot3d", avg=None, common=None, chunk=100, min_run=16, boot=2000, seed=0,
              data_root=None, gt_cache=None, workers=8):
    """runs {name: hand run dir}, cams {name: cams spec}, avg {name: [member runs]}, common: bench.world.common_support
    set or None -> dict(point, stats, diff[f"{run}-{ref}"][metric] = {point, lo, hi, sd}, ...)."""
    data_root = data_root or os.path.join(DATA_ROOT, "hot3d")
    avg = avg or {}
    names = sorted(set.intersection(*[{os.path.basename(p) for p in glob.glob(os.path.join(d, "*.pkl"))} for d in runs.values()]))
    seqs = {n: sequence_of(os.path.join(next(iter(runs.values())), n), dataset, data_root) for n in names}
    order = sorted(set(seqs.values()))
    jobs = [(d, cams[r], names, [order.index(seqs[n]) for n in names], len(order), chunk, min_run, data_root, gt_cache)
            for r, d in runs.items()]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(common,)) as ex:
        res = dict(zip(runs, ex.map(_run_sums, jobs)))
    per = {r: v[0] for r, v in res.items()}  # run -> (S, 5) sums per sequence: w, wa, n, rte_num, rte_den
    stats = {r: v[1] for r, v in res.items()}
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(order), size=(boot, len(order)))

    def metric(r, idx):
        if r in avg:  # mean of the member runs' metrics (camera seeds / SLAM runs), each on the same resample
            ms = [metrics(per[m][idx].sum(0)) for m in avg[r]]
            return {k: float(np.mean([m[k] for m in ms])) for k in KEYS}
        return metrics(per[r][idx].sum(0))

    allr = list(runs) + list(avg)
    full = np.arange(len(order))
    out = dict(dataset=dataset, chunk=chunk, min_run=min_run, boot=boot, seed=seed, num_clips=len(names),
               num_sequences=len(order), runs=runs, avg=avg, cams=cams, ref=ref,
               point={r: metric(r, full) for r in allr}, stats=stats, diff={})
    for r in allr:
        if r == ref:
            continue
        d = {k: [] for k in KEYS}
        for idx in samples:
            a, b = metric(r, idx), metric(ref, idx)
            for k in KEYS:
                d[k].append(a[k] - b[k])
        out["diff"][f"{r}-{ref}"] = {
            k: dict(point=out["point"][r][k] - out["point"][ref][k], lo=float(np.nanpercentile(v, 2.5)),
                    hi=float(np.nanpercentile(v, 97.5)), sd=float(np.nanstd(v)))
            for k, v in d.items()}
    return out


def main(args):
    runs = dict(r.split("=", 1) for r in args.runs.split(","))
    cams = dict(c.split("=", 1) for c in args.cams.split(",")) if "=" in args.cams else {k: args.cams for k in runs}
    avg = dict(a.split("=", 1) for a in args.avg.split(",")) if args.avg else {}
    avg = {k: v.split("+") for k, v in avg.items()}
    common = common_support(args.common.split(","), 0.3) if args.common else None
    out = bootstrap(runs, cams, args.ref, args.dataset, avg, common, args.chunk, args.min_run, args.boot, args.seed,
                    args.data_root, args.gt_cache, args.workers)
    out["common"] = args.common
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    for r in out["point"]:
        print(r, " / ".join(f"{out['point'][r][k]:.2f}" for k in KEYS), out["stats"].get(r, ""))
    for k, v in out["diff"].items():
        print(k, "  ".join(f"{m} {v[m]['point']:+.2f} [{v[m]['lo']:+.2f}, {v[m]['hi']:+.2f}]" for m in KEYS))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="hot3d", choices=["hot3d", "egostandard"])
    p.add_argument("--runs", required=True, help="name=<hand run dir>,... (clips common to all runs are used)")
    p.add_argument("--cams", required=True, help="one bench.world --cams spec for all runs, or name=<spec>,...")
    p.add_argument("--ref", required=True)
    p.add_argument("--avg", default="", help="name=a+b+c,...: mean of the runs' metrics (e.g. camera seeds) as one more run")
    p.add_argument("--common", default="", help="comma-separated run dirs: score only their common true positives")
    p.add_argument("--chunk", type=int, default=100)
    p.add_argument("--min_run", type=int, default=16)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_root", default=None, help="HOT3D clip tars (sequence ids, GT poses)")
    p.add_argument("--gt_cache", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    main(p.parse_args())
