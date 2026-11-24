"""The paper's world-space cells (table "Hands in world space" and the supplement's run-to-run, track and interval
tables) recomputed from stored hand runs and cameras, and checked against the published values at 2 decimals.

Inputs under $RESULTS (<ds> = hot3d: the benchmark split; egostandard: test):
  <ds>/wilor, <ds>/hawor, <ds>/fah-video   bench.run hand runs (fah-video: FAH-L, mode video; docs/evaluation.md)
  world/<ds>/slam_hawor_r{1,2,3}   3 separate runs of the camera of HaWoR's SLAM stage (HaWoR's checkout: its masked
                                   DROID-SLAM + Metric3D scale, masked with the hawor run's meshes); bench/world_slam.py
                                   --mask_run $RESULTS/<ds>/hawor runs the same algorithm (parity-checked with our masks only)
  world/<ds>/slam_fah_r{1,2,3}     3 separate runs of bench/world_slam.py --mask_run $RESULTS/<ds>/fah-video (the paper's
                                   runs: HaWoR's checkout with these masks; the two agree within the run-to-run spread,
                                   docs/world.md)
  world/<ds>/camera_s{0,1,2}       the real-time camera model, 3 separate runs with --seed 0 / 1 / 2 (docs/world.md)
  world/<ds>/offline_s{0,1,2}      python -m bench.world_offline --slam slam_fah_r{i+1} --ours camera_s{i}
    python -m bench.world_paper [--datasets hot3d,egostandard] [--workers 8] --out <json>
Every camera realisation is scored on its own (bench.world_reps, WHAM protocol, own true positives); a cell is the
mean +- SD (ddof 1) over the 3 realisations (predictions are never averaged); GT-camera rows are deterministic. Track
statistics are from the first run; intervals are paired sequence bootstraps (bench.world_bootstrap, 2000 resamples, seed 0)
of the 3-run means. Exit status 1 if any cell differs from the paper's.
"""
import argparse
import glob
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

from bench.world import DATA_ROOT, RESULTS, common_support
from bench.world_bootstrap import bootstrap, clip_sums, track_stats
from bench.world_reps import score_rows

HAWOR_SLAM = [f"slam_hawor_r{i}" for i in (1, 2, 3)]
ROWS = {  # table row: (hand run, camera realisations)
    "WiLoR, SLAM": ("wilor", HAWOR_SLAM),
    "HaWoR, own SLAM": ("hawor", HAWOR_SLAM),
    "FAH-L video hands, HaWoR's SLAM": ("fah-video", HAWOR_SLAM),
    "FAH-L video hands, SLAM with our masks": ("fah-video", [f"slam_fah_r{i}" for i in (1, 2, 3)]),
    "FAH-L world, camera model": ("fah-video", [f"camera_s{i}" for i in (0, 1, 2)]),
    "FAH-L world, offline": ("fah-video", [f"offline_s{i}" for i in (0, 1, 2)]),
}
GT_ROWS = {"WiLoR, GT": "wilor", "HaWoR, GT": "hawor", "FAH-L video hands, GT": "fah-video"}
CELLS = ("chunks W", "chunks WA", "whole W", "whole WA", "RTE")
PAPER = {  # (means, SDs) of CELLS; W / WA in mm, RTE in %
    "hot3d": {
        "WiLoR, SLAM": ((52.49, 20.64, 60.05, 23.06, 10.58), (0.19, 0.05, 0.25, 0.07, 0.02)),
        "HaWoR, own SLAM": ((31.96, 13.79, 36.45, 14.91, 7.52), (0.21, 0.05, 0.26, 0.07, 0.02)),
        "FAH-L video hands, HaWoR's SLAM": ((22.69, 8.81, 27.17, 9.81, 3.68), (0.21, 0.07, 0.41, 0.12, 0.02)),
        "FAH-L video hands, SLAM with our masks": ((22.69, 8.83, 26.90, 9.73, 3.70), (0.34, 0.09, 0.40, 0.12, 0.04)),
        "FAH-L world, camera model": ((27.03, 11.07, 32.72, 12.99, 4.78), (0.38, 0.30, 0.66, 0.46, 0.30)),
        "FAH-L world, offline": ((22.28, 8.75, 26.49, 9.92, 3.30), (0.31, 0.21, 0.52, 0.34, 0.16)),
        "WiLoR, GT": ((50.58, 19.93, 57.82, 22.10, 9.27), None),
        "HaWoR, GT": ((30.68, 13.17, 34.62, 14.03, 7.00), None),
        "FAH-L video hands, GT": ((19.91, 7.59, 23.57, 8.04, 2.50), None),
    },
    "egostandard": {
        "WiLoR, SLAM": ((83.31, 26.87, 267.54, 56.99, 1.73), (4.26, 0.23, 31.75, 1.00, 0.06)),
        "HaWoR, own SLAM": ((76.44, 28.55, 210.46, 53.62, 2.22), (4.11, 0.25, 28.90, 1.21, 0.08)),
        "FAH-L video hands, HaWoR's SLAM": ((68.20, 23.53, 260.56, 55.02, 1.35), (4.79, 0.23, 41.97, 1.08, 0.06)),
        "FAH-L video hands, SLAM with our masks": ((64.87, 23.47, 225.32, 53.10, 1.30), (1.66, 0.41, 18.97, 1.42, 0.02)),
        "FAH-L world, camera model": ((55.93, 21.28, 177.69, 45.91, 1.30), (0.14, 0.08, 3.82, 1.67, 0.03)),
        "FAH-L world, offline": ((55.19, 21.75, 152.56, 40.64, 1.18), (0.92, 0.61, 3.51, 2.02, 0.02)),
        "WiLoR, GT": ((53.91, 18.48, 130.34, 23.80, 0.78), None),
        "HaWoR, GT": ((46.73, 20.81, 90.78, 25.50, 1.19), None),
        "FAH-L video hands, GT": ((33.45, 14.45, 77.19, 17.20, 0.43), None),
    },
}
TRACKS = {  # whole tracks, own true positives, first run: scored runs, run length median, p90, top-10 share of W
    "egostandard": {"FAH-L video hands, HaWoR's SLAM": (326, 900, 900, 0.55), "FAH-L world, camera model": (326, 900, 900, 0.24),
                    "FAH-L world, offline": (326, 900, 900, 0.30), "HaWoR, own SLAM": (677, 137, 900, 0.45),
                    "WiLoR, SLAM": (461, 343, 900, 0.45)},
    "hot3d": {"FAH-L video hands, HaWoR's SLAM": (238, 130), "HaWoR, own SLAM": (271, 76), "WiLoR, SLAM": (272, 81)},
}
COMMON = {  # EgoStandard whole tracks on the hand-frames that fah-video and hawor both match: W, WA (mean over 3 runs)
    "egostandard": {"FAH-L video hands, HaWoR's SLAM": (203.00, 46.13), "FAH-L world, camera model": (141.40, 38.77),
                    "FAH-L world, offline": (124.34, 35.38), "HaWoR, own SLAM": (210.46, 53.59)},
}
INTERVALS = [  # (dataset, chunk, common TPs, row, reference row, metric, (difference, 2.5 %, 97.5 %)) of the 3-run means
    ("egostandard", 0, True, "FAH-L world, camera model", "HaWoR, own SLAM", "w_mpjpe", (-69.05, -195.14, -0.14)),
    ("egostandard", 0, True, "FAH-L world, camera model", "HaWoR, own SLAM", "wa_mpjpe", (-14.82, -36.67, -0.65)),
    ("egostandard", 0, True, "FAH-L world, offline", "HaWoR, own SLAM", "w_mpjpe", (-86.12, -207.83, -20.40)),
    ("egostandard", 0, True, "FAH-L world, offline", "HaWoR, own SLAM", "wa_mpjpe", (-18.20, -39.76, -5.87)),
    ("hot3d", 100, False, "FAH-L video hands, SLAM with our masks", "FAH-L video hands, HaWoR's SLAM", "w_mpjpe",
     (-0.00, -1.40, 1.02)),
]


def same(a, b):
    return float(f"{a:.2f}") == float(f"{b:.2f}")


def paths(ds, row):
    hands, cams = ROWS[row]
    return os.path.join(RESULTS, ds, hands), [os.path.join(RESULTS, "world", ds, c) for c in cams]


def _track_job(job):
    run, cams, data_root = job
    rr = []
    for path in sorted(glob.glob(os.path.join(run, "*.pkl"))):
        rr += clip_sums(pickle.load(open(path, "rb")), cams, 0, 16, data_root=data_root)["runs"]
    return track_stats(rr)


def main(args):
    data_root = args.data_root or os.path.join(DATA_ROOT, "hot3d")
    out, bad = {}, []

    def check(name, got, want):
        ok = same(got, want)
        print(f"  {'ok  ' if ok else 'DIFF'} {name}: {got:.2f} (paper {want:.2f})", flush=True)
        if not ok:
            bad.append(name)

    for ds in args.datasets.split(","):
        print(ds, flush=True)
        rows = {r: paths(ds, r) for r in ROWS}
        rows.update({r: (os.path.join(RESULTS, ds, h), ["gt"]) for r, h in GT_ROWS.items()})
        res = score_rows(rows, (100, 0), data_root, None, 16, args.workers)
        out[ds] = dict(cells={}, tracks={}, common={}, intervals=[])
        for r, (means, sds) in PAPER[ds].items():
            got = [res[r][100]["w_mpjpe"], res[r][100]["wa_mpjpe"], res[r][0]["w_mpjpe"], res[r][0]["wa_mpjpe"], res[r][100]["rte"]]
            out[ds]["cells"][r] = dict(mean=[g["mean"] for g in got], sd=[g["sd"] for g in got],
                                       per_realisation={str(ch): res[r][ch]["per_realisation"] for ch in (100, 0)})
            for i, c in enumerate(CELLS):
                check(f"{ds} / {r} / {c}", got[i]["mean"], means[i])
                if sds is not None:
                    check(f"{ds} / {r} / {c} SD", got[i]["sd"], sds[i])
        with ProcessPoolExecutor(args.workers) as ex:
            names = list(TRACKS.get(ds, {}))
            stats = dict(zip(names, ex.map(_track_job, [(paths(ds, r)[0], paths(ds, r)[1][0], data_root) for r in names])))
        for r, want in TRACKS.get(ds, {}).items():
            got = [stats[r]["runs"], stats[r]["len_median"], stats[r]["len_p90"], stats[r]["top10_w_share"]]
            out[ds]["tracks"][r] = stats[r]
            for name, g, w in zip(("runs", "median length", "p90 length", "top-10 share"), got, want):
                check(f"{ds} / {r} / tracks {name}", g, w)
        if ds in COMMON:
            common = common_support([os.path.join(RESULTS, ds, "fah-video"), os.path.join(RESULTS, ds, "hawor")], 0.3)
            res_c = score_rows({r: paths(ds, r) for r in COMMON[ds]}, (0,), data_root, common, 16, args.workers)
            for r, want in COMMON[ds].items():
                out[ds]["common"][r] = res_c[r][0]
                check(f"{ds} / {r} / common whole W", res_c[r][0]["w_mpjpe"]["mean"], want[0])
                check(f"{ds} / {r} / common whole WA", res_c[r][0]["wa_mpjpe"]["mean"], want[1])
        groups = {}
        for iv in INTERVALS:
            if iv[0] == ds:
                groups.setdefault(iv[1:3], []).append(iv)
        for (chunk, use_common), ivs in groups.items():
            names = list(dict.fromkeys(r for iv in ivs for r in iv[3:5]))
            runs, cams, avg = {}, {}, {}
            for r in names:
                hands, cs = paths(ds, r)
                avg[r] = []
                for i, c in enumerate(cs):
                    runs[f"{r} #{i + 1}"], cams[f"{r} #{i + 1}"] = hands, c
                    avg[r].append(f"{r} #{i + 1}")
            common = common_support([os.path.join(RESULTS, ds, "fah-video"), os.path.join(RESULTS, ds, "hawor")], 0.3) \
                if use_common else None
            refs = {iv[4] for iv in ivs}
            for ref in refs:
                b = bootstrap(runs, cams, ref, ds, avg, common, chunk, 16, args.boot, args.seed, data_root,
                              workers=args.workers)
                for iv in ivs:
                    if iv[4] != ref:
                        continue
                    d = b["diff"][f"{iv[3]}-{ref}"][iv[5]]
                    out[ds]["intervals"].append(dict(chunk=chunk, common=use_common, row=iv[3], ref=ref, metric=iv[5], **d))
                    for name, g, w in zip(("difference", "2.5 %", "97.5 %"), (d["point"], d["lo"], d["hi"]), iv[6]):
                        check(f"{ds} / {iv[3]} - {ref} / {'common ' if use_common else ''}chunk {chunk} {iv[5]} {name}", g, w)
    out["mismatches"] = bad
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"{len(bad)} cells differ from the paper" + (": " + "; ".join(bad) if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default="hot3d,egostandard")
    p.add_argument("--data_root", default=None, help="HOT3D clip tars (sequence ids, GT poses)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    sys.exit(main(p.parse_args()))
