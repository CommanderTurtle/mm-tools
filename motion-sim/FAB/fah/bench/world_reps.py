"""World metrics per camera realisation (bench.world --protocol wham, own true positives) and their mean / SD.

Each row is one method = hand run + its camera realisations (separate SLAM runs or camera-model seeds). Every
realisation is scored on its own (no averaging of predictions or cameras); the paper reports mean +- sample SD (ddof 1)
over the realisations, stored here next to the per-realisation values.
    python -m bench.world_reps --chunks 100,0 --out <json> \
        --row "hawor=<hand run>|<slam run 1>|<slam run 2>|<slam run 3>" --row "fah=<hand run>|<seed 0 cams>|..."
--common d1,d2: score every row on the hand-frames that are true positives in all of d1, d2 (bench.world --common).
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bench.world import DATA_ROOT, common_support, evaluate_world

KEYS = ("w_mpjpe", "wa_mpjpe", "rte")
_COMMON = None


def _init(common):
    global _COMMON
    _COMMON = common


def score(job):
    run, cams, chunk, data_root, min_run = job
    r = evaluate_world(run, cams=cams, data_root=data_root, common=_COMMON, protocol="wham", chunk=chunk, min_run=min_run)
    return job, {k: r[k] for k in KEYS + ("num_frames", "num_runs")}


def aggregate(per):
    """[{metric: value}] over realisations -> {metric: {mean, sd}} (sample SD, ddof 1; None for one realisation)."""
    return {k: dict(mean=float(np.mean([p[k] for p in per])),
                    sd=float(np.std([p[k] for p in per], ddof=1)) if len(per) > 1 else None) for k in KEYS}


def score_rows(rows, chunks=(100, 0), data_root=None, common=None, min_run=16, workers=8):
    """rows {name: (hand run dir, [cams spec per realisation])} -> {name: {chunk: {per_realisation, <metric>: {mean, sd}}}}."""
    data_root = data_root or os.path.join(DATA_ROOT, "hot3d")
    jobs = list(dict.fromkeys((run, c, int(ch), data_root, min_run) for run, cams in rows.values() for c in cams
                              for ch in chunks))
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(common,)) as ex:
        res = dict(ex.map(score, jobs))
    out = {}
    for name, (run, cams) in rows.items():
        out[name] = {}
        for ch in chunks:
            per = [res[(run, c, int(ch), data_root, min_run)] for c in cams]
            out[name][int(ch)] = dict(per_realisation=per, **aggregate(per))
    return out


def main(args):
    rows = {}
    for spec in args.row:
        name, rest = spec.split("=", 1)
        run, *cams = rest.split("|")
        rows[name] = (run, cams)
    chunks = [int(c) for c in args.chunks.split(",")]
    common = common_support(args.common.split(","), 0.3) if args.common else None
    res = score_rows(rows, chunks, args.data_root, common, args.min_run, args.workers)
    out = dict(protocol="wham", min_run=args.min_run, common=args.common, rows={})
    for name, (run, cams) in rows.items():
        out["rows"][name] = dict(run=run, cams=cams, chunks={str(ch): res[name][ch] for ch in chunks})
        for ch in chunks:
            r = res[name][ch]
            print(f"{name:12s} chunk {ch:>3d}: " + "  ".join(
                f"{k} {r[k]['mean']:.2f} +- {r[k]['sd'] or 0:.2f} [" + ", ".join(f"{p[k]:.2f}" for p in r["per_realisation"]) + "]"
                for k in KEYS), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--row", action="append", required=True, help="name=<hand run dir>|<cams spec>|<cams spec>|...")
    p.add_argument("--chunks", default="100,0", help="bench.world --chunk values (100: 100-frame chunks; 0: whole tracks)")
    p.add_argument("--common", default="", help="comma-separated run dirs: score only their common true positives")
    p.add_argument("--min_run", type=int, default=16)
    p.add_argument("--data_root", default=None, help="HOT3D clip tars (GT poses for pickles without T_world_cam)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    main(p.parse_args())
