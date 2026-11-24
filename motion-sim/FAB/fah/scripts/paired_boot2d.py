"""Paired sequence-bootstrap of arm differences on a 2D benchmark (bench.run --dataset bench2d pickles, matched as
bench.bench2d.evaluate): per run and GT hand the normalised joint errors (matched) or a miss; counts summed per resampling
unit = a video sequence where the key names one (HInt ND / EPIC: the video; HOT3D: the clip; also their synthetic low-light
copies), else the image; with --unit hand every GT hand is its own unit (e.g. hard pose: 175 hands in 6 images). 95 % CI of
mean(arm) - mean(ref) from B resamples of whole units (the same resamples for every run). With several runs per side (seeds),
the side's value is the seed mean of the runs' metrics on each resample. Metrics: pck05 / pck10 / tip_pck05 on matched hands
(as bench.bench2d.evaluate) and pck10_all (missed hands count as failed).

    python scripts/paired_boot2d.py --ref bench2d/v2-arm-A,bench2d/v2-arm-A-s1 --arm bench2d/v2-arm-B,bench2d/v2-arm-B-s1 [--arm ...] --out x.json
"""
import argparse
import glob
import json
import os
import pickle
import re

import numpy as np

from bench.bench2d.evaluate import TIPS, _norm_err, match

R = os.environ.get("RESULTS", os.path.join("output", "bench"))  # bench.run --out_root
KEYS = ("pck05", "pck10", "tip_pck05", "pck10_all")


def unit_of(key):
    sub, name = key.split("/", 1)
    base = sub.replace("syn_", "")
    if base in ("hint_nd", "hint_epick"):
        return sub + "/" + re.sub(r"_frame_?\d+$", "", name)
    if base == "hot3d":
        return sub + "/" + name.rsplit("_", 1)[0]
    return key


def counts(run, unit="seq"):
    """{unit: [c05, c10, matched joints, tips < 0.05, tips, visible joints of missed hands]} of one run."""
    out = {}
    for path in sorted(glob.glob(os.path.join(run if os.path.isabs(run) else os.path.join(R, run), "*.pkl"))):
        clip = pickle.load(open(path, "rb"))
        for fr in clip["frames"]:
            preds = [dict(p, joints2d=p.get("joints2d")) for p in fr["pred"]]
            m, _ = match(preds, fr["gt"])
            for gi, g in enumerate(fr["gt"]):
                u = f"{fr['key']}#{gi}" if unit == "hand" else unit_of(fr["key"])
                c = out.setdefault(u, np.zeros(6))
                if gi in m:
                    p = np.asarray(preds[m[gi]]["joints2d"])
                    e, _ = _norm_err(p, g)
                    tv = g["vis"][TIPS] > 0
                    S = max(g["bbox"][2] - g["bbox"][0], g["bbox"][3] - g["bbox"][1], 1.0)
                    t = np.linalg.norm(p[TIPS] - g["joints2d"][TIPS], axis=1)[tv] / S
                    c += [(e < 0.05).sum(), (e < 0.1).sum(), len(e), (t < 0.05).sum(), len(t), 0]
                else:
                    c[5] += int((g["vis"] > 0).sum())
    return out


def metrics(C):
    """C: (units, 6) summed counts -> the 4 metrics."""
    s = C.sum(0)
    return np.array([s[0] / max(s[2], 1), s[1] / max(s[2], 1), s[3] / max(s[4], 1), s[1] / max(s[2] + s[5], 1)])


def main(a):
    rng = np.random.default_rng(0)
    sides = [a.ref] + a.arm
    runs = {r: counts(r, a.unit) for s in sides for r in s.split(",")}
    units = sorted(set.intersection(*[set(c) for c in runs.values()]))
    mats = {r: np.stack([c[u] for u in units]) for r, c in runs.items()}
    side_val = lambda s, idx: np.mean([metrics(mats[r][idx]) for r in s.split(",")], 0)
    full = {s: side_val(s, np.arange(len(units))) for s in sides}
    resample_idx = [rng.integers(0, len(units), len(units)) for _ in range(a.boot)]
    samp = {s: np.stack([side_val(s, idx) for idx in resample_idx]) for s in sides}
    out = dict(unit=a.unit, n_units=len(units), ref=a.ref, ref_value=dict(zip(KEYS, full[a.ref].round(4).tolist())), arms={})
    for s in a.arm:
        d = samp[s] - samp[a.ref]
        out["arms"][s] = {k: dict(value=round(float(full[s][i]), 4), diff=round(float(full[s][i] - full[a.ref][i]), 4),
                                  lo=round(float(np.percentile(d[:, i], 2.5)), 4), hi=round(float(np.percentile(d[:, i], 97.5)), 4))
                          for i, k in enumerate(KEYS)}
        print(s, json.dumps(out["arms"][s]), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="comma list of run dirs (relative to $RESULTS or absolute)")
    p.add_argument("--arm", action="append", required=True, help="comma list of run dirs; repeatable")
    p.add_argument("--unit", choices=("seq", "hand"), default="seq")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--out", default="")
    main(p.parse_args())
