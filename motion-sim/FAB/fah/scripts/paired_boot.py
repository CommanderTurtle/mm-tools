"""Paired sequence-bootstrap of arm differences on a 3D benchmark (bench.run pickles, matched as bench.evaluate): per GT hand
PA-MPJPE / PA-MPVPE of every run, hands matched in all runs only, grouped by sequence (clip pickle "sequence_id"; HOT3D via
clip_definitions.json; else the clip name); 95 % CI of mean(arm) - mean(ref) from B resamples of whole sequences (the same resamples for every arm). With
several runs per side (seeds), the side's value is the seed mean of the per-hand errors.

    python scripts/paired_boot.py --ref hot3d/arm-A,hot3d/arm-A-s1 --arm hot3d/arm-B,hot3d/arm-B-s1 [--arm ...] --out x.json
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np

from bench.common import hand_metrics, match_hands

R = os.environ.get("RESULTS", os.path.join("output", "bench"))  # bench.run --out_root


def hot3d_sequences():
    """HOT3D-Clips clip name -> sequence id (HF clip_definitions.json), so HOT3D resamples whole sequences, not clips."""
    try:
        from huggingface_hub import hf_hub_download

        defs = json.load(open(hf_hub_download("bop-benchmark/hot3d", "clip_definitions.json", repo_type="dataset")))
        return {f"clip-{int(k):06d}": v["sequence_id"] for k, v in defs.items()}
    except Exception as e:  # offline: clip-level resampling
        print("no HOT3D clip definitions:", e, flush=True)
        return {}


SEQ = {}


def per_hand(run, iou=0.3):
    """{(clip, frame, side): (sequence, pa_mpjpe, pa_mpvpe or nan)} of the matched GT hands of one run."""
    out = {}
    for path in sorted(glob.glob(os.path.join(run if os.path.isabs(run) else os.path.join(R, run), "*.pkl"))):
        clip = pickle.load(open(path, "rb"))
        name = os.path.basename(path)[:-4]
        if clip.get("dataset") == "hot3d" and "hot3d" not in SEQ:
            SEQ["hot3d"] = hot3d_sequences()
        seq = clip.get("sequence_id") or clip.get("sequence") or SEQ.get(clip.get("dataset"), {}).get(name, name)
        for fr in clip["frames"]:
            matches, _ = match_hands(fr["pred"], fr["gt"], iou)
            for gi, pi in matches:
                if pi is None:
                    continue
                p, g = fr["pred"][pi], fr["gt"][gi]
                if p.get("joints3d") is None or not np.isfinite(p["joints3d"]).all():
                    continue
                v = p.get("verts3d") if p.get("verts3d") is not None and np.isfinite(p["verts3d"]).all() else None
                m = hand_metrics(p["joints3d"], g["joints3d"], v, g.get("verts3d"))
                out[(name, fr["index"], bool(g["is_right"]))] = (seq, m["pa_mpjpe"], m.get("pa_mpvpe", np.nan))
    return out


def side(runs):
    hs = [per_hand(r) for r in runs]
    keys = set.intersection(*[set(h) for h in hs])
    return {k: (hs[0][k][0], np.mean([h[k][1] for h in hs]), np.mean([h[k][2] for h in hs])) for k in keys}


def main(a):
    ref = side(a.ref.split(","))
    arms = {spec: side(spec.split(",")) for spec in a.arm}
    keys = sorted(set(ref).intersection(*[set(v) for v in arms.values()]))
    seqs = sorted({ref[k][0] for k in keys})
    sid = {s: i for i, s in enumerate(seqs)}
    g = np.array([sid[ref[k][0]] for k in keys])
    rng = np.random.default_rng(0)
    resample_idx = [rng.choice(len(seqs), len(seqs)) for _ in range(a.boot)]
    res = dict(ref=a.ref, hands=len(keys), sequences=len(seqs), arms={})
    for spec, h in arms.items():
        r = {}
        for mi, mname in ((1, "pa_mpjpe"), (2, "pa_mpvpe")):
            d = np.array([h[k][mi] - ref[k][mi] for k in keys])
            if not np.isfinite(d).any():
                continue
            ok = np.isfinite(d)
            s_sum = np.bincount(g[ok], d[ok], minlength=len(seqs))
            s_n = np.bincount(g[ok], minlength=len(seqs))
            bs = [s_sum[x].sum() / max(s_n[x].sum(), 1) for x in resample_idx]
            r[mname] = dict(ref=float(np.nanmean([ref[k][mi] for k in keys])), arm=float(np.nanmean([h[k][mi] for k in keys])),
                            delta=float(d[ok].mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        res["arms"][spec] = r
        print(spec, json.dumps(r), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="comma list of run dirs (relative to $RESULTS or absolute)")
    p.add_argument("--arm", action="append", required=True, help="comma list of run dirs; repeatable")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--out", default="")
    main(p.parse_args())
