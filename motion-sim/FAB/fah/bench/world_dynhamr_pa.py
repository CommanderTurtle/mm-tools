"""Camera-frame PA-MPJPE of Dyn-HaMR's HaMeR initialiser vs its optimised output (supplement, Dyn-HaMR paragraph), on
the hand-frames where both exist. Predictions are matched to GT hands by handedness (exactly one prediction of the GT
hand's side per frame), not by box: the initialiser's camera-frame translation comes from HaMeR's crop camera, so its
boxes are not comparable. Other runs (e.g. WiLoR, HaWoR, HaMeR) are scored on the same hand-frames for reference.
    python -m bench.world_dynhamr_pa --init $RESULTS/hot3d/dynhamr-98e_slam_fixscale-init \
        --final $RESULTS/hot3d/dynhamr-98e_slam_fixscale --ref $RESULTS/hot3d/wilor,$RESULTS/hot3d/hawor
(scripts/dynhamr/run.sh; the paper-time release on the pilot clips, 4 of 5 tracked (on clip-002742 its tracker keeps
fewer frames than its minimum track length): 9.28 -> 10.19 mm on 948 hand-frames)
"""
import argparse
import glob
import os
import pickle

import numpy as np

from bench.common import procrustes


def pa_by_side(run):
    """{(clip, frame index, GT side): PA-MPJPE (mm)} for GT hands with exactly one prediction of their side."""
    per = {}
    for f in sorted(glob.glob(os.path.join(run, "*.pkl"))):
        c = pickle.load(open(f, "rb"))
        for fr in c["frames"]:
            for g in fr["gt"]:
                cand = [p for p in fr["pred"] if bool(p["is_right"]) == bool(g["is_right"]) and np.isfinite(p["joints3d"]).all()]
                if len(cand) == 1:
                    P, G = np.asarray(cand[0]["joints3d"], np.float64), np.asarray(g["joints3d"], np.float64)
                    per[(c["clip"], fr["index"], bool(g["is_right"]))] = float(np.linalg.norm(procrustes(P, G) - G, axis=1).mean() * 1000)
    return per


def main(args):
    runs = {"initialiser": args.init, "optimised": args.final}
    runs.update({os.path.basename(os.path.normpath(r)): r for r in args.ref.split(",") if r})
    res = {k: pa_by_side(v) for k, v in runs.items()}
    keys = set(res["initialiser"]) & set(res["optimised"])
    print(f"hand-frames with both the initialiser and the optimised output: {len(keys)}")
    for k, per in res.items():
        v = [per[x] for x in keys if x in per]
        print(f"  {k:24s} n={len(v):6d}  PA-MPJPE {np.mean(v):.2f} mm (median {np.median(v):.2f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", required=True, help="bench.world_dynhamr --result_stage init run")
    p.add_argument("--final", required=True, help="bench.world_dynhamr run (its optimised output)")
    p.add_argument("--ref", default="", help="comma-separated other runs scored on the same hand-frames")
    main(p.parse_args())
