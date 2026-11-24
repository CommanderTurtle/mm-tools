"""Fingertip extent against human 2D labels on bench2d-format runs (bench.analysis.tips; the supplementary fingertip table):
median signed tip error along the labelled distal phalanx (x phalanx length; share short) and median predicted / labelled
finger length, per subset and over all.

Paper table (bench2d v2): joints = <run>:joints2d "all"; mesh = <run>:meshmax "all"; HOT3D = <run>:meshmax "hot3d" subset;
length = <run>:meshmax "all" length. FAH runs need the mesh fit (--require_fitted); without it, verts3d is the dense surface.
WiLoR / HaMeR meshes were predicted with WiLoR's default camera on the 2D sets (<run>:meshmax:wilor).

    python scripts/tip_extent.py --mano_dir $MANO_DIR --require_fitted \
        --runs $RESULTS/bench2d/v2-<fah run>:meshmax,$RESULTS/bench2d/v2-<fah run>:joints2d --out tips.json
    python scripts/tip_extent.py --mano_dir $MANO_DIR --runs $RESULTS/bench2d/v2-wilor:joints2d,$RESULTS/bench2d/v2-wilor:meshmax:wilor
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.analysis.tips import MESH_KEYS, score_run, tip_regions  # noqa: E402


def main(args):
    specs = [(q + ["joints2d", "fah"][len(q) - 1:])[:3] for q in (s.split(":") for s in args.runs.split(","))]
    region = tip_regions(args.mano_dir) if any(k in ("meshmax", "surf2d") for _, k, _ in specs) else None
    res = {}
    for d, key, intr in specs:
        r = score_run(d, key, intr, region, args.require_fitted and key in MESH_KEYS[:2])
        name = os.path.basename(d.rstrip("/")) + ":" + key
        res[name] = r
        print(f"\n{name}: tip error along the distal phalanx (median, x phalanx length; share short) | finger length pred / GT (median)")
        for s in sorted(r, key=lambda s: (s != "all", s)):
            x = r[s]
            fing = ", ".join(f"{f} {v:+.2f}" for f, v in x["fingers"].items())
            length = f" | length {x['length']:.3f}" if x["length"] is not None else ""
            print(f"  {s:12s} all {x['tip']:+.2f} ({x['short']:.0%}, n={x['n']}) | {fing}{length}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="run_dir[:key[:fah|wilor]],... ; key = a 2D joints key, or mesh / meshmax / surf2d; "
                   "the last field = whose default camera the mesh was predicted with when the set has no intrinsics")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--require_fitted", action="store_true", help="FAH mesh keys: require the saved dense surface (verts3d_dense) that "
                   "marks a fitted MANO output")
    p.add_argument("--out", default="")
    main(p.parse_args())
