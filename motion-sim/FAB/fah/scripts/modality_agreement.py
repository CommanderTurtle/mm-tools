"""Accuracy of direct and projected outputs on a 3D benchmark run (bench.analysis.agreement; the supplementary table): PCK of
the 2D output, of the projected camera-frame 3D joints and of the projected mesh joints against the GT joints projected with
the true intrinsics; PA-MPJPE of the 3D joints and of the mesh joints; the internal 2D disagreement between them.

Paper table: HOT3D, every 5th frame (--every 5), PCK@0.05 (pck05) and PA-MPJPE (pa3d / pamesh).

    python scripts/modality_agreement.py --mano_dir $MANO_DIR --every 5 \
        --runs ours=$RESULTS/hot3d/<fah run>,WiLoR=$RESULTS/hot3d/wilor,HaMeR=$RESULTS/hot3d/hamer --out agreement.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.analysis.agreement import score_run  # noqa: E402
from bench.common import VertexJoints  # noqa: E402


def main(args):
    vj = VertexJoints(args.mano_dir)
    res = {}
    for spec in args.runs.split(","):
        name, run = spec.split("=", 1)
        res[name] = r = score_run(run, vj, args.every)
        f = lambda k: "%.3f / %.3f" % (r[k]["pck05"], r[k]["pck10"]) if k in r else "-"  # noqa: E731
        print("%-12s 2D %s | 3D proj %s | mesh proj %s | PA 3D %s mesh %s | 2D-3D %s 2D-mesh %s" % (
            name, f("e2d"), f("e3d"), f("emesh"), "%.2f" % r["pa3d"] if "pa3d" in r else "-", "%.2f" % r["pamesh"] if "pamesh" in r else "-",
            "%.3f" % r["d_2d_3d"]["median"] if "d_2d_3d" in r else "-", "%.3f" % r["d_2d_mesh"]["median"] if "d_2d_mesh" in r else "-"), flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="name=run_dir,...")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--every", type=int, default=1, help="score every k-th frame of each clip")
    p.add_argument("--out", default="")
    main(p.parse_args())
