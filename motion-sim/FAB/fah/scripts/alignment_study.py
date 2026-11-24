"""Dense points versus MANO parameters in one network, and the agreement of one forward pass's outputs (bench.analysis.alignment;
the paper's dense-versus-parameters table and its supplementary paragraph).

Paper table (HOT3D, one network trained with both outputs: run its dense points + MANO fit, and its MANO-parameter head as the
mesh, both with kind "dense"), in % of hand size / mm:
  3D joints vs 2D = A_own (dense column; the parameter head's 3D joints are its mesh's joints, so its column reads B_joints of
  the MANO-head run in both 2D rows); mesh joints vs 2D = B_joints; mesh vs dense joints = D_3d; mesh vs dense surface = C_3d
  (PA-MPVPE of the mesh: bench.evaluate's pa_mpvpe of each run); supplementary: D_2d / C_2d (in 2D), B_joints / C_2d with and
  without the rigid placement of the mesh onto the 2D joints (mesh_align_2d).

    python scripts/alignment_study.py --mano_dir $MANO_DIR --every 5 \
        --runs dense=$RESULTS/hot3d/<dense-fit run>:dense,param=$RESULTS/hot3d/<MANO-head run>:dense --out alignment.json
    (the paper scores every 5th frame of each HOT3D clip)
    (FreiHAND silhouettes: --masks <bench2d-format mask annotations keyed like the run's frames>)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench.analysis.alignment import Scorer, load_masks, score_run  # noqa: E402
from bench.common import ManoLayer, VertexJoints  # noqa: E402


def main(args):
    from fah.models.heads.keypoint_head import dense_subset
    from fah.vis import render_instance_masks

    ml = ManoLayer(args.mano_dir)
    rj, rv = ml(True, np.zeros(10), np.zeros(45), np.zeros(3), np.zeros(3))
    dense = dense_subset(rv, args.dense_n, "fps", rj)
    masks = load_masks(args.masks) if args.masks else None
    sil = lambda v, faces, K, hw: render_instance_masks([(v, faces)], K, hw) == 1  # noqa: E731
    res = {}
    for spec in args.runs.split(","):
        name, rest = spec.split("=", 1)
        run, kind = (rest.split(":") + ["dense"])[:2]
        out = res[name] = score_run(run, Scorer(VertexJoints(args.mano_dir), dense, kind, ml.faces[True], masks, sil), args.every)
        f = lambda k, fmt="%.3f": (fmt % out[k]["median"]) if k in out else "-"  # noqa: E731
        print(f"{name:14s} A 3D-vs-2D own {f('A_own')} fit {f('A_fit')} | B mesh-joints-vs-2D {f('B_joints')} IoU {f('B_iou')} BF {f('B_bf')} "
              f"(dense surface IoU {f('B_iou_dense')} BF {f('B_bf_dense')}) | C dense-vs-mesh 3D {f('C_3d', '%.2f')} mm 2D {f('C_2d')} | "
              f"D joints dense-vs-mesh 3D {f('D_3d', '%.2f')} mm 2D {f('D_2d')} | 2D err {f('e2d')} mesh-joint err {f('emesh')} | "
              f"PA {f('pa3d', '%.2f')} tips {f('tip3d', '%.2f')} others {f('nontip3d', '%.2f')}", flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="name=run_dir[:dense|param],... (dense: also the dense-vs-mesh measures C / D)")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--masks", default="", help="bench2d-format mask annotations keyed like the run's frames (FreiHAND segmentation)")
    p.add_argument("--dense_n", type=int, default=400)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--out", default="")
    main(p.parse_args())
