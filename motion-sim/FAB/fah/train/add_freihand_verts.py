"""Attach FreiHAND's GT MANO vertices (training_verts.json, camera frame, metres) to the freihand_lab crop pool:
writes <crop>_v.npy (float16, 778 x 3) next to every crop and a copy of the pool index with "verts_file", as
train/build_hot3d_crops.py --verts does. All four FreiHAND training versions of an image share its GT.
The joints of the GT vertices (MANO regressor) are checked against the pool's joints3d (training_xyz.json).

    python -m train.add_freihand_verts --pool $DATA_ROOT/crops/freihand_lab --verts training_verts.json --out_name freihand_lab_mesh
"""
import argparse
import json
import os
import shutil

import numpy as np


def main(args):
    verts = np.asarray(json.load(open(args.verts)), np.float32)  # (32560, 778, 3)
    n = len(verts)
    out = os.path.join(os.path.dirname(args.pool.rstrip("/")), args.out_name)
    if not os.path.exists(out):
        shutil.copytree(args.pool, out, symlinks=True)
    recs = [json.loads(line) for line in open(os.path.join(args.pool, "index.jsonl"))]
    errs = []
    with open(os.path.join(out, "index.jsonl"), "w") as f:
        for r in recs:
            idx = int(os.path.splitext(os.path.basename(r["src"]))[0]) % n
            v = verts[idx]
            name = os.path.splitext(r["file"])[0] + "_v.npy"
            np.save(os.path.join(out, name), v.astype(np.float16))
            r["verts_file"] = name
            if len(errs) < 2000 and "joints3d" in r:  # GT consistency: vertex centroid vs joint centroid
                errs.append(np.linalg.norm(v.mean(0) - np.asarray(r["joints3d"]).mean(0)) * 1000)
            f.write(json.dumps(r) + "\n")
    print(f"{len(recs)} crops -> {out}; |vertex centroid - joint centroid| median {np.median(errs):.1f} mm")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", required=True)
    p.add_argument("--verts", required=True)
    p.add_argument("--out_name", default="freihand_lab_mesh")
    main(p.parse_args())
