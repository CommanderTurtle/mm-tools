"""Human-clicked 2D labels -> the kinematic MANO joint convention (per-source fixed 2D offsets measured by
train.egodex_mano_offsets fit2d) as new pools <pool>_kin sharing images / masks with the originals (links, as
train/convert_egodex_labels.py). A visible joint moves by -size * R @ delta[pool][j], with R / size the label's 2D
hand frame (origin wrist, y = wrist -> middle MCP, x = pinky MCP -> index MCP orthogonalised); crops whose wrist or
index / middle / pinky MCP are not labelled keep their labels.

    python -m train.convert_2d_labels --root $DATA_ROOT/crops --pools coco_wb,halpe_lab,hint_lab,cmu_lab \
        --offsets $CKPT_ROOT/label2d/label2d_offsets.json --pack
"""
import argparse
import json
import os

import numpy as np

from train.convert_egodex_labels import write_pool
from train.egodex_mano_offsets import frame2d


def convert(args, pool, delta):
    recs = [json.loads(l) for l in open(os.path.join(args.root, pool, "index.jsonl"))]
    idx = [i for i, r in enumerate(recs) if r.get("kp2d") is not None]
    kp = np.asarray([recs[i]["kp2d"] for i in idx], np.float64)
    vis = kp[..., 2] > 0
    ok = vis[:, [0, 5, 9, 17]].all(1)
    R, size = frame2d(kp[:, :, :2])
    shift = -np.einsum("nij,kj->nki", R, delta) * size[:, None, None]
    shift[~ok] = 0
    shift[~vis] = 0
    kp[..., :2] += shift
    for n, i in enumerate(idx):
        recs[i]["kp2d"] = kp[n].round(2).tolist()
        if ok[n]:
            recs[i]["label_conv"] = f"2d -> kinematic ({os.path.basename(args.offsets)})"
    moved = np.linalg.norm(shift, axis=-1)[vis & ok[:, None]]
    print(f"{pool}: {int(ok.sum())} / {len(recs)} records corrected, mean visible-joint shift {moved.mean():.2f} crop px", flush=True)
    write_pool(args.root, pool, pool + args.suffix, recs, args.pack)


def main(args):
    d = json.load(open(args.offsets))["delta2d"]
    for pool in args.pools.split(","):
        convert(args, pool, np.asarray(d[pool]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="$DATA_ROOT/crops")
    p.add_argument("--pools", default="coco_wb,halpe_lab,hint_lab,cmu_lab")
    p.add_argument("--offsets", required=True, help="label2d_offsets.json from train.egodex_mano_offsets fit2d")
    p.add_argument("--suffix", default="_kin")
    p.add_argument("--pack", action="store_true")
    main(p.parse_args())
