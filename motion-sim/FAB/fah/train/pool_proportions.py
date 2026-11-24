"""Median finger-bone proportions of labelled crop pools (index.jsonl joints3d, OpenPose order): every bone length
divided by the same hand's middle proximal phalanx (middle MCP -> PIP), median over the pool's crops.

    python -m train.pool_proportions $DATA_ROOT/crops/hot3d_lab_mesh $DATA_ROOT/crops/arctic_ego_lab ... [--md out.md]
"""
import argparse
import json
import os

import numpy as np

FINGERS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
# per finger: metacarpal (wrist -> first joint; thumb: CMC -> MCP), proximal, middle, distal (thumb: MCP->IP, IP->tip)
BONES = {}
for f, (a, b, c, d) in FINGERS.items():
    if f == "thumb":
        BONES.update({f"thumb_metacarpal": (a, b), "thumb_proximal": (b, c), "thumb_distal": (c, d)})
    else:
        BONES.update({f"{f}_metacarpal": (0, a), f"{f}_proximal": (a, b), f"{f}_middle": (b, c), f"{f}_distal": (c, d)})
REF = BONES["middle_proximal"]


def proportions(index, max_n=200000, seed=0):
    J = [json.loads(l)["joints3d"] for l in open(index) if '"joints3d"' in l]
    J = np.asarray(J, np.float64)
    if len(J) > max_n:
        J = J[np.random.default_rng(seed).choice(len(J), max_n, replace=False)]
    ref = np.linalg.norm(J[:, REF[1]] - J[:, REF[0]], axis=1)
    ok = ref > 1e-4
    out = {k: float(np.median(np.linalg.norm(J[ok, b] - J[ok, a], axis=1) / ref[ok])) for k, (a, b) in BONES.items()}
    out["middle_proximal_mm"] = float(np.median(ref[ok]) * 1000)
    return out, int(ok.sum())


def main(args):
    rows = []
    for d in args.pools:
        p, n = proportions(os.path.join(d, "index.jsonl"))
        rows.append((os.path.basename(d.rstrip("/")), n, p))
        print(os.path.basename(d.rstrip("/")), n, json.dumps({k: round(v, 3) for k, v in p.items()}), flush=True)
    if args.md:
        keys = list(rows[0][2])
        with open(args.md, "w") as f:
            f.write("| pool | crops | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 2) + "\n")
            for name, n, p in rows:
                f.write(f"| {name} | {n} | " + " | ".join(f"{p[k]:.3f}" if k != "middle_proximal_mm" else f"{p[k]:.1f}" for k in keys) + " |\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pools", nargs="+", help="crop pool directories (with index.jsonl)")
    p.add_argument("--md", default="")
    main(p.parse_args())
