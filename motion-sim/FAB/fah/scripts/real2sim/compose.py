"""Re-compose a finished sim.py run from its labelled panel videos (no re-simulation: the physics replays are not
bit-reproducible, so the video must come from the run whose metrics are reported) -> <sim dir>/<out> + poster.

    python scripts/real2sim/compose.py --sim_dirs $R2S/drill/sim_fah_recon_contact \
        --panels input,overlay,kin,phys --cols 2 --out side_by_side_2x2.mp4
    python scripts/real2sim/compose.py --sim_dirs <dir> --panels input,overlay,phys --cols 3 --width 1440 --out page_1440.mp4
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.real2sim.sim import label, write_mp4  # noqa: E402


def read(path):
    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, im = cap.read()
        if not ok:
            return out
        out.append(im)


def main(args):
    for d in args.sim_dirs.split(","):
        P = [read(os.path.join(d, f"{k}.mp4")) for k in args.panels.split(",")]
        for r in [r for r in args.relabel.split(";") if r]:  # redraw a panel's label bar (same style as sim.py's label)
            k, text = r.split("=", 1)
            P[args.panels.split(",").index(k)] = [label(im, text) for im in P[args.panels.split(",").index(k)]]
        n = min(len(p) for p in P)
        grid = []
        for i in range(n):
            tiles = [p[i] for p in P]
            while len(tiles) % args.cols:
                tiles.append(np.zeros_like(tiles[0]))
            grid.append(np.concatenate([np.concatenate(tiles[r * args.cols:(r + 1) * args.cols], 1)
                                        for r in range(len(tiles) // args.cols)], 0))
        if args.width:
            h = int(round(grid[0].shape[0] * args.width / grid[0].shape[1] / 2)) * 2
            grid = [cv2.resize(g, (args.width, h), interpolation=cv2.INTER_AREA) for g in grid]
        write_mp4(os.path.join(d, args.out), grid, args.fps)
        tr = np.load(os.path.join(d, "sim_traj.npz"))
        lift = tr["obj_ref"][:, 2, 3] - tr["obj_ref"][0, 2, 3]
        pf = int(np.argmax(lift > 0.7 * lift.max())) if lift.max() > 0.03 else n // 2
        cv2.imwrite(os.path.join(d, args.out[:-4] + ".jpg"), grid[min(pf, n - 1)], [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(d, n, "frames ->", args.out, json.load(open(os.path.join(d, "metrics.json"))).get("success"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim_dirs", required=True, help="comma-separated sim.py output dirs")
    p.add_argument("--panels", default="input,overlay,kin,phys")
    p.add_argument("--cols", type=int, default=2)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--width", type=int, default=0, help="resize the composed video to this width (e.g. 1440)")
    p.add_argument("--out", default="side_by_side_2x2.mp4")
    p.add_argument("--relabel", default="", help='"panel=new label;..." e.g. "overlay=FAH-L hand + reconstructed object"')
    main(p.parse_args())
