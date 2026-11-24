"""Where the world-space error comes from (supplement, camera-model diagnostics; uses GT, so diagnostic only): hybrid
cameras that replace one part of a predicted camera with ground truth, scored like any camera with bench.world.

The predicted camera (--cams, any bench.world camera spec of a directory) is first rigidly aligned to the GT camera
centres of each clip (GT poses: T_world_cam of the hand run's pickles); then
  gt_rotation   : GT orientations, our camera centres
  gt_centres    : our orientations, GT camera centres
  gt_clipscale  : our camera with its translation scaled about the first frame by the clip's GT Sim(3) scale
  gt_segscale<L>: our camera with each L-frame segment's translation increments scaled by that segment's GT Sim(3) scale
                  (segments that move < 1 cm keep scale 1)
  gt_rotation_segscale<L>: both
    python -m bench.world_camdiag --hands $RESULTS/egostandard_dev/fah-video --cams <camera dir> --out <dir> [--score]
-> <out>/<variant>/<clip>.npz (index, T_world_cam); --score also runs bench.world (WHAM protocol, 100-frame chunks and
whole tracks, own true positives) on the camera, the variants and the GT cameras and prints W / WA per row.
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np

from bench.world import evaluate_world, load_pred_cams, rigid_fit, sim_fit


def hybrids(T, gt, seg=81):
    """T, gt (N, 4, 4) predicted / GT T_world_cam of one clip -> {variant: (N, 4, 4)}."""
    n = len(T)
    R, t = rigid_fit(T[:, :3, 3], gt[:, :3, 3])  # predicted world -> GT world (camera centres)
    Ta = T.copy()
    Ta[:, :3, :3] = R @ T[:, :3, :3]
    Ta[:, :3, 3] = T[:, :3, 3] @ R.T + t
    p = Ta[:, :3, 3]
    s = np.ones(n)
    for a in range(0, n, seg):
        b = min(a + seg, n)
        if b - a > 2 and np.linalg.norm(gt[a:b, :3, 3] - gt[a:b, :3, 3].mean(0), axis=1).max() > 0.01:
            sc = sim_fit(p[a:b], gt[a:b, :3, 3])[0]
            s[a:b] = sc if sc > 0 else 1.0
    sc_clip = sim_fit(p, gt[:, :3, 3])[0]
    sc_clip = sc_clip if sc_clip > 0 else 1.0
    ps = np.concatenate([p[:1], p[:1] + np.cumsum(np.diff(p, axis=0) * s[1:, None], 0)])
    out = {}
    h = Ta.copy(); h[:, :3, :3] = gt[:, :3, :3]; out["gt_rotation"] = h
    h = Ta.copy(); h[:, :3, 3] = gt[:, :3, 3]; out["gt_centres"] = h
    h = Ta.copy(); h[:, :3, 3] = p[:1] + (p - p[:1]) * sc_clip; out["gt_clipscale"] = h
    h = Ta.copy(); h[:, :3, 3] = ps; out[f"gt_segscale{seg}"] = h
    h = Ta.copy(); h[:, :3, :3] = gt[:, :3, :3]; h[:, :3, 3] = ps; out[f"gt_rotation_segscale{seg}"] = h
    return out


def main(args):
    n_clips = 0
    for path in sorted(glob.glob(os.path.join(args.hands, "*.pkl"))):
        clip = pickle.load(open(path, "rb"))
        gt = np.stack([np.asarray(fr["T_world_cam"], np.float64) for fr in clip["frames"]])
        pc = load_pred_cams(args.cams, clip)
        if not pc:
            continue
        assert all(i in pc for i in range(len(gt))), f"{clip['clip']}: the camera must cover every frame"
        T = np.stack([pc[i] for i in range(len(gt))]).astype(np.float64)
        for k, Th in hybrids(T, gt, args.seg).items():
            os.makedirs(os.path.join(args.out, k), exist_ok=True)
            np.savez(os.path.join(args.out, k, f"{clip['clip']}.npz"), index=np.arange(len(Th)), T_world_cam=Th)
        n_clips += 1
    print(f"{n_clips} clips -> {args.out}")
    if not args.score:
        return
    rows = [("camera", args.cams)] + [(k, os.path.join(args.out, k)) for k in
            ("gt_rotation", "gt_centres", "gt_clipscale", f"gt_segscale{args.seg}", f"gt_rotation_segscale{args.seg}")] + [("GT cameras", "gt")]
    res = {}
    for name, spec in rows:
        res[name] = {ch: evaluate_world(args.hands, spec, protocol="wham", chunk=ch, min_run=16) for ch in (100, 0)}
        r = res[name]
        print(f"{name:26s} chunks {r[100]['w_mpjpe']:.2f} / {r[100]['wa_mpjpe']:.2f}   whole tracks {r[0]['w_mpjpe']:.2f} / {r[0]['wa_mpjpe']:.2f}",
              flush=True)
    json.dump({n: {str(ch): {k: v[k] for k in ("w_mpjpe", "wa_mpjpe", "rte", "num_frames")} for ch, v in r.items()}
               for n, r in res.items()}, open(os.path.join(args.out, "camdiag.json"), "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hands", required=True, help="hand run (bench.run pickles with GT T_world_cam per frame)")
    p.add_argument("--cams", required=True, help="predicted camera (bench.world --cams spec of a <clip>.npz dir)")
    p.add_argument("--out", required=True)
    p.add_argument("--seg", type=int, default=81, help="segment length (frames) of the per-segment GT scale")
    p.add_argument("--score", action="store_true")
    main(p.parse_args())
