"""GT-free camera up vectors from GeoCalib (cvg/GeoCalib; ViPE's copy, vipe.priors.geocalib) on every --step-th frame
of a video, with the ViPE camera's focal (bench/world_vipe.py fx) as a prior: per frame the up direction in the camera
frame (OpenCV axes: x right, y down, z forward) and GeoCalib's roll / pitch uncertainties (rad)
-> <out>/<name>.npz: frames, up_cam (n, 3), roll_unc, pitch_unc, fx_prior.
GeoCalib's vector at zero roll / pitch is (0, -1, 0) (camera up); the sign is kept so that up_cam . (0, -1, 0) > 0 for
most frames (hand-held / mounted cameras are upright). For world-space evaluation (world up error vs GT) and
the world-mode focal rule. Runs in ViPE's env (bench/baselines/setup_vipe.sh):
    $FAB_ENVS/vipe/bin/python -m bench.world_geocalib --list clips.json --cams <vipe out>/cams --out <dir>
    $FAB_ENVS/vipe/bin/python -m bench.world_geocalib --emdb <EMDB root> --subset emdb2 --cams <vipe out>/cams --out <dir>
"""

import argparse
import os
import sys

import numpy as np

FAB_HOME = os.path.expanduser(os.environ.get("FAB_HOME", "~/.cache/fab"))  # as fab.paths (this runs in ViPE's env)
FAB_ENVS = os.path.expanduser(os.environ.get("FAB_ENVS", os.path.join(FAB_HOME, "envs")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emdb", default="")
    p.add_argument("--subset", default="emdb2")
    p.add_argument("--threedpw", default="")
    p.add_argument("--split", default="test")
    p.add_argument("--list", default="", help="as bench/world_vipe.py --list")
    p.add_argument("--only", default="")
    p.add_argument("--cams", required=True, help="ViPE cams dir (fx prior)")
    p.add_argument("--out", required=True)
    p.add_argument("--step", type=int, default=10)
    p.add_argument("--vipe", default=os.environ.get("VIPE_DIR", os.path.join(FAB_ENVS, "src", "vipe")))
    args = p.parse_args()
    os.environ.setdefault("TORCH_HOME", os.path.join(FAB_HOME, "hf-cache", "vipe", "torch"))
    sys.path.insert(0, args.vipe)
    import imageio.v2 as imageio
    import torch
    from vipe.priors.geocalib import GeoCalib

    from bench.world_vipe import videos

    model = GeoCalib(weights="pinhole").cuda()
    os.makedirs(args.out, exist_ok=True)
    for name, imgs, _ in videos(args):
        out_path = os.path.join(args.out, f"{name}.npz")
        cam_path = os.path.join(args.cams, f"{name}.npz")
        if os.path.exists(out_path) or not os.path.exists(cam_path):
            continue
        fx = float(np.load(cam_path)["fx"])
        frames = np.arange(0, len(imgs), args.step)
        up, ru, pu = [], [], []
        for f in frames:
            img = torch.from_numpy(imageio.imread(imgs[f])).permute(2, 0, 1).float().cuda() / 255.0
            res = model.calibrate(img, priors={"focal": torch.tensor(fx).cuda()})
            up.append(res["gravity"].vec3d[0].float().cpu().numpy())
            ru.append(float(res.get("roll_uncertainty", torch.zeros(1)).reshape(-1)[0]))
            pu.append(float(res.get("pitch_uncertainty", torch.zeros(1)).reshape(-1)[0]))
        up = np.array(up, np.float64)
        if np.median(-up[:, 1]) < 0:  # make it point up (OpenCV y is down)
            up = -up
        np.savez(out_path, frames=frames, up_cam=up, roll_unc=np.array(ru), pitch_unc=np.array(pu), fx_prior=fx)
        tilt = np.degrees(np.arccos(np.clip(-up[:, 1], -1, 1)))
        print(f"[geocalib] {name}: {len(frames)} frames, camera tilt from upright median {np.median(tilt):.1f} deg",
              flush=True)


if __name__ == "__main__":
    main()
