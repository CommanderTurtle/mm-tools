"""ViPE's dense artifacts of a clip (bench/world_vipe.py --depth) -> one compact npz for scene fusion: metric depth
(float16, downsampled by --down, nearest), a dynamic mask (any ViPE instance: person / animal / vehicle / ball / sky;
dilated by --dilate pixels at the reduced size) and the per-frame intrinsics at the reduced size.
Runs in ViPE's env (reads the zipped EXR / PNG artifacts with vipe.utils.io):
    $FAB_ENVS/vipe12/bin/python -m bench.videomimic.vipe_depth --vipe_out <vipe out> --vipe $FAB_ENVS/src/vipe12 --name <clip>
-> <vipe_out>/dense/<name>.npz: depth (T, h, w) f16 (m, ViPE's own metric scale), dyn (T, h, w) bool, K (T, 3, 3),
   frames (T,), down, phrases (json)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

FAB_HOME = os.path.expanduser(os.environ.get("FAB_HOME", "~/.cache/fab"))  # as fab.paths (this runs in ViPE's env)
FAB_ENVS = os.path.expanduser(os.environ.get("FAB_ENVS", os.path.join(FAB_HOME, "envs")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vipe_out", required=True)
    p.add_argument("--name", default="", help="clip names, comma separated (default: every depth zip)")
    p.add_argument("--down", type=int, default=4)
    p.add_argument("--dilate", type=int, default=2)
    p.add_argument("--vipe", default=os.environ.get("VIPE_DIR", os.path.join(FAB_ENVS, "src", "vipe")))
    args = p.parse_args()
    sys.path.insert(0, args.vipe)
    import cv2
    from vipe.utils.io import read_depth_artifacts, read_instance_artifacts, read_instance_phrases

    base = Path(args.vipe_out)
    names = args.name.split(",") if args.name else sorted(f.stem for f in (base / "depth").glob("*.zip"))
    os.makedirs(base / "dense", exist_ok=True)
    for name in names:
        out = base / "dense" / f"{name}.npz"
        if out.exists() or not (base / "depth" / f"{name}.zip").exists():
            continue
        d = args.down
        dep = {i: x.numpy() for i, x in read_depth_artifacts(base / "depth" / f"{name}.zip")}
        frames = np.array(sorted(dep))
        H, W = dep[frames[0]].shape
        h, w = H // d, W // d
        depth = np.stack([cv2.resize(dep[i], (w, h), interpolation=cv2.INTER_NEAREST) for i in frames]).astype(np.float16)
        dyn = np.zeros((len(frames), h, w), bool)
        mz = base / "mask" / f"{name}.zip"
        phrases = {}
        if mz.exists():
            pos = {int(f): k for k, f in enumerate(frames)}
            ker = np.ones((2 * args.dilate + 1, 2 * args.dilate + 1), np.uint8)
            for i, m in read_instance_artifacts(mz):
                if i in pos:
                    mm = cv2.resize((m.numpy() > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    dyn[pos[i]] = cv2.dilate(mm, ker) > 0 if args.dilate else mm > 0
            pp = base / "mask" / f"{name}.txt"
            phrases = read_instance_phrases(pp) if pp.exists() else {}
        intr = np.load(base / "intrinsics" / f"{name}.npz")
        ii = {int(k): v for k, v in zip(intr["inds"], intr["data"])}
        K = np.zeros((len(frames), 3, 3))
        for k, f in enumerate(frames):
            fx, fy, cx, cy = ii.get(int(f), intr["data"][0])[:4]
            K[k] = [[fx / d, 0, cx / d], [0, fy / d, cy / d], [0, 0, 1]]
        np.savez_compressed(out, depth=depth, dyn=dyn, K=K, frames=frames, down=d, full_hw=np.array([H, W]),
                            phrases=json.dumps({str(k): v for k, v in phrases.items()}))
        print(f"[dense] {name}: {len(frames)} frames {H}x{W} -> {h}x{w}, dynamic {dyn.mean():.3f}, "
              f"depth median {np.nanmedian(depth.astype(np.float32)):.2f} m", flush=True)


if __name__ == "__main__":
    main()
