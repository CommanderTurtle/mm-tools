"""Run FAH with the paths and output options in configs/local.json."""
import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._recipes import MODELS, args, config, execute, module, path


def commands(c, source, mode=None, model=None):
    model = model or c.get("model", "fah")
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    source = Path(source).expanduser().resolve()
    video = source.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    mode = mode or c.get("inference", {}).get("mode") or ("video" if video else "image")
    if mode not in ("image", "video", "world") or (not video and mode != "image"):
        raise ValueError("Video/world mode requires a video input.")
    opts = c.get("inference", {})
    cmd = module(c, "fah.cli", *args(**{"vid" if video else "img": source}),
                 *args(backbone=MODELS[model][0], mode=mode, output_dir=path(c, "output_dir", "output/fah")))
    for key in ("weights", "mano_dir", "sam3d"):
        if c.get(key):
            cmd += args(**{key: path(c, key)})
    cmd += args(device=opts.get("device", "auto"), focal=opts.get("focal"), fov=opts.get("fov"))
    for key in ("dense", "parts", "contact", "masks"):
        if opts.get(key, False):
            cmd.append("--" + key)
    backend = opts.get("backend", "torch")
    if backend not in ("torch", "trt", "coreml"):
        raise ValueError("inference.backend must be torch, trt or coreml.")
    if backend != "torch":
        cmd.append("--" + backend)
    if mode == "world":
        world = c.get("world", {})
        if world.get("camera_version", "realtime") not in ("realtime", "offline"):
            raise ValueError("world.camera_version must be realtime or offline.")
        for key, value in world.items():
            if key not in {"camera_version", "dpvo_dir", "dpvo_python", "depth_python", "droid_dir", "droid_weights",
                           "metric3d_dir", "metric3d_weights", "slam_python", "cameras"}:
                raise ValueError(f"Unknown world option: {key}")
            if value:
                cmd += args(**{key: value if key == "camera_version" else path(world, key)})
    return [cmd]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="image, image folder or video")
    p.add_argument("--config", default="configs/local.json")
    p.add_argument("--model", choices=list(MODELS))
    p.add_argument("--mode", choices=["image", "video", "world"])
    p.add_argument("--dry-run", action="store_true", help="print the command without running inference")
    a = p.parse_args()
    try:
        c = config(a.config)
        execute(commands(c, a.input, a.mode, a.model), c, a.dry_run)
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        p.exit(1, f"{e}\n")


if __name__ == "__main__":
    main()
