"""Core ML exports for one FAH model on Apple silicon -> <weights>/coreml/ (what `fah --coreml` and `fah-live` load):
<model>-pose.mlpackage (the whole decoder-free pose network, float32 by default on the GPU, batch functions b1-b4: crossed / interacting
hands give 3-4 crops per pass), <model>-pose-noparts.mlpackage (the same without the part head: what runs while the part
maps are off, the default) and detector-640.mlpackage (the YOLO-seg hand detector shared/detector.pt; fp16
on the Neural Engine, one function per letterbox shape: --frames, by default the common landscape and portrait video
aspects, DEFAULT_FRAMES), via
scripts/build_coreml.py; then, with --parity_frames, the end-to-end parity of the Core ML configuration against the
PyTorch fp32 model on those frames (per matched hand: worst joint / vertex differences).
macOS only; needs coremltools (pip install -e ".[coreml]").

    fah-build-coreml --backbone s --parity_frames <dir of <stream>_<index>.jpg frames with hands>
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from fah.api import DEFAULT_SAM3D, DEFAULT_WEIGHTS, REPO, coreml_noparts, coreml_paths, detector_weights, method_args
from fah.cli import BACKBONES

# frame sizes whose letterbox shapes (imgsz 640, stride 32) the detector export holds: 16:9, 4:3, 3:2, 1:1 (384 / 480 / 448 /
# 640 x 640), 1.85:1 (352 x 640), 2:1 and 19.5:9 phone video (320 x 640), 16:10 (416 x 640) and the portrait 9:16, 3:4, 2:3, 1:2
# (640 x 384 / 480 / 448 / 320); other frames are padded up to the nearest built shape (fah.coreml.fit_letterbox)
DEFAULT_FRAMES = ("1920x1080,1440x1080,1620x1080,1080x1080,1134x612,2160x1080,1920x1200,1080x1920,1080x1440,1080x1620,"
                  "1080x2160")


def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _package_digest(path):
    h = hashlib.sha256()
    for p in sorted(Path(path).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).encode())
            h.update(b"\0")
            h.update(_digest(p).encode())
    return h.hexdigest()


def _build_identity(kind, args, parts=True, cache=None):
    cache = {} if cache is None else cache

    def source(p):
        p = os.path.abspath(p)
        if p not in cache:
            cache[p] = _digest(p) if os.path.isfile(p) else None
        return cache[p]

    code = [Path(REPO) / "scripts/build_coreml.py", Path(REPO) / "bench/methods/fah.py"]
    code += sorted((Path(REPO) / "fah").rglob("*.py"))
    versions = {}
    for name in ("torch", "coremltools", "timm", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    identity = dict(kind=kind, versions=versions, macos=platform.mac_ver()[0],
                    code={str(p.relative_to(REPO)): source(p) for p in code})
    if kind == "pose":
        identity.update(precision=getattr(args, "pose_precision", "float32"), parts=parts,
                        batch=sorted(set(args.batch)),
                        inputs={"checkpoint": source(os.path.join(args.weights, BACKBONES[args.backbone], "model.pt")),
                                "sam_config": source(os.path.join(args.sam3d, "model_config.yaml")),
                                "sam_checkpoint": source(os.path.join(args.sam3d, "model.ckpt"))
                                if os.path.isfile(os.path.join(args.sam3d, "model.ckpt")) else "absent",
                                "mhr": source(os.path.join(args.sam3d, "assets/mhr_model.pt"))})
    else:
        identity.update(precision="float16", frames=sorted(set(args.frames.split(","))), imgsz=640,
                        inputs={"checkpoint": source(detector_weights(args.weights))})
    return identity


def _manifest_path(out):
    return out + ".fah-build.json"


def _is_current(out, identity):
    if not os.path.isdir(out) or any(v is None for v in identity["inputs"].values()):
        return False
    try:
        with open(_manifest_path(out)) as f:
            manifest = json.load(f)
        return (manifest["schema"] == 1 and manifest["identity"] == identity
                and manifest["package_sha256"] == _package_digest(out))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _write_manifest(out, identity):
    if not os.path.isdir(out):
        raise RuntimeError(f"Core ML exporter did not create {out}")
    manifest = dict(schema=1, identity=identity, package_sha256=_package_digest(out))
    fd, tmp = tempfile.mkstemp(prefix=".fah-build-", dir=os.path.dirname(out))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        os.replace(tmp, _manifest_path(out))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def commands(args):
    model = BACKBONES[args.backbone]
    pose, det = coreml_paths(model, args.weights)
    script = os.path.join(REPO, "scripts", "build_coreml.py")
    detector = detector_weights(args.weights)
    noparts = coreml_noparts(pose)
    cmds = []
    cache = {}
    args._build_identities = {}
    for out, extra in ((pose, []), (noparts, ["--no_parts"])):  # with / without the part head (fah --parts, key m of fah-live)
        identity = _build_identity("pose", args, parts=not extra, cache=cache)
        if args.force or not _is_current(out, identity):
            args._build_identities[out] = identity
            cmds.append([sys.executable, script, "pose", "--finetune_ckpt", os.path.join(args.weights, model, "model.pt"), "--heads",
                         *extra, "--detector_weights", detector, "--sam3d", args.sam3d, "--batch", *map(str, args.batch),
                         "--precision", getattr(args, "pose_precision", "float32"), "--out", out])
    identity = _build_identity("det", args, cache=cache)
    if args.force or not _is_current(det, identity):
        args._build_identities[det] = identity
        cmds.append([sys.executable, script, "det", "--weights", detector, "--imgsz", "640", "--frames", args.frames, "--out", det])
    if args.parity_frames:
        base = method_args(model, "image", args.weights, "mps", args.sam3d, mesh=args.mesh, dtype="float32", det_half=False,
                           mesh_fit_device=None, mesh_fit_jac="fd")
        test = dict(pose_coreml=noparts, det_coreml=det, mesh_fit_device="cpu", mesh_fit_jac="fdb")  # part maps off (the default)
        cmds.append([sys.executable, script, "parity", "--frames_dir", args.parity_frames, "--base_args", json.dumps(base),
                     "--args", json.dumps(test), "--names", "coreml", "--focal", str(args.focal),
                     "--out", os.path.join(os.path.dirname(pose), f"{model}-parity.json")])
    return cmds


def main(argv=None):
    p = argparse.ArgumentParser(prog="fah-build-coreml", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="s", choices=list(BACKBONES))
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--sam3d", default=DEFAULT_SAM3D)
    p.add_argument("--frames", default=DEFAULT_FRAMES,
                   help="frame sizes WxH of the detector's letterbox shapes (default: 16:9, 4:3, 3:2, 1:1, 1.85:1, 2:1, 16:10 and the "
                        "portrait 9:16, 3:4, 2:3, 1:2)")
    p.add_argument("--batch", type=int, nargs="+", default=[1, 2, 3, 4], help="pose network batch sizes (one function each)")
    p.add_argument("--parity_frames", default="", help="dir of <stream>_<index>.jpg frames for the parity check (skipped if empty)")
    p.add_argument("--mesh", default="mano", choices=["none", "mano"], help="parity with / without the mesh fit")
    p.add_argument("--focal", type=float, default=610.0, help="parity frames' focal length (px)")
    p.add_argument("--pose_precision", choices=("float32", "float16"), default="float32", help="pose compute precision (float32 for reliable outputs; float16 needs parity validation)")
    p.add_argument("--force", action="store_true", help="rebuild existing exports")
    p.add_argument("--dry_run", action="store_true", help="print the commands only")
    args = p.parse_args(argv)
    for k in ("weights", "sam3d", "parity_frames"):  # the steps run from the repository directory
        if getattr(args, k):
            setattr(args, k, os.path.abspath(os.path.expanduser(getattr(args, k))))
    if sys.platform != "darwin" and not args.dry_run:
        p.error("Core ML exports are built on macOS (Apple silicon)")
    cmds = commands(args)
    if not args.parity_frames:
        print("no --parity_frames: the parity check is skipped", flush=True)
    for c in cmds:
        print(" ".join(c[:3] + ["..."]), flush=True)
        if not args.dry_run:
            subprocess.run(c, check=True, cwd=REPO, env=dict(os.environ, PYTHONPATH=REPO))
            if c[2] in ("pose", "det"):
                out = c[c.index("--out") + 1]
                identity = args._build_identities[out]
                current = _build_identity(c[2], args, parts="--no_parts" not in c)
                if current != identity:
                    raise RuntimeError("Core ML build inputs changed during export; run the builder again")
                _write_manifest(out, identity)


if __name__ == "__main__":
    main()
