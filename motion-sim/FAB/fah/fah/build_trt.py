"""TensorRT FP16 engines for one FAH model on this GPU -> <weights>/trt/ (what `fah --trt` loads):
<model>-pose-fp16.engine (the pose encoder, scripts/build_pose_engine.py export + build; when <model>/contact.pt has a
contact branch, the engine also outputs the branch's input, used with and without contact) and detector-640-fp16.engine
(the hand detector shared/detector.pt; scripts/build_det_engine.py). Engines depend on the GPU model and the TensorRT version. The conversion needs onnx +
nvidia-modelopt[onnx] + tensorrt (pip install -e ".[engine,trt]"; modelopt may want its own environment: --trt_python).

    fah-build-trt --backbone l --calib_images <dir of images with hands>
"""
import argparse
import os
import subprocess
import sys
import tempfile

from fah.api import DEFAULT_SAM3D, DEFAULT_WEIGHTS, DETECTOR, FAH_HOME, REPO, detector_weights, trt_dir
from fah.cli import BACKBONES


def commands(args):
    model = BACKBONES[args.backbone]
    out = args.out or trt_dir(args.weights)
    work = args.work or tempfile.mkdtemp(prefix=f"fah_trt_{model}_")
    script = lambda name: os.path.join(REPO, "scripts", name)
    pose, det = os.path.join(out, f"{model}-pose-fp16.engine"), os.path.join(out, f"{DETECTOR}-640-fp16.engine")
    cmds = []
    if args.force or not os.path.exists(pose):
        contact = os.path.join(args.weights, model, "contact.pt")  # a contact head with a branch: + its input as a second output
        branch = ["--contact_ckpt", contact] if os.path.exists(contact) and not getattr(args, "no_contact_branch", False) else []
        cmds += [[sys.executable, script("build_pose_engine.py"), "export", "--finetune_ckpt", os.path.join(args.weights, model, "model.pt"),
                  "--detector", detector_weights(args.weights), "--sam3d", args.sam3d,
                  "--calib_frames", args.calib_images, "--work", work, *branch],
                 [args.trt_python, script("build_pose_engine.py"), "build", "--work", work, "--out", pose]]
    if args.force or not os.path.exists(det):
        cmds.append([args.trt_python, script("build_det_engine.py"), "--weights", detector_weights(args.weights),
                     "--imgsz", "640", "--out", det])
    return cmds


def main(argv=None):
    p = argparse.ArgumentParser(prog="fah-build-trt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="l", choices=list(BACKBONES))
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--sam3d", default=DEFAULT_SAM3D)
    p.add_argument("--calib_images", required=True, help="a few images with hands (value ranges of the fp16 conversion)")
    p.add_argument("--out", default="", help="default: <weights>/trt")
    p.add_argument("--work", default="", help="scratch dir (ONNX)")
    p.add_argument("--trt_python", default=sys.executable, help="python with tensorrt + nvidia-modelopt for the build steps")
    p.add_argument("--force", action="store_true", help="rebuild existing engines")
    p.add_argument("--no_contact_branch", action="store_true", help="pose engine without the contact branch's input (an "
                   "engine for use without contact=True; with a branch contact head, contact then runs the PyTorch encoder)")
    p.add_argument("--dry_run", action="store_true", help="print the commands only")
    args = p.parse_args(argv)
    for k in ("weights", "sam3d", "calib_images", "out", "work"):  # the steps run from the repository directory
        if getattr(args, k):
            setattr(args, k, os.path.abspath(os.path.expanduser(getattr(args, k))))
    env = dict(os.environ, PYTHONPATH=REPO)
    if "MANO_DIR" not in env and os.path.isdir(os.path.join(FAH_HOME, "mano")):  # the default of fah.FAH
        env["MANO_DIR"] = os.path.join(FAH_HOME, "mano")
    for c in commands(args):
        print(" ".join(c), flush=True)
        if not args.dry_run:
            subprocess.run(c, check=True, cwd=REPO, env=env)


if __name__ == "__main__":
    main()
