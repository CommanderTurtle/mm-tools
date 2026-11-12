"""Verify the pinned model bundles and expose them to the tracked runtime; offline."""

import argparse
import json
import sys
from pathlib import Path

from state import PIN, ROOT, SOURCE, BACKEND_TREES

REPO = next(p for p in ROOT.parents if (p / "models/download_models.py").is_file())
CACHE = REPO / "models" / PIN["cache"]


def link(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() and path.resolve() == target.resolve():
        return
    if path.exists() or path.is_symlink():
        raise SystemExit(f"Preserving existing path: {path}; expected link to {target}")
    path.symlink_to(target, target_is_directory=target.is_dir())


def backend_links():
    """Link the committed pinned trees into the path the runtime resolves."""
    for name in BACKEND_TREES:
        link(SOURCE / "local_app" / name, ROOT / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phc", action="store_true")
    parser.add_argument("--physics-environment", choices=("check", "record"))
    args = parser.parse_args()
    if args.physics_environment:
        from importlib import metadata, util
        from state import atomic, physics_contract

        if sys.version_info[:2] != (3, 8):
            raise SystemExit(
                "PHC requires the Isaac Gym Preview 4 Python 3.8 environment."
            )
        try:
            torch_version = metadata.version("torch")
        except metadata.PackageNotFoundError:
            torch_version = ""
        if not torch_version.startswith("1.13.") or util.find_spec("isaacgym") is None:
            raise SystemExit(
                "Prepare Isaac Gym Preview 4 and PyTorch 1.13 / CUDA 11.7 in this interpreter first."
            )
        receipt = ROOT / ".physics-ready"
        if args.physics_environment == "check":
            receipt.unlink(missing_ok=True)
        else:
            executable = Path(sys.executable).absolute()
            atomic(
                receipt,
                dict(
                    interpreter=str(executable.parent.resolve() / executable.name),
                    prefix=sys.prefix,
                    python=list(sys.version_info[:2]),
                    torch=torch_version,
                    contract=physics_contract(),
                ),
            )
        return
    backend_links()
    entries = {
        "YOLOX": "setup.py",
        "KimodoMotionCorrection": "setup.py",
        "RGBMocapCorrection": "pyproject.toml",
        "PerpetualHumanoidControl": "phc/env/tasks/humanoid.py",
        "sam-3d-body": "sam_3d_body/__init__.py",
        "vggt-omega": "vggt_omega/__init__.py",
        "smplsim": "smpl_sim/__init__.py",
        "smplx": "setup.py",
        "chumpy": "chumpy/__init__.py",
    }
    absent = [
        str(ROOT / name / entry)
        for name, entry in entries.items()
        if not (ROOT / name / entry).is_file()
    ]
    if absent:
        raise SystemExit("Unpacked source is incomplete:\n" + "\n".join(absent))
    required = [
        "checkpoints/flowhmr_latest/flowhmr_latest.ckpt",
        "checkpoints/flowhmr_latest/config.yml",
        "ckpts/yolox/yolox_l.pth",
        "ckpts/sam-3d-body-dinov3/model.ckpt",
        "ckpts/sam-3d-body-dinov3/model_config.yaml",
        "ckpts/sam-3d-body-dinov3/assets/mhr_model.pt",
        "assets/body_models/smplh/neutral/model.npz",
        "assets/body_models/smpl_neutral_J_regressor.pt",
    ]
    if args.phc:
        required += [
            "data/smpl/SMPL_NEUTRAL.pkl",
            "data/smpl/SMPL_MALE.pkl",
            "data/smpl/SMPL_FEMALE.pkl",
            "output/HumanoidIm/phc_comp_3/Humanoid.pth",
            "output/HumanoidIm/phc_3/Humanoid.pth",
            "assets/example/amass_isaac_gender_betas.pkl",
            "assets/example/amass_isaac_gender_betas_unique.pkl",
        ]
    missing = [str(CACHE / f) for f in required if not (CACHE / f).is_file()]
    if missing:
        raise SystemExit(
            "Select flowhmr in the model picker; place the SMPL body files listed "
            "in models/which-ones.txt:\n" + "\n".join(missing)
        )
    link(ROOT / "models", CACHE)
    for relative in (
        "checkpoints",
        "ckpts",
        "assets/body_models/smplh",
        "assets/body_models/smpl_neutral_J_regressor.pt",
    ):
        link(SOURCE / relative, CACHE / relative)
    if args.phc:
        for relative in ("data/smpl", "output/HumanoidIm"):
            link(SOURCE / relative, CACHE / relative)
        for file in (
            "amass_isaac_gender_betas.pkl",
            "amass_isaac_gender_betas_unique.pkl",
        ):
            link(SOURCE / "assets/example" / file, CACHE / "assets/example" / file)


if __name__ == "__main__":
    main()
