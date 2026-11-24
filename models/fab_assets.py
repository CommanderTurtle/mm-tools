# SPDX-License-Identifier: AGPL-3.0-only
"""Resolve FAB's native inputs from existing motion studios; never edit a donor model."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

MODELS = Path(__file__).resolve().parent
MOTION = MODELS.parent / "motion-sim"
STUDIO = MOTION / "FAB"
FAB = MODELS / "fab"
NEUTRAL = ("SMPL_NEUTRAL.pkl", "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
           "basicmodel_n_lbs_10_207_0_v1.1.0.pkl")
HANDS = ("MANO_RIGHT.pkl", "MANO_LEFT.pkl")


def env(name, default=None):
    value = os.environ.get(name) or default
    return Path(value).expanduser() if value else None


def find(roots, names):
    for root in roots:
        if not root:
            continue
        if root.is_file() and root.name.lower() in {n.lower() for n in names}:
            yield root
        if root.is_dir():
            for name in names:
                if (root / name).is_file():
                    yield root / name
            # Only a selected body-asset directory is walked, never the checkout or home.
            for p in sorted(root.rglob("*.pkl")):
                if p.name.lower() in {n.lower() for n in names} and p.is_file():
                    yield p


def soma():
    return env("SOMA_ASSETS", MOTION / "nvidia-sim/SOMA-X/assets")


def find_smpl():
    return next(find((env("FAB_SMPL_SOURCE"), env("FAB_INPUT_SMPL_DIR"), env("FAB_SMPL_DIR"),
                      env("FAB_HOME", FAB) / "body_models/smpl", soma() / "SMPL",
                      MODELS / "flowhmr/data/smpl", MOTION / "reftoanimesh/flowhmr/data/smpl",
                      MOTION / "4d-ify/models/smpl", FAB / "soma/SMPL",
                      MOTION / "nvidia-sim/SOMA-X/assets/SMPL"), NEUTRAL), None)


def find_hands():
    roots = (env("MANO_SOURCE"), env("FAB_INPUT_MANO_DIR"), soma() / "MANO",
             env("FAB_HOME", FAB) / "mano", MODELS / "flowhmr/assets/body_models/mano",
             MOTION / "4d-ify/models/mano", MOTION / "nvidia-sim/SOMA-X/assets/MANO")
    return next((p.parent for p in find(roots, HANDS[:1])
                 if all((p.parent / n).is_file() for n in HANDS)), None)


def link(source, dest):
    if not source.is_file():
        raise FileNotFoundError(f"{source} is missing; select fab in the model picker.")
    if dest.is_file() and dest.samefile(source):
        return
    if dest.exists() and not dest.is_symlink():
        raise FileExistsError(f"Refusing to replace your existing {dest} with {source}.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        staged = Path(tmp) / dest.name
        staged.symlink_to(source.resolve())
        staged.replace(dest)


def prepare():
    home = env("FAB_HOME", FAB)
    smpl = find_smpl()
    if smpl is None:
        raise SystemExit("SMPL_NEUTRAL.pkl not found in SOMA-X or FlowHMR. Select fab in the model picker.")
    link(smpl, home / "body_models/smpl/SMPL_NEUTRAL.pkl")
    if home.resolve() != FAB.resolve():
        for src in (FAB / "body_models").glob("*/*"):
            if src.name != "SMPL_NEUTRAL.pkl" and src.is_file():
                link(src, home / src.relative_to(FAB))
    src = find_hands()
    if src is None:
        return
    # FAH needs NumPy pickles. Convert beside its output, never over the SOMA/MANO originals.
    signature = hashlib.sha256()
    for name in HANDS:
        with (src / name).open("rb") as f:
            signature.update(hashlib.file_digest(f, "sha256").digest())
    root = home / "mano-cache"
    dest = root / signature.hexdigest()
    if not all((dest / name).is_file() for name in HANDS):
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            staged = Path(tmp) / "models"
            subprocess.run([STUDIO / ".hands/bin/python", STUDIO / ".runtime/fah/bench/convert_mano.py",
                            src, staged], check=True)
            staged.rename(dest)
    # Publish both hands together. A failed conversion leaves the last good pair intact.
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        alias = Path(tmp) / "current"
        alias.symlink_to(dest.resolve(), target_is_directory=True)
        alias.replace(root / "current")


if __name__ == "__main__":
    prepare()
