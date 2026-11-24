# SPDX-License-Identifier: AGPL-3.0-only
"""Resolve FAB's native inputs from existing motion studios; never edit a donor model."""
import hashlib
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

MODELS = Path(__file__).resolve().parent
MOTION = MODELS.parent / "motion-sim"
STUDIO = MOTION / "FAB"
FAB = MODELS / "fab"
NEUTRAL = ("SMPL_NEUTRAL.pkl", "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
           "basicmodel_n_lbs_10_207_0_v1.1.0.pkl", "SMPL_NEUTRAL.npz")
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
            for p in sorted(root.rglob("*")):
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


def g1_target():
    return env("FAB_G1", env("FAB_HOME", FAB) / "g1")


def g1_ready(root):
    if not all((root / name).is_file() for name in
               ("g1_29dof.urdf", "g1_gate_robot.xml", "scene_g1_gate.xml")):
        return False
    try:
        for name in ("g1_29dof.xml", "g1_mjlab.xml"):
            xml = ET.parse(root / name)
            folder = root / xml.find("compiler").attrib.get("meshdir", "")
            meshes = xml.findall("./asset/mesh[@file]")
            if not meshes or not all((folder / m.attrib["file"]).is_file() for m in meshes):
                return False
    except (OSError, ET.ParseError, AttributeError):
        return False
    return True


def prepare_g1():
    root = g1_target()
    if g1_ready(root):
        return
    sources = FAB / "g1-sources"
    if not (sources / "unitree/g1_29dof.xml").is_file():
        raise SystemExit("Select fab in the model picker to prepare the G1 meshes.")
    # An explicit custom rig is read-only; never rebuild over the user's configuration.
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"Incomplete G1 at {root}. Point FAB_G1 at a complete rig or an empty directory.")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as tmp:
        stage = Path(tmp) / "g1"
        subprocess.run([sys.executable, STUDIO / "fab" / "scripts" / "build_g1.py",
                        "--src", sources / "unitree", "--out", stage], check=True)
        link(sources / "mjlab/g1.xml", stage / "g1_mjlab.xml")
        (stage / "assets").symlink_to((sources / "mjlab/assets").resolve(), target_is_directory=True)
        if not g1_ready(stage):
            raise SystemExit("Incomplete G1 build: reselect fab in the picker.")
        stage.replace(root)


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
    prepare_g1()
    home = env("FAB_HOME", FAB)
    smpl = find_smpl()
    if smpl is None:
        raise SystemExit("SMPL_NEUTRAL.pkl not found in SOMA-X or FlowHMR. Select fab in the model picker.")
    if smpl.suffix.lower()==".npz":
        # SOMA also accepts NumPy archives. FAB's native loader specifically expects a pickle.
        # Put the same arrays in a private cache; do not rename a ZIP file to .pkl.
        import numpy as np
        with smpl.open("rb") as f: checksum=hashlib.file_digest(f,"sha256").hexdigest()
        cached=home/"body-cache"/checksum/"SMPL_NEUTRAL.pkl"
        if not cached.is_file():
            cached.parent.mkdir(parents=True,exist_ok=True)
            with np.load(smpl,allow_pickle=True,encoding="latin1") as src, tempfile.TemporaryDirectory(dir=cached.parent) as tmp:
                keys=("v_template","shapedirs","posedirs","J_regressor","weights","kintree_table","f")
                arrays={k:src["faces" if k=="f" and "f" not in src else k] for k in keys}
                arrays={k:v.item() if v.dtype.hasobject and v.shape==() else v for k,v in arrays.items()}
                if arrays["v_template"].shape!=(6890,3) or arrays["weights"].shape!=(6890,24):
                    raise ValueError(f"{smpl} is not a neutral SMPL body.")
                with (Path(tmp)/"model").open("wb") as f: pickle.dump(arrays,f)
                (Path(tmp)/"model").replace(cached)
        smpl=cached
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
            subprocess.run([STUDIO / ".hands/bin/python", STUDIO / "fah" / "bench" / "convert_mano.py",
                            src, staged], check=True)
            staged.rename(dest)
    # Publish both hands together. A failed conversion leaves the last good pair intact.
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        alias = Path(tmp) / "current"
        alias.symlink_to(dest.resolve(), target_is_directory=True)
        alias.replace(root / "current")


if __name__ == "__main__":
    prepare()
