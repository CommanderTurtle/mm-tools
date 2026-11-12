"""Small on-disk job contract shared by the HTTP process and disposable workers."""

import json
import hashlib
import math
import os
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "flowhmr"

# Pinned upstream trees committed at this lane root; setup links each one into
# flowhmr/local_app/, the path the runtime resolves them at.
BACKEND_TREES = (
    "KimodoMotionCorrection",
    "RGBMocapCorrection",
    "PerpetualHumanoidControl",
    "YOLOX",
    "sam-3d-body",
    "vggt-omega",
    "smplsim",
    "smplx",
    "chumpy",
)
WORK = ROOT / "workspace"
PIN = {
    'cache': 'flowhmr',
}
# Upstream revisions the vendored trees were cut from, as recorded in the
# expansion-era source.json; the physics receipt contract binds these plus
# requirements, so it stays stable while models or data land in the trees.
PHYSICS_PINS = dict(
    flowhmr="df15cadf56e738bf80c42c93d5e73687ff1002d2",
    dinov3="6876159a11b4df116f30f667f8c9888617df0751",
    smplsim="b5c08720503ad5fff64050c4d289c42d947fcf8d",
    smplx="a5b8e4ac14f79f3f33fd2cf2a16e6f507146b813",
    chumpy="580566eafc9ac68b2614b64d6f7aaa84eebb70da",
)

def tree_stamp(tree: Path) -> str:
    """Identity of the source this lane runs, derived from the files on disk."""
    generated = {"__pycache__", ".venv", "node_modules", "workspace", ".work",
                 "runs", ".git-archives"}
    scratch = {".env", ".physics-ready", ".snapshot-members", "MUJOCO_LOG.TXT"}
    digest = hashlib.sha256()
    for path in sorted(p for p in tree.rglob("*")
                       if p.is_file() and p.suffix not in (".pyc", ".lst")
                       and p.name not in scratch
                       and not generated & set(p.relative_to(tree).parts)):
        digest.update(path.relative_to(tree).as_posix().encode())
        digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()[:40]


LANE_STAMP = tree_stamp(ROOT)

DEFAULTS = dict(
    seed=0,
    cfg=1.0,
    steps=20,
    max_frames=900,
    camera_stride=1,
    static_camera=False,
    sam_chunk=16,
    manual_person=True,
    postprocess=False,
    attention="original",
    query_chunk=256,
    sam_compile=False,
    device=0,
    physics=False,
    reselect=False,
)


def atomic(path, value):
    path = Path(path)
    pending = path.with_name("." + uuid.uuid4().hex + ".partial")
    try:
        pending.write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def read(path, fallback=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


def job_path(key):
    if (
        not isinstance(key, str)
        or len(key) != 32
        or any(c not in "0123456789abcdef" for c in key)
    ):
        raise ValueError("Invalid job ID")
    path = WORK / key
    if path.is_symlink():
        raise ValueError("Job must belong to this workspace")
    return path


def settings(data):
    if not isinstance(data, dict) or set(data) - DEFAULTS.keys():
        raise ValueError("Unknown reconstruction settings")
    result = {**DEFAULTS, **data}
    for name, lo, hi in (
        ("seed", 0, 2147483647),
        ("steps", 1, 100),
        ("max_frames", 0, 10000),
        ("camera_stride", 1, 120),
        ("sam_chunk", 1, 128),
        ("query_chunk", 1, 8192),
        ("device", 0, 31),
    ):
        value = result[name]
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    for name in (
        "static_camera",
        "manual_person",
        "postprocess",
        "sam_compile",
        "physics",
        "reselect",
    ):
        if type(result[name]) is not bool:
            raise ValueError(f"{name} must be true or false")
    if result["attention"] not in ("original", "query_batched"):
        raise ValueError("Choose original or query_batched attention")
    if (
        type(result["cfg"]) not in (int, float)
        or not math.isfinite(result["cfg"])
        or not 0 <= result["cfg"] <= 10
    ):
        raise ValueError("CFG must be between 0 and 10")
    return result


def physics_python():
    path = (
        Path(os.environ.get("FLOWHMR_PHC_PYTHON", str(ROOT / ".venv-phc/bin/python")))
        .expanduser()
        .absolute()
    )
    # Resolve the environment directory, not its python symlink into the base install.
    return path.parent.resolve() / path.name


def physics_contract():
    return hashlib.sha256(
        json.dumps(PHYSICS_PINS, sort_keys=True).encode()
        + (ROOT / "requirements-physics.txt").read_bytes()
    ).hexdigest()


def physics_ready():
    receipt = read(ROOT / ".physics-ready", {})
    files = (
        "data/smpl/SMPL_NEUTRAL.pkl",
        "data/smpl/SMPL_MALE.pkl",
        "data/smpl/SMPL_FEMALE.pkl",
        "output/HumanoidIm/phc_comp_3/Humanoid.pth",
        "output/HumanoidIm/phc_3/Humanoid.pth",
        "assets/example/amass_isaac_gender_betas.pkl",
        "assets/example/amass_isaac_gender_betas_unique.pkl",
    )
    return (
        physics_python().is_file()
        and all((SOURCE / f).is_file() for f in files)
        and isinstance(receipt, dict)
        and receipt.get("interpreter") == str(physics_python())
        and receipt.get("contract") == physics_contract()
        and receipt.get("python") == [3, 8]
    )
