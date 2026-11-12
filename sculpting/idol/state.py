"""Small durable jobs; the serving process never imports the model runtime."""

import json
import math
import re
import secrets
import uuid
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT.parent.parent / "models/idol"
WORK = ROOT / "workspace"
SOURCE = ROOT / "idol-sdk"

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
    mode="synthetic",
    steps=50,
    cfg=3.0,
    rescale_t=3.0,
    seed=42,
    randomize=False,
    num_gaussians=262144,
)
REQUIRED = (
    "weights/config.json",
    "weights/denoiser.safetensors",
    "weights/decoder.safetensors",
    "weights/decoder_gs.safetensors",
    "weights/birefnet.safetensors",
    "dinov3/config.json",
    "dinov3/model.safetensors",
    "flux2/vae/config.json",
    "flux2/vae/diffusion_pytorch_model.safetensors",
    "demo/examples/presets.json",
    "demo/examples/default-obj-backpack.ply",
)


def atomic(path, value):
    temporary = path.with_name("." + uuid.uuid4().hex + ".partial")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read(path, fallback=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


def job_path(key):
    if not re.fullmatch(r"[a-f0-9]{32}", key):
        raise ValueError("Invalid run ID")
    path = WORK / key
    if path.is_symlink():
        raise ValueError("Run must belong to this workspace")
    return path


def settings(value):
    if not isinstance(value, dict) or set(value) - DEFAULTS.keys():
        raise ValueError("Unknown generation settings")
    result = {**DEFAULTS, **value}
    if result["mode"] not in ("synthetic", "real"):
        raise ValueError("Choose synthetic or real photo")
    for key, low, high in (("steps", 10, 100), ("seed", 0, 2147483647)):
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}")
    for key, low, high in (("cfg", 0, 10), ("rescale_t", 1, 6)):
        if (
            type(result[key]) not in (float, int)
            or not math.isfinite(result[key])
            or not low <= result[key] <= high
        ):
            raise ValueError(f"{key} must be between {low} and {high}")
    if type(result["num_gaussians"]) is not int or result["num_gaussians"] not in (
        65536,
        131072,
        262144,
        524288,
        1048576,
    ):
        raise ValueError("Choose a listed Gaussian count")
    if type(result["randomize"]) is not bool:
        raise ValueError("Random seed must be true or false")
    if result["randomize"]:
        result["seed"] = secrets.randbelow(2147483648)
    return result


def presets():
    return read(MODELS / "demo/examples/presets.json", [])


def require_models():
    missing = [name for name in REQUIRED if not (MODELS / name).is_file()]
    for preset in presets():
        image = (MODELS / "demo" / preset["image"]).resolve()
        if (
            not image.is_relative_to((MODELS / "demo/examples").resolve())
            or not image.is_file()
        ):
            missing.append(preset["image"])
    if missing:
        raise ValueError(
            "Choose idol in ../../models/download_models.py first; missing "
            + ", ".join(missing)
        )
