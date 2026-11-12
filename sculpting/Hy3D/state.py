"""Disk is the stage boundary; the HTTP process never imports GPU runtimes."""

import json
import re
import secrets
import uuid
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT.parents[1] / "models/hy3d"
WORK = ROOT / "workspace"
PIN = {
    'model': 'd502e938cbedc6e87aba8884614c5ca6d177a4b6',
    'dinov2': '47b73eefe95e8d44ec3623f8890bd894b6ea2d6c',
    'clip': '32bd64288804d66eefd0ccbe215aa642df71cc41',
    'coarse_model': 'f90a0f7df7d5e6f71109cf333f6a95a0ae3194a6',
}

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

SOURCES = ROOT
COARSE = "hunyuan3d-dit-v2-mini-turbo"
DEFAULTS = dict(
    route="image",
    resolution=1024,
    steps=25,
    seed=0,
    randomize=False,
    offload=False,
    remove_background=False,
)
REQUIRED = (
    "weights/model_index.json",
    "weights/scheduler/scheduler_config.json",
    "weights/vae/config.json",
    "weights/vae/model.safetensors",
    "weights/transformer_512/config.json",
    "weights/transformer_512/model.safetensors",
    "weights/transformer_1024/config.json",
    "weights/transformer_1024/model.safetensors",
    "dinov2/config.json",
    "dinov2/model.safetensors",
    "clip/config.json",
    "clip/model.safetensors",
)
MESH_TYPES = (".glb", ".ply", ".obj", ".stl")


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
    if result["route"] not in ("image", "mesh"):
        raise ValueError("Choose image generation or mesh refinement")
    if type(result["resolution"]) is not int or result["resolution"] not in (
        512,
        1024,
        1536,
    ):
        raise ValueError("Choose a listed extraction grid")
    for key, low, high in (("steps", 1, 200), ("seed", 0, 2147483647)):
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}")
    for key in ("randomize", "offload", "remove_background"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be true or false")
    if result["randomize"]:
        result["seed"] = secrets.randbelow(2147483648)
    return result


def coarse_ready():
    return all(
        (MODELS / "coarse" / COARSE / name).is_file()
        for name in ("config.yaml", "model.fp16.safetensors")
    )


def require_models(config=None):
    required = list(REQUIRED)
    if config and config["route"] == "image":
        required += [
            f"coarse/{COARSE}/{name}"
            for name in ("config.yaml", "model.fp16.safetensors")
        ]
    if config and config["remove_background"]:
        required.append("rembg/u2net.onnx")
    missing = [name for name in required if not (MODELS / name).is_file()]
    if missing:
        raise ValueError(
            "Select hy3d (and hy3d-coarse for image generation) in "
            "../../models/download_models.py; missing " + ", ".join(missing)
        )


def source_paths():
    return (
        (SOURCES / "hi3d", LANE_STAMP),
        (SOURCES / "hunyuan", LANE_STAMP),
    )
