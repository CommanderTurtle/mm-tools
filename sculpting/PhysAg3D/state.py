"""Durable case/attempt records; model imports belong to disposable workers."""

import json
import math
import os
import hashlib
from pathlib import Path
import re
import time
import uuid

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "workspace"
SOURCE = ROOT / "PhysAg3D"
TRELLIS = ROOT.parent / "trellis"
PIN = {
    'attention': 'sdpa',
    'sparse_attention': 'sdpa',
    'sparse_convolution': 'flex_gemm',
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

DEFAULTS = dict(
    mode="local",
    model="",
    device=0,
    seed=42,
    resolution=1024,
    texture=2048,
    steps=24,
    min_free_gb=28,
    minutes=60,
    blender="blender",
    feedback="",
    resume=True,
)


def atomic(path, value):
    path = Path(path)
    temp = path.with_name("." + uuid.uuid4().hex + ".partial")
    try:
        temp.write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def read(path, fallback=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


def identity(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("Invalid case or attempt ID")
    return value


def inside(root, name):
    root = root.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("File must remain inside its case")
    return path


def case_path(key):
    path = WORK / identity(key)
    if path.is_symlink():
        raise ValueError("Case must belong to this workspace")
    return path


def attempt_path(case, key):
    return inside(case / "runs", identity(key))


def settings(value):
    if not isinstance(value, dict) or set(value) - DEFAULTS.keys():
        raise ValueError("Unknown stage settings")
    result = {**DEFAULTS, **value}
    for key, lo, hi in (
        ("device", 0, 31),
        ("seed", 0, 2147483647),
        ("steps", 1, 100),
        ("minutes", 1, 240),
    ):
        if type(result[key]) is not int or not lo <= result[key] <= hi:
            raise ValueError(f"{key} must be an integer from {lo} to {hi}")
    if result["resolution"] not in (512, 1024) or result["texture"] not in (
        1024,
        2048,
        4096,
    ):
        raise ValueError("Choose an offered generation/texture resolution")
    if (
        type(result["min_free_gb"]) not in (int, float)
        or not math.isfinite(result["min_free_gb"])
        or not 0 <= result["min_free_gb"] <= 192
    ):
        raise ValueError("Free VRAM guard must be 0–192 GiB")
    if result["mode"] not in ("local", "nvidia") or type(result["resume"]) is not bool:
        raise ValueError("Choose a provider mode and resume setting")
    for key, maximum in (("model", 512), ("blender", 4096), ("feedback", 32000)):
        if (
            not isinstance(result[key], str)
            or len(result[key]) > maximum
            or "\x00" in result[key]
        ):
            raise ValueError(f"Invalid {key}")
    return result


def attempts(case):
    rows = [
        read(p / "run.json")
        for p in (case / "runs").glob("*")
        if p.is_dir() and not p.is_symlink()
    ]
    return sorted((r for r in rows if r), key=lambda r: r["created"])


def latest(case, phase, successful=True):
    return next(
        (
            r
            for r in reversed(attempts(case))
            if r["phase"] == phase
            and (not successful or r["status"] in ("complete", "partial"))
        ),
        None,
    )


def update(path, **fields):
    atomic(
        path / "run.json",
        {**read(path / "run.json", {}), **fields, "updated": time.time()},
    )


def native_python():
    setting = read(ROOT / ".native-python.json", {})
    path = (
        Path(
            os.environ.get(
                "PHYS_TRELLIS_PYTHON",
                setting.get("path", str(ROOT / ".venv-native/bin/python")),
            )
        )
        .expanduser()
        .absolute()
    )
    return path.parent.resolve() / path.name
