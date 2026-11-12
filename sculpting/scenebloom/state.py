"""Durable stage boundaries; HTTP never imports the model stack."""

import json
import math
import hashlib
from pathlib import Path
import re
import secrets
import uuid

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MODELS = REPO / "models/scenebloom"
WORK = ROOT / "workspace"
NATIVE = ROOT / "scenebloom"

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

VLM = REPO / "models/qwen/text-encoder-vl-nvfp4/qwen3_vl_4b_nvfp4_full.safetensors"
STAGES = (
    "index",
    "retrieve",
    "zone",
    "proposals",
    "compile",
    "complete",
    "materials",
    "render",
    "export",
)
DEFAULTS = dict(
    random_seed=42,
    randomize=True,
    retrieval_candidate_k=12,
    retrieval_context_n=3,
    retrieval_temperature=0.04,
    retrieval_max_score_drop=0.08,
    small_object_max_candidates=20,
    small_object_retry_rounds=2,
    wall_object_max_per_zone=4,
    small_object_supports="desk,coffee_table,nightstand,side_table,dining_table,cabinet,tv_stand,chest_of_drawers,dressing_table",
    texture_top_k=5,
    texture_sample_top_k=3,
)


def read(path, fallback=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


def atomic(path, data):
    temporary = path.with_name("." + uuid.uuid4().hex + ".partial")
    try:
        temporary.write_text(
            json.dumps(data, indent=2, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def job_path(key):
    if not re.fullmatch(r"[a-f0-9]{32}", key):
        raise ValueError("Invalid run ID")
    path = WORK / key
    if path.is_symlink():
        raise ValueError("Run must belong to this workspace")
    return path


def settings(value):
    if not isinstance(value, dict) or set(value) - DEFAULTS.keys():
        raise ValueError("Unknown settings")
    result = {**DEFAULTS, **value}
    for key, maximum in (
        ("random_seed", 2147483647),
        ("retrieval_candidate_k", 100),
        ("retrieval_context_n", 12),
        ("small_object_max_candidates", 100),
        ("small_object_retry_rounds", 10),
        ("wall_object_max_per_zone", 30),
        ("texture_top_k", 5),
        ("texture_sample_top_k", 5),
    ):
        minimum = 0 if key in ("random_seed", "small_object_retry_rounds") else 1
        if type(result[key]) is not int or not minimum <= result[key] <= maximum:
            raise ValueError(f"{key} must be an integer from {minimum} to {maximum}")
    for key in ("retrieval_temperature", "retrieval_max_score_drop"):
        v = result[key]
        if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 10:
            raise ValueError(f"Invalid {key}")
    if result["retrieval_temperature"] == 0:
        raise ValueError("Retrieval temperature must be positive")
    if type(result["randomize"]) is not bool:
        raise ValueError("Randomize must be true or false")
    support = result["small_object_supports"]
    if (
        not isinstance(support, str)
        or len(support) > 1000
        or not re.fullmatch(r"[a-z_, ]+", support)
    ):
        raise ValueError("Supports must be comma-separated native category names")
    if result["randomize"]:
        result["random_seed"] = secrets.randbelow(2147483648)
    return result


def require_models():
    if (
        read(MODELS / "assets-ready.json", {}).get("source")
        != read(REPO / "models/manifests/scenebloom.json", {}).get("source")
        or not VLM.is_file()
    ):
        raise ValueError("Run the model downloader's scenebloom bundle first")


NATIVE_SOURCES = ("scenebloom", "ovmono3d", "dinov2", "pytorch3d", "detectron2",
                  "groundingdino", "depthpro", "sam")


def source_paths():
    """Each source tree the studio runs is committed at this lane root."""
    return [
        (
            ROOT / name,
            tree_stamp(ROOT / name),
        )
        for name in NATIVE_SOURCES
    ]
