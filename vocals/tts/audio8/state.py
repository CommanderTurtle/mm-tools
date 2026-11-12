import json
import math
import os
import re
import uuid
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT.parents[2] / "models/audio8-realistic-air"
WORK = ROOT / ".work"
RUNS = WORK / "runs"
PIN = {
    'model': 'Edge0/Audio8-TTS-Realistic-Air',
    'model_revision': '23c305cc013fde378edb459f538317043c993a74',
    'space_revision': '0e7d7ed6626f49e7ac5898365cb116321b610a5d',
    'precision': 'BF16 CUDA; native 44.1 kHz codec',
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
    text="",
    voice="example10",
    transcript="",
    paragraphs=False,
    seed=42,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    frames=1024,
    retry_frames=2000,
    batch_size=1,
    sample=True,
    codes=False,
)
VOICES = {
    "example10": dict(
        name="Example 10 · English",
        file="sample-10-reference.wav",
        transcript="What a season! Amy figured out what I was up to, and now she's gonna use my DNA to make a multiverse waypoint.",
    ),
    "example18": dict(
        name="Example 18 · 中文",
        file="sample-18-reference.wav",
        transcript="又顺着床沿摸到了台灯的开关，按了下去，屋子里仍然漆黑一片。",
    ),
}
LANGUAGES = "Arabic,Cantonese,Chinese,Dutch,English,Finnish,French,German,Hindi,Indonesian,Italian,Japanese,Korean,Malay,Norwegian,Polish,Portuguese,Russian,Spanish,Swedish,Thai,Turkish,Vietnamese".split(
    ","
)


def atomic(path, data):
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def read(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def run_path(key):
    if not re.fullmatch(r"[a-f0-9]{32}", key):
        raise ValueError("Invalid take")
    path = RUNS / key
    if path.is_symlink():
        raise ValueError("Invalid take path")
    return path


def options(value):
    if not isinstance(value, dict):
        raise ValueError("Settings must be an object")
    cfg = DEFAULTS | value
    for key in ("text", "transcript"):
        if not isinstance(cfg[key], str):
            raise ValueError(f"{key} must be text")
    if not cfg["text"].strip():
        raise ValueError("Enter text to speak")
    for key in ("sample", "codes", "paragraphs"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    parts = (
        [s.strip() for s in re.split(r"\n\s*\n", cfg["text"]) if s.strip()]
        if cfg["paragraphs"]
        else [cfg["text"].strip()]
    )
    if not 1 <= len(parts) <= 32 or any(len(p) > 600 for p in parts):
        raise ValueError("Use up to 32 paragraphs, each at most 600 characters")
    if not isinstance(cfg["voice"], str) or cfg["voice"] not in {
        *VOICES,
        "custom",
        "none",
    }:
        raise ValueError("Choose a reference voice")
    if len(cfg["transcript"]) > 600:
        raise ValueError("Reference transcript must be at most 600 characters")
    if cfg["voice"] == "custom" and not cfg["transcript"].strip():
        raise ValueError("Enter the exact reference transcript")
    for key, low, high in (
        ("seed", 0, 2**32 - 1),
        ("frames", 32, 2000),
        ("retry_frames", 32, 2000),
        ("top_k", 0, 4096),
        ("batch_size", 1, 8),
    ):
        if type(cfg[key]) is not int or not low <= cfg[key] <= high:
            raise ValueError(f"{key} must be an integer between {low} and {high}")
    for key, low, high in (("temperature", 0.05, 2), ("top_p", 0.01, 1)):
        if (
            type(cfg[key]) not in (int, float)
            or not math.isfinite(cfg[key])
            or not low <= cfg[key] <= high
        ):
            raise ValueError(f"{key} must be between {low} and {high}")
    if cfg["retry_frames"] < cfg["frames"]:
        raise ValueError("Retry frame budget must be at least the initial budget")
    cfg["text"] = cfg["text"].strip()
    return {k: cfg[k] for k in DEFAULTS}, parts


def require_models():
    names = (
        "model.safetensors",
        "codec.pth",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "processor_config.json",
        "processing_arktts.py",
        "configuration_arktts.py",
        "modeling_arktts.py",
        "modeling_arktts_codec.py",
    )
    for name in names:
        file = MODELS / name
        if not file.is_file() or file.stat().st_size < 20:
            raise ValueError(
                "Download the audio8 bundle in models/download_models.py first"
            )
        with file.open("rb") as stream:
            if stream.read(45).startswith(b"version https://git-lfs.github.com/spec"):
                raise ValueError(f"Incomplete model download: {name}")


def offline_env():
    return dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        DO_NOT_TRACK="1",
        WANDB_DISABLED="true",
        OTEL_SDK_DISABLED="true",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
        AUDIO8_PARENT_PID=str(os.getpid()),
    )
