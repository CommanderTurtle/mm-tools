import json
import os
import hashlib
from pathlib import Path
import re
import uuid

ROOT = Path(__file__).resolve().parent
MODELS = ROOT.parents[1] / "models/unlimited-ocr2"
NATIVE = ROOT / "Unlimited-OCR"
WORK = ROOT / ".work"
RUNS = WORK / "documents"
PIN = {
    'model_repository': 'baidu/Unlimited-OCR-2',
    'model_revision': '0480966938291445591f69f08a13b041c073a96a',
    'space_revision': '78c27d176793811cf8b5e74b62ceb04dd773c29a',
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
    mode="auto", instruction="", max_page_tokens=20000, max_length=131072, dpi=200
)


def atomic(path, value):
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def read(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def document(key):
    if not re.fullmatch(r"[a-f0-9]{32}", key):
        raise ValueError("Invalid document")
    path = RUNS / key
    if path.is_symlink():
        raise ValueError("Invalid document path")
    return path


def options(value):
    if not isinstance(value, dict):
        raise ValueError("Settings must be an object")
    out = DEFAULTS | value
    if out["mode"] not in ("auto", "gundam", "base"):
        raise ValueError("Choose Auto, Gundam or Base")
    for key, low, high in (
        ("max_page_tokens", 256, 20000),
        ("max_length", 4096, 786432),
        ("dpi", 72, 400),
    ):
        if type(out[key]) is not int or not low <= out[key] <= high:
            raise ValueError(f"{key} must be an integer between {low} and {high}")
    if not isinstance(out["instruction"], str) or len(out["instruction"]) > 16000:
        raise ValueError("Instruction must be at most 16000 characters")
    return {key: out[key] for key in DEFAULTS}


def require_models():
    for name in (
        "config.json",
        "modeling_unlimitedocr.py",
        "tokenizer.json",
        "model.safetensors.index.json",
    ):
        if not (MODELS / name).is_file():
            raise ValueError("Select unlimited-ocr2 in models/download_models.py first")
    index = read(MODELS / "model.safetensors.index.json")
    for name in set(index["weight_map"].values()):
        path = MODELS / name
        if not path.is_file() or path.stat().st_size < 1024:
            raise ValueError("Complete the unlimited-ocr2 model download first")


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
        OCR_PARENT_PID=str(os.getpid()),
    )
