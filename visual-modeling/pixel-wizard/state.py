import json
import math
import os
import hashlib
from pathlib import Path
import re
import uuid

ROOT = Path(__file__).resolve().parent
MODELS = ROOT.parents[1] / "models/wevisone-vl-4b"
NATIVE = ROOT / "WeVisOne-VL"
WORK = ROOT / ".work"
RUNS = WORK / "runs"
PIN = {
    'model_repository': 'tencent/WeVisOne-VL-4B',
    'model_revision': '6b44a45873b4bbc7dca383922504430dc60eca23',
    'space_revision': 'ef77acac09937039d2d8cf318dc613090d8f7cba',
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

TASKS = {
    "referring": "Grounding", "detection": "Dense detection",
    "count_box": "Counting · boxes", "count_point": "Counting · points",
    "point": "Pointing", "ocr": "OCR · all text",
    "ocr_grounding": "OCR · locate text", "freeform": "Caption / VQA",
    "segmentation": "Segmentation", "matting": "Matting",
    "box_matting": "Matting · box", "visual_prompt": "Similar objects · boxes",
}
DEFAULTS = dict(task="referring", query="", prompt="", boxes=[],
                max_new_tokens=1024, temperature=0.0, top_p=0.95,
                top_k=50, repetition_penalty=1.05)
EXAMPLES = [
    dict(image="boys.jpg", task="referring", query="boys holding microphone", title="Singers"),
    dict(image="livingroom.jpg", task="detection", query="sofa\npillow\ncoffee table\nbook\nlamp\nvase", title="Living room"),
    dict(image="chess.jpg", task="count_box", query="white chess pieces", title="Chess"),
    dict(image="dogs.jpg", task="point", query="dogs", title="Dogs"),
    dict(image="coffee_sign.jpg", task="ocr", query="", title="Coffee sign"),
    dict(image="livingroom.jpg", task="freeform", query="describe the image with bounding boxes in detail", title="Grounded caption"),
    dict(image="parasol.jpg", task="segmentation", query="parasol with alternating rings of light green and dark green with flowers", title="Parasol"),
    dict(image="seabreeze.jpg", task="matting", query="person", title="Portrait"),
]


def atomic(path, value):
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def read(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def run_path(key):
    if not re.fullmatch(r"[a-f0-9]{32}", key):
        raise ValueError("Invalid run")
    path = RUNS / key
    if path.is_symlink():
        raise ValueError("Invalid run path")
    return path


def options(value):
    if not isinstance(value, dict):
        raise ValueError("Settings must be an object")
    out = DEFAULTS | value
    if out["task"] not in TASKS:
        raise ValueError("Choose a supported task")
    for key in ("query", "prompt"):
        if not isinstance(out[key], str) or len(out[key]) > 16000:
            raise ValueError(f"{key} must be text of at most 16000 characters")
    for key, low, high in (("max_new_tokens", 64, 8192), ("top_k", 1, 100)):
        if type(out[key]) is not int or not low <= out[key] <= high:
            raise ValueError(f"{key} must be an integer between {low} and {high}")
    for key, low, high in (("temperature", 0, 1.5), ("top_p", 0.01, 1), ("repetition_penalty", 1, 1.5)):
        if type(out[key]) not in (float, int) or not math.isfinite(out[key]) or not low <= out[key] <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
    boxes = out["boxes"]
    if not isinstance(boxes, list) or len(boxes) > 32:
        raise ValueError("Choose at most 32 reference boxes")
    for box in boxes:
        if (not isinstance(box, list) or len(box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in box)
                or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError("Boxes must be normalized xyxy rectangles")
    if out["task"] in ("box_matting", "visual_prompt") and not boxes:
        raise ValueError("Draw a reference box on the input image")
    if out["task"] == "box_matting" and len(boxes) != 1:
        raise ValueError("Box matting takes one reference box")
    if out["task"] not in ("ocr", "box_matting", "visual_prompt") and not (out["query"].strip() or out["prompt"].strip()):
        raise ValueError("Enter a query or an explicit prompt")
    return {key: out[key] for key in DEFAULTS}


def require_models():
    for name in ("config.json", "processor_config.json", "tokenizer_config.json", "chat_template.jinja"):
        if not (MODELS / name).is_file():
            raise ValueError("Select pixel-wizard in models/download_models.py first")
    for name in ("model.safetensors", "mask_decoder.pt", "tokenizer.json"):
        path = MODELS / name
        if not path.is_file() or path.stat().st_size < 1024:
            raise ValueError("Complete the pixel-wizard model download first")


def offline_env():
    return dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", WANDB_DISABLED="true",
                OTEL_SDK_DISABLED="true", TOKENIZERS_PARALLELISM="false",
                PYTHONUNBUFFERED="1", PIXEL_PARENT_PID=str(os.getpid()))
