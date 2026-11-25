import json
import math
import os
import re
import uuid
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT.parents[2] / "models/ming3"
NATIVE = ROOT / "Ming-3.0-tts"
WORK = ROOT / ".work"
RUNS = WORK / "runs"
PIN = {
    'models': {'0.5B': {'repo': 'inclusionAI/Ming-3.0-tts-0.5B', 'revision': '7247f4c94a5b2e987404ed28a5cfec41b1a4ca61'}, 'tiny': {'repo': 'inclusionAI/Ming-3.0-tts-tiny', 'revision': '4715ef5a2f02901f75687d538b6920fd0dd11e6b'}},
    'voices': {'repo': 'hugging-apps/ming-3-tts-05b', 'revision': 'd4ecae7bc5c59fb0b8e80461e503e53a8a6ba6bd'},
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
    "seed": "Voice clone",
    "multilingual": "Multilingual",
    "voice_design": "Voice design",
    "freeform": "Director",
    "ip": "Built-in voice",
    "emotion": "Emotion",
    "basic": "Attributes",
    "dialect": "Dialect",
    "paralinguistic": "Vocal tags",
    "podcast": "Podcast",
}
LANGUAGES = ["English", "Chinese", "Japanese", "Korean", "German", "French", "Spanish"]
PRESETS = {
    "lingguang": {
        "name": "灵小柔",
        "file": "lingguang.wav",
        "transcript": "诶，你今天有空吗？我们能不能聊聊这件事啊？",
    },
    "db30": {"name": "灵楷", "file": "db30.wav", "transcript": "感谢你的认可。"},
    "emotion": {
        "name": "Emotion reference",
        "file": "emotion_prompt.wav",
        "transcript": "",
    },
    "yue": {"name": "Cantonese reference", "file": "yue_prompt.wav", "transcript": ""},
}
DEFAULTS = dict(
    model="0.5B",
    task="ip",
    text="",
    language="English",
    ip="US_Emma",
    instruction="",
    style="",
    emotion="高兴",
    dialect="cantonese",
    speed="中速",
    pitch="中",
    volume="中",
    speakers=[dict(voice="lingguang", transcript=PRESETS["lingguang"]["transcript"])],
    seed=None,
    cfg=2.0,
    sigma=0.25,
    temperature=0.0,
    flow_steps=10,
    max_length=24,
    max_decode_steps=None,
    min_decode_steps=None,
    stop_threshold=None,
    min_frames=None,
    max_frames=None,
    normalization=True,
    trim=True,
    raw=False,
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


def reference_mode(cfg):
    if cfg["task"] in ("ip", "voice_design"):
        return "none"
    return "optional" if cfg["task"] == "freeform" else "required"


def transcript_mode(cfg):
    return cfg["task"] in ("seed", "multilingual", "paralinguistic", "podcast")


def options(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError("Unknown settings")
    cfg = DEFAULTS | value
    for key in (
        "model",
        "task",
        "text",
        "language",
        "ip",
        "instruction",
        "style",
        "emotion",
        "dialect",
        "speed",
        "pitch",
        "volume",
    ):
        if not isinstance(cfg[key], str):
            raise ValueError(f"{key} must be text")
    if cfg["model"] not in PIN["models"] or cfg["task"] not in TASKS:
        raise ValueError("Choose a model and task")
    if cfg["task"] == "podcast" and cfg["model"] != "tiny":
        raise ValueError("Podcast requires tiny")
    if cfg["task"] != "podcast":
        cfg["min_frames"] = cfg["max_frames"] = None
    else:
        cfg["min_decode_steps"] = cfg["max_decode_steps"] = None
    if cfg["model"] == "0.5B":
        cfg["min_decode_steps"] = cfg["stop_threshold"] = None
        cfg["flow_steps"] = DEFAULTS["flow_steps"]
    if not cfg["text"].strip() or len(cfg["text"]) > 24000:
        raise ValueError("Enter between 1 and 24000 characters")
    for key in ("normalization", "trim", "raw"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    for key, low, high in (
        ("seed", -1, 2**32 - 1),
        ("flow_steps", 1, 100),
        ("max_length", 1, 500),
        ("min_decode_steps", 1, 4000),
        ("max_decode_steps", 1, 4000),
        ("min_frames", 8, 600),
        ("max_frames", 8, 600),
    ):
        if cfg[key] is None and key not in ("flow_steps", "max_length"):
            continue
        if type(cfg[key]) is not int or not low <= cfg[key] <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}")
    for key, low, high in (
        ("cfg", 0, 10),
        ("sigma", 0, 2),
        ("temperature", 0, 2),
        ("stop_threshold", 0, 1),
    ):
        if cfg[key] is None and key == "stop_threshold":
            continue
        if (
            type(cfg[key]) not in (int, float)
            or not math.isfinite(cfg[key])
            or not low <= cfg[key] <= high
        ):
            raise ValueError(f"{key} must be from {low} to {high}")
    for lo, hi in (
        ("min_frames", "max_frames"),
        ("min_decode_steps", "max_decode_steps"),
    ):
        if cfg[lo] is not None and cfg[hi] is not None and cfg[lo] > cfg[hi]:
            raise ValueError(f"{lo} must be at most {hi}")
    if (cfg["min_frames"] is None) != (cfg["max_frames"] is None):
        raise ValueError("Set both podcast frame bounds, or leave both automatic")
    task = cfg["task"]
    if task in ("seed", "paralinguistic", "multilingual"):
        allowed = LANGUAGES[2:] if task == "multilingual" else LANGUAGES[:2]
        if cfg["language"] not in allowed:
            raise ValueError("Choose a language supported by this task")
    if task == "emotion" and cfg["emotion"] not in (
        "高兴",
        "悲伤",
        "愤怒",
        "恐惧",
        "厌恶",
        "惊讶",
        "中性",
    ):
        raise ValueError("Choose an emotion")
    if task == "dialect" and cfg["dialect"] not in ("cantonese", "chuanyu dialect"):
        raise ValueError("Choose a dialect")
    if task == "basic" and (
        cfg["speed"] not in ("慢速", "中速", "快速")
        or cfg["pitch"] not in ("低", "中", "高")
        or cfg["volume"] not in ("低", "中", "高")
    ):
        raise ValueError("Choose speed, pitch and volume")
    for mode, field in (
        ("freeform", "instruction"),
        ("voice_design", "style"),
        ("ip", "ip"),
    ):
        if task == mode and (not cfg[field].strip() or len(cfg[field]) > 4000):
            raise ValueError(f"Enter {field} (at most 4000 characters)")
    speakers = cfg["speakers"]
    if not isinstance(speakers, list) or len(speakers) > 16:
        raise ValueError("Use up to 16 reference speakers")
    mode = reference_mode(cfg)
    if mode == "none":
        speakers = []
    elif task == "podcast":
        if len(speakers) < 2:
            raise ValueError("Podcast needs at least two reference speakers")
    elif len(speakers) != 1 and not (mode == "optional" and len(speakers) == 0):
        raise ValueError("Choose one reference speaker")
    clean = []
    for speaker in speakers:
        if (
            not isinstance(speaker, dict)
            or not isinstance(speaker.get("voice"), str)
            or not isinstance(speaker.get("transcript", ""), str)
        ):
            raise ValueError("Invalid reference speaker")
        voice = speaker["voice"]
        if voice == "none" and mode == "optional":
            continue
        if voice not in (*PRESETS, "custom"):
            raise ValueError("Choose a reference recording")
        transcript = (
            PRESETS[voice]["transcript"]
            if voice in PRESETS
            else speaker.get("transcript", "").strip()
        )
        if cfg["model"] == "tiny" and transcript_mode(cfg) and not transcript:
            raise ValueError(
                "This task needs the exact reference transcript; choose a transcribed preset or upload"
            )
        if len(transcript) > 4000:
            raise ValueError("Reference transcript is too long")
        clean.append(dict(voice=voice, transcript=transcript))
    cfg["speakers"] = clean
    if task == "podcast":
        seen = set()
        for line in cfg["text"].splitlines():
            if not line.strip():
                continue
            match = re.fullmatch(r"\s*speaker_(\d+):\s*(.+)", line)
            if not match or not match[2].strip():
                raise ValueError("Each podcast line must read speaker_N: speech")
            seen.add(int(match[1]))
        if seen != set(range(1, len(clean) + 1)):
            raise ValueError(
                "Use every reference speaker, numbered 1 through the speaker count"
            )
    return cfg


def require_models(model):
    path = MODELS / model
    names = ["config.json", "tokenizer.json", "tokenizer_config.json", "campplus.onnx"]
    if model == "tiny":
        index = read(path / "model.safetensors.index.json")
        if not index or not isinstance(index.get("weight_map"), dict):
            raise ValueError("Select the ming3 tiny tier (2) in the model picker first")
        names += sorted(set(index["weight_map"].values()))
    else:
        names.append("model.safetensors")
    for name in names:
        file = path / name
        if (
            not file.resolve().is_relative_to(path.resolve())
            or not file.is_file()
            or file.stat().st_size < 20
        ):
            raise ValueError(f"Complete ming3-{model.lower()} download: {name}")
        with file.open("rb") as stream:
            if stream.read(45).startswith(b"version https://git-lfs.github.com/spec"):
                raise ValueError(f"Incomplete model download: {name}")
    return path


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
        MING3_PARENT_PID=str(os.getpid()),
    )
