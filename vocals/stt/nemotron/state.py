import json
import math
import os
import re
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT.parents[2] / "models/nemotron-transcribe"
NATIVE = ROOT / "Speech"
WORK = ROOT / ".work"
RUNS = WORK / "runs"
PIN = {
    "repository": "https://github.com/NVIDIA-NeMo/Speech.git",
    "revision": "3d381ffad4a513885c97a48a577013ed805d88da",
    "model": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4",
    "model_revision": "66c6adc4020179567074d0399cda2a4eeb2bb347",
    "reference": "https://huggingface.co/spaces/nvidia/nemotron-transcribe",
    "reference_revision": "51f2b415708906b7d3782d5ce0187c6b3a04f00f",
    "vllm": "0.32.0",
    "demucs": "4.0.1",
    "stem_sha256": "8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4"
}
TASKS = {
    "speakers": "Podcast / speakers", "verbatim": "Verbatim", "clean": "Readable",
    "subtitles": "Movie subtitles", "karaoke": "Karaoke", "context": "Context terms",
    "translate": "Translate", "language": "Identify language", "rich": "Events / style",
    "nearfield": "Nearest speaker", "ask": "Ask audio",
}
PROMPTS = {
    "verbatim": "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only.",
    "clean": "Audio transcription. Language: as spoken. Wording: omit fillers and accidental repetitions; keep the remaining spoken wording unchanged. Formatting: punctuation and capitalization. Output: transcript only.",
    "speakers": "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. Write numbers in their standard written form. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription.",
    "context": "Transcribe with punctuation and capitalization. Context terms: ",
    "translate": "Please translate the given speech to English.",
    "language": "Identify the spoken language. Respond with the language name only.",
    "rich": "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. Write numbers in their standard written form. Use <ev:*> and <sty:*> special tokens to recognize all non-speech verbatims and speaking style.",
    "nearfield": "Transcribe this audio. Include all spoken words and verbal fillers. Write numbers as spoken. No punctuation or capitalization. Transcribe only the speaker nearest to the microphone. Do not transcribe any other voices.",
    "ask": "What is happening in this audio?",
    "karaoke": "Transcribe the sung lyrics with punctuation and capitalization. Output the lyrics only.",
}
PROMPTS["subtitles"] = PROMPTS["speakers"]
DEFAULTS = dict(task="speakers", prompt=PROMPTS["speakers"], system="", reasoning=False,
                timestamps=True, max_tokens=16384, context=65536, memory=0.85,
                mtp=True, continue_chat=False, source_url="")
EXPORTS = {"transcript.txt", "response.json", "words.json", "transcript.srt",
           "transcript.vtt", "lyrics.lrc", "conversation.json", "result.zip"}
MEDIA = {"audio.wav", "preview.mp4", "vocals.wav", "music.wav", "original.wav"}


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
    for key in ("prompt", "system", "source_url"):
        if not isinstance(out[key], str) or len(out[key]) > 32000:
            raise ValueError(f"{key} must be text of at most 32000 characters")
    if not out["prompt"].strip():
        raise ValueError("Enter an instruction")
    for key in ("reasoning", "timestamps", "mtp", "continue_chat"):
        if type(out[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    for key, low, high in (("max_tokens", 128, 32768), ("context", 4096, 65536)):
        if type(out[key]) is not int or not low <= out[key] <= high:
            raise ValueError(f"{key} must be an integer between {low} and {high}")
    if out["max_tokens"] >= out["context"]:
        raise ValueError("Output budget must leave context space for audio and instructions")
    if (type(out["memory"]) not in (int, float) or not math.isfinite(out["memory"])
            or not 0.1 <= out["memory"] <= 0.95):
        raise ValueError("GPU memory fraction must be between 0.1 and 0.95")
    if out["task"] in ("translate", "language", "ask"):
        out["timestamps"] = False
    if out["task"] != "ask":
        out["reasoning"] = out["continue_chat"] = False
    if out["reasoning"] and out["max_tokens"] < 18432:
        raise ValueError("Reasoning needs an output budget of at least 18432 tokens")
    return {key: out[key] for key in DEFAULTS}


def require_models():
    root = MODELS / "checkpoint"
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "llm_backbone/config.json", "model.safetensors.index.json"):
        if not (root / name).is_file():
            raise ValueError("Select nemotron in models/download_models.py first")
    index = read(root / "model.safetensors.index.json")
    for name in set(index["weight_map"].values()) | {
            "perception.safetensors", "model-ctc-bf16.safetensors", "model-mtp-fp8.safetensors"}:
        path = root / name
        if not path.is_file() or path.stat().st_size < 1024:
            raise ValueError("Complete the nemotron model download first")


def offline_env():
    return dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", WANDB_DISABLED="true",
                OTEL_SDK_DISABLED="true", VLLM_NO_USAGE_STATS="1", TOKENIZERS_PARALLELISM="false",
                PYTHONUNBUFFERED="1", NEMOTRON_PARENT_PID=str(os.getpid()),
                VLLM_PLUGINS="nemo_speechlm", VLLM_MAX_AUDIO_DECODE_DURATION_S="3600",
                VLLM_MAX_AUDIO_CLIP_FILESIZE_MB="180", VLLM_USE_V2_MODEL_RUNNER="0",
                VLLM_DISABLED_KERNELS="FlashInferFP8ScaledMMLinearKernel",
                FLASHINFER_WORKSPACE_BASE=str(WORK / "kernels-vllm032"))
