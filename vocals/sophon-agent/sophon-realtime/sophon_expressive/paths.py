# Copyright 2026 Adventists AI. Apache-2.0.
"""Where the released files live. Set SOPHON_MODEL_DIR to the downloaded Sophon-Realtime-32B-E2E directory."""
import os


def model_dir() -> str:
    d = os.environ.get("SOPHON_MODEL_DIR", "")
    if not d or not os.path.isdir(d):
        raise RuntimeError("Set SOPHON_MODEL_DIR to the Sophon-Realtime-32B-E2E model directory "
                           "(modelscope download --model AdventistsAI/Sophon-Realtime-32B-E2E --local_dir ...)")
    return os.path.abspath(d)


def base_dir() -> str:
    return os.path.join(model_dir(), "base")


def tag_adapter_dir() -> str:
    return os.environ.get("SOPHON_REPLY_LORA") or os.path.join(model_dir(), "adapters", "tag")


def talker_weights() -> str:   # 32B talker of the two-GPU build (weights tagged v1.1.0 and earlier)
    return os.path.join(model_dir(), "talker", "model.safetensors")


def talker_dir() -> str:       # talker of the single-GPU build
    return os.path.join(model_dir(), "talker220")


def e2_dir() -> str:
    return os.path.join(model_dir(), "e2")


def voice_file(name: str = "chelsie") -> str:
    return os.path.join(model_dir(), "voices", f"{name}.safetensors")
