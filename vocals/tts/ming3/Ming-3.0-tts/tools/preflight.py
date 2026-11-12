#!/usr/bin/env python3
# Copyright (c) Ant Group. All rights reserved.
"""Check a checkpoint directory before paying to load it.

Verifies that the directory is self-contained and that its tokenizer matches
the trained token contract. Loads only the tokenizer and the JSON configs, so
it finishes in seconds and runs on a CPU-only box.

    python tools/preflight.py --model_path /path/to/Ming-3.0-tts-0.5b
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

REQUIRED_FILES = ("config.json", "model.safetensors")
# Either the fast tokenizer or the BPE pair is enough.
TOKENIZER_ALTERNATIVES = (("tokenizer.json",), ("vocab.json", "merges.txt"))

OK, WARN, BAD = "ok  ", "warn", "FAIL"


def report(status, message):
    print(f"  [{status}] {message}")
    return status != BAD


def check_files(model_path):
    healthy = True
    for name in REQUIRED_FILES:
        path = os.path.join(model_path, name)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 ** 2)
            healthy &= report(OK, f"{name} ({size:.1f} MiB)")
        else:
            healthy &= report(BAD, f"{name} is missing")

    if any(all(os.path.exists(os.path.join(model_path, n)) for n in group)
           for group in TOKENIZER_ALTERNATIVES):
        healthy &= report(OK, "tokenizer files present")
    else:
        healthy &= report(BAD, "no tokenizer.json and no vocab.json+merges.txt; "
                               "the checkpoint is not self-contained")

    campplus = os.path.join(model_path, "campplus.onnx")
    if os.path.exists(campplus):
        report(OK, "campplus.onnx (speaker embedding)")
    else:
        report(WARN, "campplus.onnx missing; --use_spk_emb will fail")
    return healthy


def check_config(model_path):
    path = os.path.join(model_path, "config.json")
    if not os.path.exists(path):
        return False
    with open(path) as handle:
        config = json.load(handle)

    healthy = True
    model_type = config.get("model_type")
    if model_type == "dense":
        healthy &= report(OK, f"model_type = {model_type}")
    else:
        healthy &= report(BAD, f"model_type = {model_type!r}; this repository "
                               "ships the dense 0.5b model only")

    ditar = config.get("ditar_config", {})
    for field in ("patch_size", "history_patch_size"):
        if field in ditar:
            report(OK, f"ditar_config.{field} = {ditar[field]}")
        else:
            healthy &= report(BAD, f"ditar_config.{field} is missing")

    sample_rate = config.get("audio_tokenizer_config", {}).get("sample_rate")
    if sample_rate:
        report(OK, f"sample_rate = {sample_rate}")
    else:
        healthy &= report(BAD, "audio_tokenizer_config.sample_rate is missing")
    return healthy


def check_tokenizer(model_path):
    import prompt_contract

    with open(os.path.join(model_path, "config.json")) as handle:
        model_type = json.load(handle).get("model_type", "dense")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
    except Exception as error:  # noqa: BLE001
        return report(BAD, f"tokenizer failed to load: {error}")

    try:
        prompt_contract.validate_tokenizer(tokenizer, model_type)
    except ValueError as error:
        return report(BAD, str(error))
    return report(OK, f"tokenizer matches the {model_type} token contract "
                      f"({len(tokenizer)} tokens)")


def check_optional_deps():
    for module, note in (
        ("pynini", "text normalization; run with --no_text_normalization without it"),
        ("flash_attn", "attention speedup; falls back to PyTorch SDPA"),
    ):
        if importlib.util.find_spec(module) is not None:
            report(OK, f"{module} available")
        else:
            report(WARN, f"{module} not installed - {note}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        raise SystemExit(f"not a directory: {args.model_path}")

    print(f"checkpoint: {args.model_path}\n")
    print("files")
    healthy = check_files(args.model_path)
    print("\nconfig")
    healthy &= check_config(args.model_path)
    print("\ntokenizer")
    healthy &= check_tokenizer(args.model_path)
    print("\noptional dependencies")
    check_optional_deps()

    print()
    if healthy:
        print("preflight passed; see README.md for inference examples")
        return 0
    print("preflight FAILED; fix the items above before running inference")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
