#!/usr/bin/env python3
# Copyright (c) Ant Group. All rights reserved.
"""Check a Ming-3.0-tts-tiny checkpoint before paying to load it.

Loads only the tokenizer and the JSON configs, so it finishes in seconds on a
CPU-only box. Ming-3.0-tts-0.5b has its own check in ``preflight.py``; the two
models share no contract, so one script cannot cover both.

    python tools/preflight_tiny.py --model_path /path/to/Ming-3.0-tts-tiny
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

OK, WARN, BAD = "ok  ", "warn", "FAIL"


def report(status, message):
    print(f"  [{status}] {message}")
    return status != BAD


def check_files(model_path):
    healthy = True
    if not os.path.isdir(model_path):
        return report(BAD, f"not a directory: {model_path}")

    shards = [n for n in os.listdir(model_path) if n.endswith(".safetensors")]
    if shards:
        total = sum(
            os.path.getsize(os.path.join(model_path, n)) for n in shards
        ) / (1024 ** 3)
        report(OK, f"{len(shards)} safetensors shard(s), {total:.1f} GiB")
    else:
        healthy &= report(BAD, "no .safetensors weights found")

    if os.path.exists(os.path.join(model_path, "config.json")):
        report(OK, "config.json")
    else:
        healthy &= report(BAD, "config.json is missing")

    if os.path.exists(os.path.join(model_path, "tokenizer.json")):
        report(OK, "tokenizer.json (checkpoint is self-contained)")
    else:
        healthy &= report(BAD, "tokenizer.json is missing; the tokenizer ships with the checkpoint")

    if os.path.exists(os.path.join(model_path, "campplus.onnx")):
        report(OK, "campplus.onnx")
    else:
        healthy &= report(BAD, "campplus.onnx is missing; it ships with the checkpoint")
    return healthy


def check_release(model_path):
    """The contract MingTinyTTS.from_pretrained enforces, without the weights."""

    from ming_tiny.api import REQUIRED_TRANSFORMERS, require_transformers
    from ming_tiny.task_protocol import validate_release_bundle

    healthy = True
    try:
        require_transformers()
        report(OK, f"transformers=={REQUIRED_TRANSFORMERS}")
    except RuntimeError as error:
        healthy &= report(BAD, str(error))
    try:
        validate_release_bundle(model_path)
        report(OK, "checkpoint matches the released decode and prompt contract")
    except (OSError, RuntimeError, ValueError) as error:
        healthy &= report(BAD, str(error))
    return healthy


def check_config(model_path):
    from ming_tiny.prompt_contract import (
        BASE_PROTOCOL,
        THINKING_V1_PROTOCOL,
        protocol_version,
    )

    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as fh:
        raw = json.load(fh)

    healthy = True
    tts_config = raw.get("tts_config")
    if not isinstance(tts_config, dict):
        return report(BAD, "config.json has no tts_config block")

    try:
        protocol = protocol_version(tts_config)
    except ValueError as error:
        return report(BAD, str(error))
    report(OK, f"protocol: {protocol}")

    llm = raw.get("llm_config") or {}
    # Both levels tie independently: the wrapper through PreTrainedModel, the
    # language model through its own config.
    tied = [
        level
        for level, config in (("top level", raw), ("llm_config", llm))
        if config.get("tie_word_embeddings", False)
    ]
    if tied:
        # tiny's embedding and lm_head are independently trained matrices.
        healthy &= report(
            BAD,
            f"tie_word_embeddings is true in {' and '.join(tied)}; this "
            "collapses two independently trained matrices and produces "
            "garbled speech",
        )
    else:
        report(OK, "tie_word_embeddings: false")

    layer_types = llm.get("layer_types") or llm.get("attention_layer_types")
    needs_fla = bool(layer_types) and any(t != "attention" for t in layer_types)
    if needs_fla:
        import importlib.util

        if importlib.util.find_spec("fla") is not None:
            report(OK, "linear-attention layers present, fla-core installed")
        else:
            healthy &= report(
                BAD, "checkpoint has linear-attention layers: pip install -U fla-core"
            )
    else:
        report(OK, "no linear-attention layers; fla-core not needed")

    if protocol == THINKING_V1_PROTOCOL:
        report(OK, "thinking_v1: portable MoE loop with fp32 recurrent state")
    elif protocol == BASE_PROTOCOL:
        try:
            from ming_tiny.modeling_bailing_moe_v3 import _ATORCH_MOE_AVAILABLE
        except ImportError:
            _ATORCH_MOE_AVAILABLE = False
        if _ATORCH_MOE_AVAILABLE:
            report(OK, "atorch present: fused grouped-GEMM MoE path")
        else:
            report(
                WARN,
                "atorch absent: portable MoE loop, slower and not numerically "
                "identical to the validated trajectory",
            )
    return healthy


def check_tokenizer(model_path, tokenizer_path):
    from transformers import AutoTokenizer

    from ming_tiny.prompt_contract import validate_tts_tokenizer

    source = tokenizer_path or model_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source, trust_remote_code=True, use_fast=True
        )
    except Exception as error:  # noqa: BLE001 - surface any loader failure
        # Transformers lists every known model_type on failure; one line is enough.
        reason = str(error).split("Should have")[0].strip()
        return report(BAD, f"tokenizer failed to load from {source}: {reason}")

    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as fh:
        raw = json.load(fh)

    # The engine reads the EOS off the language model, which is where the
    # converter writes it; the top level usually has none.
    config_eos = raw.get("llm_config", {}).get("eos_token_id", raw.get("eos_token_id"))
    if config_eos is None:
        return report(BAD, "config.json declares no llm_config.eos_token_id")

    try:
        validate_tts_tokenizer(tokenizer, raw["tts_config"], config_eos)
    except ValueError as error:
        return report(BAD, str(error))
    report(OK, f"tokenizer matches the trained contract ({len(tokenizer)} tokens)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Preflight a Ming-3.0-tts-tiny checkpoint"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    args = parser.parse_args()

    print(f"\nMing-3.0-tts-tiny preflight: {args.model_path}\n")
    print("files")
    healthy = check_files(args.model_path)
    if healthy:
        print("config")
        healthy &= check_config(args.model_path)
        print("tokenizer")
        healthy &= check_tokenizer(args.model_path, args.tokenizer_path)
        print("release contract")
        healthy &= check_release(args.model_path)

    print()
    if healthy:
        print("ready: run infer_tiny.py")
        return 0
    print("not ready: fix the FAIL lines above")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
