#!/usr/bin/env python3
"""Validate a released checkpoint directory against the public decoder config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


PUBLIC_DECODER_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "wevisone_vl_v1_decoder.json"
REQUIRED = (
    "README.md",
    "LICENSE",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "mask_decoder.pt",
)
TOKENIZER_LOCAL_KEYS = ("is_local", "local_files_only")


def check_files(root: Path, errors: list) -> None:
    """Require exactly the public file set."""
    present = {path.name for path in root.iterdir() if not path.name.startswith(".")}
    missing = sorted(set(REQUIRED) - present)
    extra = sorted(present - set(REQUIRED))
    if missing:
        errors.append(f"missing files: {missing}")
    if extra:
        errors.append(f"unexpected files: {extra}")


def check_sidecar(root: Path, public: dict, errors: list) -> dict:
    """Validate mask_decoder.pt against the public decoder config."""
    sidecar = torch.load(root / "mask_decoder.pt", map_location="cpu", weights_only=True)
    contract = public["checkpoint_contract"]
    if set(sidecar) != {"state_dict", "cfg", "seg_token_id", "spatial_merge_size"}:
        errors.append(f"mask_decoder.pt has unexpected top-level keys: {sorted(sidecar)}")
    cfg = sidecar.get("cfg") or {}
    extra = sorted(set(cfg) - set(public["cfg"]))
    if extra:
        errors.append(f"mask_decoder.pt cfg has non-public fields: {extra}")
    for key, expected in public["cfg"].items():
        actual = cfg.get(key)
        if isinstance(actual, tuple):
            actual = list(actual)
        if actual != expected:
            errors.append(f"decoder cfg mismatch for {key}: checkpoint={actual!r}, public={expected!r}")
    state_dict = sidecar.get("state_dict") or {}
    if len(state_dict) != contract["state_dict_tensors"]:
        errors.append(f"unexpected decoder tensor count: {len(state_dict)}")
    params = sum(tensor.numel() for tensor in state_dict.values())
    if params != contract["parameters"]:
        errors.append(f"unexpected decoder parameter count: {params}")
    for key in ("seg_token_id", "spatial_merge_size"):
        if sidecar.get(key) != contract[key]:
            errors.append(f"mask_decoder.pt {key}={sidecar.get(key)!r}, expected {contract[key]!r}")
    return sidecar


def check_tokenizer(root: Path, sidecar: dict, errors: list) -> None:
    """Ensure the tokenizer agrees with the sidecar and carries no local-save keys."""
    tokenizer = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
    seg_id = tokenizer.get("model", {}).get("vocab", {}).get("<OBJ>")
    if seg_id is None:
        seg_id = next(
            (t.get("id") for t in tokenizer.get("added_tokens", []) if t.get("content") == "<OBJ>"),
            None,
        )
    if seg_id is None or int(seg_id) != int(sidecar.get("seg_token_id", -1)):
        errors.append("tokenizer and sidecar disagree on <OBJ> id")
    config = json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8"))
    leftover = [key for key in TOKENIZER_LOCAL_KEYS if key in config]
    if leftover:
        errors.append(f"tokenizer_config.json still has local-save keys: {leftover}")


def count_base_params(root: Path) -> int:
    """Count the base-model parameters stored in model.safetensors."""
    params = 0
    with safe_open(root / "model.safetensors", framework="pt", device="cpu") as model:
        for name in model.keys():
            count = 1
            for dimension in model.get_slice(name).get_shape():
                count *= dimension
            params += count
    return params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    root = args.checkpoint.expanduser().resolve()
    public = json.loads(PUBLIC_DECODER_CONFIG.read_text(encoding="utf-8"))
    errors: list = []

    check_files(root, errors)
    sidecar = check_sidecar(root, public, errors)
    check_tokenizer(root, sidecar, errors)
    params = count_base_params(root)

    if errors:
        raise SystemExit("Checkpoint validation failed:\n- " + "\n- ".join(errors))
    print(
        f"OK: {public['release']} · base={params:,} params · "
        f"decoder={public['checkpoint_contract']['parameters']:,} params · "
        f"<OBJ>={sidecar['seg_token_id']}"
    )


if __name__ == "__main__":
    main()
