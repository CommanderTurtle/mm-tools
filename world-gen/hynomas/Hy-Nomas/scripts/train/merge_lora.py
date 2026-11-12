#!/usr/bin/env python3
"""Merge a Hy-Nomas LoRA adapter into a DiT safetensors checkpoint.

The merge is performed as::

    merged_weight = base_weight + (alpha / rank) * (lora_B @ lora_A)

Activation-checkpoint wrapper components in adapter keys are removed before
matching them to base checkpoint keys. The output is written atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def normalize_adapter_stem(key: str, suffix: str) -> str:
    stem = key[: -len(suffix)]
    return stem.replace("._checkpoint_wrapped_module.", ".").replace(
        "_checkpoint_wrapped_module.", ""
    )


def collect_pairs(adapter: dict[str, torch.Tensor]):
    pairs: dict[str, dict[str, str]] = {}
    for key in adapter:
        if key.endswith(".lora_A.weight"):
            kind, suffix = "A", ".lora_A.weight"
        elif key.endswith(".lora_B.weight"):
            kind, suffix = "B", ".lora_B.weight"
        else:
            raise ValueError(f"unexpected adapter tensor: {key}")
        stem = normalize_adapter_stem(key, suffix)
        if kind in pairs.setdefault(stem, {}):
            raise ValueError(f"duplicate LoRA {kind} tensor for {stem}")
        pairs[stem][kind] = key
    incomplete = [stem for stem, pair in pairs.items() if set(pair) != {"A", "B"}]
    if incomplete:
        raise ValueError(f"incomplete LoRA A/B pairs: {incomplete[:10]}")
    return pairs


def merge(base_path: Path, adapter_dir: Path, output_path: Path, device: str):
    config_path = adapter_dir / "adapter_config.json"
    adapter_path = adapter_dir / "adapter.safetensors"
    if not base_path.is_file():
        raise FileNotFoundError(f"base checkpoint not found: {base_path}")
    if not adapter_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"adapter directory must contain adapter.safetensors and "
            f"adapter_config.json: {adapter_dir}"
        )
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    rank = int(config["rank"])
    alpha = float(config["alpha"])
    if rank <= 0 or alpha <= 0:
        raise ValueError(f"invalid LoRA rank/alpha: rank={rank}, alpha={alpha}")
    scale = alpha / rank

    print(f"loading base checkpoint: {base_path}", flush=True)
    base = load_file(str(base_path), device="cpu")
    print(f"loading adapter: {adapter_path}", flush=True)
    adapter = load_file(str(adapter_path), device="cpu")
    pairs = collect_pairs(adapter)

    missing = []
    mismatched = []
    mappings = []
    for stem, pair in sorted(pairs.items()):
        weight_key = stem + ".weight"
        if weight_key not in base:
            missing.append(weight_key)
            continue
        weight_shape = tuple(base[weight_key].shape)
        a_shape = tuple(adapter[pair["A"]].shape)
        b_shape = tuple(adapter[pair["B"]].shape)
        if (
            len(weight_shape) != 2
            or a_shape != (rank, weight_shape[1])
            or b_shape != (weight_shape[0], rank)
        ):
            mismatched.append((weight_key, weight_shape, a_shape, b_shape))
        mappings.append((weight_key, pair["A"], pair["B"]))
    if missing:
        raise ValueError(
            f"adapter targets missing from base checkpoint: {missing[:10]}"
        )
    if mismatched:
        raise ValueError(f"LoRA/base shape mismatches: {mismatched[:5]}")

    merge_device = torch.device(device)
    if merge_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA merge requested but CUDA is unavailable")
    print(
        f"merging {len(mappings)} weights with rank={rank}, alpha={alpha}, "
        f"scale={scale:g}, device={merge_device}",
        flush=True,
    )
    with torch.no_grad():
        for index, (weight_key, a_key, b_key) in enumerate(mappings, 1):
            weight = base[weight_key]
            compute_dtype = torch.float32
            a = adapter[a_key].to(device=merge_device, dtype=compute_dtype)
            b = adapter[b_key].to(device=merge_device, dtype=compute_dtype)
            delta = torch.matmul(b, a).mul_(scale)
            merged = weight.to(device=merge_device, dtype=compute_dtype).add_(delta)
            base[weight_key] = merged.to(device="cpu", dtype=weight.dtype).contiguous()
            if index % 25 == 0 or index == len(mappings):
                print(f"merged {index}/{len(mappings)} weights", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        print(f"writing merged checkpoint: {output_path}", flush=True)
        save_file(base, str(temporary))
        os.replace(temporary, output_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"merged checkpoint ready: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="device used for B@A multiplication (default: cuda:0 when available)",
    )
    args = parser.parse_args()
    merge(
        args.base.expanduser().resolve(),
        args.adapter.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.device,
    )


if __name__ == "__main__":
    main()
