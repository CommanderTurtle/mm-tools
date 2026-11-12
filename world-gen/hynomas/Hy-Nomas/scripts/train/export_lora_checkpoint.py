#!/usr/bin/env python3
"""Export a sharded LoRA training checkpoint as adapter.safetensors."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from hymm.models.checkpointing.lora_checkpoint import (
    ADAPTER_CONFIG_FILE,
    COMPLETE_FILE,
    FORMAT,
    SHARD_TEMPLATE,
)


def export_adapter(checkpoint_dir: Path, output_dir: Path):
    manifest = json.loads((checkpoint_dir / COMPLETE_FILE).read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("complete") is not True:
        raise ValueError(f"Not a complete LoRA HSDP checkpoint: {checkpoint_dir}")

    shard_count = int(manifest["shard_count"])
    shards = [
        torch.load(
            checkpoint_dir / SHARD_TEMPLATE.format(shard_rank=index),
            map_location="cpu",
            weights_only=False,
        )
        for index in range(shard_count)
    ]
    keys = set(shards[0]["adapter"])
    if any(set(shard["adapter"]) != keys for shard in shards[1:]):
        raise ValueError("LoRA shard tensor keys are inconsistent")

    merged = {}
    for key in sorted(keys):
        metadata = shards[0]["param_metadata"][key]
        global_shape = tuple(int(value) for value in metadata["global_shape"])
        shard_dim = metadata.get("shard_dim")
        pieces = [shard["adapter"][key] for shard in shards]
        if shard_dim is None:
            if any(not torch.equal(pieces[0], piece) for piece in pieces[1:]):
                raise ValueError(f"Replicated LoRA tensor differs across shards: {key}")
            tensor = pieces[0]
        else:
            shard_dim = int(shard_dim)
            tensor = torch.cat(pieces, dim=shard_dim)
            slices = tuple(slice(0, size) for size in global_shape)
            tensor = tensor[slices]
        if tuple(tensor.shape) != global_shape:
            raise ValueError(
                f"Reconstructed shape mismatch for {key}: "
                f"{tuple(tensor.shape)} != {global_shape}"
            )
        merged[key] = tensor.contiguous()

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(output_dir / "adapter.safetensors"))
    shutil.copyfile(
        checkpoint_dir / ADAPTER_CONFIG_FILE,
        output_dir / ADAPTER_CONFIG_FILE,
    )
    return len(merged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    count = export_adapter(args.checkpoint_dir, args.output_dir)
    print(f"Exported {count} LoRA tensors to {args.output_dir}")


if __name__ == "__main__":
    main()
