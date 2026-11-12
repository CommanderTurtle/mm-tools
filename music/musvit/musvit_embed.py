#!/usr/bin/env python3
"""Extract local MuSViT page or staff representations to compressed NPZ."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import ViTModel


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("MUSVIT_MODEL_PATH", "")),
    )
    parser.add_argument("--device", default=os.environ.get("MUSVIT_DEVICE", "cuda"))
    parser.add_argument(
        "--layout",
        choices=("page", "pad", "native"),
        default="page",
        help="page=rescale square; pad=preserve aspect on white; native=interpolate positions.",
    )
    parser.add_argument(
        "--patches", action="store_true", help="Store all patch embeddings, not summaries only."
    )
    return parser


def _image_tensor(path: Path, layout: str) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    original = image.size
    if layout == "page":
        image = image.resize((1024, 1024), Image.Resampling.BILINEAR)
    elif layout == "pad":
        image.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (1024, 1024), "white")
        canvas.paste(image, (0, 0))
        image = canvas
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
    return tensor.unsqueeze(0), original


def main() -> int:
    args = _parser().parse_args()
    source = args.input.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Input not found: {source}")
    if not (model_path / "model.safetensors").is_file():
        raise SystemExit("MuSViT model is incomplete. Run ./download-models.sh or edit .env.")

    image, original_size = _image_tensor(source, args.layout)
    model = ViTModel.from_pretrained(str(model_path), local_files_only=True).to(args.device)
    model.eval()
    image = image.to(args.device)
    kwargs = {"interpolate_pos_encoding": True} if args.layout == "native" else {}
    with torch.inference_mode():
        hidden = model(image, **kwargs).last_hidden_state.float().cpu().numpy()

    output = (args.output or source.with_suffix(".musvit.npz")).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "cls": hidden[:, 0, :],
        "patch_mean": hidden[:, 1:, :].mean(axis=1),
        "original_size": np.asarray(original_size, dtype=np.int32),
        "input_size": np.asarray(image.shape[-2:], dtype=np.int32),
    }
    if args.patches:
        payload["patches"] = hidden[:, 1:, :]
    np.savez_compressed(output, **payload)
    print(f"Saved MuSViT representation to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
