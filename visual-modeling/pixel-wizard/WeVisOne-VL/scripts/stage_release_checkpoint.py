#!/usr/bin/env python3
"""Create a clean model-hub upload directory from a final HF snapshot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
HF_FILES = (
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
)
PUBLIC_DECODER_CONFIG = REPO_ROOT / "configs" / "wevisone_vl_v1_decoder.json"
LICENSE = REPO_ROOT / "LICENSE"
TOKENIZER_LOCAL_KEYS = ("is_local", "local_files_only")
MODEL_SIZES = {2048: "2B", 2560: "4B"}
MODEL_CARD_TEMPLATE = """---
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - vision-language
  - object-detection
  - referring-expression-comprehension
  - object-counting
  - point-grounding
  - ocr
  - referring-expression-segmentation
  - image-matting
---

# WeVisOne-VL: Unified Visual Perception from Objects to Pixels

**WeVisOne-VL** (**We**Chat **Vis**ion All-in-**One**) is a unified vision-language model developed by WeChat Vision, Tencent Inc. Built upon Qwen3-VL, it unifies object-level perception (open-vocabulary object detection, referring localization, object counting, object pointing, and OCR) and pixel-level perception (referring segmentation and promptable alpha matting) through structured next-token prediction and a lightweight 9.6M-parameter pixel decoder.

## Quick Start

Install the inference package from [GitHub](https://github.com/Tencent/WeVisOne-VL), then:

```python
from PIL import Image
from wevisone_vl import WeVisOneVL

model = WeVisOneVL("tencent/WeVisOne-VL-{size}", device="auto", dtype="bfloat16")
image = Image.open("example.jpg").convert("RGB")
result = model.predict(image, task="matting", query="person", temperature=0.0)
print(result.response, len(result.masks))
```

## Architecture & Released Revision

The model uses dedicated coordinate tokens (`<0>` ... `<1000>`) to represent normalized bounding boxes and points, returns `None` when a queried target is absent, and emits a special instance token (`<OBJ>`, token ID `152678`) after each predicted box for dense tasks.

The attached `mask_decoder.pt` (9,609,764 parameters) is a required core component that combines:
- A **Visual Feature Adapter** over multi-level Qwen3-VL pre-merging ViT features;
- A **Box-Guided Query Decoder** with 6 deformable cross-attention blocks initialized from the `<OBJ>` hidden state and Fourier-encoded box prior;
- An **Instance-Conditioned Dense Prediction Head** with `FullCanvasAlphaCascadeHead` (refining predictions across S/4, S/2, and S/1 to output full-resolution continuous alpha mattes at stride 1).

## Intended Use

The release is intended for research in visual grounding, referring localization, pointing, counting, OCR localization, referring segmentation, and image matting. It is not intended to make autonomous high-impact decisions or to identify individuals.

## Limitations

- Fine alpha boundaries remain sensitive to image resolution, severe occlusion, complex transparency, and prompt phrasing.
- For inference efficiency, the model directly generates structured outputs without explicit chain-of-thought (CoT) reasoning, and currently targets single-turn image perception (video grids are rejected by the pixel decoder).

"""


def write_public_sidecar(source: Path, destination: Path) -> None:
    """Re-save mask_decoder.pt with only the fields needed for inference."""
    public = json.loads(PUBLIC_DECODER_CONFIG.read_text(encoding="utf-8"))
    sidecar = torch.load(source, map_location="cpu", weights_only=False)
    source_cfg = sidecar.get("cfg") or {}
    for key, expected in public["cfg"].items():
        if key == "architecture_version":
            continue
        actual = source_cfg.get(key)
        if isinstance(actual, tuple):
            actual = list(actual)
        if actual != expected:
            raise SystemExit(f"source decoder cfg mismatch for {key}: {actual!r} != {expected!r}")
    contract = public["checkpoint_contract"]
    state_dict = {name: tensor.contiguous() for name, tensor in sidecar["state_dict"].items()}
    if len(state_dict) != contract["state_dict_tensors"]:
        raise SystemExit(f"unexpected decoder tensor count: {len(state_dict)}")
    torch.save(
        {
            "state_dict": state_dict,
            "cfg": dict(public["cfg"]),
            "seg_token_id": int(contract["seg_token_id"]),
            "spatial_merge_size": int(contract["spatial_merge_size"]),
        },
        destination,
    )


def write_tokenizer_config(source: Path, destination: Path) -> None:
    """Drop keys that only describe the machine the tokenizer was saved on."""
    config = json.loads(source.read_text(encoding="utf-8"))
    for key in TOKENIZER_LOCAL_KEYS:
        config.pop(key, None)
    destination.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_model_card(config_path: Path, destination: Path) -> None:
    """Render the model card with the model id that matches this checkpoint size."""
    hidden_size = int(json.loads(config_path.read_text(encoding="utf-8"))["text_config"]["hidden_size"])
    if hidden_size not in MODEL_SIZES:
        raise SystemExit(f"unknown text hidden size {hidden_size}; add it to MODEL_SIZES")
    size = MODEL_SIZES[hidden_size]
    destination.write_text(MODEL_CARD_TEMPLATE.format(size=size), encoding="utf-8")


def stage_files(source: Path, destination: Path, copy) -> None:
    for name in HF_FILES:
        copy(source / name, destination / name)
    write_tokenizer_config(source / "tokenizer_config.json", destination / "tokenizer_config.json")
    write_public_sidecar(source / "mask_decoder.pt", destination / "mask_decoder.pt")
    write_model_card(source / "config.json", destination / "README.md")
    shutil.copy2(LICENSE, destination / "LICENSE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="Use hard links for the large files (source and destination must share a filesystem)",
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    required = HF_FILES + ("tokenizer_config.json", "mask_decoder.pt")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise SystemExit(f"source is missing required files: {', '.join(missing)}")
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    stage_files(source, destination, os.link if args.hardlink else shutil.copy2)
    staged = sorted(path.name for path in destination.iterdir())
    print(f"Staged {len(staged)} files in {destination}: {', '.join(staged)}")


if __name__ == "__main__":
    main()
