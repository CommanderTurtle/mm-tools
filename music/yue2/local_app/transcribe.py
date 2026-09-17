#!/usr/bin/env python3
"""Offline SheetSage2 launcher with an explicit local MERT-v2 parent."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--preset", choices=("default", "paper"), default="default")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--overlap", type=float)
    parser.add_argument("--lookahead", type=float)
    parser.add_argument("--prompts", nargs="+")
    parser.add_argument("--melody-only", action="store_true")
    parser.add_argument("--export-logits", action="store_true")
    parser.add_argument("--export-scores", action="store_true")
    parser.add_argument("--export-embeddings", action="store_true")
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if not args.audio.is_file() or not args.model.is_dir() or not args.base_model.is_dir():
        parser.error("Audio, SheetSage2, and MERT-v2 paths must exist locally.")
    sys.path.insert(0, str(args.model))
    import torch
    from transformers import AutoModel

    if args.device != "cuda" or not torch.cuda.is_available():
        parser.error("The mm-tools SheetSage2 profile is CUDA-only.")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = AutoModel.from_pretrained(
        str(args.model),
        base_model_path=str(args.base_model),
        local_files_only=True,
        trust_remote_code=True,
    ).eval().to("cuda")

    def progress(value: dict) -> None:
        if value.get("stage") == "encoding":
            print(f"Window {value['window']}/{value['windows']}", flush=True)

    options = {
        "dtype": args.dtype,
        "preset": args.preset,
        "max_seconds": args.max_seconds,
        "overlap_seconds": args.overlap,
        "lookahead_seconds": args.lookahead,
        "export_logits": args.export_logits,
        "export_scores": args.export_scores,
        "export_embeddings": args.export_embeddings,
        "output_hidden_states": args.all_layers,
        "melody_only": args.melody_only,
        "progress": progress,
    }
    if args.prompts:
        options["prompts"] = args.prompts
    args.output.mkdir(parents=True, exist_ok=True)
    result = model.transcribe(args.audio, output_dir=args.output, **options)
    if result.get("abc_error") or not result.get("abc"):
        raise RuntimeError(result.get("abc_error") or "SheetSage2 produced no ABC score")
    print(f"Saved transcription to {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
