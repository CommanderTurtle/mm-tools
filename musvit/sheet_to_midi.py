#!/usr/bin/env python3
"""Transcribe piano-form sheet images to beKern and MIDI, fully locally.

MuSViT's released checkpoint is a foundation encoder and has no notation
decoder. This command therefore uses the authors' official full-page Sheet
Music Transformer for OMR, then Verovio for deterministic Humdrum -> MIDI.
The companion `musvit_embed.py` exposes the actual MuSViT encoder.
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import verovio

ROOT = Path(__file__).resolve().parent
SMT_SOURCE = Path(os.environ.get("SMT_SOURCE_PATH", ROOT / "vendor" / "SMT"))
if not (SMT_SOURCE / "smt_model").is_dir():
    raise SystemExit("SMT source is missing. Run ./uvsetup.sh first.")
sys.path.insert(0, str(SMT_SOURCE))

from smt_model import SMTModelForCausalLM  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a score image or PDF into local beKern, MIDI, and SVG."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("SMT_MODEL_PATH", "")),
        help="Local PRAIG/smt-fp-grandstaff directory.",
    )
    parser.add_argument("--device", default=os.environ.get("MUSVIT_DEVICE", "cuda"))
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument(
        "--page",
        type=int,
        help="For a PDF, process only this 1-based page. Default: every page.",
    )
    parser.add_argument("--pdf-dpi", type=int, default=220)
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Optional emergency cap for autoregressive decoding.",
    )
    parser.add_argument("--no-svg", action="store_true")
    return parser


def _natural_page_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    return (int(match.group(1)) if match else 0, path.name)


def _score_pages(source: Path, temp_dir: Path, dpi: int) -> list[Path]:
    if source.suffix.lower() != ".pdf":
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise SystemExit(f"Unsupported score format: {source.suffix}")
        return [source]
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise SystemExit("pdftoppm (poppler-utils) is required for PDF input")
    prefix = temp_dir / "page"
    subprocess.run(
        [pdftoppm, "-png", "-r", str(dpi), str(source), str(prefix)], check=True
    )
    pages = sorted(temp_dir.glob("page-*.png"), key=_natural_page_key)
    if not pages:
        raise SystemExit("PDF rasterization produced no pages")
    return pages


def _fit_page(path: Path, max_height: int, max_width: int) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    height, width = image.shape
    scale = min(max_height / height, max_width / width)
    new_height = max(16, min(max_height, int(round(height * scale))))
    new_width = max(16, min(max_width, int(round(width * scale))))
    # ConvNeXt reduces both axes four times. Multiples of 16 avoid silently
    # dropping a partial border patch.
    new_height = max(16, (new_height // 16) * 16)
    new_width = max(16, (new_width // 16) * 16)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.ascontiguousarray(resized)).float().div_(255.0)
    return tensor.unsqueeze(0).unsqueeze(0)


def _decode_bekern(tokens: list[str]) -> str:
    body = "".join(tokens)
    body = (
        body.replace("<t>", "\t")
        .replace("<b>", "\n")
        .replace("<s>", " ")
        .replace("@", "")
        .replace("·", "")
        .strip()
    )
    lines = [line.rstrip() for line in body.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("The OMR decoder produced an empty score")

    if lines[0].startswith("**kern"):
        score = lines
    else:
        # The published full-page checkpoint was trained after removing its
        # two-spine piano header. Restore only that deterministic wrapper.
        score = ["**kern\t**kern", *lines]
    if not score[-1].lstrip().startswith("*-"):
        score.append("*-\t*-")
    return "\n".join(score) + "\n"


def _validate_bekern(kern: str) -> None:
    """Reject malformed spine counts before handing data to native Verovio."""
    lines = [line for line in kern.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("The OMR decoder produced an empty score")

    spines = len(lines[0].split("\t"))
    if spines < 1:
        raise RuntimeError("The OMR decoder produced no beKern spines")

    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) != spines:
            raise RuntimeError(
                "The OMR decoder produced malformed beKern at line "
                f"{line_number}: expected {spines} tab-separated fields, "
                f"found {len(fields)}. The decoded .krn file was preserved."
            )

        joins = 0
        index = 0
        while index < len(fields):
            if fields[index] != "*v":
                index += 1
                continue
            end = index + 1
            while end < len(fields) and fields[end] == "*v":
                end += 1
            run_length = end - index
            if run_length > 1:
                joins += run_length - 1
            index = end

        spines += fields.count("*^") + fields.count("*+")
        spines -= fields.count("*-") + joins
        if spines < 0:
            raise RuntimeError(
                f"The OMR decoder terminated too many spines at line {line_number}."
            )


def _write_derivatives(kern: str, midi_path: Path, write_svg: bool) -> None:
    kern_path = midi_path.with_suffix(".krn")
    kern_path.write_text(kern, encoding="utf-8")
    _validate_bekern(kern)

    toolkit = verovio.toolkit()
    toolkit.setOptions({"inputFrom": "humdrum", "breaks": "none"})
    midi_base64 = toolkit.convertHumdrumToMIDI(kern)
    if not midi_base64:
        raise RuntimeError(f"Verovio rejected the decoded beKern:\n{toolkit.getLog()}")
    midi_path.write_bytes(base64.b64decode(midi_base64))

    if write_svg:
        preview = verovio.toolkit()
        preview.setOptions({"inputFrom": "humdrum", "breaks": "auto"})
        if preview.loadData(kern):
            midi_path.with_suffix(".svg").write_text(
                preview.renderToSVG(1), encoding="utf-8"
            )


def _output_for(source: Path, requested: Path | None, page: int, total: int) -> Path:
    if total == 1:
        target = requested or source.with_suffix(".mid")
        if target.suffix.lower() not in {".mid", ".midi"}:
            target = target / f"{source.stem}.mid"
        return target.expanduser().resolve()
    directory = requested or source.with_name(f"{source.stem}_omr")
    if directory.suffix:
        raise SystemExit("Multiple PDF pages require --output to be a directory")
    return directory.expanduser().resolve() / f"page-{page:03d}.mid"


def main() -> int:
    args = _parser().parse_args()
    source = args.input.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Input not found: {source}")
    if not (model_path / "model.safetensors").is_file():
        raise SystemExit("SMT model is incomplete. Run ./download-models.sh or edit .env.")
    if args.page is not None and args.page < 1:
        raise SystemExit("--page is 1-based")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    dtype = getattr(torch, args.dtype)
    if args.device == "cpu" and dtype != torch.float32:
        raise SystemExit("CPU inference requires --dtype float32")

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"Loading full-page OMR decoder from {model_path}")
    model = SMTModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, use_safetensors=True
    ).to(device=args.device, dtype=dtype)
    model.eval()
    if args.max_tokens:
        model.maxlen = min(model.maxlen, args.max_tokens)

    with tempfile.TemporaryDirectory(prefix="musvit-omr-") as temp:
        pages = _score_pages(source, Path(temp), args.pdf_dpi)
        selected = [(index, path) for index, path in enumerate(pages, start=1)]
        if args.page is not None:
            selected = [item for item in selected if item[0] == args.page]
            if not selected:
                raise SystemExit(f"PDF has no page {args.page}")

        for page_number, page_path in selected:
            destination = _output_for(source, args.output, page_number, len(selected))
            destination.parent.mkdir(parents=True, exist_ok=True)
            image = _fit_page(page_path, model.config.maxh, model.config.maxw).to(
                device=args.device, dtype=dtype
            )
            print(f"Transcribing page {page_number}: {tuple(image.shape)}")
            with torch.inference_mode():
                tokens, _ = model.predict(image, convert_to_str=True)
            kern = _decode_bekern(tokens)
            _write_derivatives(kern, destination, write_svg=not args.no_svg)
            print(f"Saved {destination}, {destination.with_suffix('.krn')}" )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
