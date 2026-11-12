#!/usr/bin/env python3
"""Local workstation entry point for audio -> MIDI.

The constants below are intentionally small and editable. Command-line flags
override them, while .env owns machine-specific paths. Input is normalized by
ffmpeg first, so MP3, WAV, M4A, FLAC, OGG, and video containers behave alike.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from muscriptor.tokenizer.mt3 import resolve_instrument_names
from muscriptor.transcription_model import TranscriptionModel

# Easy defaults for direct `uv run --no-sync python local_transcribe.py ...` use.
DEFAULT_DEVICE = os.environ.get("MUSCRIPTOR_DEVICE", "cuda")
DEFAULT_DTYPE = os.environ.get("MUSCRIPTOR_DTYPE") or None
DEFAULT_INSTRUMENTS = ""  # e.g. "piano,drums,bass"
DEFAULT_DETECT_TEMPO = "best-effort"
DEFAULT_BEAM_SIZE = 1
DEFAULT_PRELUDE_FORCING = True
DEFAULT_BATCH_SIZE: int | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe any ffmpeg-readable audio file into MIDI locally."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("MUSCRIPTOR_MODEL_PATH", "")),
        help="Local model.safetensors path (defaults to MUSCRIPTOR_MODEL_PATH).",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument(
        "--instruments",
        default=DEFAULT_INSTRUMENTS,
        help="Optional comma-separated hard constraint, such as piano,drums,bass.",
    )
    parser.add_argument(
        "--detect-tempo",
        choices=("true", "false", "best-effort"),
        default=DEFAULT_DETECT_TEMPO,
    )
    parser.add_argument("--beam-size", type=int, default=DEFAULT_BEAM_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--no-prelude-forcing",
        action="store_true",
        help="Permit batched chunks at the cost of chunk-boundary quality.",
    )
    return parser


def _normalize_audio(source: Path, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is required on PATH")
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    subprocess.run(command, check=True)


def main() -> int:
    args = _parser().parse_args()
    source = args.input.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Input file not found: {source}")
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(
            "Local model not found. Set MUSCRIPTOR_MODEL_PATH in .env or pass --model."
        )
    if args.beam_size < 1:
        raise SystemExit("--beam-size must be at least 1")
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    prelude_forcing = not args.no_prelude_forcing
    if prelude_forcing and args.batch_size not in (None, 1):
        raise SystemExit("--batch-size > 1 requires --no-prelude-forcing")

    instruments = None
    if args.instruments.strip():
        instruments = resolve_instrument_names(
            [item.strip() for item in args.instruments.split(",") if item.strip()]
        )

    output = (args.output or source.with_suffix(".mid")).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    tempo_mode: bool | str = {
        "true": True,
        "false": False,
        "best-effort": "best-effort",
    }[args.detect_tempo]

    print(f"Loading MuScriptor from {model_path}")
    model = TranscriptionModel.load_model(
        weights_path=model_path,
        device=args.device,
        dtype=args.dtype,
    )
    with tempfile.TemporaryDirectory(prefix="muscriptor-") as temp_dir:
        normalized = Path(temp_dir) / "input.wav"
        print(f"Normalizing {source.name} with ffmpeg")
        _normalize_audio(source, normalized)
        print("Transcribing; the large model is optimized for accuracy, not latency")
        midi = model.transcribe_to_midi(
            audio=normalized,
            instruments=instruments,
            batch_size=args.batch_size,
            no_eos_is_ok=True,
            beam_size=args.beam_size,
            prelude_forcing=prelude_forcing,
            detect_tempo=tempo_mode,
        )
    output.write_bytes(midi)
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
