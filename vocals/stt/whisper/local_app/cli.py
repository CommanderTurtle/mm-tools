from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .core import manager, normalize_audio


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Offline CrisperWhisper 2.0 transcription")
    command.add_argument("input", type=Path, help="MP3, WAV, M4A, or any ffmpeg-supported audio")
    command.add_argument("--operation", choices=("both", "verbatim", "intended", "verbatimize", "align"), default="both")
    command.add_argument("--language", default="en")
    command.add_argument("--transcript", default="", help="Required for verbatimize or align")
    command.add_argument("--timestamps", action=argparse.BooleanOptionalAction, default=True)
    command.add_argument("--strategy", choices=("continuation", "chunked_lcs", "token_lcs"), default="continuation")
    command.add_argument("--chunk-duration", type=float, default=30.0)
    command.add_argument("--stride", type=float, default=26.0)
    command.add_argument("--context-words", type=int, default=12)
    command.add_argument("--max-new-tokens", type=int, default=256)
    command.add_argument("--hotword", action="append", default=[])
    command.add_argument("--output", type=Path)
    return command


def main() -> None:
    args = parser().parse_args()
    source = args.input.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Input does not exist: {source}")
    with tempfile.TemporaryDirectory(prefix="cw2-") as temp:
        wav = normalize_audio(source, Path(temp) / "audio.wav")
        results = manager.run(
            wav,
            operation=args.operation,
            language=args.language,
            transcript=args.transcript,
            word_timestamps=args.timestamps,
            strategy=args.strategy,
            chunk_duration=args.chunk_duration,
            stride=args.stride,
            context_words=args.context_words,
            max_new_tokens=args.max_new_tokens,
            hotwords=args.hotword,
        )
    payload = json.dumps({"source": str(source), "results": results}, ensure_ascii=False, indent=2)
    if args.output:
        args.output.expanduser().write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
