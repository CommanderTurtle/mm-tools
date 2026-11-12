from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import soundfile as sf

from .core import LongCatEngine, SynthesisOptions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local LongCat 3.5B text-to-speech")
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--prompt-audio", type=Path)
    parser.add_argument("--prompt-text")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--guidance-strength", type=float, default=4.0)
    parser.add_argument("--guidance-method", choices=("cfg", "apg"), default="apg")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if bool(args.text) == bool(args.text_file):
        raise SystemExit("Provide either positional text or --text-file")
    text = args.text or args.text_file.expanduser().read_text(encoding="utf-8")
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = Path(os.environ.get("LONGCAT_OUTPUT_DIR", "outputs")) / f"longcat-{stamp}.wav"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    options = SynthesisOptions(
        steps=args.steps,
        guidance_strength=args.guidance_strength,
        guidance_method=args.guidance_method,
        seed=args.seed,
        duration_scale=args.duration_scale,
    )
    result = LongCatEngine().synthesize(
        text,
        prompt_audio=args.prompt_audio,
        prompt_text=args.prompt_text,
        options=options,
    )
    sf.write(output, result.waveform, result.sample_rate, subtype="PCM_24")
    print(f"Saved {output}")
    print(
        f"Audio {result.audio_seconds:.2f}s; generation {result.generation_seconds:.2f}s; "
        f"{args.guidance_method.upper()} {args.steps} steps"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
