"""Load the self-contained online video rewriter system prompt."""

from __future__ import annotations

import argparse
from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parent
BASE_SYSTEM_PROMPT_PATH = PROMPT_DIR / "online_user_video_rewriter_system_prompt.md"


def load_system_prompt(
    base_path: Path = BASE_SYSTEM_PROMPT_PATH,
) -> str:
    """Read the self-contained system prompt without external ICL composition."""
    prompt = base_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"system prompt is empty: {base_path}")
    return prompt


# Import this symbol from inference code that needs the fully composed prompt.
SYSTEM_PROMPT = load_system_prompt()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print or materialize the online video rewriter system prompt."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional generated output path. The base prompt cannot be overwritten.",
    )
    args = parser.parse_args()

    prompt = load_system_prompt()
    if args.output is None:
        print(prompt)
        return 0

    output = args.output.resolve()
    if output == BASE_SYSTEM_PROMPT_PATH.resolve():
        parser.error("refusing to overwrite the base system prompt")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{prompt}\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
