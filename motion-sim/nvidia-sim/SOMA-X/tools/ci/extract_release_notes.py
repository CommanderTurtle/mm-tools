#!/usr/bin/env python3

"""Extract release notes for a tag from CHANGELOG.md."""

import argparse
import sys
from pathlib import Path


def extract_release_notes(changelog: Path, tag: str) -> str:
    heading = f"## {tag}"
    lines = changelog.read_text(encoding="utf-8").splitlines()

    start = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == heading or stripped.startswith(f"{heading} "):
            start = index + 1
            break
    if start is None:
        raise ValueError(f"Could not find changelog heading: {heading}")

    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break

    notes = "\n".join(lines[start:end]).strip()
    if not notes:
        raise ValueError(f"Changelog section is empty: {heading}")
    return notes + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Release tag, for example v0.2.0.")
    parser.add_argument(
        "--changelog",
        type=Path,
        default=Path("CHANGELOG.md"),
        help="Path to CHANGELOG.md.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write notes to this file instead of stdout.",
    )
    args = parser.parse_args()

    try:
        notes = extract_release_notes(args.changelog, args.tag)
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.output:
        args.output.write_text(notes, encoding="utf-8")
    else:
        print(notes, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
