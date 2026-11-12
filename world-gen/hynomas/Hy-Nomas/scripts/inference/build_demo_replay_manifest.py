#!/usr/bin/env python3
"""Build self-contained recorded-parameter replay manifests (no GPU required)."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from apps.seg_ar_replay import build_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.input)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"Wrote {len(manifest['sessions'])} sessions, "
        f"{sum(len(s['turns']) for s in manifest['sessions'])} original turns "
        f"to {args.output}"
    )
    print("Historical checkpoint/config/SP identity is NOT verified by this manifest.")


if __name__ == "__main__":
    main()
