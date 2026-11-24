#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Fig. 5 (LingBot FastFrames rows) from the JSON records of ``latency.py``.

  python apps/lingbot/summarize.py outputs/fig5/ [--output outputs/fig5/fig5.json]

Each row is the median of the per-process medians; fig5.sh runs three fresh processes per row.
"""
import argparse
import json
import statistics
from pathlib import Path

ROWS = [  # (model, variant, GPUs, paper ms, paper label)
    ("lingbot-1.3b", "fastframes", 1, 224, "w/ FastFrames - 1 GPU"),
    ("lingbot-1.3b", "fastframes", 8, 52, "w/ FastFrames - 8 GPU"),
]
NOTE = "Measurements use NVIDIA B200 GPUs."


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = {}
    for path in sorted(args.results.rglob("*.json")):
        try:
            r = json.loads(path.read_text())
        except ValueError:
            continue
        if isinstance(r, dict) and {"model", "variant", "gpus", "model_ms"} <= r.keys():
            records.setdefault((r["model"], r["variant"], r["gpus"]), []).append((str(path), r["model_ms"]))
    rows = []
    print(f"{'model':13s} {'configuration':30s} {'paper':>7s} {'measured':>9s}  processes")
    for model, variant, gpus, paper, label in ROWS:
        found = records.get((model, variant, gpus), [])
        measured = statistics.median(ms for _, ms in found) if found else None
        rows.append(dict(model=model, configuration=label, gpus=gpus, paper_ms=paper, measured_ms=measured,
                         process_ms=[ms for _, ms in found], sources=[p for p, _ in found]))
        shown = f"{measured:9.1f}" if measured is not None else "        -"
        print(f"{model:13s} {label:30s} {paper:7d} {shown}  {len(found)}")
    print(NOTE)
    if args.output:
        args.output.write_text(json.dumps(dict(rows=rows, note=NOTE), indent=2) + "\n")


if __name__ == "__main__":
    main()
