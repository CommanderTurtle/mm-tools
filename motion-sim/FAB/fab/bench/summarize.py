"""Merge eval_hmr shards into one metrics table.

    python -m bench.summarize --results $FAB_HOME/bench/results --out $FAB_HOME/bench/results/leaderboard.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ["pa_mpjpe", "mpjpe", "pve", "pa_mpjpe24", "mpjpe24"]
# curated hard subsets of a benchmark: (subset name, source dataset, sequence-name prefixes of the sample ids)
SUBSETS = [
    ("emdb1_hard", "emdb1", ("P8/68_outdoor_handstand", "P8/69_outdoor_cartwheel", "P5/42_indoor_dancing",
                             "P6/49_outdoor_big_stairs_down", "P6/50_outdoor_workout")),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="dir of <method>/<dataset>_shard*.npz")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    table = defaultdict(dict)
    for method_dir in sorted(p for p in Path(args.results).iterdir() if p.is_dir()):
        shards = defaultdict(list)
        for f in method_dir.glob("*_shard*.npz"):
            if "_pred_shard" in f.name:  # --save_pred arrays, not metrics
                continue
            shards[f.name.rsplit("_shard", 1)[0]].append(np.load(f))
        for dataset, parts in shards.items():
            merged = {k: np.concatenate([p[k] for p in parts]) for k in METRICS + ["ms", "ids"]}
            groups = [(dataset, np.ones(len(merged["ids"]), bool))]
            for name, src, prefixes in SUBSETS:
                if src == dataset.split("_dark")[0] and len(parts) == len(list(method_dir.glob(f"{dataset}_shard*.npz"))):
                    sub = name + dataset[len(src):]
                    groups.append((sub, np.array([str(i).startswith(prefixes) for i in merged["ids"]])))
            for name, m in groups:
                if not m.any():
                    continue
                row = {k: float(np.nanmean(merged[k][m])) for k in METRICS}
                row["n"] = int(m.sum())
                row["ms_median"] = float(np.median(merged["ms"][m]))
                table[name][method_dir.name] = row

    lines = []
    for dataset, methods in sorted(table.items()):
        lines += [f"## {dataset}", "", "| method | n | PA-MPJPE | MPJPE | PVE | PA-MPJPE24 | MPJPE24 | ms (median) |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, r in sorted(methods.items(), key=lambda kv: kv[1]["pa_mpjpe"]):
            lines.append(f"| {name} | {r['n']} | {r['pa_mpjpe']:.1f} | {r['mpjpe']:.1f} | {r['pve']:.1f} | "
                         f"{r['pa_mpjpe24']:.1f} | {r['mpjpe24']:.1f} | {r['ms_median']:.1f} |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        Path(args.out).with_suffix(".json").write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
