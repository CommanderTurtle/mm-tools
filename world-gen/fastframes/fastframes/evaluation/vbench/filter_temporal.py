#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""RAFT static filter for the temporal-flickering candidates.

For each of the 75 temporal-flickering prompts, candidates 0..24 are checked in
numeric order with VBench's RAFT static filter and the first (up to) five
static clips are copied to ``<prompt>-0.mp4`` .. ``<prompt>-4.mp4``. Prompts
are sharded across GPUs; ``--merge-only`` validates the shards and writes the
filter contract consumed by ``score.sh`` and ``summarize.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import protocol

VBENCH_ROOT = Path(__file__).resolve().parent / "VBench"
FILTER_POLICY = "upstream_vbench_raft_static_filter_first_up_to_five_v2"


def _atomic_copy(source: Path, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def temporal_prompts(suite: protocol.PromptSuite) -> list[str]:
    return [prompt for prompt in suite.prompts if prompt in suite.temporal_prompts]


def filter_shard(args: argparse.Namespace, suite: protocol.PromptSuite) -> None:
    if not 0 <= args.shard < args.num_shards:
        raise ValueError("invalid shard index")
    sys.path.insert(0, str(VBENCH_ROOT))
    from static_filter import StaticFilter

    seeds = {job.filename: job.seed for job in protocol.build_sample_specs(suite) if job.temporal_flickering}
    prompts = temporal_prompts(suite)[args.shard :: args.num_shards]
    # The RAFT code probes optional fields with `'name' in args`, so this must
    # be an argparse.Namespace.
    raft_args = argparse.Namespace(model=str(args.model), small=False, mixed_precision=False, alternate_corr=False)
    static_filter = StaticFilter(raft_args, device="cuda")
    selections = []
    for index, prompt in enumerate(prompts, start=1):
        selected = []
        for sample_index in range(protocol.TEMPORAL_CANDIDATES):
            source = args.raw_dir / f"{prompt}-{sample_index}.mp4"
            if not source.is_file():
                raise FileNotFoundError(source)
            if static_filter.infer(str(source)):
                target = args.filtered_dir / f"{prompt}-{len(selected)}.mp4"
                _atomic_copy(source, target)
                selected.append({
                    "output_filename": target.name,
                    "source_filename": source.name,
                    "source_sample_index": sample_index,
                    "source_seed": seeds[source.name],
                    "source_sha256": protocol.sha256_file(source),
                })
                if len(selected) == protocol.BASE_SAMPLES:
                    break
        selections.append({"prompt": prompt, "selected": selected})
        print(f"[shard {args.shard}] {index}/{len(prompts)} selected={len(selected)} "
              f"sources={[row['source_sample_index'] for row in selected]}", flush=True)
    protocol.atomic_json(args.shard_output, {
        "shard": args.shard,
        "num_shards": args.num_shards,
        "raft_model_sha256": protocol.sha256_file(args.model),
        "prompts": selections,
    })


def merge(args: argparse.Namespace, suite: protocol.PromptSuite) -> None:
    merged: dict[str, list] = {}
    raft_hashes = set()
    for shard in range(args.num_shards):
        payload = json.loads((args.shard_dir / f"shard_{shard}.json").read_text(encoding="utf-8"))
        if payload["shard"] != shard or payload["num_shards"] != args.num_shards:
            raise RuntimeError(f"shard identity mismatch for shard {shard}")
        raft_hashes.add(payload["raft_model_sha256"])
        for item in payload["prompts"]:
            if item["prompt"] in merged:
                raise RuntimeError(f"duplicate filtered prompt: {item['prompt']}")
            merged[item["prompt"]] = item["selected"]
    if set(merged) != set(suite.temporal_prompts) or len(raft_hashes) != 1:
        raise RuntimeError(f"filter merge mismatch: prompts={len(merged)} raft_hashes={raft_hashes}")
    contract = {
        "protocol": FILTER_POLICY,
        "prompt_count": len(merged),
        "output_video_count": sum(len(rows) for rows in merged.values()),
        "full_coverage_prompt_count": sum(len(rows) == protocol.BASE_SAMPLES for rows in merged.values()),
        "shortfall_prompt_count": sum(len(rows) < protocol.BASE_SAMPLES for rows in merged.values()),
        "zero_static_prompt_count": sum(len(rows) == 0 for rows in merged.values()),
        "raft_model_sha256": raft_hashes.pop(),
        "selection_by_prompt": merged,
        "vbench": protocol.protocol_summary(suite),
    }
    expected = protocol.expected_filtered_names(suite, contract)
    observed = {p.name for p in args.filtered_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"}
    if observed != expected:
        raise RuntimeError(f"filtered outputs mismatch: missing={len(expected - observed)} "
                           f"extra={len(observed - expected)}")
    protocol.atomic_json(args.output, contract)
    print(f"PASS: {contract['prompt_count']} prompts, {contract['output_video_count']} filtered videos "
          f"(full={contract['full_coverage_prompt_count']}, shortfall={contract['shortfall_prompt_count']}, "
          f"zero={contract['zero_static_prompt_count']}) -> {args.output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--filtered-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="raft-things.pth")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-output", type=Path)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--shard-dir", type=Path)
    parser.add_argument("--output", type=Path, help="temporal_filter_contract.json")
    protocol.add_suite_arguments(parser)
    args = parser.parse_args()
    if args.num_shards <= 0:
        parser.error("--num-shards must be positive")
    if args.merge_only and (args.shard_dir is None or args.output is None):
        parser.error("--merge-only requires --shard-dir and --output")
    if not args.merge_only and (args.model is None or args.shard_output is None):
        parser.error("filtering requires --model and --shard-output")
    for name in ("raw_dir", "filtered_dir", "model", "shard_output", "shard_dir", "output"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    args.filtered_dir.mkdir(parents=True, exist_ok=True)
    suite = protocol.load_suite(args.protocol, args.full_info, args.original_prompts, args.extended_prompts)
    (merge if args.merge_only else filter_shard)(args, suite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
