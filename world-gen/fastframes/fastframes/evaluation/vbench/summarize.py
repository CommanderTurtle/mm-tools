#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Combine VBench results into the official Quality, Semantic, and Total scores.

Each raw dimension score is min-max normalized with VBench's constants,
weighted (dynamic degree 0.5, others 1), averaged within the quality and
semantic groups, and Total = (4 * Quality + 1 * Semantic) / 5.

    python summarize.py --non-temporal RUN/vbench_results/non_temporal/*_eval_results.json \
        --temporal RUN/vbench_results/temporal_flickering/*_eval_results.json \
        --filter-contract RUN/temporal_filter_contract.json --output RUN/score_summary.json

    python summarize.py --from-summary score_summary.json   # recompute from recorded raw scores
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import protocol

VBENCH_ROOT = Path(__file__).resolve().parent / "VBench"


def _load_constants():
    spec = importlib.util.spec_from_file_location("vbench_constant", VBENCH_ROOT / "scripts" / "constant.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = _load_constants()
TEMPORAL = {"temporal_flickering"}
NON_TEMPORAL = {name.replace(" ", "_") for name in C.TASK_INFO} - TEMPORAL


def load_result(path: Path, expected: set[str]) -> dict[str, float]:
    """Read one ``*_eval_results.json`` holding exactly the ``expected`` dimensions."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(payload) != expected:
        raise RuntimeError(f"dimension set mismatch in {path}: missing={sorted(expected - set(payload))} "
                           f"extra={sorted(set(payload) - expected)}")
    scores = {}
    for dimension, value in payload.items():
        if not isinstance(value, list) or len(value) < 2:
            raise RuntimeError(f"malformed result for {dimension} in {path}")
        scores[dimension.replace("_", " ")] = float(value[0])
    return scores


def normalize(raw: dict[str, float], dimension: str) -> float:
    bounds = C.NORMALIZE_DIC[dimension]
    return (raw[dimension] - bounds["Min"]) / (bounds["Max"] - bounds["Min"])


def weighted(raw: dict[str, float], dimensions: list[str]) -> float:
    # Plain left-to-right accumulation (Python >= 3.12 `sum` uses compensated
    # summation, which can change the last bit).
    numerator = denominator = 0.0
    for dimension in dimensions:
        numerator += normalize(raw, dimension) * C.DIM_WEIGHT[dimension]
        denominator += C.DIM_WEIGHT[dimension]
    return numerator / denominator


def aggregate(raw: dict[str, float]) -> dict:
    """Official VBench aggregation of the 16 raw dimension scores (fractions in [0, 1])."""
    if set(raw) != set(C.TASK_INFO):
        raise ValueError(f"expected the 16 VBench dimensions, got {sorted(raw)}")
    quality = weighted(raw, C.QUALITY_LIST)
    semantic = weighted(raw, C.SEMANTIC_LIST)
    total = (quality * C.QUALITY_WEIGHT + semantic * C.SEMANTIC_WEIGHT) / (C.QUALITY_WEIGHT + C.SEMANTIC_WEIGHT)
    return {
        "raw_scores": dict(raw),
        "normalized_scores": {d: normalize(raw, d) for d in C.TASK_INFO},
        "quality_score": quality,
        "semantic_score": semantic,
        "total_score": total,
    }


def scorer_tree_sha256(root: Path = VBENCH_ROOT) -> str:
    """Hash of (relative path, sha256) over evaluate.py, scripts/constant.py, and vbench/**/*.py."""
    if not (root / "vbench").is_dir():
        raise FileNotFoundError(f"{root / 'vbench'} is missing; build it with setup_scorer.py")
    paths = sorted([root / "evaluate.py", root / "scripts" / "constant.py"] + list((root / "vbench").rglob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(f"{path.relative_to(root)}\0{protocol.sha256_file(path)}\n".encode("utf-8"))
    return digest.hexdigest()


def filter_coverage(contract_path: Path) -> dict:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    selections = contract["selection_by_prompt"]
    count = sum(len(rows) for rows in selections.values())
    if len(selections) != protocol.EXPECTED_TEMPORAL_PROMPTS or count != contract["output_video_count"]:
        raise RuntimeError("temporal filter contract coverage mismatch")
    if not 0 < count <= protocol.EXPECTED_TEMPORAL_FILTERED_VIDEOS or any(
        len(rows) > protocol.BASE_SAMPLES for rows in selections.values()
    ):
        raise RuntimeError("temporal filter contract is outside official bounds")
    return {
        "selected_video_count": count,
        "full_coverage_prompt_count": sum(len(r) == protocol.BASE_SAMPLES for r in selections.values()),
        "shortfall_prompt_count": sum(len(r) < protocol.BASE_SAMPLES for r in selections.values()),
        "zero_static_prompt_count": sum(len(r) == 0 for r in selections.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--non-temporal", type=Path, help="VBench result for the 15 non-temporal dimensions")
    parser.add_argument("--temporal", type=Path, help="VBench result for temporal_flickering")
    parser.add_argument("--filter-contract", type=Path)
    parser.add_argument("--from-summary", type=Path, help="recompute from the raw_scores of a score summary")
    parser.add_argument("--model", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.from_summary:
        raw = json.loads(args.from_summary.read_text(encoding="utf-8"))["raw_scores"]
        payload = {"source": str(args.from_summary), **aggregate(raw)}
    else:
        if args.non_temporal is None or args.temporal is None:
            parser.error("--non-temporal and --temporal are required")
        raw = load_result(args.non_temporal, NON_TEMPORAL)
        raw.update(load_result(args.temporal, TEMPORAL))
        payload = {
            "model": args.model,
            "inputs": {label: {"path": str(path), "sha256": protocol.sha256_file(path)}
                       for label, path in (("non_temporal", args.non_temporal), ("temporal", args.temporal),
                                           ("filter_contract", args.filter_contract)) if path is not None},
            "scorer_tree_sha256": scorer_tree_sha256(),
            **aggregate(raw),
        }
        if args.filter_contract is not None:
            payload["temporal_filter_coverage"] = filter_coverage(args.filter_contract)
    if args.output:
        protocol.atomic_json(args.output, payload)
    q, s, t = (100 * payload[k] for k in ("quality_score", "semantic_score", "total_score"))
    print(f"{args.model or 'scores'}: total={t:.2f} quality={q:.2f} semantic={s:.2f} "
          f"(fractions: total={payload['total_score']:.8f} quality={payload['quality_score']:.8f} "
          f"semantic={payload['semantic_score']:.8f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
