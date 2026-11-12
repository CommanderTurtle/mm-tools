#!/usr/bin/env python3
"""Build an offline token-packing plan for one public SFT Arrow/Parquet file.

The runtime collator concatenates flattened video latents along its sequence
axis, so one row's packing cost is exactly ``T * H * W`` from
``latent_shape=[C,T,H,W]``. Text tokens are stored in a separate padded tensor
and are therefore not part of ``total_packed_tokens`` or ``max_seq_len``.

The output JSON is consumed directly by
``dataset.token_packing_sorted_groups_path``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def read_latent_shapes(path: Path) -> list[list[int]]:
    shapes: list[list[int]] = []
    if path.suffix == ".arrow":
        with pa.memory_map(str(path), "r") as source:
            reader = pa.ipc.open_file(source)
            if "latent_shape" not in reader.schema.names:
                raise ValueError("Arrow file is missing latent_shape")
            column_index = reader.schema.get_field_index("latent_shape")
            for index in range(reader.num_record_batches):
                shapes.extend(reader.get_batch(index).column(column_index).to_pylist())
    elif path.suffix == ".parquet":
        table = pq.read_table(path, columns=["latent_shape"])
        shapes = table.column("latent_shape").to_pylist()
    else:
        raise ValueError("input must end in .arrow or .parquet")

    for index, shape in enumerate(shapes):
        if (
            not isinstance(shape, list)
            or len(shape) != 4
            or any(not isinstance(value, int) or value <= 0 for value in shape)
        ):
            raise ValueError(f"row {index}: invalid latent_shape {shape!r}")
    if not shapes:
        raise ValueError("input dataset has no rows")
    return shapes


def sample_tokens(shapes: list[list[int]], patch_size: tuple[int, int, int]):
    pt, ph, pw = patch_size
    costs = []
    for index, (_, frames, height, width) in enumerate(shapes):
        if frames % pt or height % ph or width % pw:
            raise ValueError(
                f"row {index}: latent shape {(frames, height, width)} is not "
                f"divisible by patch size {patch_size}"
            )
        costs.append((frames // pt) * (height // ph) * (width // pw))
    return costs


def ordered_indices(costs: list[int], seed: int) -> list[int]:
    # Seeded tie-breaking makes equal-cost rows reproducible without preserving
    # source-order clustering.
    rng = random.Random(seed)
    ties = [rng.random() for _ in costs]
    return sorted(range(len(costs)), key=lambda i: (-costs[i], ties[i], i))


def first_fit_decreasing(costs: list[int], max_seq_len: int, seed: int):
    groups: list[list[int]] = []
    totals: list[int] = []
    for sample_index in ordered_indices(costs, seed):
        cost = costs[sample_index]
        for group_index, total in enumerate(totals):
            if total + cost <= max_seq_len:
                groups[group_index].append(sample_index)
                totals[group_index] += cost
                break
        else:
            groups.append([sample_index])
            totals.append(cost)
    return groups, totals


def balanced_fixed_groups(
    costs: list[int], max_seq_len: int, num_groups: int, seed: int
):
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if num_groups > len(costs):
        raise ValueError("num_groups cannot exceed the number of samples")
    if sum(costs) > num_groups * max_seq_len:
        raise ValueError(
            f"total tokens {sum(costs)} exceed fixed capacity "
            f"{num_groups} * {max_seq_len}"
        )

    def score(totals):
        overflow = sum(max(0, total - max_seq_len) for total in totals)
        average = sum(totals) / len(totals)
        variance = sum((total - average) ** 2 for total in totals)
        return overflow, max(totals), variance

    # LPT produces a very even initial assignment, but with a tight capacity
    # it can leave several bins slightly over the limit. Deterministic local
    # moves and pair swaps repair those bins while preserving exactly the
    # requested group count. Seeded tie ordering provides reproducible restarts.
    for attempt in range(32):
        attempt_seed = seed + attempt
        rng = random.Random(attempt_seed)
        groups = [[] for _ in range(num_groups)]
        totals = [0] * num_groups
        for sample_index in ordered_indices(costs, attempt_seed):
            group_index = min(
                range(num_groups), key=lambda index: (totals[index], index)
            )
            groups[group_index].append(sample_index)
            totals[group_index] += costs[sample_index]

        current_score = score(totals)
        for _iteration in range(10_000):
            if current_score[0] == 0:
                if any(not group for group in groups):
                    raise AssertionError("fixed-group packing produced an empty group")
                return groups, totals

            improved = False
            overloaded = sorted(
                (index for index, total in enumerate(totals) if total > max_seq_len),
                key=lambda index: (-totals[index], index),
            )
            for source_group in overloaded:
                source_items = list(groups[source_group])
                rng.shuffle(source_items)
                source_items.sort(key=lambda index: costs[index], reverse=True)
                destination_groups = sorted(
                    range(num_groups), key=lambda index: (totals[index], index)
                )
                for source_item in source_items:
                    source_cost = costs[source_item]

                    # First try the cheaper operation: move one item.
                    for destination_group in destination_groups:
                        if destination_group == source_group:
                            continue
                        candidate_totals = list(totals)
                        candidate_totals[source_group] -= source_cost
                        candidate_totals[destination_group] += source_cost
                        candidate_score = score(candidate_totals)
                        if candidate_score < current_score:
                            groups[source_group].remove(source_item)
                            groups[destination_group].append(source_item)
                            totals = candidate_totals
                            current_score = candidate_score
                            improved = True
                            break
                    if improved:
                        break

                    # If no direct move helps, exchange one item with another
                    # group. Shuffle equal-cost candidates reproducibly so
                    # restarts explore different local-search paths.
                    for destination_group in destination_groups:
                        if destination_group == source_group:
                            continue
                        destination_items = list(groups[destination_group])
                        rng.shuffle(destination_items)
                        for destination_item in destination_items:
                            destination_cost = costs[destination_item]
                            candidate_totals = list(totals)
                            candidate_totals[source_group] += (
                                destination_cost - source_cost
                            )
                            candidate_totals[destination_group] += (
                                source_cost - destination_cost
                            )
                            candidate_score = score(candidate_totals)
                            if candidate_score < current_score:
                                groups[source_group].remove(source_item)
                                groups[destination_group].remove(destination_item)
                                groups[source_group].append(destination_item)
                                groups[destination_group].append(source_item)
                                totals = candidate_totals
                                current_score = candidate_score
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

    raise ValueError(
        f"could not construct exactly {num_groups} groups below "
        f"{max_seq_len} tokens after 32 deterministic local-search restarts"
    )


def validate_plan(
    groups: list[list[int]], totals: list[int], costs: list[int], max_seq_len: int
):
    if len(groups) != len(totals):
        raise AssertionError("group/token-total count mismatch")
    flat = [sample for group in groups for sample in group]
    expected = list(range(len(costs)))
    if sorted(flat) != expected:
        raise AssertionError("packing plan must contain every row exactly once")
    for index, (group, total) in enumerate(zip(groups, totals)):
        actual = sum(costs[sample] for sample in group)
        if actual != total:
            raise AssertionError(f"group {index}: token total mismatch")
        if total > max_seq_len:
            raise AssertionError(f"group {index}: {total} exceeds {max_seq_len}")


def atomic_write_json(payload: dict, output: Path):
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=360_000)
    parser.add_argument(
        "--num-groups",
        type=int,
        default=None,
        help="Produce exactly this many balanced groups; omit for FFD minimum bins.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(1, 1, 1))
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    shapes = read_latent_shapes(input_path)
    costs = sample_tokens(shapes, tuple(args.patch_size))
    if max(costs) > args.max_seq_len:
        row = max(range(len(costs)), key=costs.__getitem__)
        raise ValueError(
            f"row {row} has {costs[row]} tokens, exceeding {args.max_seq_len}"
        )

    if args.num_groups is None:
        groups, totals = first_fit_decreasing(costs, args.max_seq_len, args.seed)
        strategy = "first_fit_decreasing"
    else:
        groups, totals = balanced_fixed_groups(
            costs, args.max_seq_len, args.num_groups, args.seed
        )
        strategy = "lpt_balanced_local_search"
    validate_plan(groups, totals, costs, args.max_seq_len)

    payload = {
        "groups": groups,
        "group_tokens": totals,
        "num_samples": len(costs),
        "num_groups": len(groups),
        "config": {
            "input": str(input_path),
            "max_seq_len": args.max_seq_len,
            "patch_size": list(args.patch_size),
            "strategy": strategy,
            "heuristic": "latent_token_count",
            "seed": args.seed,
            "num_groups_requested": args.num_groups,
        },
    }
    atomic_write_json(payload, output_path)

    print(f"output: {output_path}")
    print(
        f"samples={len(costs)} groups={len(groups)} "
        f"sample_tokens[min/mean/max]={min(costs)}/{sum(costs)/len(costs):.1f}/{max(costs)} "
        f"group_tokens[min/mean/max]={min(totals)}/{sum(totals)/len(totals):.1f}/{max(totals)}"
    )


if __name__ == "__main__":
    main()
