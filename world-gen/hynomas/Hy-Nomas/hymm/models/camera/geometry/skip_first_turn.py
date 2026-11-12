"""Build a rank-local input view omitting the first generation turn.

Source JSON/NPZ files are never modified. Remaining precomputed camera turns
are sliced without resampling; frame boundaries and captions are rebased.
"""

from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path

import numpy as np

from .demo_precomputed import FORMAT, load_precomputed


def seconds(value):
    parts = str(value).split(":")
    result = Decimal(0)
    for part in parts:
        result = result * 60 + Decimal(part)
    return result


def timestamp(value):
    ms = int((value * 1000).to_integral_value())
    if ms < 0:
        raise ValueError("Negative timestamp after removing first turn")
    minutes, rest = divmod(ms, 60000)
    sec, millis = divmod(rest, 1000)
    return f"{minutes:02d}:{sec:02d}.{millis:03d}"


def rebase(value, offset):
    if isinstance(value, dict):
        return {
            k: timestamp(seconds(v) - offset)
            if k in ("start_time", "end_time") and v is not None
            else rebase(v, offset)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [rebase(v, offset) for v in value]
    return value


def slice_npz(data):
    bounds = data["segment_boundaries_pixel"].tolist()
    flags = json.loads(str(data["normalization_json"].item()))
    load_precomputed(data, bounds[-1], bounds, flags)
    if len(bounds) < 3:
        raise ValueError("skip_first_turn requires at least two turns")
    result = {k: data[k].copy() for k in data.files}
    offset = int(data["turn_camera_offsets"][1])
    for k in ("norm_extrinsics", "norm_intrinsics", "action_labels"):
        result[k] = result[k][offset:]
    result["turn_camera_offsets"] = result["turn_camera_offsets"][1:] - offset
    result["segment_boundaries_pixel"] = (
        result["segment_boundaries_pixel"][1:] - bounds[1]
    )
    for k in ("turn_num_frames", "turn_has_camera", "metadata_sha256"):
        if k in result:
            result[k] = result[k][1:]
    new_bounds = result["segment_boundaries_pixel"].tolist()
    load_precomputed(result, new_bounds[-1], new_bounds, flags)
    return result


def prepare_skipped_input(source, destination, fps):
    source, destination = Path(source).resolve(), Path(destination)
    if not source.is_file() or source.suffix.lower() != ".json":
        raise ValueError("skip_first_turn currently requires a benchmark JSON file")
    entries = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("skip_first_turn requires a nonempty benchmark array")
    result = []
    for index, original in enumerate(entries):
        entry = deepcopy(original)
        segments = entry.get("segments")
        if not isinstance(segments, list) or len(segments) < 2:
            raise ValueError(
                f"Entry {index}: skip_first_turn requires at least two segments"
            )
        first, remaining = segments[0], segments[1:]
        offset = seconds(remaining[0]["start_time"])
        entry["segments"] = rebase(remaining, offset)
        # Preserve static environment context, not the omitted turn's actions.
        if not entry["segments"][0].get("visual_description_long"):
            entry["segments"][0]["visual_description_long"] = first.get(
                "visual_description_long"
            )
        entry["total_duration"] = entry["segments"][-1]["end_time"]
        for i, segment in enumerate(entry["segments"]):
            if "segment_index" in segment:
                segment["segment_index"] = i
        spec = entry.get("camera_trajectory_spec")
        if spec:
            params = spec.get("params") or {}
            if params.get("format") != FORMAT:
                raise ValueError(
                    "skip_first_turn with camera control requires precomputed demo NPZ"
                )
            if spec.get("fps") != fps:
                raise ValueError("skip_first_turn camera FPS mismatch")
            with np.load(
                source.parent / spec["ground_truth_npz"], allow_pickle=False
            ) as data:
                arrays = slice_npz(data)
            target = destination / f"camera_{index:05d}.npz"
            np.savez_compressed(target, **arrays)
            spec["ground_truth_npz"] = str(target.resolve())
            bounds = arrays["segment_boundaries_pixel"].tolist()
            params["segment_boundaries_pixel"] = bounds
            for k in ("original_turn_frames", "right_padding_frames"):
                params[k] = params[k][1:]
            params["original_total_frames"] = sum(params["original_turn_frames"])
            params["offline_total_frames"] = bounds[-1]
            params["skipped_first_turn"] = True
            params["initialization_kv_available"] = False
            joint = entry["joint_control_spec"]
            joint["segment_boundaries_pixel"] = bounds
            joint["segments"] = [
                {
                    **rebase(old, offset),
                    "segment_index": i,
                    "pixel_frame_start": bounds[i],
                    "pixel_frame_end_exclusive": bounds[i + 1],
                }
                for i, old in enumerate(joint["segments"][1:])
            ]
            entry["camera_control_pattern"] = entry["camera_control_pattern"][1:]
            joint["alternation_pattern"] = entry["camera_control_pattern"]
        result.append(entry)
    output = destination / "skip_first_turn.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return str(output)
