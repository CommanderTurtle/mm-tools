#!/usr/bin/env python3
"""Render matched or sampled GT versus percept+reconstruction scene figures."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.rendering.rerender_imaginarium_exact_camera_rgb import (
    fully_opaque_blend_material_names,
)


DEFAULT_RECONSTRUCTION_ROOT = (
    REPO_ROOT
    / "results/percept_recon/reconstruction"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/percept_recon_mesh_comparison"
DEFAULT_SCENES = (
    ("ithor", "iTHOR_FloorPlan415_physics"),
    ("imaginarium", "diningroom_05"),
)
BLENDER_SCRIPT = REPO_ROOT / "eval/rendering/blender_render_percept_recon_mesh_comparison.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        metavar="DATASET:SCENE_ID",
        help="Repeat for multiple scenes; defaults to one iTHOR and one Imaginarium scene.",
    )
    parser.add_argument(
        "--fire3d-test-root",
        type=Path,
        default=REPO_ROOT / "data",
    )
    parser.add_argument(
        "--reconstruction-root", type=Path, default=DEFAULT_RECONSTRUCTION_ROOT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--blender",
        default=str(REPO_ROOT / "blender/blender-4.5.1-linux-x64/blender"),
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--recipe", default="canonical_pbr")
    parser.add_argument(
        "--render-profile",
        choices=("legacy-comparison", "inference-rgb"),
        default="legacy-comparison",
        help=(
            "inference-rgb applies the same GLB-aware two-sided and verified-"
            "opaque material policy used to create inference RGB inputs."
        ),
    )
    parser.add_argument(
        "--view-mode",
        choices=("matched-two", "sampled-grid"),
        default="matched-two",
    )
    parser.add_argument("--num-views", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument(
        "--sample-profile",
        choices=("balanced", "high-overview"),
        default="balanced",
        help=(
            "Sampled-grid camera policy. high-overview favors elevated source "
            "anchors with downward-looking, coverage-preserving perturbations."
        ),
    )
    parser.add_argument(
        "--skip-background",
        action="store_true",
        help="Omit the GT layout and predicted background instance from renders.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.width * 9 != args.height * 16:
        parser.error("render dimensions must have an exact 16:9 aspect ratio")
    if args.samples <= 0:
        parser.error("--samples must be positive")
    grid_side = math.isqrt(args.num_views)
    if args.view_mode == "sampled-grid" and grid_side**2 != args.num_views:
        parser.error("--num-views must be a perfect square in sampled-grid mode")
    if args.view_mode != "sampled-grid" and args.sample_profile != "balanced":
        parser.error("--sample-profile only applies to sampled-grid mode")
    return args


def parse_scene_specs(values: list[str]) -> list[tuple[str, str]]:
    if not values:
        return list(DEFAULT_SCENES)
    parsed = []
    for value in values:
        dataset, separator, scene_id = value.partition(":")
        dataset = dataset.strip().lower()
        scene_id = scene_id.strip()
        if not separator or dataset not in {"ithor", "imaginarium"} or not scene_id:
            raise ValueError(
                f"Invalid scene {value!r}; expected ithor:SCENE or imaginarium:SCENE"
            )
        parsed.append((dataset, scene_id))
    return parsed


def load_scene_record(dataset: str, scene_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = (
        REPO_ROOT
        / "benchmarks/scene_reconstruction/manifests"
        / f"{dataset}_v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [scene for scene in manifest["scenes"] if scene["scene_id"] == scene_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one {dataset}/{scene_id} in {manifest_path}")
    return manifest, matches[0]


def load_mask(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != 1:
            raise ValueError(f"Expected one mask array in {path}")
        return np.asarray(archive[archive.files[0]])


def infer_predicted_background_instance_id(prediction_glb: Path) -> int:
    """Recover the perception instance with the largest point assignment."""

    scene_root = prediction_glb.parent.parent
    obb_path = scene_root / "object_obbs.json"
    obb_payload = json.loads(obb_path.read_text(encoding="utf-8"))
    perception_root = Path(obb_payload["source_perception_result"])
    masks_path = perception_root / "point_instance_masks.pkl"
    with masks_path.open("rb") as handle:
        masks_payload = pickle.load(handle)
    pred = np.asarray(masks_payload["pred"]).reshape(-1)
    instance_ids, counts = np.unique(pred[pred >= 0], return_counts=True)
    if len(instance_ids) == 0:
        raise ValueError(f"No predicted instance assignments in {masks_path}")
    background_id = int(instance_ids[int(np.argmax(counts))])

    appearance_path = prediction_glb.parent / "appearance_summary.json"
    appearance = json.loads(appearance_path.read_text(encoding="utf-8"))
    composed = appearance.get("composed_world_scene") or {}
    composed_ids = {
        int(record["instance_id"])
        for record in composed.get("objects", [])
        if "instance_id" in record
    }
    if background_id not in composed_ids:
        raise ValueError(
            f"Majority instance {background_id} from {masks_path} is absent from "
            f"{appearance_path}"
        )
    return background_id


def camera_forward(frame: dict[str, Any]) -> np.ndarray:
    direction = np.asarray(frame["lookat"], dtype=np.float64) - np.asarray(
        frame["eye"], dtype=np.float64
    )
    length = float(np.linalg.norm(direction))
    if not math.isfinite(length) or length <= 1e-8:
        raise ValueError(f"Invalid camera frame: {frame}")
    return direction / length


def build_view_statistics(
    camera: dict[str, Any], mask_paths: list[Path], visible_object_ids: list[int]
) -> tuple[list[dict[str, Any]], int]:
    frames = camera["frames"]
    if len(frames) != len(mask_paths):
        raise ValueError(
            f"Camera/mask count mismatch: {len(frames)} versus {len(mask_paths)}"
        )
    visible_ids = np.asarray(sorted(set(map(int, visible_object_ids))), dtype=np.int64)
    if not len(visible_ids):
        raise ValueError("Scene has no visible GT objects")
    statistics = []
    for index, path in enumerate(mask_paths):
        mask = load_mask(path)
        counts = np.asarray([(mask == object_id).sum() for object_id in visible_ids])
        visible_pixels = int(counts.sum())
        statistics.append(
            {
                "index": index,
                "visible_objects": int(np.count_nonzero(counts >= 16)),
                "visible_pixels": visible_pixels,
                "visible_pixel_fraction": visible_pixels / max(mask.size, 1),
                "largest_object_pixel_fraction": float(counts.max(initial=0))
                / max(mask.size, 1),
                "object_pixel_balance": (
                    1.0 - float(counts.max(initial=0)) / visible_pixels
                    if visible_pixels
                    else 0.0
                ),
                "forward": camera_forward(frames[index]),
                "eye": np.asarray(frames[index]["eye"], dtype=np.float64),
            }
        )
    return statistics, len(visible_ids)


def select_two_views(
    camera: dict[str, Any], mask_paths: list[Path], visible_object_ids: list[int]
) -> tuple[list[int], list[dict[str, Any]]]:
    statistics, total_objects = build_view_statistics(
        camera, mask_paths, visible_object_ids
    )
    first = max(
        statistics,
        key=lambda row: (row["visible_objects"], row["visible_pixels"], -row["index"]),
    )
    max_pixels = max(row["visible_pixels"] for row in statistics) or 1
    def second_score(row: dict[str, Any]) -> tuple[float, int, int]:
        cosine = float(np.clip(np.dot(first["forward"], row["forward"]), -1.0, 1.0))
        angular_separation = math.acos(cosine) / math.pi
        score = (
            float(row["visible_objects"])
            + 0.75 * total_objects * angular_separation
            + 0.10 * total_objects * row["visible_pixels"] / max_pixels
        )
        return score, row["visible_objects"], -row["index"]

    second = max(
        (row for row in statistics if row["index"] != first["index"]),
        key=second_score,
    )
    selected = [int(first["index"]), int(second["index"])]
    diagnostics = []
    for row in (first, second):
        diagnostics.append(
            {
                "view_index": int(row["index"]),
                "visible_gt_objects": int(row["visible_objects"]),
                "visible_gt_pixels": int(row["visible_pixels"]),
                "forward": row["forward"].tolist(),
            }
        )
    return selected, diagnostics


def rotate_vector(vector: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    radians = math.radians(float(degrees))
    return (
        vector * math.cos(radians)
        + np.cross(axis, vector) * math.sin(radians)
        + axis * np.dot(axis, vector) * (1.0 - math.cos(radians))
    )


def downward_angle_degrees(forward: np.ndarray) -> float:
    forward = np.asarray(forward, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    return math.degrees(math.asin(float(np.clip(-forward[2], -1.0, 1.0))))


def select_diverse_anchors(
    statistics: list[dict[str, Any]], total_objects: int, count: int
) -> list[dict[str, Any]]:
    if count > len(statistics):
        raise ValueError(f"Requested {count} anchors from {len(statistics)} frames")
    max_pixels = max(row["visible_pixels"] for row in statistics) or 1
    eyes = np.stack([row["eye"] for row in statistics])
    scene_span = float(np.linalg.norm(eyes.max(axis=0) - eyes.min(axis=0))) or 1.0

    def coverage(row: dict[str, Any]) -> float:
        return (
            row["visible_objects"] / max(total_objects, 1)
            + 0.15 * row["visible_pixels"] / max_pixels
        )

    selected = [max(statistics, key=lambda row: (coverage(row), -row["index"]))]
    while len(selected) < count:
        selected_indices = {row["index"] for row in selected}
        remaining = [
            row for row in statistics if row["index"] not in selected_indices
        ]

        def diversity_score(row: dict[str, Any]) -> tuple[float, int]:
            angular = min(
                math.acos(
                    float(np.clip(np.dot(row["forward"], prior["forward"]), -1, 1))
                )
                / math.pi
                for prior in selected
            )
            spatial = min(
                float(np.linalg.norm(row["eye"] - prior["eye"])) / scene_span
                for prior in selected
            )
            return coverage(row) + 0.55 * angular + 0.20 * spatial, -row["index"]

        selected.append(max(remaining, key=diversity_score))
    return selected


def select_source_views(
    camera: dict[str, Any],
    mask_paths: list[Path],
    visible_object_ids: list[int],
    *,
    count: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Select exact source cameras with foreground coverage and view diversity."""
    if count <= 0:
        raise ValueError("View count must be positive")
    if count == 2:
        return select_two_views(camera, mask_paths, visible_object_ids)
    statistics, total_objects = build_view_statistics(
        camera, mask_paths, visible_object_ids
    )
    selected_rows = select_diverse_anchors(statistics, total_objects, count)
    selected = [int(row["index"]) for row in selected_rows]
    diagnostics = [
        {
            "view_index": int(row["index"]),
            "visible_gt_objects": int(row["visible_objects"]),
            "visible_gt_pixels": int(row["visible_pixels"]),
            "forward": row["forward"].tolist(),
        }
        for row in selected_rows
    ]
    return selected, diagnostics


def select_high_overview_anchors(
    statistics: list[dict[str, Any]],
    total_objects: int,
    count: int,
    overview_target: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Select elevated, downward-looking anchors without sacrificing coverage."""
    if count > len(statistics):
        raise ValueError(f"Requested {count} anchors from {len(statistics)} frames")
    heights = np.asarray([row["eye"][2] for row in statistics], dtype=np.float64)
    height_min = float(heights.min())
    height_span = float(heights.max() - height_min) or 1.0
    height_cutoff = float(np.quantile(heights, 0.55))

    def height_score(row: dict[str, Any]) -> float:
        return float((row["eye"][2] - height_min) / height_span)

    def downward_score(row: dict[str, Any]) -> float:
        angle = downward_angle_degrees(row["forward"])
        return float(np.clip((angle - 15.0) / 55.0, 0.0, 1.0))

    def overview_quality(row: dict[str, Any]) -> float:
        object_coverage = row["visible_objects"] / max(total_objects, 1)
        frame_coverage = min(row["visible_pixel_fraction"] / 0.45, 1.0)
        closeup_penalty = float(
            np.clip(
                (row["largest_object_pixel_fraction"] - 0.25) / 0.50,
                0.0,
                1.0,
            )
        )
        return (
            object_coverage
            + 0.10 * frame_coverage
            + 0.40 * height_score(row)
            + 0.30 * downward_score(row)
            + 0.15 * row["object_pixel_balance"]
            - 0.35 * closeup_penalty
        )

    candidates = [
        row
        for row in statistics
        if row["eye"][2] >= height_cutoff
        and downward_angle_degrees(row["forward"]) >= 15.0
    ]
    minimum_visible_objects = min(
        total_objects, max(2, math.ceil(total_objects * 0.05))
    )
    usable = [
        row
        for row in statistics
        if row["visible_objects"] >= minimum_visible_objects
        and row["largest_object_pixel_fraction"] <= 0.80
    ]
    if len(usable) >= count:
        usable_indices = {row["index"] for row in usable}
        candidates = [row for row in candidates if row["index"] in usable_indices]
    if overview_target is not None and len(candidates) > count:
        target_xy = np.asarray(overview_target[:2], dtype=np.float64)
        retained_count = max(count, math.ceil(0.85 * len(candidates)))
        candidates = sorted(
            candidates,
            key=lambda row: float(np.linalg.norm(row["eye"][:2] - target_xy)),
            reverse=True,
        )[:retained_count]
    if len(candidates) < count:
        pool = usable if len(usable) >= count else statistics
        pool_size = min(len(pool), max(count, count * 2))
        candidates = sorted(
            pool,
            key=lambda row: (overview_quality(row), -row["index"]),
            reverse=True,
        )[:pool_size]

    candidate_eyes = np.stack([row["eye"][:2] for row in candidates])
    horizontal_span = float(
        np.linalg.norm(candidate_eyes.max(axis=0) - candidate_eyes.min(axis=0))
    ) or 1.0

    def horizontal_forward(row: dict[str, Any]) -> np.ndarray:
        value = (
            row["eye"][:2] - np.asarray(overview_target[:2], dtype=np.float64)
            if overview_target is not None
            else np.asarray(row["forward"][:2], dtype=np.float64)
        )
        length = float(np.linalg.norm(value))
        if length <= 1e-8:
            return np.array([1.0, 0.0], dtype=np.float64)
        return value / length

    selected = [
        max(candidates, key=lambda row: (overview_quality(row), -row["index"]))
    ]
    while len(selected) < count:
        selected_indices = {row["index"] for row in selected}
        remaining = [
            row for row in candidates if row["index"] not in selected_indices
        ]

        def diversity_score(row: dict[str, Any]) -> tuple[float, int]:
            direction = horizontal_forward(row)
            azimuth = min(
                math.acos(
                    float(
                        np.clip(
                            np.dot(direction, horizontal_forward(prior)), -1.0, 1.0
                        )
                    )
                )
                / math.pi
                for prior in selected
            )
            spatial = min(
                float(np.linalg.norm(row["eye"][:2] - prior["eye"][:2]))
                / horizontal_span
                for prior in selected
            )
            return overview_quality(row) + 0.45 * azimuth + 0.15 * spatial, -row[
                "index"
            ]

        selected.append(max(remaining, key=diversity_score))
    return selected


def sample_novel_views(
    camera: dict[str, Any],
    mask_paths: list[Path],
    visible_object_ids: list[int],
    *,
    count: int,
    seed: int,
    profile: str = "balanced",
    overview_target: np.ndarray | None = None,
    statistics: list[dict[str, Any]] | None = None,
    total_objects: int | None = None,
) -> list[dict[str, Any]]:
    """Anchor selection and view construction for the Aug-28 profiles.

    `statistics`/`total_objects` let a caller supply the per-frame scores
    directly. ScanNet++ has no GT instance masks, so `build_view_statistics`
    cannot run there; the scannetpp path in
    `eval/unified_sample_render_views.py` derives the same three quantities by
    projecting the predicted objects instead, and everything downstream --
    height cutoff, downward angle, azimuth diversity, overview framing -- is
    then shared with the GT-scored datasets.
    """

    if statistics is None:
        statistics, total_objects = build_view_statistics(
            camera, mask_paths, visible_object_ids
        )
    elif total_objects is None:
        raise ValueError("total_objects is required when statistics are supplied")
    if profile == "high-overview":
        if overview_target is None:
            overview_target = np.median(
                np.asarray([frame["lookat"] for frame in camera["frames"]]), axis=0
            )
        overview_target = np.asarray(overview_target, dtype=np.float64)
        if overview_target.shape != (3,) or not np.all(np.isfinite(overview_target)):
            raise ValueError(f"Invalid overview target: {overview_target}")
    if profile == "balanced":
        anchors = select_diverse_anchors(statistics, total_objects, count)
    elif profile == "high-overview":
        anchors = select_high_overview_anchors(
            statistics, total_objects, count, overview_target
        )
    else:
        raise ValueError(f"Unsupported sample profile: {profile}")
    eyes = np.stack([row["eye"] for row in statistics])
    camera_span = float(np.linalg.norm(eyes.max(axis=0) - eyes.min(axis=0))) or 1.0
    rng = np.random.default_rng(seed)
    views = []
    for rank, anchor in enumerate(anchors):
        frame = camera["frames"][anchor["index"]]
        eye = np.asarray(frame["eye"], dtype=np.float64)
        lookat = np.asarray(frame["lookat"], dtype=np.float64)
        up_hint = np.asarray(frame["up"], dtype=np.float64)
        forward = lookat - eye
        focus_distance = float(np.linalg.norm(forward))
        forward /= focus_distance
        right = np.cross(forward, up_hint)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        if profile == "balanced":
            yaw = float(rng.uniform(-12.0, 12.0))
            pitch = float(rng.uniform(-5.0, 5.0))
            if abs(yaw) < 3.0:
                yaw = math.copysign(3.0, yaw if yaw else 1.0)
            lateral = float(rng.uniform(-0.035, 0.035) * camera_span)
            vertical = float(rng.uniform(-0.012, 0.012) * camera_span)
            sampled_eye = eye + right * lateral + up * vertical
            sampled_forward = rotate_vector(forward, up, yaw)
            sampled_right = np.cross(sampled_forward, up)
            sampled_right /= np.linalg.norm(sampled_right)
            sampled_forward = rotate_vector(sampled_forward, sampled_right, pitch)
            sampled_forward /= np.linalg.norm(sampled_forward)
            sampled_up = np.cross(sampled_right, sampled_forward)
            sampled_up /= np.linalg.norm(sampled_up)
        else:
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            yaw = float(rng.uniform(-7.0, 7.0))
            if abs(yaw) < 2.0:
                yaw = math.copysign(2.0, yaw if yaw else 1.0)
            # Source camera positions are known to be inside the room shell.
            # Moving a high anchor can cross a wall or ceiling and blank the GT.
            lateral = 0.0
            vertical = 0.0
            sampled_eye = eye.copy()
            base_horizontal = overview_target - sampled_eye
            base_horizontal[2] = 0.0
            if np.linalg.norm(base_horizontal) <= 1e-8:
                base_horizontal = forward.copy()
                base_horizontal[2] = 0.0
            if np.linalg.norm(base_horizontal) <= 1e-8:
                base_horizontal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            base_horizontal /= np.linalg.norm(base_horizontal)
            horizontal_right = np.cross(base_horizontal, world_up)
            horizontal_right /= np.linalg.norm(horizontal_right)
            sampled_eye += horizontal_right * lateral
            target_delta = overview_target - sampled_eye
            horizontal_distance = float(np.linalg.norm(target_delta[:2]))
            target_downward = math.degrees(
                math.atan2(max(sampled_eye[2] - overview_target[2], 0.0),
                           max(horizontal_distance, 1e-8))
            )
            horizontal_forward = rotate_vector(base_horizontal, world_up, yaw)
            horizontal_forward /= np.linalg.norm(horizontal_forward)
            anchor_downward = downward_angle_degrees(forward)
            sampled_downward = float(
                np.clip(
                    max(target_downward, 32.0) + rng.uniform(-2.0, 4.0),
                    30.0,
                    68.0,
                )
            )
            downward_radians = math.radians(sampled_downward)
            sampled_forward = (
                horizontal_forward * math.cos(downward_radians)
                - world_up * math.sin(downward_radians)
            )
            sampled_right = np.cross(sampled_forward, world_up)
            sampled_right /= np.linalg.norm(sampled_right)
            sampled_up = np.cross(sampled_right, sampled_forward)
            sampled_up /= np.linalg.norm(sampled_up)
            pitch = sampled_downward - anchor_downward
        sampled_lookat = sampled_eye + sampled_forward * focus_distance
        views.append(
            {
                "candidate_id": f"C{rank + 1:02d}",
                "kind": (
                    "high_overview_perturbed_source_anchor"
                    if profile == "high-overview"
                    else "novel_perturbed_source_anchor"
                ),
                "anchor_frame_index": int(anchor["index"]),
                "anchor_visible_gt_objects": int(anchor["visible_objects"]),
                "anchor_visible_gt_pixels": int(anchor["visible_pixels"]),
                "sample_profile": profile,
                "overview_target": (
                    overview_target.tolist() if profile == "high-overview" else None
                ),
                "anchor_eye_height": float(eye[2]),
                "anchor_downward_angle_degrees": downward_angle_degrees(forward),
                "sampled_eye_height": float(sampled_eye[2]),
                "sampled_downward_angle_degrees": downward_angle_degrees(
                    sampled_forward
                ),
                "yaw_offset_degrees": yaw,
                "pitch_offset_degrees": pitch,
                "lateral_offset": lateral,
                "vertical_offset": vertical,
                "frame": {
                    "eye": sampled_eye.tolist(),
                    "lookat": sampled_lookat.tolist(),
                    "up": sampled_up.tolist(),
                },
            }
        )
    return views


def compose_matrix(record: dict[str, Any]) -> np.ndarray:
    import trimesh

    scale = float(record["scale"])
    return trimesh.transformations.compose_matrix(
        scale=[scale, scale, scale],
        angles=np.asarray(record["angles"], dtype=np.float64),
        translate=np.asarray(record["trans"], dtype=np.float64),
    )


def force_opaque_materials_for_profile(
    path: Path, render_profile: str
) -> list[str]:
    if render_profile == "legacy-comparison":
        return []
    if render_profile != "inference-rgb":
        raise ValueError(f"Unsupported render profile: {render_profile}")
    return fully_opaque_blend_material_names(path)


def scene_overview_target(
    transforms: dict[str, Any], visible_object_ids: list[int]
) -> np.ndarray:
    centers = np.asarray(
        [
            transforms[f"object_{int(object_id):04d}"]["trans"]
            for object_id in visible_object_ids
        ],
        dtype=np.float64,
    )
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("Expected visible-object translations with shape [N, 3]")
    return np.median(centers, axis=0)


def build_task(
    args: argparse.Namespace, dataset: str, scene_id: str
) -> tuple[Path, list[Path], dict[str, Any]]:
    manifest, scene = load_scene_record(dataset, scene_id)
    dataset_root = args.fire3d_test_root / manifest["dataset_subdir"]
    camera_path = dataset_root / scene["camera_path"]
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    mask_paths = sorted((dataset_root / scene["masks_dir"]).glob("mask_*.npz"))
    transform_path = dataset_root / scene["transforms_path"]
    with transform_path.open("rb") as handle:
        transforms = pickle.load(handle)
    if args.view_mode == "matched-two":
        selected_views, view_diagnostics = select_two_views(
            camera, mask_paths, scene["visible_object_ids"]
        )
        views = [
            {
                "candidate_id": f"V{rank + 1}",
                "kind": "source_camera",
                "source_frame_index": int(view_index),
                "frame": camera["frames"][view_index],
                **view_diagnostics[rank],
            }
            for rank, view_index in enumerate(selected_views)
        ]
    else:
        scene_seed = args.seed + sum(scene_id.encode("utf-8"))
        views = sample_novel_views(
            camera,
            mask_paths,
            scene["visible_object_ids"],
            count=args.num_views,
            seed=scene_seed,
            profile=args.sample_profile,
            overview_target=(
                scene_overview_target(transforms, scene["visible_object_ids"])
                if args.sample_profile == "high-overview"
                else None
            ),
        )
    layout_keys = sorted(key for key in transforms if key.startswith("layout_"))
    if len(layout_keys) != 1:
        raise ValueError(f"Expected one layout transform in {transform_path}")
    object_keys = [
        f"object_{int(object_id):04d}" for object_id in scene["visible_object_ids"]
    ]
    gt_assets = []
    asset_keys = object_keys if args.skip_background else [layout_keys[0], *object_keys]
    for key in asset_keys:
        mesh_path = dataset_root / scene["mesh_dir"] / f"{key}.glb"
        if key not in transforms or not mesh_path.is_file():
            raise FileNotFoundError(f"Missing GT asset or transform: {mesh_path}")
        gt_assets.append(
            {
                "name": key,
                "path": str(mesh_path.resolve()),
                "object_to_world": compose_matrix(transforms[key]).tolist(),
                "force_opaque_materials": force_opaque_materials_for_profile(
                    mesh_path, args.render_profile
                ),
            }
        )
    prediction_glb = (
        args.reconstruction_root
        / dataset
        / scene_id
        / "appearance/predicted_textured_world_scene.glb"
    )
    if not prediction_glb.is_file():
        raise FileNotFoundError(prediction_glb)
    predicted_background_instance_id = (
        infer_predicted_background_instance_id(prediction_glb)
        if args.skip_background
        else None
    )
    scene_output = args.output_root / dataset / scene_id
    task = {
        "schema": "ff_percept_recon_mesh_comparison_task_v1",
        "dataset": dataset,
        "scene_id": scene_id,
        "camera_path": str(camera_path.resolve()),
        "source_width": int(camera["width"]),
        "source_height": int(camera["height"]),
        "render_width": int(args.width),
        "render_height": int(args.height),
        "samples": int(args.samples),
        "recipe": args.recipe,
        "render_profile": args.render_profile,
        "material_policy": {
            "two_sided": args.render_profile == "inference-rgb",
            "shadow_two_sided": args.render_profile == "inference-rgb",
            "fully_opaque_blend_detection": (
                "glb_alpha_factor_and_texture_scan"
                if args.render_profile == "inference-rgb"
                else None
            ),
        },
        "view_mode": args.view_mode,
        "sample_profile": (
            args.sample_profile if args.view_mode == "sampled-grid" else None
        ),
        "views": views,
        "skip_background": bool(args.skip_background),
        "predicted_background_instance_id": predicted_background_instance_id,
        "gt_assets": gt_assets,
        "prediction_glb": str(prediction_glb.resolve()),
        "prediction_force_opaque_materials": force_opaque_materials_for_profile(
            prediction_glb, args.render_profile
        ),
        "prediction_label": "Ours: perception + reconstruction",
        "output_dir": str(scene_output.resolve()),
    }
    scene_output.mkdir(parents=True, exist_ok=True)
    task_path = scene_output / "render_task.json"
    task_path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
    if args.view_mode == "matched-two":
        outputs = [scene_output / "comparison_2view_16x9.png"]
    else:
        outputs = [
            scene_output / "candidate_views_gt_16x9.png",
            scene_output / "candidate_views_ours_16x9.png",
        ]
    return task_path, outputs, task


def render_task(
    args: argparse.Namespace, task_path: Path, outputs: list[Path]
) -> None:
    blender = Path(shutil.which(args.blender) or args.blender).resolve()
    command = [
        str(blender),
        "-b",
        "--factory-startup",
        "--python",
        str(BLENDER_SCRIPT),
        "--",
        "--task",
        str(task_path),
    ]
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    grid_side = math.isqrt(args.num_views)
    expected = (
        (args.width * 2, args.height * 2)
        if args.view_mode == "matched-two"
        else (args.width * grid_side, args.height * grid_side)
    )
    for output in outputs:
        if not output.is_file():
            raise RuntimeError(f"Blender did not produce {output}")
        with Image.open(output) as image:
            if image.size != expected or image.width * 9 != image.height * 16:
                raise RuntimeError(
                    f"Unexpected montage dimensions {image.size}; expected {expected}"
                )


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset, scene_id in parse_scene_specs(args.scene):
        task_path, outputs, task = build_task(args, dataset, scene_id)
        render_task(args, task_path, outputs)
        records.append(
            {
                "dataset": dataset,
                "scene_id": scene_id,
                "outputs": [str(output) for output in outputs],
                "view_mode": task["view_mode"],
                "sample_profile": task["sample_profile"],
                "render_profile": task["render_profile"],
                "views": task["views"],
            }
        )
    grid_side = math.isqrt(args.num_views)
    montage_dimensions = (
        [args.width * 2, args.height * 2]
        if args.view_mode == "matched-two"
        else [args.width * grid_side, args.height * grid_side]
    )
    summary = {
        "schema": "ff_percept_recon_mesh_comparison_16x9_v1",
        "panel_dimensions": [args.width, args.height],
        "montage_dimensions": montage_dimensions,
        "aspect_ratio": "16:9",
        "view_mode": args.view_mode,
        "sample_profile": (
            args.sample_profile if args.view_mode == "sampled-grid" else None
        ),
        "render_profile": args.render_profile,
        "records": records,
    }
    summary_path = args.output_root / "visualization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
