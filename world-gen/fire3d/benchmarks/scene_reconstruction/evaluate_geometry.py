from __future__ import annotations

"""Evaluate oracle-ID/oracle-pose geometry on iTHOR or Imaginarium.

The benchmark intentionally excludes perception and appearance. Each predicted
mesh is keyed by its GT scene/object ID and, unless it is already in world space,
is placed with the GT object transform before FF geometry metrics are computed.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scene_reconstruction.geometry_protocol import (  # noqa: E402
    METRIC_KEYS,
    aggregate_records,
    find_prediction,
    jsonable,
    load_gt_world_mesh,
    load_manifest,
    load_mesh,
    load_scene_transforms,
    object_key,
    prediction_to_world,
    parse_prediction_manifest,
    stable_object_seed,
)
from eval.reconstruction.geometry_eval_utils import calculate_geometry_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--fire3d-test-root",
        type=Path,
        default=REPO_ROOT / "data",
        help="Directory containing the manifest's dataset_subdir.",
    )
    parser.add_argument("--prediction-root", type=Path, default=None)
    parser.add_argument("--prediction-manifest", type=Path, default=None)
    parser.add_argument(
        "--prediction-space",
        choices=("ff_canonical", "raw_glb", "world"),
        default="ff_canonical",
        help=(
            "ff_canonical: decoder output in the benchmark's Z-up unit object frame; "
            "raw_glb: source GLB frame before +90deg X conversion; world: already placed."
        ),
    )
    parser.add_argument("--object-set", choices=("visible", "all"), default="visible")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--n-points", type=int, default=100000)
    parser.add_argument("--fscore-threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--scene", action="append", default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument(
        "--identity-gt-prediction",
        action="store_true",
        help="Use each world-space GT mesh as prediction for protocol smoke testing.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write partial diagnostics and exit zero when predictions are missing/failed.",
    )
    args = parser.parse_args()
    if args.n_points <= 0:
        parser.error("--n-points must be positive")
    if args.fscore_threshold <= 0:
        parser.error("--fscore-threshold must be positive")
    if args.max_scenes is not None and args.max_scenes <= 0:
        parser.error("--max-scenes must be positive")
    if not args.identity_gt_prediction and args.prediction_root is None and args.prediction_manifest is None:
        parser.error("pass --prediction-root, --prediction-manifest, or --identity-gt-prediction")
    return args


def select_scenes(manifest: dict[str, Any], names: list[str] | None, max_scenes: int | None) -> list[dict[str, Any]]:
    scenes = manifest["scenes"]
    if names:
        requested = set(names)
        available = {scene["scene_id"] for scene in scenes}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Requested scene IDs are absent from the manifest: {missing}")
        scenes = [scene for scene in scenes if scene["scene_id"] in requested]
    if max_scenes is not None:
        scenes = scenes[:max_scenes]
    if not scenes:
        raise ValueError("No scenes selected")
    return scenes


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "dataset",
        "scene_id",
        "object_id",
        "status",
        "visible_pixels",
        "gt_mesh_path",
        "pred_mesh_path",
        "prediction_space",
        "error",
        *METRIC_KEYS,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            metrics = record.get("metrics") or {}
            writer.writerow(
                {
                    "dataset": record["dataset"],
                    "scene_id": record["scene_id"],
                    "object_id": record["object_id"],
                    "status": record["status"],
                    "visible_pixels": record.get("visible_pixels"),
                    "gt_mesh_path": record["gt_mesh_path"],
                    "pred_mesh_path": record.get("pred_mesh_path"),
                    "prediction_space": record.get("prediction_space"),
                    "error": record.get("error"),
                    **{key: metrics.get(key) for key in METRIC_KEYS},
                }
            )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    dataset = manifest["dataset"]
    dataset_root = args.fire3d_test_root.resolve() / manifest["dataset_subdir"]
    scenes = select_scenes(manifest, args.scene, args.max_scenes)

    prediction_root = args.prediction_root.resolve() if args.prediction_root else None
    explicit_predictions = (
        parse_prediction_manifest(args.prediction_manifest.resolve())
        if args.prediction_manifest
        else {}
    )

    records: list[dict[str, Any]] = []
    for scene in scenes:
        scene_id = scene["scene_id"]
        object_ids = scene["visible_object_ids"] if args.object_set == "visible" else scene["gt_object_ids"]
        transforms_path = dataset_root / scene["transforms_path"]
        transforms = load_scene_transforms(transforms_path)
        for object_id in object_ids:
            object_name = f"object_{int(object_id):04d}"
            gt_mesh_path = dataset_root / scene["mesh_dir"] / f"{object_name}.glb"
            record: dict[str, Any] = {
                "key": object_key(scene_id, object_id),
                "dataset": dataset,
                "scene_id": scene_id,
                "benchmark_index": scene["benchmark_index"],
                "legacy_eval_index": scene["legacy_eval_index"],
                "object_id": int(object_id),
                "visible_pixels": int(scene.get("visible_pixel_counts", {}).get(str(object_id), 0)),
                "gt_mesh_path": str(gt_mesh_path),
                "pred_mesh_path": None,
                "prediction_space": args.prediction_space,
                "status": "pending",
                "metrics": None,
                "error": None,
            }
            try:
                gt_world = load_gt_world_mesh(dataset_root, scene, object_id, transforms)
                if args.identity_gt_prediction:
                    pred_world = gt_world.copy()
                    record["prediction_source"] = "identity_gt_smoke"
                    record["prediction_space"] = "world"
                else:
                    pred_path, ambiguous, explicit = find_prediction(
                        prediction_root=prediction_root,
                        prediction_manifest=explicit_predictions,
                        dataset=dataset,
                        scene=scene,
                        object_id=object_id,
                    )
                    if ambiguous:
                        record["status"] = "ambiguous_prediction" if len(ambiguous) > 1 else "missing_prediction"
                        record["prediction_candidates"] = ambiguous
                        records.append(record)
                        continue
                    if pred_path is None:
                        record["status"] = "missing_prediction"
                        records.append(record)
                        continue
                    record["pred_mesh_path"] = str(pred_path)
                    prediction_space = (
                        explicit.get("prediction_space", args.prediction_space)
                        if explicit is not None
                        else args.prediction_space
                    )
                    record["prediction_space"] = prediction_space
                    pred_world = prediction_to_world(
                        load_mesh(pred_path),
                        transforms[object_name],
                        prediction_space,
                    )

                np.random.seed(stable_object_seed(args.seed, dataset, scene_id, object_id))
                metrics = calculate_geometry_metrics(
                    pred_world,
                    gt_world,
                    n_points=args.n_points,
                    f1_threshold=args.fscore_threshold,
                )
                if not np.isfinite(metrics["CD"]):
                    raise ValueError("Geometry metric returned non-finite CD")
                metrics["CD_x100"] = float(metrics["CD"]) * 100.0
                record["metrics"] = {key: float(metrics[key]) for key in METRIC_KEYS}
                record["status"] = "evaluated"
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)

    expected_scene_ids = [scene["scene_id"] for scene in scenes]
    aggregation = aggregate_records(records, expected_scene_ids)
    output = {
        "protocol": {
            "name": "ff_ithor_imaginarium_geometry_v1",
            "scope": "geometry_only_oracle_object_id_and_pose",
            "manifest": str(manifest_path),
            "dataset": dataset,
            "dataset_subdir": manifest["dataset_subdir"],
            "video_id": manifest["video_id"],
            "object_set": args.object_set,
            "prediction_root": str(prediction_root) if prediction_root else None,
            "prediction_manifest": str(args.prediction_manifest.resolve()) if args.prediction_manifest else None,
            "prediction_space": args.prediction_space,
            "n_points": args.n_points,
            "seed": args.seed,
            "geometry": {
                "source": "eval/reconstruction/geometry_eval_utils.py",
                "coordinate_space": "world coordinates using GT object pose and scale",
                "chamfer": "symmetric Chamfer-L1",
                "reported_chamfer": "CD x100",
                "fscore_threshold": args.fscore_threshold,
                "normal_consistency": "absolute nearest-surface normal dot product",
            },
            "excluded": ["detection", "segmentation", "pose_prediction", "appearance", "rendering"],
        },
        "selection": {
            "num_selected_scenes": len(scenes),
            "scene_ids": expected_scene_ids,
        },
        **aggregation,
        "objects": records,
    }
    return output


def main() -> None:
    args = parse_args()
    output = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(output), indent=2) + "\n")
    csv_output = args.csv_output or args.output.with_suffix(".csv")
    write_csv(csv_output, output["objects"])

    report = {
        "complete": output["complete"],
        "coverage": output["coverage"],
        "primary_metrics": output["primary_metrics"],
        "partial_scene_macro_metrics": output["partial_scene_macro_metrics"],
    }
    print(json.dumps(jsonable(report), indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {csv_output}")
    if not output["complete"] and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
