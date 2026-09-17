from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh


MESH_EXTENSIONS = (".ply", ".glb", ".obj")
METRIC_KEYS = ("CD", "CD_x100", "F1", "NC", "precision", "recall")
BOUNDED_METRIC_KEYS = ("F1", "NC", "precision", "recall")
OBJECT_MESH_RE = re.compile(r"^object_(\d{4})\.glb$")

# A GLB exported from an FF-canonical Z-up trimesh is represented as glTF
# Y-up. Blender's glTF importer materializes that convention change in the
# imported vertex coordinates even when the object matrix is identity:
#     (x, y, z)_ff -> (x, -z, y)_blender
# The inverse must be composed before an FF canonical object-to-world pose.
FF_CANONICAL_TO_BLENDER_IMPORTED_GLTF = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
BLENDER_IMPORTED_GLTF_TO_FF_CANONICAL = np.linalg.inv(
    FF_CANONICAL_TO_BLENDER_IMPORTED_GLTF
)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_key(scene_id: str, object_id: int) -> str:
    return f"{scene_id}/object_{int(object_id):04d}"


def as_mesh(value: Any) -> trimesh.Trimesh:
    if isinstance(value, trimesh.Trimesh):
        return value
    if isinstance(value, trimesh.Scene):
        dumped = value.dump()
        meshes = [mesh for mesh in dumped if isinstance(mesh, trimesh.Trimesh)]
        if not meshes:
            return trimesh.Trimesh()
        return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    raise TypeError(f"Unsupported mesh value: {type(value)!r}")


def load_mesh(path: Path) -> trimesh.Trimesh:
    return as_mesh(trimesh.load(path, force="scene", process=False))


def raw_glb_to_ff_canonical(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    result = mesh.copy()
    result.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])
    )
    return result


def object_world_transform(transform_record: dict[str, Any]) -> np.ndarray:
    scale = float(transform_record["scale"])
    angles = np.asarray(transform_record["angles"], dtype=np.float64).reshape(3)
    translate = np.asarray(transform_record["trans"], dtype=np.float64).reshape(3)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid uniform object scale: {scale}")
    if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(translate)):
        raise ValueError("Object angles/translation must be finite")
    return trimesh.transformations.compose_matrix(
        scale=[scale, scale, scale],
        shear=None,
        angles=angles,
        translate=translate,
        perspective=None,
    )


def load_scene_transforms(path: Path) -> dict[str, dict[str, Any]]:
    # These are trusted, locally generated benchmark annotations. Do not use this
    # helper with untrusted pickle files.
    with path.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, dict):
        raise TypeError(f"Expected transform dictionary in {path}, got {type(records)!r}")
    return records


def load_gt_world_mesh(
    dataset_root: Path,
    scene_record: dict[str, Any],
    object_id: int,
    transforms: dict[str, dict[str, Any]] | None = None,
) -> trimesh.Trimesh:
    object_name = f"object_{int(object_id):04d}"
    if transforms is None:
        transforms = load_scene_transforms(dataset_root / scene_record["transforms_path"])
    if object_name not in transforms:
        raise KeyError(f"{scene_record['scene_id']} is missing transform {object_name}")
    mesh_path = dataset_root / scene_record["mesh_dir"] / f"{object_name}.glb"
    mesh = raw_glb_to_ff_canonical(load_mesh(mesh_path))
    mesh.apply_transform(object_world_transform(transforms[object_name]))
    return mesh


def prediction_to_world(
    mesh: trimesh.Trimesh,
    transform_record: dict[str, Any],
    prediction_space: str,
) -> trimesh.Trimesh:
    result = mesh.copy()
    if prediction_space == "world":
        return result
    if prediction_space == "raw_glb":
        result = raw_glb_to_ff_canonical(result)
    elif prediction_space != "ff_canonical":
        raise ValueError(f"Unsupported prediction space: {prediction_space}")
    result.apply_transform(object_world_transform(transform_record))
    return result


def stable_object_seed(base_seed: int, dataset: str, scene_id: str, object_id: int) -> int:
    payload = f"{base_seed}:{dataset}:{scene_id}:{int(object_id)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def parse_prediction_manifest(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "predictions" in payload:
        payload = payload["predictions"]

    records: dict[str, Any]
    if isinstance(payload, dict):
        records = payload
    elif isinstance(payload, list):
        records = {}
        for record in payload:
            if not isinstance(record, dict):
                raise ValueError("Prediction-manifest list entries must be JSON objects")
            scene_id = record.get("scene_id") or record.get("scene")
            object_id = record.get("object_id") or record.get("index")
            path_value = record.get("path") or record.get("mesh_path") or record.get("pred_mesh_path")
            if scene_id is None or object_id is None or path_value is None:
                raise ValueError(f"Prediction record lacks scene/object/path: {record}")
            records[object_key(str(scene_id), int(object_id))] = record
    else:
        raise ValueError("Prediction manifest must be a mapping or list")

    parsed: dict[str, dict[str, Any]] = {}
    for key, value in records.items():
        if isinstance(value, str):
            record = {"path": value}
        elif isinstance(value, dict):
            record = dict(value)
        else:
            raise ValueError(f"Unsupported prediction value for {key}: {value!r}")
        path_value = record.get("path") or record.get("mesh_path") or record.get("pred_mesh_path")
        if not path_value:
            raise ValueError(f"Prediction record lacks a mesh path: {key}")
        mesh_path = Path(path_value).expanduser()
        if not mesh_path.is_absolute():
            mesh_path = path.parent / mesh_path
        record["path"] = str(mesh_path.resolve())
        parsed[str(key)] = record
    return parsed


def _candidate_prediction_paths(
    prediction_root: Path,
    dataset: str,
    scene: dict[str, Any],
    object_id: int,
) -> list[Path]:
    scene_id = scene["scene_id"]
    stem = f"object_{int(object_id):04d}"
    numeric = str(int(object_id))
    legacy_index = int(scene["legacy_eval_index"])
    benchmark_index = int(scene["benchmark_index"])
    bases = [
        prediction_root / scene_id,
        prediction_root / scene_id / "objects",
        prediction_root / dataset / scene_id,
        prediction_root / dataset / scene_id / "objects",
        prediction_root / f"val_{legacy_index}" / "objects",
    ]
    if benchmark_index != legacy_index:
        bases.append(prediction_root / f"benchmark_{benchmark_index}" / "objects")

    candidates: list[Path] = []
    for base in bases:
        for extension in MESH_EXTENSIONS:
            candidates.append(base / f"{stem}{extension}")
            candidates.append(base / f"{numeric}{extension}")
    return candidates


def find_prediction(
    *,
    prediction_root: Path | None,
    prediction_manifest: dict[str, dict[str, Any]],
    dataset: str,
    scene: dict[str, Any],
    object_id: int,
) -> tuple[Path | None, list[str], dict[str, Any] | None]:
    key = object_key(scene["scene_id"], object_id)
    explicit = prediction_manifest.get(key)
    if explicit is not None:
        path = Path(explicit["path"])
        return (path if path.is_file() else None), ([] if path.is_file() else [str(path)]), explicit
    if prediction_root is None:
        return None, [], None

    existing = sorted(
        {path.resolve() for path in _candidate_prediction_paths(prediction_root, dataset, scene, object_id) if path.is_file()},
        key=str,
    )
    if len(existing) == 1:
        return existing[0], [], None
    if len(existing) > 1:
        return None, [str(path) for path in existing], None
    return None, [], None


def scalar_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "median": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
    }


def aggregate_records(
    records: list[dict[str, Any]],
    expected_scene_ids: list[str],
) -> dict[str, Any]:
    by_scene: dict[str, Any] = {}
    for scene_id in expected_scene_ids:
        scene_records = [record for record in records if record["scene_id"] == scene_id]
        evaluated = [record for record in scene_records if record["status"] == "evaluated"]
        conditional = {
            key: scalar_summary(record["metrics"][key] for record in evaluated)
            for key in METRIC_KEYS
        }
        expected_count = len(scene_records)
        evaluated_count = len(evaluated)
        coverage = evaluated_count / expected_count if expected_count else 1.0
        zero_for_missing = {
            key: (
                float(sum(record["metrics"][key] for record in evaluated) / expected_count)
                if expected_count
                else None
            )
            for key in BOUNDED_METRIC_KEYS
        }
        by_scene[scene_id] = {
            "num_expected_objects": expected_count,
            "num_evaluated_objects": evaluated_count,
            "num_missing_predictions": sum(record["status"] == "missing_prediction" for record in scene_records),
            "num_ambiguous_predictions": sum(record["status"] == "ambiguous_prediction" for record in scene_records),
            "num_failed_objects": sum(record["status"] == "failed" for record in scene_records),
            "coverage": coverage,
            "conditional_metrics": conditional,
            "zero_for_missing_metrics": zero_for_missing,
        }

    scene_macro = {}
    for key in METRIC_KEYS:
        scene_means = [
            scene["conditional_metrics"][key]["mean"]
            for scene in by_scene.values()
            if scene["conditional_metrics"][key]["mean"] is not None
        ]
        scene_macro[key] = scalar_summary(scene_means)

    zero_for_missing_scene_macro = {
        key: scalar_summary(
            scene["zero_for_missing_metrics"][key]
            for scene in by_scene.values()
            if scene["zero_for_missing_metrics"][key] is not None
        )
        for key in BOUNDED_METRIC_KEYS
    }
    object_micro = {
        key: scalar_summary(
            record["metrics"][key]
            for record in records
            if record["status"] == "evaluated"
        )
        for key in METRIC_KEYS
    }

    num_expected = len(records)
    num_evaluated = sum(record["status"] == "evaluated" for record in records)
    num_missing = sum(record["status"] == "missing_prediction" for record in records)
    num_ambiguous = sum(record["status"] == "ambiguous_prediction" for record in records)
    num_failed = sum(record["status"] == "failed" for record in records)
    complete = num_evaluated == num_expected and not (num_missing or num_ambiguous or num_failed)

    return {
        "complete": complete,
        "coverage": {
            "num_expected_scenes": len(expected_scene_ids),
            "num_complete_scenes": sum(
                scene["num_evaluated_objects"] == scene["num_expected_objects"]
                and scene["num_missing_predictions"] == 0
                and scene["num_ambiguous_predictions"] == 0
                and scene["num_failed_objects"] == 0
                for scene in by_scene.values()
            ),
            "num_expected_objects": num_expected,
            "num_evaluated_objects": num_evaluated,
            "num_missing_predictions": num_missing,
            "num_ambiguous_predictions": num_ambiguous,
            "num_failed_objects": num_failed,
            "object_coverage": num_evaluated / num_expected if num_expected else 1.0,
        },
        "primary_metrics": scene_macro if complete else None,
        "partial_scene_macro_metrics": scene_macro,
        "zero_for_missing_scene_macro_metrics": zero_for_missing_scene_macro,
        "object_micro_metrics": object_micro,
        "by_scene": by_scene,
        "aggregation": {
            "primary": "scene_macro",
            "formula": "mean_over_scenes(mean_over_objects_in_scene(metric))",
            "incomplete_run_policy": "primary_metrics is null",
            "bounded_missing_policy": "also report missing predictions as zero",
        },
    }


def manifest_scene_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("Manifest must contain a scenes list")
    result = {scene["scene_id"]: scene for scene in scenes}
    if len(result) != len(scenes):
        raise ValueError("Manifest contains duplicate scene IDs")
    return result


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "ff_scene_geometry_manifest_v1":
        raise ValueError(f"Unsupported manifest schema: {manifest.get('schema')!r}")
    manifest_scene_map(manifest)
    return manifest
