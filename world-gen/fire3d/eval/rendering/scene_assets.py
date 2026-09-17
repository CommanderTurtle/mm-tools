"""Ground-truth scene asset contracts used by the public comparison renderer."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from eval.rendering.rerender_imaginarium_exact_camera_rgb import (
    compose_matrix,
    fully_opaque_blend_material_names,
)


def load_scene_transforms(path: Path) -> dict[str, dict[str, Any]]:
    """Load trusted benchmark transforms generated with the release data."""

    with path.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, dict):
        raise TypeError(f"Expected transform dictionary in {path}, got {type(records)!r}")
    return records


def gt_object_asset(
    dataset_root: Path,
    scene: dict[str, Any],
    transforms: dict[str, dict[str, Any]],
    object_id: int,
) -> dict[str, Any]:
    name = f"object_{int(object_id):04d}"
    path = dataset_root / scene["mesh_dir"] / f"{name}.glb"
    if name not in transforms or not path.is_file():
        raise FileNotFoundError(f"Missing GT object asset or transform: {path}")
    return {
        "name": name,
        "object_id": int(object_id),
        "scene_role": "object",
        "kind": "gt_raw_glb",
        "path": str(path.resolve()),
        "object_to_world": compose_matrix(transforms[name]).tolist(),
        "force_opaque_materials": fully_opaque_blend_material_names(path),
    }


def gt_layout_asset(
    dataset_root: Path,
    scene: dict[str, Any],
    transforms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    layout_names = sorted(name for name in transforms if name.startswith("layout_"))
    if len(layout_names) != 1:
        raise ValueError(f"Expected one layout transform, found {layout_names}")
    name = layout_names[0]
    path = dataset_root / scene["mesh_dir"] / f"{name}.glb"
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "name": name,
        "scene_role": "layout",
        "kind": "gt_raw_glb",
        "path": str(path.resolve()),
        "object_to_world": compose_matrix(transforms[name]).tolist(),
        "force_opaque_materials": fully_opaque_blend_material_names(path),
    }


def full_gt_scene_assets(
    dataset_root: Path,
    scene: dict[str, Any],
    transforms: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    object_ids = [
        int(value) for value in scene.get("gt_object_ids", scene["visible_object_ids"])
    ]
    if len(set(object_ids)) != len(object_ids):
        raise ValueError(f"Duplicate GT object IDs: {object_ids}")
    objects = [
        gt_object_asset(dataset_root, scene, transforms, object_id)
        for object_id in object_ids
    ]
    return objects, gt_layout_asset(dataset_root, scene, transforms)
