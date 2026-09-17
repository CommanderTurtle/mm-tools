#!/usr/bin/env python3
"""Blender-side exact-camera RGB renderer for benchmark scene overlays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Matrix


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scene_reconstruction.render_appearance_blender import (  # noqa: E402
    set_camera,
)
from eval.rendering.blender_pbr_lighting_recipes import (  # noqa: E402
    align_probe_to_camera,
    apply_recipe,
)
from eval.rendering.blender_glb_material_policy import (  # noqa: E402
    apply_imported_material_policy,
)
from eval.rendering.blender_render_percept_recon_mesh_comparison import (  # noqa: E402
    clear_scene,
    import_glb,
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def configure_scene(task: dict[str, Any]) -> tuple[Any, Any]:
    scene = bpy.context.scene
    scene.render.engine = task.get("engine", "BLENDER_EEVEE_NEXT")
    scene.render.resolution_x = int(task["width"])
    scene.render.resolution_y = int(task["height"])
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = int(task["samples"])
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except (TypeError, ValueError):
        pass
    camera_settings = task.get("camera_settings") or {}
    camera_data = bpy.data.cameras.new("exact_dataset_camera")
    camera_data.type = camera_settings.get("type", "PERSP")
    camera_data.clip_start = float(camera_settings.get("clip_start", 0.01))
    camera_data.clip_end = float(camera_settings.get("clip_end", 1000.0))
    camera = bpy.data.objects.new("exact_dataset_camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    return scene, camera


def import_gt_with_material_policy(task: dict[str, Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    imported = []
    changes = []
    for asset in task["gt_assets"]:
        objects = import_glb(Path(asset["path"]), Matrix(asset["object_to_world"]))
        imported.extend(objects)
        changes.extend(
            apply_imported_material_policy(
                objects,
                asset_name=asset["name"],
                force_opaque_materials=asset.get("force_opaque_materials", []),
            )
        )
    return imported, changes


def main() -> None:
    args = parse_args()
    task = json.loads(args.task.read_text(encoding="utf-8"))
    clear_scene()
    scene, camera = configure_scene(task)
    _, material_changes = import_gt_with_material_policy(task)
    light = apply_recipe(
        task["recipe"], scene=scene, camera=camera, disable_other_lights=True
    )
    # Lighting recipes set shared PNG defaults for visualization renders.
    # Reassert the dataset's JPEG contract after applying the recipe.
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.quality = int(task["jpeg_quality"])
    scene.render.use_file_extension = False
    renders = []
    for record in task["frames"]:
        output = Path(record["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        set_camera(
            camera,
            record["camera"],
            task["K"],
            int(task["width"]),
            int(task["height"]),
        )
        align_probe_to_camera(light, camera)
        if output.is_file() and not args.overwrite:
            status = "existing"
        else:
            scene.render.filepath = str(output)
            bpy.ops.render.render(write_still=True)
            status = "rendered"
        renders.append(
            {
                "frame_index": int(record["frame_index"]),
                "output": str(output),
                "status": status,
            }
        )
        print(
            f"{task['scene_id']} frame {int(record['frame_index']):04d}: {status}",
            flush=True,
        )
    summary = {
        "schema": "ff_exact_camera_rgb_blender_render_v2",
        "dataset": task.get("dataset", "imaginarium"),
        "scene_id": task["scene_id"],
        "engine": scene.render.engine,
        "recipe": task["recipe"],
        "resolution": [int(task["width"]), int(task["height"])],
        "K": task["K"],
        "camera_path": task["camera_path"],
        "num_assets": len(task["gt_assets"]),
        "forced_opaque_materials": material_changes,
        "renders": renders,
    }
    output = Path(task["output_dir"]).parent / "exact_camera_rgb_blender_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()
