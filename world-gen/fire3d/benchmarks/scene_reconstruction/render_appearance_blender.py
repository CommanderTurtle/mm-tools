#!/usr/bin/env python3
"""Render isolated predicted PBR objects at exact scene cameras and GT poses."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Matrix, Vector

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.rendering.blender_pbr_lighting_recipes import (  # noqa: E402
    align_probe_to_camera,
    apply_recipe,
)
def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--recipe", default="canonical_pbr")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def configure_scene(width: int, height: int) -> tuple[Any, Any]:
    scene = bpy.context.scene
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    camera_data = bpy.data.cameras.new("exact_benchmark_camera")
    camera_data.type = "PERSP"
    camera_data.clip_start = 0.01
    camera_data.clip_end = 1000.0
    camera = bpy.data.objects.new("exact_benchmark_camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    return scene, camera


def camera_pose(frame: dict[str, Any]) -> Matrix:
    eye = Vector(frame["eye"])
    lookat = Vector(frame["lookat"])
    up_hint = Vector(frame["up"]).normalized()
    forward = (lookat - eye).normalized()
    right = forward.cross(up_hint).normalized()
    up = right.cross(forward).normalized()
    return Matrix(
        (
            (right.x, up.x, -forward.x, eye.x),
            (right.y, up.y, -forward.y, eye.y),
            (right.z, up.z, -forward.z, eye.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def set_camera(camera: Any, frame: dict[str, Any], intrinsics: list[list[float]], width: int, height: int) -> None:
    camera.matrix_world = camera_pose(frame)
    fx, fy = float(intrinsics[0][0]), float(intrinsics[1][1])
    cx, cy = float(intrinsics[0][2]), float(intrinsics[1][2])
    camera.data.sensor_fit = "VERTICAL" if fy >= fx else "HORIZONTAL"
    if camera.data.sensor_fit == "VERTICAL":
        camera.data.sensor_height = 32.0
        camera.data.lens = fy * camera.data.sensor_height / float(height)
    else:
        camera.data.sensor_width = 32.0
        camera.data.lens = fx * camera.data.sensor_width / float(width)
    camera.data.shift_x = (float(width) * 0.5 - cx) / float(width)
    camera.data.shift_y = (cy - float(height) * 0.5) / float(height)


def import_prediction(path: Path, object_to_world: Matrix) -> list[Any]:
    prior = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.context.scene.objects if obj not in prior]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects imported from {path}")
    imported_set = set(imported)
    # Keep this Blender-side correction self-contained: importing the general
    # geometry protocol would pull trimesh into Blender's bundled Python.
    imported_glb_to_ff = Matrix.Rotation(-math.pi / 2.0, 4, "X")
    for obj in imported:
        if obj.parent not in imported_set:
            # Blender has already converted the glTF's Y-up coordinates into
            # Blender Z-up, changing FF canonical (x,y,z) to (x,-z,y). Undo
            # that import conversion before applying the dataset's FF-canonical
            # object-to-world pose. This matches the established exact-camera
            # PBR renderer in eval/rendering/render_pbr_x2_flow_posed_pair_blender.py.
            obj.matrix_world = object_to_world @ imported_glb_to_ff @ obj.matrix_world
    for obj in meshes:
        for polygon in obj.data.polygons:
            polygon.use_smooth = False
        for slot in obj.material_slots:
            if slot.material is not None and hasattr(slot.material, "use_backface_culling"):
                slot.material.use_backface_culling = False
    return imported


def set_visible(objects: list[list[Any]], selected: int) -> None:
    for index, group in enumerate(objects):
        visible = index == selected
        for obj in group:
            obj.hide_render = not visible
            obj.hide_viewport = not visible


def main() -> None:
    args = parse_args()
    payload = json.loads(args.tasks.read_text())
    camera_data = json.loads(Path(payload["camera_path"]).read_text())
    width, height = int(camera_data["width"]), int(camera_data["height"])
    clear_scene()
    scene, camera = configure_scene(width, height)
    groups = [
        import_prediction(Path(task["textured_glb"]), Matrix(task["object_to_world"]))
        for task in payload["objects"]
    ]
    light = apply_recipe(
        args.recipe, scene=scene, camera=camera, disable_other_lights=True
    )
    renders = []
    for object_index, task in enumerate(payload["objects"]):
        set_visible(groups, object_index)
        for view in task["views"]:
            view_index = int(view["view_index"])
            set_camera(camera, camera_data["frames"][view_index], camera_data["K"], width, height)
            align_probe_to_camera(light, camera)
            output = Path(view["prediction_rgba"])
            if output.is_file() and not args.overwrite:
                status = "existing"
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                scene.render.filepath = str(output)
                bpy.ops.render.render(write_still=True)
                status = "rendered"
            renders.append(
                {
                    "object_id": task["object_id"],
                    "view_index": view_index,
                    "prediction_rgba": str(output),
                    "status": status,
                }
            )
            print(
                f"object_{int(task['object_id']):04d} view {view_index}: {status}",
                flush=True,
            )
    summary = {
        "schema": "ff_scene_appearance_exact_camera_render_v1",
        "tasks": str(args.tasks),
        "recipe": args.recipe,
        "camera_path": payload["camera_path"],
        "canonical_frame_correction": "blender_imported_gltf_to_ff_canonical_x_minus_90",
        "renders": renders,
    }
    output = args.tasks.parent / "render_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
