#!/usr/bin/env python3
"""Blender renderer for GT versus percept+reconstruction scene figures."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Matrix
from PIL import Image, ImageDraw, ImageFont


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


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def configure_scene(task: dict[str, Any]) -> tuple[Any, Any]:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = int(task["render_width"])
    scene.render.resolution_y = int(task["render_height"])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = int(task["samples"])
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except (TypeError, ValueError):
        pass
    camera_data = bpy.data.cameras.new("matched_scene_camera")
    camera_data.type = "PERSP"
    camera_data.clip_start = 0.01
    camera_data.clip_end = 1000.0
    camera = bpy.data.objects.new("matched_scene_camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    return scene, camera


def import_glb(path: Path, world_transform: Matrix) -> list[Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    prior = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.context.scene.objects if obj not in prior]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No meshes imported from {path}")
    imported_set = set(imported)
    for obj in imported:
        if obj.parent not in imported_set:
            obj.matrix_world = world_transform @ obj.matrix_world
    for obj in meshes:
        for slot in obj.material_slots:
            material = slot.material
            if material is not None and hasattr(material, "use_backface_culling"):
                material.use_backface_culling = False
    return imported


def uses_inference_rgb_material_policy(task: dict[str, Any]) -> bool:
    return task.get("render_profile", "legacy-comparison") == "inference-rgb"


def import_gt(task: dict[str, Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    objects = []
    changes = []
    for asset in task["gt_assets"]:
        object_to_world = Matrix(asset["object_to_world"])
        # Blender's source-GLB import conversion is the dataset's established
        # raw-GLB to FF-canonical transform. Only generated FF GLBs need the
        # inverse import correction used in import_prediction().
        imported = import_glb(Path(asset["path"]), object_to_world)
        objects.extend(imported)
        if uses_inference_rgb_material_policy(task):
            changes.extend(
                apply_imported_material_policy(
                    imported,
                    asset_name=asset["name"],
                    force_opaque_materials=asset.get(
                        "force_opaque_materials", []
                    ),
                )
            )
    return objects, changes


def belongs_to_instance(obj: Any, instance_id: int) -> bool:
    expected = (
        "background" if instance_id == 0 else f"instance_{instance_id:04d}"
    )
    current = obj
    while current is not None:
        if expected in current.name.lower():
            return True
        current = current.parent
    return False


def import_prediction(
    task: dict[str, Any],
) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
    correction = Matrix.Rotation(-math.pi / 2.0, 4, "X")
    imported = import_glb(Path(task["prediction_glb"]), correction)
    changes = []
    if uses_inference_rgb_material_policy(task):
        changes = apply_imported_material_policy(
            imported,
            asset_name="predicted_textured_world_scene",
            force_opaque_materials=task.get(
                "prediction_force_opaque_materials", []
            ),
        )
    if not task.get("skip_background", False):
        return imported, [], changes
    background_instance_id = int(task["predicted_background_instance_id"])
    background = [
        obj for obj in imported if belongs_to_instance(obj, background_instance_id)
    ]
    foreground = [obj for obj in imported if obj not in background]
    if not any(obj.type == "MESH" for obj in background):
        raise RuntimeError("skip_background requested but no predicted background mesh was found")
    for obj in background:
        obj.hide_render = True
        obj.hide_viewport = True
    return foreground, background, changes


def set_group_visibility(gt_objects: list[Any], pred_objects: list[Any], gt: bool) -> None:
    for obj in gt_objects:
        obj.hide_render = not gt
        obj.hide_viewport = not gt
    for obj in pred_objects:
        obj.hide_render = gt
        obj.hide_viewport = gt


def label_font(size: int, bold: bool = False):
    candidates = (
        Path(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        Path(
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/dejavu/DejaVuSans.ttf"
        ),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def add_panel_label(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    result = image.convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width = max(390, int(result.width * 0.38))
    draw.rectangle((0, 0, width, 86), fill=(13, 18, 25, 196))
    draw.text((22, 14), title, fill=(255, 255, 255, 255), font=label_font(28, True))
    draw.text((22, 52), subtitle, fill=(220, 226, 234, 255), font=label_font(17))
    return Image.alpha_composite(result, overlay).convert("RGB")


def assemble_montage(task: dict[str, Any], tiles: list[dict[str, Any]]) -> Path:
    width = int(task["render_width"])
    height = int(task["render_height"])
    montage = Image.new("RGB", (width * 2, height * 2), (235, 238, 242))
    scope = " | foreground only" if task.get("skip_background", False) else ""
    subtitle = f"{task['dataset']} / {task['scene_id']}{scope}"
    for tile in tiles:
        row = 0 if tile["method"] == "gt" else 1
        column = int(tile["view_rank"])
        title = (
            "Ground truth mesh"
            if tile["method"] == "gt"
            else task["prediction_label"]
        )
        with Image.open(tile["path"]) as source:
            if source.size != (width, height):
                raise RuntimeError(f"Unexpected render dimensions: {source.size}")
            panel = add_panel_label(source, f"{title} | View {column + 1}", subtitle)
            montage.paste(panel, (column * width, row * height))
    output = Path(task["output_dir"]) / "comparison_2view_16x9.png"
    montage.save(output)
    return output


def assemble_gallery(
    task: dict[str, Any], tiles: list[dict[str, Any]], method: str
) -> Path:
    width = int(task["render_width"])
    height = int(task["render_height"])
    method_tiles = sorted(
        (tile for tile in tiles if tile["method"] == method),
        key=lambda tile: int(tile["view_rank"]),
    )
    side = math.isqrt(len(method_tiles))
    if side**2 != len(method_tiles):
        raise ValueError("Sampled-view galleries require a square number of views")
    gallery = Image.new("RGB", (width * side, height * side), (235, 238, 242))
    scope = " | foreground only" if task.get("skip_background", False) else ""
    subtitle = (
        f"{task['dataset']} / {task['scene_id']} | sampled novel view{scope}"
    )
    for tile in method_tiles:
        rank = int(tile["view_rank"])
        title = (
            f"Ground truth mesh | {tile['candidate_id']}"
            if method == "gt"
            else f"{task['prediction_label']} | {tile['candidate_id']}"
        )
        with Image.open(tile["path"]) as source:
            if source.size != (width, height):
                raise RuntimeError(f"Unexpected render dimensions: {source.size}")
            panel = add_panel_label(source, title, subtitle)
            gallery.paste(panel, ((rank % side) * width, (rank // side) * height))
    output = Path(task["output_dir"]) / f"candidate_views_{method}_16x9.png"
    gallery.save(output)
    return output


def main() -> None:
    args = parse_args()
    task = json.loads(args.task.read_text(encoding="utf-8"))
    output_dir = Path(task["output_dir"])
    view_dir = output_dir / "views"
    view_dir.mkdir(parents=True, exist_ok=True)
    camera_payload = json.loads(Path(task["camera_path"]).read_text(encoding="utf-8"))

    clear_scene()
    scene, camera = configure_scene(task)
    gt_objects, gt_material_changes = import_gt(task)
    pred_objects, pred_background, pred_material_changes = import_prediction(task)
    light = apply_recipe(
        task["recipe"], scene=scene, camera=camera, disable_other_lights=True
    )
    renders = []
    for method, use_gt in (("gt", True), ("ours", False)):
        set_group_visibility(gt_objects, pred_objects, use_gt)
        for view_rank, view in enumerate(task["views"]):
            set_camera(
                camera,
                view["frame"],
                camera_payload["K"],
                int(task["source_width"]),
                int(task["source_height"]),
            )
            align_probe_to_camera(light, camera)
            candidate_id = str(view["candidate_id"])
            output = view_dir / f"{method}_{candidate_id}.png"
            if output.is_file() and not args.overwrite:
                status = "existing"
            else:
                scene.render.filepath = str(output)
                bpy.ops.render.render(write_still=True)
                status = "rendered"
            renders.append(
                {
                    "method": method,
                    "view_rank": view_rank,
                    "candidate_id": candidate_id,
                    "view": view,
                    "path": str(output),
                    "status": status,
                }
            )
            print(f"{method} view {view_rank + 1}: {status}", flush=True)

    if task["view_mode"] == "matched-two":
        outputs = [assemble_montage(task, renders)]
    else:
        outputs = [
            assemble_gallery(task, renders, "gt"),
            assemble_gallery(task, renders, "ours"),
        ]
    grid_side = math.isqrt(len(task["views"]))
    montage_dimensions = (
        [task["render_width"] * 2, task["render_height"] * 2]
        if task["view_mode"] == "matched-two"
        else [task["render_width"] * grid_side, task["render_height"] * grid_side]
    )
    summary = {
        "schema": "ff_percept_recon_mesh_comparison_render_v1",
        "dataset": task["dataset"],
        "scene_id": task["scene_id"],
        "camera_path": task["camera_path"],
        "view_mode": task["view_mode"],
        "views": task["views"],
        "panel_dimensions": [task["render_width"], task["render_height"]],
        "montage_dimensions": montage_dimensions,
        "aspect_ratio": "16:9",
        "recipe": task["recipe"],
        "render_profile": task.get("render_profile", "legacy-comparison"),
        "material_policy": task.get("material_policy"),
        "gt_forced_opaque_materials": gt_material_changes,
        "prediction_forced_opaque_materials": pred_material_changes,
        "gt_assets": len(task["gt_assets"]),
        "skip_background": bool(task.get("skip_background", False)),
        "predicted_background_instance_id": task.get(
            "predicted_background_instance_id"
        ),
        "predicted_background_objects_hidden": len(pred_background),
        "prediction_glb": task["prediction_glb"],
        "renders": renders,
        "outputs": [str(output) for output in outputs],
    }
    (output_dir / "render_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for output in outputs:
        print(output, flush=True)


if __name__ == "__main__":
    main()
