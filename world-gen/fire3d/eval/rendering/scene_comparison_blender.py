#!/usr/bin/env python3
"""Render GT and external-baseline world meshes from identical saved cameras."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Matrix
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scene_reconstruction.render_appearance_blender import set_camera  # noqa: E402
from eval.rendering.render_common import (  # noqa: E402
    background_exclusion_filters_disagree,
    replace_empty_background,
)
from eval.rendering.blender_glb_material_policy import (  # noqa: E402
    apply_imported_material_policy,
)
from eval.rendering.blender_render_percept_recon_mesh_comparison import (  # noqa: E402
    import_glb,
)
from eval.rendering.blender_rerender_imaginarium_rgb import (  # noqa: E402
    configure_scene as configure_inference_rgb_scene,
)
from eval.rendering.blender_pbr_lighting_recipes import (  # noqa: E402
    align_probe_to_camera,
    apply_recipe,
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Render only this method key; repeat as needed.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def configure_scene(task: dict[str, Any]) -> tuple[Any, Any]:
    return configure_inference_rgb_scene(task)


def configure_inference_rgb_beauty_output(scene: Any, task: dict[str, Any]) -> None:
    """Keep the refreshed-input opaque-film rasterization with lossless output."""
    output = task.get("output_settings") or {}
    scene.render.image_settings.file_format = output.get("file_format", "PNG")
    scene.render.image_settings.color_mode = output.get("color_mode", "RGB")
    scene.render.image_settings.color_depth = output.get("color_depth", "8")
    scene.render.film_transparent = bool(output.get("film_transparent", False))
    scene.render.use_file_extension = bool(output.get("use_file_extension", True))


LEGACY_INSTANCE_PALETTE = (
    (0.43, 0.61, 0.71, 1.0),
    (0.69, 0.54, 0.39, 1.0),
    (0.45, 0.67, 0.49, 1.0),
    (0.67, 0.47, 0.58, 1.0),
    (0.62, 0.61, 0.39, 1.0),
    (0.49, 0.52, 0.70, 1.0),
)


def _diverse_instance_color(object_id: int) -> tuple[float, float, float, float]:
    """Golden-ratio hue stepping: adjacent ids land far apart on the wheel."""
    import colorsys

    hue = (object_id * 0.6180339887498949) % 1.0
    saturation = 0.62 if object_id % 2 == 0 else 0.78
    value = 0.86 if object_id % 3 else 0.68
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (r, g, b, 1.0)


def instance_material(
    object_id: int,
    palette: str = "legacy",
    override_rgba: tuple[float, float, float, float] | None = None,
) -> Any:
    material = bpy.data.materials.new(f"shaper_object_{object_id:04d}")
    material.use_nodes = True
    node = material.node_tree.nodes.get("Principled BSDF")
    if override_rgba is not None:
        color = tuple(override_rgba)
    elif palette == "diverse":
        color = _diverse_instance_color(object_id)
    else:
        color = LEGACY_INSTANCE_PALETTE[object_id % len(LEGACY_INSTANCE_PALETTE)]
    node.inputs["Base Color"].default_value = color
    node.inputs["Metallic"].default_value = 0.0
    node.inputs["Roughness"].default_value = 0.72
    return material


def belongs_to_instance(obj: Any, instance_id: int) -> bool:
    """Match a node to an instance id under either composer's naming.

    `lc64_shape_pbr_decode.compose_textured_scene` names instance 0
    `background_position_0000`, while `unified_baseline_inference.compose_scene`
    names it `instance_0000_position_0000`. Accepting both is what lets the
    imported baselines (Gen3DSR ships a background mesh) use the same
    exclude/gray paths as our own reconstructions.
    """

    expected = [f"instance_{instance_id:04d}"]
    if instance_id == 0:
        expected.append("background")
    current = obj
    while current is not None:
        name = current.name.lower()
        if any(candidate in name for candidate in expected):
            return True
        current = current.parent
    return False


def belongs_to_named_background(obj: Any) -> bool:
    current = obj
    while current is not None:
        if "background" in current.name.lower():
            return True
        current = current.parent
    return False


def belongs_to_node_name(obj: Any, node_name: str) -> bool:
    """Match an imported GLB node by exact name through its parent chain."""

    current = obj
    while current is not None:
        if current.name == node_name:
            return True
        current = current.parent
    return False


def apply_view_node_exclusions(
    asset_groups: list[tuple[dict[str, Any], list[tuple[Any, bool, bool]]]],
    view_index: int,
) -> int:
    """Apply and validate task-recorded per-view GLB node exclusions."""

    hidden_meshes = 0
    for asset, object_states in asset_groups:
        names = tuple((asset.get("exclude_nodes_by_view") or {}).get(str(view_index), []))
        matched = {name: False for name in names}
        for obj, base_render, base_viewport in object_states:
            excluded = False
            for name in names:
                if belongs_to_node_name(obj, name):
                    excluded = True
                    if obj.type == "MESH":
                        matched[name] = True
            obj.hide_render = base_render or excluded
            obj.hide_viewport = base_viewport or excluded
            if excluded and obj.type == "MESH":
                hidden_meshes += 1
        missing = [name for name, found in matched.items() if not found]
        if missing:
            raise RuntimeError(
                f"Asset {asset['path']} has no rendered mesh below requested "
                f"view-exclusion nodes {missing}"
            )
    return hidden_meshes


def object_instance_id(obj: Any) -> int | None:
    current = obj
    while current is not None:
        name = current.name.lower()
        match = re.search(r"instance_(\d+)", name)
        if match is not None:
            return int(match.group(1))
        if "background" in name:
            # Composition names this node by transform index instead of giving it
            # an instance_NNNN token. It is instance 0, matching the convention
            # already used by belongs_to_instance().
            return 0
        current = current.parent
    return None


def apply_geometry_material_mode(imported: list[Any], asset: dict[str, Any]) -> None:
    mode = asset.get("material_mode", "textured")
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if mode == "textured":
        return
    if mode == "instance_color":
        material = instance_material(
            int(asset["object_id"]),
            palette=asset.get("instance_palette", "legacy"),
            override_rgba=(
                tuple(asset["override_rgba"]) if asset.get("override_rgba") else None
            ),
        )
        assign_material(meshes, material)
        return
    if mode == "instance_color_by_instance":
        palette = asset.get("instance_palette", "legacy")
        gray_instance = asset.get("gray_instance_id")
        gray_rgba = tuple(asset.get("gray_rgba", (0.5, 0.5, 0.5, 1.0)))
        materials = {}
        for obj in meshes:
            instance_id = object_instance_id(obj)
            if instance_id is None:
                raise RuntimeError(
                    f"Cannot recover instance ID for mesh {obj.name} in {asset['path']}"
                )
            material = materials.get(instance_id)
            if material is None:
                override = (
                    gray_rgba
                    if gray_instance is not None and instance_id == int(gray_instance)
                    else None
                )
                material = instance_material(
                    instance_id, palette=palette, override_rgba=override
                )
                materials[instance_id] = material
            assign_material([obj], material)
        return
    raise ValueError(f"Unsupported material mode: {mode}")


def import_asset(asset: dict[str, Any]) -> list[Any]:
    path = Path(asset["path"])
    kind = asset["kind"]
    if kind == "gt_raw_glb":
        imported = import_glb(path, Matrix(asset["object_to_world"]))
    elif kind == "ff_canonical_glb":
        correction = Matrix.Rotation(-math.pi / 2.0, 4, "X")
        imported = import_glb(path, Matrix(asset["object_to_world"]) @ correction)
    elif kind in {"world_glb", "world_scene_glb"}:
        correction = Matrix.Rotation(-math.pi / 2.0, 4, "X")
        imported = import_glb(path, correction)
    elif kind == "world_ply":
        before = set(bpy.context.scene.objects)
        bpy.ops.wm.ply_import(filepath=str(path))
        imported = [obj for obj in bpy.context.scene.objects if obj not in before]
        meshes = [obj for obj in imported if obj.type == "MESH"]
        if len(meshes) != 1:
            raise RuntimeError(f"Expected one PLY mesh from {path}, found {len(meshes)}")
        material = instance_material(int(asset["object_id"]))
        meshes[0].data.materials.clear()
        meshes[0].data.materials.append(material)
    else:
        raise ValueError(f"Unsupported asset kind: {kind}")

    apply_imported_material_policy(
        imported,
        asset_name=asset["name"],
        force_opaque_materials=asset.get("force_opaque_materials", []),
    )
    include_filters = sum(
        key in asset
        for key in ("include_named_background", "include_instance_id")
    )
    exclude_filters = sum(
        key in asset
        for key in ("exclude_named_background", "exclude_instance_id")
    )
    if include_filters > 1 or (include_filters and exclude_filters):
        raise ValueError(
            "An asset must use at most one include filter, or one or more exclude filters"
        )
    if asset.get("include_named_background"):
        included = [obj for obj in imported if belongs_to_named_background(obj)]
        if not any(obj.type == "MESH" for obj in included):
            raise RuntimeError(f"Named background was not found in composed asset {path}")
        for obj in imported:
            if obj.type == "MESH" and obj not in included:
                obj.hide_render = True
                obj.hide_viewport = True
        imported = [obj for obj in imported if obj.type != "MESH" or obj in included]
    if "include_instance_id" in asset:
        instance_id = int(asset["include_instance_id"])
        included = [obj for obj in imported if belongs_to_instance(obj, instance_id)]
        if not any(obj.type == "MESH" for obj in included):
            raise RuntimeError(
                f"Instance {instance_id} was not found in composed asset {path}"
            )
        for obj in imported:
            if obj.type == "MESH" and obj not in included:
                obj.hide_render = True
                obj.hide_viewport = True
        imported = [obj for obj in imported if obj.type != "MESH" or obj in included]
    excluded = []
    exclusion_labels = []
    instance_excluded: list[Any] = []
    named_excluded: list[Any] = []
    if "exclude_instance_id" in asset:
        instance_id = int(asset["exclude_instance_id"])
        exclusion_labels.append(f"instance {instance_id}")
        instance_excluded = [
            obj for obj in imported if belongs_to_instance(obj, instance_id)
        ]
        excluded.extend(instance_excluded)
    if asset.get("exclude_named_background"):
        exclusion_labels.append("named background")
        named_excluded = [obj for obj in imported if belongs_to_named_background(obj)]
        excluded.extend(named_excluded)
    excluded = list(dict.fromkeys(excluded))
    if exclusion_labels and not any(obj.type == "MESH" for obj in excluded):
        labels = " or ".join(exclusion_labels)
        raise RuntimeError(f"Could not find {labels} in composed asset {path}")
    instance_meshes = {obj.name for obj in instance_excluded if obj.type == "MESH"}
    named_meshes = {obj.name for obj in named_excluded if obj.type == "MESH"}
    if background_exclusion_filters_disagree(instance_meshes, named_meshes):
        raise RuntimeError(
            "Background exclusion filters disagree in composed asset "
            f"{path}: the instance filter removed {sorted(instance_meshes)} while "
            f"the named-background filter removed {sorted(named_meshes)}. The "
            "composed background node name is assigned by transform index, so one "
            "of these filters is deleting foreground geometry."
        )
    for obj in excluded:
        obj.hide_render = True
        obj.hide_viewport = True
    imported = [obj for obj in imported if obj not in excluded]
    apply_geometry_material_mode(imported, asset)
    return imported


def emission_material(name: str, color: tuple[float, float, float, float]) -> Any:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def assign_material(objects: list[Any], material: Any) -> None:
    for obj in objects:
        if obj.type != "MESH":
            continue
        obj.data.materials.clear()
        obj.data.materials.append(material)


def visible_mask_pixels(path: Path) -> int:
    with Image.open(path) as image:
        histogram = image.convert("L").histogram()
    return int(sum(histogram[1:]))


def _scene_bounds(objects):
    points = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Matrix.Translation(corner).to_translation())
    if not points:
        return None, None
    lows = [min(p[i] for p in points) for i in range(3)]
    highs = [max(p[i] for p in points) for i in range(3)]
    centre = [(lows[i] + highs[i]) / 2 for i in range(3)]
    extent = [highs[i] - lows[i] for i in range(3)]
    return centre, extent


def _lift_dark_texels(material, black_point: float):
    """Insert an RGB curve lifting the black point on the base-color input."""

    if not material.use_nodes:
        return
    tree = material.node_tree
    principled = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if principled is None:
        return
    base = principled.inputs["Base Color"]
    if not base.is_linked:
        return
    source = base.links[0].from_socket
    curves = tree.nodes.new("ShaderNodeRGBCurve")
    curves.label = "beauty_dark_lift"
    for channel in range(3):
        curve = curves.mapping.curves[channel]
        curve.points[0].location = (0.0, black_point)
    curves.mapping.update()
    tree.links.new(source, curves.inputs["Color"])
    tree.links.new(curves.outputs["Color"], base)


def _floor_roughness(material, floor: float):
    if not material.use_nodes:
        return
    tree = material.node_tree
    principled = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if principled is None:
        return
    rough = principled.inputs["Roughness"]
    if rough.is_linked:
        source = rough.links[0].from_socket
        clamp = tree.nodes.new("ShaderNodeMath")
        clamp.operation = "MAXIMUM"
        clamp.inputs[1].default_value = floor
        tree.links.new(source, clamp.inputs[0])
        tree.links.new(clamp.outputs[0], rough)
    else:
        rough.default_value = max(float(rough.default_value), floor)


def _beauty_compositor(scene, bloom: bool, vignette: bool):
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
    tail = layers.outputs["Image"]
    if bloom:
        glare = tree.nodes.new("CompositorNodeGlare")
        glare.glare_type = "BLOOM" if "BLOOM" in {
            item.identifier for item in glare.bl_rna.properties["glare_type"].enum_items
        } else "FOG_GLOW"
        glare.quality = "MEDIUM"
        glare.mix = -0.7
        glare.threshold = 1.0
        tree.links.new(tail, glare.inputs["Image"])
        tail = glare.outputs["Image"]
    if vignette:
        ellipse = tree.nodes.new("CompositorNodeEllipseMask")
        ellipse.width = 1.35
        ellipse.height = 1.15
        blur = tree.nodes.new("CompositorNodeBlur")
        blur.size_x = 400
        blur.size_y = 400
        blur.use_relative = False
        ramp = tree.nodes.new("CompositorNodeMath")
        ramp.operation = "MULTIPLY_ADD"
        ramp.inputs[1].default_value = 0.25
        ramp.inputs[2].default_value = 0.75
        mix = tree.nodes.new("CompositorNodeMixRGB")
        mix.blend_type = "MULTIPLY"
        mix.inputs[0].default_value = 1.0
        tree.links.new(ellipse.outputs["Mask"], blur.inputs["Image"])
        tree.links.new(blur.outputs["Image"], ramp.inputs[0])
        tree.links.new(tail, mix.inputs[1])
        tree.links.new(ramp.outputs["Value"], mix.inputs[2])
        tail = mix.outputs["Image"]
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(tail, composite.inputs["Image"])
    scene.render.use_compositing = True


def _despeckle_image(image, threshold: float, erosions: int, fill_iterations: int) -> int:
    """Remove SMALL dark specks from a base-color image, in memory only.

    Our atlases carry two distinct dark populations: isolated specks (median
    4-17 texels) which read as black dots, and huge connected regions (up to
    116 k texels, 44% of a 512 atlas) which are unfilled atlas area and real
    dark undersides. A blanket black-point lift hits both and grays out real
    shadows; a morphological OPENING separates them, so only the specks are
    refilled from their surrounding colour.

    Pure numpy (Blender's bundled Python has no scipy) and operates on the
    image datablock, so nothing on disk changes.
    """

    import numpy as np

    width, height = image.size
    if width == 0 or height == 0:
        return 0
    buffer = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(buffer)
    pixels = buffer.reshape(height, width, 4)
    rgb = pixels[..., :3]
    dark = rgb.max(axis=2) < threshold
    if not dark.any():
        return 0

    def erode(mask):
        out = mask.copy()
        out[1:, :] &= mask[:-1, :]
        out[:-1, :] &= mask[1:, :]
        out[:, 1:] &= mask[:, :-1]
        out[:, :-1] &= mask[:, 1:]
        return out

    def dilate(mask):
        out = mask.copy()
        out[1:, :] |= mask[:-1, :]
        out[:-1, :] |= mask[1:, :]
        out[:, 1:] |= mask[:, :-1]
        out[:, :-1] |= mask[:, 1:]
        return out

    large = dark
    for _ in range(max(1, erosions)):
        large = erode(large)
    for _ in range(max(1, erosions)):
        large = dilate(large)
    specks = dark & ~large
    speck_count = int(specks.sum())
    if speck_count == 0:
        return 0

    filled = rgb.copy()
    holes = specks.copy()
    for _ in range(max(1, fill_iterations)):
        if not holes.any():
            break
        valid = ~holes
        accumulated = np.zeros_like(filled)
        counts = np.zeros(holes.shape, dtype=np.float32)
        for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour_valid = np.roll(valid, shift, axis=(0, 1))
            neighbour_rgb = np.roll(filled, shift, axis=(0, 1))
            accumulated += np.where(neighbour_valid[..., None], neighbour_rgb, 0.0)
            counts += neighbour_valid
        fillable = holes & (counts > 0)
        if not fillable.any():
            break
        filled[fillable] = accumulated[fillable] / counts[fillable][:, None]
        holes &= ~fillable

    pixels[..., :3] = filled
    image.pixels.foreach_set(pixels.reshape(-1))
    image.update()
    return speck_count


def _label_components(dark, height, width):
    """4-connected labels via one raster pass plus union-find (numpy only).

    Exact component sizes are what let the fill be size-aware; the morphological
    opening used by the v1 despeckle can only approximate "small". A flat Python
    list is used for the scan because per-pixel numpy scalar indexing dominates
    the runtime otherwise.
    """

    import numpy as np

    flat = dark.reshape(-1)
    labels = [0] * (height * width)
    parent = [0]

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    next_label = 1
    for index in np.nonzero(flat)[0].tolist():
        row = index // width
        column = index - row * width
        up = labels[index - width] if row > 0 else 0
        left = labels[index - 1] if column > 0 else 0
        if up and left:
            labels[index] = up
            root_up, root_left = find(up), find(left)
            if root_up != root_left:
                parent[root_left] = root_up
        elif up:
            labels[index] = up
        elif left:
            labels[index] = left
        else:
            labels[index] = next_label
            parent.append(next_label)
            next_label += 1

    lookup = np.fromiter(
        (find(i) for i in range(next_label)), dtype=np.int32, count=next_label
    )
    return lookup[np.asarray(labels, dtype=np.int32)].reshape(height, width), next_label


def _repair_dark_regions(
    image, threshold: float, max_size: int, contrast: float,
    fill_iterations: int, blur: bool,
) -> tuple[int, int]:
    """Refill dark regions that are punched into lighter material, in memory.

    Measured on our atlases: EVERY dark component sits inside a much brighter
    border (luminance gap 0.28-0.72, including a 117 k-texel one at luminance
    0.017), so they are all holes rather than genuinely dark surfaces. The
    border-contrast test is therefore a safety net -- it protects a real dark
    underside if one ever appears -- while `max_size` governs fill quality,
    since neighbour-averaging a very large hole smears rather than repairs.
    """

    import numpy as np

    width, height = image.size
    if width == 0 or height == 0:
        return 0, 0
    buffer = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(buffer)
    pixels = buffer.reshape(height, width, 4)
    rgb = pixels[..., :3]
    luminance = rgb.max(axis=2)
    dark = luminance < threshold
    if not dark.any():
        return 0, 0

    labels, label_count = _label_components(dark, height, width)
    sizes = np.bincount(labels.reshape(-1), minlength=label_count)
    sums = np.bincount(labels.reshape(-1), weights=luminance.reshape(-1),
                       minlength=label_count)
    means = np.divide(sums, np.maximum(sizes, 1))

    # border luminance per component: every bright pixel adjacent to a dark one
    border_sum = np.zeros(label_count, dtype=np.float64)
    border_count = np.zeros(label_count, dtype=np.float64)
    for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        neighbour_dark = np.roll(dark, shift, axis=(0, 1))
        neighbour_lum = np.roll(luminance, shift, axis=(0, 1))
        edge = dark & ~neighbour_dark
        if not edge.any():
            continue
        np.add.at(border_sum, labels[edge], neighbour_lum[edge])
        np.add.at(border_count, labels[edge], 1.0)
    border_mean = np.divide(border_sum, np.maximum(border_count, 1.0))

    repairable = np.zeros(label_count, dtype=bool)
    valid = (sizes > 0) & (border_count > 0)
    repairable[valid] = (
        (sizes[valid] <= max_size)
        & ((border_mean[valid] - means[valid]) >= contrast)
    )
    repairable[0] = False
    holes = repairable[labels]
    repaired = int(holes.sum())
    if repaired == 0:
        return 0, int(label_count - 1)

    filled = rgb.copy()
    remaining = holes.copy()
    for _ in range(max(1, fill_iterations)):
        if not remaining.any():
            break
        available = ~remaining
        accumulated = np.zeros_like(filled)
        counts = np.zeros(remaining.shape, dtype=np.float32)
        for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour_ok = np.roll(available, shift, axis=(0, 1))
            neighbour_rgb = np.roll(filled, shift, axis=(0, 1))
            accumulated += np.where(neighbour_ok[..., None], neighbour_rgb, 0.0)
            counts += neighbour_ok
        fillable = remaining & (counts > 0)
        if not fillable.any():
            break
        filled[fillable] = accumulated[fillable] / counts[fillable][:, None]
        remaining &= ~fillable

    if blur:
        # the iterative fill leaves faint radial banding inside big holes
        smoothed = filled.copy()
        for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            smoothed += np.roll(filled, shift, axis=(0, 1))
        smoothed /= 5.0
        filled[holes] = smoothed[holes]

    pixels[..., :3] = filled
    image.pixels.foreach_set(pixels.reshape(-1))
    image.update()
    return repaired, int(label_count - 1)


def _principled_input(node, names):
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            return socket
    return None


def _smooth_material(material, style: dict):
    """Material-side smoothing: interpolation, metallic/specular taming, sheen.

    Our baked 512 atlases are viewed well above native texel scale, so Cubic
    interpolation smooths texel noise and the piecewise-linear gradient breaks
    that make chart seams pop. Predicted metallic is noisy and indoor objects
    are rarely metal, so scaling it down removes patchy mirror sheen; a low
    specular level kills the plastic look flat shading produces; a touch of
    sheen and subsurface reads as soft fabric rather than vinyl.
    """

    if not material.use_nodes:
        return
    tree = material.node_tree
    if style.get("texture_interpolation"):
        for node in tree.nodes:
            if node.type == "TEX_IMAGE":
                node.interpolation = str(style["texture_interpolation"]).title()
    principled = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if principled is None:
        return
    if style.get("metallic_scale") is not None:
        metallic = principled.inputs.get("Metallic")
        if metallic is not None:
            factor = float(style["metallic_scale"])
            if metallic.is_linked:
                source = metallic.links[0].from_socket
                scale = tree.nodes.new("ShaderNodeMath")
                scale.operation = "MULTIPLY"
                scale.inputs[1].default_value = factor
                tree.links.new(source, scale.inputs[0])
                tree.links.new(scale.outputs[0], metallic)
            else:
                metallic.default_value = float(metallic.default_value) * factor
    if style.get("specular_level") is not None:
        socket = _principled_input(
            principled, ("Specular IOR Level", "Specular")
        )
        if socket is not None and not socket.is_linked:
            socket.default_value = float(style["specular_level"])
    if style.get("sheen_weight") is not None:
        socket = _principled_input(principled, ("Sheen Weight", "Sheen"))
        if socket is not None and not socket.is_linked:
            socket.default_value = float(style["sheen_weight"])
    if style.get("subsurface_weight") is not None:
        socket = _principled_input(
            principled, ("Subsurface Weight", "Subsurface")
        )
        if socket is not None and not socket.is_linked:
            socket.default_value = float(style["subsurface_weight"])


def _beauty_grade(scene, saturation: float, contrast: float):
    """Gentle pop: saturation and an S-curve, composited after everything else."""

    scene.use_nodes = True
    tree = scene.node_tree
    composite = next(
        (n for n in tree.nodes if n.type == "COMPOSITE"), None
    )
    if composite is None:
        tree.nodes.clear()
        layers = tree.nodes.new("CompositorNodeRLayers")
        composite = tree.nodes.new("CompositorNodeComposite")
        tree.links.new(layers.outputs["Image"], composite.inputs["Image"])
    upstream = composite.inputs["Image"].links[0].from_socket
    tail = upstream
    if saturation and saturation != 1.0:
        huesat = tree.nodes.new("CompositorNodeHueSat")
        huesat.inputs["Saturation"].default_value = saturation
        tree.links.new(tail, huesat.inputs["Image"])
        tail = huesat.outputs["Image"]
    if contrast:
        curves = tree.nodes.new("CompositorNodeCurveRGB")
        curve = curves.mapping.curves[3]  # combined
        curve.points.new(0.25, max(0.0, 0.25 - contrast))
        curve.points.new(0.75, min(1.0, 0.75 + contrast))
        curves.mapping.update()
        tree.links.new(tail, curves.inputs["Image"])
        tail = curves.outputs["Image"]
    for link in list(composite.inputs["Image"].links):
        tree.links.remove(link)
    tree.links.new(tail, composite.inputs["Image"])
    scene.render.use_compositing = True


def apply_beauty_style(scene, style: dict, objects, sun=None) -> None:
    """Presentation-only overrides on top of the measurement recipe.

    Every knob is task-driven and recorded upstream in the render index; the
    default (no ``beauty_style`` in the task) leaves the recipe untouched, so
    ``normal`` renders stay byte-comparable with every previous run.
    """

    # The canonical recipe keeps a shadowless camera-aligned sun at 2.75; a
    # beauty rig that merely ADDS its key/world/GI on top overexposes any scene
    # that is already bright (v0 blew out bedroom_01 and FloorPlan24). Scaling
    # the sun down is what lets the soft key take over as the dominant light.
    if style.get("sun_scale") is not None and sun is not None:
        try:
            sun.data.energy = float(sun.data.energy) * float(style["sun_scale"])
        except AttributeError:
            print("beauty: recipe light has no energy attribute; sun_scale skipped")

    view = style.get("view_transform")
    if view:
        scene.view_settings.view_transform = view
    look = style.get("look")
    if look:
        try:
            scene.view_settings.look = look
        except TypeError:
            print(f"beauty: look {look!r} unavailable; keeping default")
    if style.get("exposure") is not None:
        scene.view_settings.exposure = float(style["exposure"])
    if style.get("samples"):
        scene.eevee.taa_render_samples = int(style["samples"])

    if style.get("sky_world"):
        world = scene.world
        world.use_nodes = True
        tree = world.node_tree
        tree.nodes.clear()
        sky = tree.nodes.new("ShaderNodeTexSky")
        background = tree.nodes.new("ShaderNodeBackground")
        background.inputs["Strength"].default_value = float(
            style.get("sky_strength", 0.5)
        )
        out = tree.nodes.new("ShaderNodeOutputWorld")
        tree.links.new(sky.outputs["Color"], background.inputs["Color"])
        tree.links.new(background.outputs["Background"], out.inputs["Surface"])

    if style.get("soft_key"):
        centre, extent = _scene_bounds(objects)
        if centre is not None:
            span = max(extent[0], extent[1], 1.0)
            key_data = bpy.data.lights.new("beauty_key", type="AREA")
            key_data.energy = float(style.get("key_energy", 400.0)) * span
            key_data.size = span * 0.9
            key_data.use_shadow = True
            if style.get("key_color"):
                key_data.color = tuple(style["key_color"])
            key = bpy.data.objects.new("beauty_key", key_data)
            key.location = (centre[0], centre[1], centre[2] + extent[2] * 1.2 + 1.0)
            scene.collection.objects.link(key)
            rim_data = bpy.data.lights.new("beauty_rim", type="AREA")
            rim_data.energy = float(style.get("rim_energy", 120.0)) * span
            rim_data.size = span * 0.6
            rim_data.color = (0.75, 0.85, 1.0)
            rim = bpy.data.objects.new("beauty_rim", rim_data)
            rim.location = (
                centre[0] - extent[0], centre[1] - extent[1],
                centre[2] + extent[2] * 0.8,
            )
            rim.rotation_euler = (0.9, 0.0, -0.785)
            scene.collection.objects.link(rim)

    if style.get("auto_smooth_degrees"):
        import math

        angle = math.radians(float(style["auto_smooth_degrees"]))
        meshes = [
            obj for obj in objects if obj.type == "MESH" and len(obj.data.polygons) > 0
        ]
        if not meshes:
            raise RuntimeError("beauty: smooth-by-angle requested with no meshes")
        bpy.ops.object.select_all(action="DESELECT")
        for obj in meshes:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        status = bpy.ops.object.shade_smooth_by_angle(
            angle=angle,
            keep_sharp_edges=bool(style.get("auto_smooth_keep_sharp", True)),
        )
        if "FINISHED" not in status:
            raise RuntimeError(f"beauty: smooth-by-angle returned {status}")
        smooth_faces = sum(
            int(polygon.use_smooth) for obj in meshes for polygon in obj.data.polygons
        )
        if smooth_faces == 0:
            raise RuntimeError(
                "beauty: smooth-by-angle finished but changed no polygon shading"
            )
        print(
            f"beauty: smooth-by-angle {style['auto_smooth_degrees']} degrees "
            f"applied to {len(meshes)} meshes / {smooth_faces} faces",
            flush=True,
        )

    if style.get("smooth_factor"):
        for obj in objects:
            if obj.type != "MESH":
                continue
            modifier = obj.modifiers.new("beauty_smooth", "SMOOTH")
            modifier.factor = float(style["smooth_factor"])
            modifier.iterations = int(style.get("smooth_iterations", 2))

    if style.get("weighted_normal"):
        # Area+angle weighted corner normals: the documented fix for shading
        # artifacts on imperfect imported geometry. Needs smooth shading, which
        # auto_smooth_degrees above has already applied.
        for obj in objects:
            if obj.type != "MESH":
                continue
            modifier = obj.modifiers.new("beauty_weighted_normal", "WEIGHTED_NORMAL")
            modifier.mode = "FACE_AREA_WITH_ANGLE"
            modifier.keep_sharp = bool(style.get("weighted_normal_keep_sharp", True))
            modifier.weight = int(style.get("weighted_normal_weight", 50))

    if style.get("despeckle_components"):
        threshold = float(style.get("despeckle_threshold", 0.16))
        max_size = int(style.get("despeckle_max_size", 20000))
        contrast = float(style.get("despeckle_contrast", 0.15))
        fill_iterations = int(style.get("despeckle_fill_iterations", 96))
        blur = bool(style.get("despeckle_blur", False))
        done, repaired, components = set(), 0, 0
        for obj in objects:
            if obj.type != "MESH":
                continue
            for slot in obj.material_slots:
                material = slot.material
                if material is None or not material.use_nodes:
                    continue
                for node in material.node_tree.nodes:
                    if node.type != "TEX_IMAGE" or node.image is None:
                        continue
                    base = node.outputs.get("Color")
                    if base is None or not base.is_linked:
                        continue
                    targets = {link.to_socket.name for link in base.links}
                    if not targets & {"Base Color", "Color"}:
                        continue
                    if node.image.name in done:
                        continue
                    done.add(node.image.name)
                    count, total = _repair_dark_regions(
                        node.image, threshold, max_size, contrast,
                        fill_iterations, blur,
                    )
                    repaired += count
                    components += total
        print(f"beauty: repaired {repaired} dark texels "
              f"({components} components seen) across {len(done)} images",
              flush=True)
    elif style.get("despeckle"):
        threshold = float(style.get("despeckle_threshold", 0.16))
        erosions = int(style.get("despeckle_erosions", 2))
        fill_iterations = int(style.get("despeckle_fill_iterations", 12))
        done, total = set(), 0
        for obj in objects:
            if obj.type != "MESH":
                continue
            for slot in obj.material_slots:
                material = slot.material
                if material is None or not material.use_nodes:
                    continue
                for node in material.node_tree.nodes:
                    if node.type != "TEX_IMAGE" or node.image is None:
                        continue
                    base = node.outputs.get("Color")
                    if base is None or not base.is_linked:
                        continue
                    # base-colour images only; leave metallic/roughness alone
                    targets = {link.to_socket.name for link in base.links}
                    if not targets & {"Base Color", "Color"}:
                        continue
                    if node.image.name in done:
                        continue
                    done.add(node.image.name)
                    total += _despeckle_image(
                        node.image, threshold, erosions, fill_iterations
                    )
        print(f"beauty: despeckled {total} dark texels across {len(done)} images",
              flush=True)

    seen = set()
    for obj in objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            material = slot.material
            if material is None or material.name in seen:
                continue
            seen.add(material.name)
            if style.get("dark_lift"):
                _lift_dark_texels(material, float(style["dark_lift"]))
            if style.get("roughness_floor"):
                _floor_roughness(material, float(style["roughness_floor"]))
            _smooth_material(material, style)

    if style.get("world_color") or style.get("world_strength") is not None:
        world = scene.world
        if world is not None and world.use_nodes:
            background = world.node_tree.nodes.get("Background")
            if background is not None:
                if style.get("world_color"):
                    background.inputs["Color"].default_value = (
                        *style["world_color"], 1.0
                    )
                if style.get("world_strength") is not None:
                    background.inputs["Strength"].default_value = float(
                        style["world_strength"]
                    )

    if style.get("raytracing"):
        try:
            scene.eevee.use_raytracing = True
        except AttributeError:
            print("beauty: eevee raytracing unavailable in this build")

    if style.get("bloom") or style.get("vignette"):
        _beauty_compositor(
            scene, bool(style.get("bloom")), bool(style.get("vignette"))
        )
    if style.get("grade_saturation") or style.get("grade_contrast"):
        _beauty_grade(
            scene,
            float(style.get("grade_saturation", 1.0)),
            float(style.get("grade_contrast", 0.0)),
        )


def render_method(
    task: dict[str, Any], method: dict[str, Any], camera_payload: dict[str, Any], overwrite: bool
) -> list[dict[str, Any]]:
    intentionally_blank = bool(method.get("intentionally_blank"))
    if intentionally_blank and method.get("assets"):
        raise ValueError(
            f"Intentionally blank method {method['key']} cannot contain assets"
        )
    if intentionally_blank:
        output_dir = Path(task["output_dir"]) / "views"
        output_dir.mkdir(parents=True, exist_ok=True)
        size = (int(task["width"]), int(task["height"]))
        records = []
        for view in task["views"]:
            view_index = int(view["view_index"])
            beauty = output_dir / f"{method['key']}_v{view_index:04d}_beauty.png"
            mask = output_dir / f"{method['key']}_v{view_index:04d}_mask.png"
            if overwrite or not beauty.is_file():
                Image.new("RGB", size, (255, 255, 255)).save(beauty)
            if overwrite or not mask.is_file():
                Image.new("L", size, 0).save(mask)
            records.append(
                {
                    "method": method["key"],
                    "view_index": view_index,
                    "beauty": str(beauty),
                    "mask": str(mask),
                    "intentionally_blank": True,
                    "blank_reason": method["blank_reason"],
                }
            )
        return records
    clear_scene()
    scene, camera = configure_scene(task)
    context_objects = []
    for asset in task.get("context_assets", []):
        context_objects.extend(import_asset(asset))
    method_objects = []
    asset_groups = []
    background_objects = []
    for asset in method["assets"]:
        imported = import_asset(asset)
        method_objects.extend(imported)
        asset_groups.append(
            (
                asset,
                [(obj, bool(obj.hide_render), bool(obj.hide_viewport)) for obj in imported],
            )
        )
        if asset.get("scene_role") == "ours_background":
            background_objects.extend(imported)
    light = apply_recipe(
        task["recipe"], scene=scene, camera=camera, disable_other_lights=True
    )
    configure_inference_rgb_beauty_output(scene, task)
    # Presentation styles apply to the TEXTURED render only. The geometry
    # protocol is an instance-colour measurement figure: beauty lighting, the
    # material knobs and the texture repair would all corrupt it, and its
    # materials are synthetic anyway. Gating here (rather than dropping the
    # geometry protocol) is what lets a beauty style be the default for our
    # renders without losing the geometry sheets.
    style = task.get("beauty_style") or {}
    style_targets = task.get("beauty_style_applies_to", ["texture"])
    method_protocol = (
        "texture" if "texture" in str(method.get("key", "")) else "geometry"
    )
    if style and method_protocol in style_targets:
        apply_beauty_style(
            scene, style, method_objects + context_objects, sun=light
        )
    output_dir = Path(task["output_dir"]) / "views"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    intrinsics = task.get("K", camera_payload["K"])

    for view in task["views"]:
        view_index = int(view["view_index"])
        hidden_meshes = apply_view_node_exclusions(asset_groups, view_index)
        if hidden_meshes:
            print(
                f"{method['key']} view {view_index}: excluded "
                f"{hidden_meshes} configured layout meshes",
                flush=True,
            )
        set_camera(
            camera,
            view["frame"],
            intrinsics,
            int(task["width"]),
            int(task["height"]),
        )
        align_probe_to_camera(light, camera)
        beauty = output_dir / f"{method['key']}_v{view_index:04d}_beauty.png"
        if overwrite or not beauty.is_file():
            scene.render.filepath = str(beauty)
            bpy.ops.render.render(write_still=True)

    black = emission_material("mask_context_black", (0.0, 0.0, 0.0, 1.0))
    white = emission_material("mask_method_white", (1.0, 1.0, 1.0, 1.0))
    assign_material(context_objects, black)
    scene.render.film_transparent = False
    if scene.world is not None and scene.world.use_nodes:
        background = scene.world.node_tree.nodes.get("Background")
        if background is not None:
            background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            background.inputs["Strength"].default_value = 0.0

    background_masks = {}
    if background_objects:
        assign_material(method_objects, black)
        assign_material(background_objects, white)
        for view in task["views"]:
            view_index = int(view["view_index"])
            apply_view_node_exclusions(asset_groups, view_index)
            set_camera(
                camera,
                view["frame"],
                intrinsics,
                int(task["width"]),
                int(task["height"]),
            )
            mask = output_dir / f"{method['key']}_v{view_index:04d}_background_mask.png"
            if overwrite or not mask.is_file():
                scene.render.filepath = str(mask)
                bpy.ops.render.render(write_still=True)
            background_masks[view_index] = {
                "path": str(mask),
                "visible_pixels": visible_mask_pixels(mask),
            }

    assign_material(method_objects, white)
    for view in task["views"]:
        view_index = int(view["view_index"])
        apply_view_node_exclusions(asset_groups, view_index)
        set_camera(
            camera,
            view["frame"],
            intrinsics,
            int(task["width"]),
            int(task["height"]),
        )
        beauty = output_dir / f"{method['key']}_v{view_index:04d}_beauty.png"
        mask = output_dir / f"{method['key']}_v{view_index:04d}_mask.png"
        if overwrite or not mask.is_file():
            scene.render.filepath = str(mask)
            bpy.ops.render.render(write_still=True)
        if (
            task.get("background_policy", {}).get(method["key"]) == "none"
            and task.get("empty_background_rgb") is not None
        ):
            if not overwrite:
                raise RuntimeError(
                    "Solid empty-background compositing requires --overwrite to avoid "
                    "recompositing antialiased edges"
                )
            replace_empty_background(
                beauty, mask, list(task["empty_background_rgb"])
            )
        record = {
            "method": method["key"],
            "view_index": view_index,
            "beauty": str(beauty),
            "mask": str(mask),
        }
        if intentionally_blank:
            record["intentionally_blank"] = True
            record["blank_reason"] = method["blank_reason"]
        if view_index in background_masks:
            record["background_mask"] = background_masks[view_index]["path"]
            record["background_visible_pixels"] = background_masks[view_index][
                "visible_pixels"
            ]
        records.append(record)
        print(
            f"{method['key']} view {view_index}: beauty + foreground mask rendered",
            flush=True,
        )
    return records


def main() -> None:
    args = parse_args()
    task = json.loads(args.task.read_text(encoding="utf-8"))
    camera_payload = json.loads(Path(task["camera_path"]).read_text(encoding="utf-8"))
    methods_by_key = {method["key"]: method for method in task["methods"]}
    requested = list(dict.fromkeys(args.method))
    unknown = sorted(set(requested) - set(methods_by_key))
    if unknown:
        raise ValueError(f"Unknown method keys: {unknown}")
    selected_methods = (
        [methods_by_key[key] for key in requested]
        if requested
        else list(task["methods"])
    )
    records = []
    for method in selected_methods:
        records.extend(render_method(task, method, camera_payload, args.overwrite))
    rendered_keys = {method["key"] for method in selected_methods}
    for method in task["methods"]:
        if method["key"] in rendered_keys:
            continue
        for view in task["views"]:
            view_index = int(view["view_index"])
            beauty = (
                Path(task["output_dir"])
                / "views"
                / f"{method['key']}_v{view_index:04d}_beauty.png"
            )
            mask = beauty.with_name(beauty.name.replace("_beauty.png", "_mask.png"))
            if not beauty.is_file() or not mask.is_file():
                raise FileNotFoundError(
                    f"Partial render requires existing beauty and mask: {beauty}, {mask}"
                )
            record = {
                "method": method["key"],
                "view_index": view_index,
                "beauty": str(beauty),
                "mask": str(mask),
            }
            if method.get("intentionally_blank"):
                record["intentionally_blank"] = True
                record["blank_reason"] = method["blank_reason"]
            if any(
                asset.get("scene_role") == "ours_background"
                for asset in method.get("assets", [])
            ):
                background_mask = beauty.with_name(
                    beauty.name.replace("_beauty.png", "_background_mask.png")
                )
                if not background_mask.is_file():
                    raise FileNotFoundError(background_mask)
                record["background_mask"] = str(background_mask)
                record["background_visible_pixels"] = visible_mask_pixels(
                    background_mask
                )
            records.append(record)
    summary = {
        "schema": "ff_external_baseline_scene_render_v2",
        "task": str(args.task.resolve()),
        "coordinate_contract": "dataset_world_without_posthoc_alignment",
        "engine": task.get("engine", "BLENDER_EEVEE_NEXT"),
        "recipe": task["recipe"],
        "render_profile": task.get("render_profile", "inference-rgb"),
        "beauty_style": task.get("beauty_style"),
        "beauty_style_applies_to": task.get(
            "beauty_style_applies_to", ["texture"]
        ),
        "output_settings": task.get("output_settings"),
        "camera_settings": task.get("camera_settings"),
        "beauty_render_profile": "inference_rgb_opaque",
        "beauty_film_transparent": bool(
            (task.get("output_settings") or {}).get("film_transparent", False)
        ),
        "beauty_color_mode": (task.get("output_settings") or {}).get(
            "color_mode", "RGB"
        ),
        "resolution": [int(task["width"]), int(task["height"])],
        "K": task.get("K", camera_payload["K"]),
        "aspect_policy": task.get("aspect_policy"),
        "background_policy": task.get("background_policy", {}),
        "empty_background_rgb": task.get("empty_background_rgb"),
        "evaluation_object_ids": task.get("evaluation_object_ids"),
        "rendered_gt_object_ids": task.get("rendered_gt_object_ids"),
        "rendered_methods": [method["key"] for method in selected_methods],
        "intentionally_blank_methods": [
            method["key"]
            for method in task["methods"]
            if method.get("intentionally_blank")
        ],
        "records": records,
    }
    output = Path(task["output_dir"]) / "render_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    output.chmod(0o644)


if __name__ == "__main__":
    main()
