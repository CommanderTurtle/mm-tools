#!/usr/bin/env python3
"""Canonical Blender lighting recipes for decoded PBR objects and scenes.

The recipe definitions can be inspected with ordinary Python. Applying a
recipe requires Blender's Python runtime::

    blender input.blend -b --python eval/rendering/blender_pbr_lighting_recipes.py \
      -- --recipe canonical_pbr --apply --render-output output.png

For programmatic use, import ``apply_recipe`` after the camera and materials
have been created. Call ``align_probe_to_camera`` after every camera update.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class LightingRecipe:
    name: str
    use_case: str
    engine: str
    view_transform: str
    world_strength: float
    sun_energy: float
    material_mode: str
    emission_strength: float = 0.0
    world_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    sun_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    sun_shadows: bool = False
    sun_camera_aligned: bool = True
    transparent_film: bool = True
    composite_background_rgb: tuple[int, int, int] = (255, 255, 255)
    exposure: float = 0.0
    gamma: float = 1.0

    def profile_dict(self) -> dict[str, Any]:
        """Return the profile schema used by the existing render scripts."""
        return {
            "engine": self.engine,
            "world_strength": self.world_strength,
            "sun_energy": self.sun_energy,
            "view_transform": self.view_transform,
            "material_mode": self.material_mode,
            "emission_strength": self.emission_strength,
        }


RECIPES: dict[str, LightingRecipe] = {
    "object_max_psnr": LightingRecipe(
        name="object_max_psnr",
        use_case="Match the existing isolated-object pyrender observations.",
        engine="BLENDER_EEVEE_NEXT",
        view_transform="Standard",
        world_strength=0.4,
        sun_energy=2.75,
        material_mode="full_pbr",
    ),
    "scene_soft_pbr_e080_s050": LightingRecipe(
        name="scene_soft_pbr_e080_s050",
        use_case="Soft canonical material probe for reconstructed enclosed rooms.",
        engine="BLENDER_EEVEE_NEXT",
        view_transform="Standard",
        world_strength=0.0,
        sun_energy=0.5,
        material_mode="pbr_emission_ambient",
        emission_strength=0.8,
    ),
    "albedo_unlit": LightingRecipe(
        name="albedo_unlit",
        use_case="Lighting-independent base-color diagnostic; not a full-PBR probe.",
        engine="BLENDER_EEVEE_NEXT",
        view_transform="Standard",
        world_strength=0.0,
        sun_energy=0.0,
        material_mode="albedo_emission",
        emission_strength=1.0,
    ),
}

ALIASES = {
    "canonical_pbr": "object_max_psnr",
    "eevee_match_w040_s275": "object_max_psnr",
    "scene_soft_pbr": "scene_soft_pbr_e080_s050",
}

# Stable public default for both future object and scene rendering. The alias
# resolves to the benchmark-proven ``object_max_psnr`` recipe above.
DEFAULT_RECIPE = "canonical_pbr"

BENCHMARKS: dict[str, dict[str, Any]] = {
    "object_max_psnr": {
        "protocol": "25 objects x 8 views, 200 total views",
        "full_psnr": 29.101673578944574,
        "full_ssim": 0.9347569977492094,
        "crop_psnr": 24.49558640732357,
        "crop_ssim": 0.8529718304425478,
        "mask_iou": 0.9632178273604519,
        "scene_crosscheck_five_dataset_macro": {
            "psnr": 14.126371992754333,
            "ssim": 0.6423982073863348,
        },
    },
    "scene_soft_pbr_e080_s050": {
        "protocol": "one scene from each of five datasets, 6 views per scene",
        "dataset_macro_psnr": 14.684388903338599,
        "dataset_macro_ssim": 0.6215827643871308,
        "object_crosscheck_200_views": {
            "full_psnr": 28.06562632218028,
            "full_ssim": 0.9378566573932767,
            "crop_psnr": 23.459425139762335,
            "crop_ssim": 0.8614337246492505,
            "mask_iou": 0.9632178273604519,
        },
    },
}


def get_recipe(name: str) -> LightingRecipe:
    canonical = ALIASES.get(name, name)
    try:
        return RECIPES[canonical]
    except KeyError as exc:
        choices = sorted(set(RECIPES) | set(ALIASES))
        raise KeyError(f"Unknown lighting recipe {name!r}; choices: {choices}") from exc


def recipe_profile(name: str) -> dict[str, Any]:
    return get_recipe(name).profile_dict()


def _require_bpy() -> Any:
    try:
        import bpy  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Applying a lighting recipe requires Blender's Python runtime") from exc
    return bpy


def _iter_principled_nodes(materials: Iterable[Any] | None = None) -> Iterable[tuple[Any, Any]]:
    bpy = _require_bpy()
    source = bpy.data.materials if materials is None else materials
    for material in source:
        tree = getattr(material, "node_tree", None)
        if tree is None:
            continue
        for node in tree.nodes:
            if node.bl_idname == "ShaderNodeBsdfPrincipled":
                yield tree, node


def _set_socket_constant(tree: Any, node: Any, names: tuple[str, ...], value: Any) -> None:
    socket = next((node.inputs.get(name) for name in names if node.inputs.get(name) is not None), None)
    if socket is None:
        return
    for link in list(socket.links):
        tree.links.remove(link)
    socket.default_value = value


def _connect_base_color_to_emission(tree: Any, node: Any, strength: float) -> None:
    base = node.inputs.get("Base Color")
    emission = node.inputs.get("Emission Color") or node.inputs.get("Emission")
    emission_strength = node.inputs.get("Emission Strength")
    if base is None or emission is None:
        return
    for link in list(emission.links):
        tree.links.remove(link)
    if base.links:
        tree.links.new(base.links[0].from_socket, emission)
    else:
        emission.default_value = base.default_value
    if emission_strength is not None:
        emission_strength.default_value = strength


def apply_material_recipe(recipe: LightingRecipe, materials: Iterable[Any] | None = None) -> None:
    """Apply only the recipe's intentional material override.

    ``full_pbr`` leaves imported material channels unchanged. The scene recipe
    adds base-color emission while preserving metallic and roughness. The
    albedo diagnostic disables metallic/specular response.
    """
    if recipe.material_mode == "full_pbr":
        return
    for tree, node in _iter_principled_nodes(materials):
        if recipe.material_mode == "albedo_emission":
            _set_socket_constant(tree, node, ("Metallic",), 0.0)
            _set_socket_constant(tree, node, ("Roughness",), 1.0)
            _set_socket_constant(tree, node, ("Specular IOR Level", "Specular"), 0.0)
        _connect_base_color_to_emission(tree, node, recipe.emission_strength)


def configure_world(scene: Any, recipe: LightingRecipe) -> Any:
    bpy = _require_bpy()
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("canonical_pbr_world")
        scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is None:
        background = world.node_tree.nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = recipe.world_color
    background.inputs["Strength"].default_value = recipe.world_strength
    return world


def configure_render(scene: Any, recipe: LightingRecipe) -> None:
    scene.render.engine = recipe.engine
    scene.render.film_transparent = recipe.transparent_film
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = recipe.view_transform
    try:
        scene.view_settings.look = "None"
    except TypeError:
        pass
    scene.view_settings.exposure = recipe.exposure
    scene.view_settings.gamma = recipe.gamma
    if hasattr(scene.view_settings, "use_curve_mapping"):
        scene.view_settings.use_curve_mapping = False


def align_probe_to_camera(light: Any, camera: Any) -> None:
    """Copy the camera pose so the Sun remains a camera-facing headlight."""
    light.matrix_world = camera.matrix_world.copy()


def ensure_camera_probe_sun(
    scene: Any,
    camera: Any,
    recipe: LightingRecipe,
    *,
    disable_other_lights: bool = True,
) -> Any:
    bpy = _require_bpy()
    name = "canonical_camera_probe_sun"
    light = bpy.data.objects.get(name)
    if light is None or light.type != "LIGHT" or light.data.type != "SUN":
        data = bpy.data.lights.new(name, type="SUN")
        light = bpy.data.objects.new(name, data)
        scene.collection.objects.link(light)
    light.hide_render = False
    light.data.color = recipe.sun_color
    light.data.energy = recipe.sun_energy
    light.data.angle = 0.0
    if hasattr(light.data, "use_shadow"):
        light.data.use_shadow = recipe.sun_shadows
    if disable_other_lights:
        for obj in scene.objects:
            if obj.type == "LIGHT" and obj != light:
                obj.hide_render = True
    if recipe.sun_camera_aligned:
        align_probe_to_camera(light, camera)
    return light


def apply_recipe(
    name: str,
    *,
    scene: Any | None = None,
    camera: Any | None = None,
    materials: Iterable[Any] | None = None,
    disable_other_lights: bool = True,
) -> Any:
    """Apply a named recipe and return its camera-aligned Sun object."""
    bpy = _require_bpy()
    recipe = get_recipe(name)
    scene = bpy.context.scene if scene is None else scene
    camera = scene.camera if camera is None else camera
    if camera is None:
        raise RuntimeError("The scene needs an active camera before applying a lighting recipe")
    configure_render(scene, recipe)
    configure_world(scene, recipe)
    apply_material_recipe(recipe, materials)
    return ensure_camera_probe_sun(
        scene,
        camera,
        recipe,
        disable_other_lights=disable_other_lights,
    )


def _payload(name: str) -> dict[str, Any]:
    recipe = get_recipe(name)
    return {"recipe": asdict(recipe), "benchmark": BENCHMARKS.get(recipe.name)}


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", default=DEFAULT_RECIPE, choices=sorted(set(RECIPES) | set(ALIASES)))
    parser.add_argument("--list", action="store_true", help="Print every recipe and benchmark as JSON.")
    parser.add_argument("--apply", action="store_true", help="Apply the recipe to the open Blender scene.")
    parser.add_argument("--camera", help="Blender camera object name; defaults to scene.camera.")
    parser.add_argument("--render-output", type=Path, help="Apply the recipe and render one still image.")
    parser.add_argument("--opaque-background", action="store_true", help="Render the world instead of RGBA transparency.")
    parser.add_argument("--dump-json", type=Path, help="Save the resolved recipe and benchmark metadata.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.list:
        payload = {name: _payload(name) for name in RECIPES}
    else:
        payload = _payload(args.recipe)

    if args.dump_json is not None:
        args.dump_json.parent.mkdir(parents=True, exist_ok=True)
        args.dump_json.write_text(json.dumps(payload, indent=2) + "\n")

    if args.apply or args.render_output is not None:
        bpy = _require_bpy()
        camera = bpy.data.objects.get(args.camera) if args.camera else bpy.context.scene.camera
        light = apply_recipe(args.recipe, camera=camera)
        if args.opaque_background:
            bpy.context.scene.render.film_transparent = False
        if args.render_output is not None:
            args.render_output.parent.mkdir(parents=True, exist_ok=True)
            bpy.context.scene.render.filepath = str(args.render_output)
            align_probe_to_camera(light, camera)
            bpy.ops.render.render(write_still=True)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
