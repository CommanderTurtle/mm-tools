"""Shared Blender-side material policy for scene GLB rendering."""

from __future__ import annotations

from typing import Any, Iterable


def material_name_matches(material_name: str, source_name: str) -> bool:
    return material_name == source_name or material_name.startswith(source_name + ".")


def set_material_two_sided(material: Any) -> None:
    if hasattr(material, "use_backface_culling"):
        material.use_backface_culling = False
    if hasattr(material, "use_backface_culling_shadow"):
        material.use_backface_culling_shadow = False


def force_material_opaque(material: Any) -> dict[str, Any]:
    """Promote a GLB material already proven to have no effective alpha."""
    before_method = getattr(material, "surface_render_method", None)
    removed_alpha_links = 0
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "DITHERED"
    if hasattr(material, "blend_method"):
        material.blend_method = "OPAQUE"
    diffuse = list(material.diffuse_color)
    diffuse[3] = 1.0
    material.diffuse_color = diffuse
    if material.node_tree is not None:
        for node in material.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED" or "Alpha" not in node.inputs:
                continue
            alpha = node.inputs["Alpha"]
            for link in list(alpha.links):
                material.node_tree.links.remove(link)
                removed_alpha_links += 1
            alpha.default_value = 1.0
    return {
        "material": material.name,
        "method_before": before_method,
        "method_after": getattr(material, "surface_render_method", None),
        "removed_alpha_links": removed_alpha_links,
    }


def apply_imported_material_policy(
    objects: Iterable[Any],
    *,
    asset_name: str,
    force_opaque_materials: Iterable[str],
) -> list[dict[str, Any]]:
    """Apply the inference-RGB two-sided and verified-opaque policy."""
    source_names = tuple(force_opaque_materials)
    seen: set[str] = set()
    changes = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            material = slot.material
            if material is None:
                continue
            set_material_two_sided(material)
            if material.name in seen:
                continue
            seen.add(material.name)
            if any(
                material_name_matches(material.name, source_name)
                for source_name in source_names
            ):
                changes.append(
                    {
                        "asset": asset_name,
                        **force_material_opaque(material),
                    }
                )
    return changes
