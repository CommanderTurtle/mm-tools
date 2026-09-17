# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blender add-on for the SOMA procedural-control reference implementation."""

from .blender_reference import (
    clear_armature_configuration,
    configure_armature,
    evaluate_armature,
    find_repo_definition,
    update_scene,
)

bl_info = {
    "name": "SOMA Procedural Transforms",
    "author": "NVIDIA",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "Properties > Object > SOMA Procedural",
    "description": "Reference consumer for SOMA procedural twist transforms",
    "category": "Rigging",
}

try:
    import bpy
    from bpy.app.handlers import persistent
    from bpy.props import BoolProperty, PointerProperty, StringProperty
    from bpy.types import Operator, Panel, PropertyGroup
except ModuleNotFoundError:
    bpy = None


if bpy is None:

    def register() -> None:
        raise RuntimeError(
            "The SOMA procedural Blender add-on can only be registered inside Blender"
        )

    def unregister() -> None:
        return None

else:

    def _active_armature(context):
        obj = context.object
        if obj is None or obj.type != "ARMATURE":
            return None
        return obj

    def _default_definition_path() -> str:
        path = find_repo_definition()
        return "" if path is None else str(path)

    class SomaProceduralProperties(PropertyGroup):
        """User-facing Blender properties for SOMA procedural armature setup."""

        definition_path: StringProperty(
            name="Definition Path",
            description="Path to assets/SOMA_procedural_transforms.json",
            subtype="FILE_PATH",
            default="",
        )
        enabled: BoolProperty(
            name="Evaluate",
            description="Evaluate SOMA procedural transforms from Blender handlers",
            default=True,
        )

    class SOMA_OT_configure_procedural_armature(Operator):
        """Configure the active armature for SOMA procedural evaluation."""

        bl_idname = "soma.configure_procedural_armature"
        bl_label = "Configure"
        bl_options = {"REGISTER", "UNDO"}

        def execute(self, context):
            armature_object = _active_armature(context)
            if armature_object is None:
                self.report({"ERROR"}, "Select an armature object")
                return {"CANCELLED"}

            props = armature_object.soma_procedural
            definition_path = props.definition_path or _default_definition_path()
            try:
                metadata = configure_armature(
                    armature_object,
                    definition_path=definition_path,
                    enabled=props.enabled,
                )
            except (FileNotFoundError, TypeError, ValueError) as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}

            props.definition_path = definition_path
            self.report(
                {"INFO"},
                f"Configured {len(metadata['twist_joint_names'])} SOMA procedural twist bones",
            )
            return {"FINISHED"}

    class SOMA_OT_update_procedural_armature(Operator):
        """Evaluate SOMA procedural transforms on the active armature once."""

        bl_idname = "soma.update_procedural_armature"
        bl_label = "Update Now"
        bl_options = {"REGISTER", "UNDO"}

        def execute(self, context):
            armature_object = _active_armature(context)
            if armature_object is None:
                self.report({"ERROR"}, "Select an armature object")
                return {"CANCELLED"}

            try:
                written = evaluate_armature(armature_object)
            except (FileNotFoundError, TypeError, ValueError) as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}

            context.view_layer.update()
            self.report({"INFO"}, f"Updated {len(written)} SOMA procedural twist bones")
            return {"FINISHED"}

    class SOMA_OT_clear_procedural_armature(Operator):
        """Disable SOMA procedural evaluation on the active armature."""

        bl_idname = "soma.clear_procedural_armature"
        bl_label = "Clear"
        bl_options = {"REGISTER", "UNDO"}

        def execute(self, context):
            armature_object = _active_armature(context)
            if armature_object is None:
                self.report({"ERROR"}, "Select an armature object")
                return {"CANCELLED"}

            clear_armature_configuration(armature_object)
            armature_object.soma_procedural.enabled = False
            self.report({"INFO"}, "Cleared SOMA procedural armature configuration")
            return {"FINISHED"}

    class SOMA_PT_procedural_armature(Panel):
        """Object properties panel for SOMA procedural evaluation."""

        bl_label = "SOMA Procedural"
        bl_idname = "SOMA_PT_procedural_armature"
        bl_space_type = "PROPERTIES"
        bl_region_type = "WINDOW"
        bl_context = "object"

        @classmethod
        def poll(cls, context):
            return _active_armature(context) is not None

        def draw(self, context):
            layout = self.layout
            armature_object = _active_armature(context)
            props = armature_object.soma_procedural

            layout.prop(props, "definition_path")
            layout.prop(props, "enabled")

            row = layout.row(align=True)
            row.operator(SOMA_OT_configure_procedural_armature.bl_idname)
            row.operator(SOMA_OT_update_procedural_armature.bl_idname)
            layout.operator(SOMA_OT_clear_procedural_armature.bl_idname)

    @persistent
    def _soma_depsgraph_update_post(scene, depsgraph):
        update_scene(scene)

    @persistent
    def _soma_frame_change_post(scene, depsgraph=None):
        update_scene(scene)

    CLASSES = (
        SomaProceduralProperties,
        SOMA_OT_configure_procedural_armature,
        SOMA_OT_update_procedural_armature,
        SOMA_OT_clear_procedural_armature,
        SOMA_PT_procedural_armature,
    )

    def _append_handler(handler_list, handler) -> None:
        if handler not in handler_list:
            handler_list.append(handler)

    def _remove_handler(handler_list, handler) -> None:
        if handler in handler_list:
            handler_list.remove(handler)

    def register() -> None:
        for cls in CLASSES:
            bpy.utils.register_class(cls)
        bpy.types.Object.soma_procedural = PointerProperty(type=SomaProceduralProperties)
        _append_handler(bpy.app.handlers.depsgraph_update_post, _soma_depsgraph_update_post)
        _append_handler(bpy.app.handlers.frame_change_post, _soma_frame_change_post)

    def unregister() -> None:
        _remove_handler(bpy.app.handlers.frame_change_post, _soma_frame_change_post)
        _remove_handler(bpy.app.handlers.depsgraph_update_post, _soma_depsgraph_update_post)
        if hasattr(bpy.types.Object, "soma_procedural"):
            del bpy.types.Object.soma_procedural
        for cls in reversed(CLASSES):
            bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
