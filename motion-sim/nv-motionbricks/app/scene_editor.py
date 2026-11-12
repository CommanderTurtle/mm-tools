# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Build scenes with Smart Primitives, Solid Cubes, and a CharacterStart."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

import warp as wp

from motionbricks.model.contract import MODEL_POSES_PER_TOKEN, ModelRig
from motionbricks.renderers.base_renderer import BaseRenderer
from motionbricks.renderers.wire_box_renderer import WireBoxRenderer
from motionbricks.renderers.picking import raycast_lines
from motionbricks.renderers.solid_box_renderer import SolidBoxRenderer
from motionbricks.runtime.smart_primitive import (
    InteractionBox,
    SmartPrimitive,
)

from .utils.cli import configure_logging, create_gl_parser
from .utils.file_dialog import choose_file
from .runtime.common.rig import RigAssets, create_mesh_renderer, load_rig_assets
from .runtime.common.smart_primitive_behaviors import behavior_types
from .runtime.common.io.smart_primitive_document import (
    SMART_PRIMITIVE_DOCUMENT,
    SmartPrimitiveAssetPool,
    load_smart_primitive_rig,
)
from .runtime.common.io.scene_document import (
    SCENE_DOCUMENT,
    SceneDefinition,
    SceneItem,
    SolidCube,
    SmartPrimitiveInstance,
    check_reference_rigs,
    load_scene_document,
    save_scene_document,
)

from .authoring.common import (
    EditorViewer,
    InlineRename,
    NAME_TEXT_COLOR,
    PROPERTIES_WINDOW_HEIGHT,
    PROPERTIES_WINDOW_WIDTH,
    WINDOW_MARGIN,
    accept_drop,
    apply_transform_locks,
    drag_source,
    draw_add_remove_buttons,
    draw_box_dimensions,
    draw_character_selector,
    draw_clip_selector,
    draw_transform_editor,
    draw_tree_row,
    log_transform_gizmo,
    transforms_equal,
)


_SCENE_WINDOW_WIDTH = 320.0
_SCENE_WINDOW_HEIGHT = 380.0
_MIN_SOLID_CUBE_DIMENSION = 0.01
_CHARACTER_START_RADIUS = 0.35
_CHARACTER_START_CIRCLE_SEGMENTS = 32
_CHARACTER_START_ARROW_LENGTH = 0.8
_CHARACTER_START_MARKER_HEIGHT = 0.01
_CHARACTER_START_COLOR = wp.vec3(0.25, 0.85, 1.0)
_INTERACTION_BOX_COLOR = wp.vec3(1.0, 0.15, 0.10)
_SCENE_ITEM_PAYLOAD = "SCENE_ITEM"


@wp.kernel
def _transform_marker_lines_kernel(
    transform: wp.transform,
    local_starts: wp.array(dtype=wp.vec3),
    local_ends: wp.array(dtype=wp.vec3),
    world_starts: wp.array(dtype=wp.vec3),
    world_ends: wp.array(dtype=wp.vec3),
):
    index = wp.tid()
    world_starts[index] = wp.transform_point(transform, local_starts[index])
    world_ends[index] = wp.transform_point(transform, local_ends[index])


class _CharacterStartRenderer(BaseRenderer):
    _NAME = "/scene_character_start"

    def __init__(self, device=None) -> None:
        super().__init__()
        starts = []
        ends = []
        for segment in range(_CHARACTER_START_CIRCLE_SEGMENTS):
            angle0 = math.tau * segment / _CHARACTER_START_CIRCLE_SEGMENTS
            angle1 = math.tau * (segment + 1) / _CHARACTER_START_CIRCLE_SEGMENTS
            starts.append(
                (
                    _CHARACTER_START_RADIUS * math.cos(angle0),
                    _CHARACTER_START_RADIUS * math.sin(angle0),
                    _CHARACTER_START_MARKER_HEIGHT,
                )
            )
            ends.append(
                (
                    _CHARACTER_START_RADIUS * math.cos(angle1),
                    _CHARACTER_START_RADIUS * math.sin(angle1),
                    _CHARACTER_START_MARKER_HEIGHT,
                )
            )

        arrow_length = _CHARACTER_START_ARROW_LENGTH
        arrow_head_x = arrow_length - 0.18
        starts.extend(
            (
                (0.0, 0.0, _CHARACTER_START_MARKER_HEIGHT),
                (arrow_length, 0.0, _CHARACTER_START_MARKER_HEIGHT),
                (arrow_length, 0.0, _CHARACTER_START_MARKER_HEIGHT),
            )
        )
        ends.extend(
            (
                (arrow_length, 0.0, _CHARACTER_START_MARKER_HEIGHT),
                (arrow_head_x, 0.12, _CHARACTER_START_MARKER_HEIGHT),
                (arrow_head_x, -0.12, _CHARACTER_START_MARKER_HEIGHT),
            )
        )

        self._pick_lines = [(wp.vec3(*start), wp.vec3(*end)) for start, end in zip(starts, ends)]

        self.local_starts = wp.array(starts, dtype=wp.vec3, device=device)
        self.local_ends = wp.array(ends, dtype=wp.vec3, device=device)
        self.world_starts = wp.empty_like(self.local_starts)
        self.world_ends = wp.empty_like(self.local_ends)
        self.colors = wp.array(
            [_CHARACTER_START_COLOR] * len(starts),
            dtype=wp.vec3,
            device=device,
        )

    def raycast(self, transform: wp.transform, origin: wp.vec3, direction: wp.vec3, max_distance: float) -> float:
        return raycast_lines(transform, self._pick_lines, origin, direction, max_distance)

    def draw(self, viewer, transform: wp.transform) -> None:
        wp.launch(
            _transform_marker_lines_kernel,
            dim=self.local_starts.shape[0],
            inputs=[transform, self.local_starts, self.local_ends],
            outputs=[self.world_starts, self.world_ends],
            device=self.local_starts.device,
        )
        self._log_visible(
            viewer.log_lines,
            self._NAME,
            self.world_starts,
            self.world_ends,
            self.colors,
        )

    def clear(self, viewer) -> None:
        self._clear(viewer.lines)


class _SceneSmartPrimitiveView:
    """Draw one placed Smart Primitive and keep its previews up to date."""

    def __init__(
        self,
        definition: SmartPrimitiveInstance,
        runtime: SmartPrimitive,
        model,
        reference_pose: wp.array,
        transform: wp.transform,
    ) -> None:
        self.definition = definition
        self.runtime = runtime
        with wp.ScopedDevice(runtime.device):
            self.target_poses = wp.empty(
                (MODEL_POSES_PER_TOKEN, *reference_pose.shape),
                dtype=reference_pose.dtype,
            )
            self.interaction_boxes = wp.empty(
                max(1, runtime.interaction_box_count),
                dtype=InteractionBox,
            )
            self.mesh_renderer = create_mesh_renderer(runtime.rig, model)
            self.wire_box_renderer = WireBoxRenderer(
                max_boxes=max(1, runtime.interaction_box_count)
            )
        self._target_pose_sequence_index: int | None = None
        self._target_pose_node_indices: frozenset[int] | None = None
        self.set_transform(transform)

    @property
    def target_pose_sequence_index(self) -> int | None:
        """The previewed sequence, or None to show all target poses."""

        return self._target_pose_sequence_index

    @target_pose_sequence_index.setter
    def target_pose_sequence_index(self, sequence_index: int | None) -> None:
        if sequence_index is None:
            self._target_pose_sequence_index = None
            self._target_pose_node_indices = None
            return

        target_names = self.runtime.target_names_for_sequence(sequence_index)
        self._target_pose_sequence_index = sequence_index
        self._target_pose_node_indices = frozenset(
            self.runtime.node_index(name) for name in target_names
        )

    def set_transform(self, transform: wp.transform) -> None:
        self.runtime.set_node_world_transform(0, transform)

    def _target_is_visible(self, node_index: int) -> bool:
        return self.definition.show_target_poses and (
            self._target_pose_node_indices is None
            or node_index in self._target_pose_node_indices
        )

    def raycast(self, origin: wp.vec3, direction: wp.vec3, max_distance: float) -> float:
        for node_index in range(self.runtime.node_count):
            if self.runtime.has_target_pose(node_index) and self._target_is_visible(node_index):
                self.runtime.get_world_target_poses(node_index, self.target_poses)
                max_distance = self.mesh_renderer.raycast_pose(
                    self.target_poses[0], wp.transform_identity(), origin, direction, max_distance,
                )
        box_count = self.runtime.interaction_box_count
        if self.definition.show_interaction_boxes and box_count:
            self.runtime.get_world_interaction_boxes(self.interaction_boxes)
            max_distance = self.wire_box_renderer.raycast_batch(
                self.interaction_boxes, box_count, origin, direction, max_distance,
            )
        return max_distance

    def draw(self, viewer) -> None:
        runtime = self.runtime
        definition = self.definition
        for node_index in range(runtime.node_count):
            if not runtime.has_target_pose(node_index):
                continue
            renderer_id = f"scene_{definition.id}_target_{node_index}"
            if self._target_is_visible(node_index):
                runtime.get_world_target_poses(
                    node_index,
                    self.target_poses,
                )
                self.mesh_renderer.draw_pose(
                    viewer,
                    self.target_poses[0],
                    wp.transform_identity(),
                    definition.target_pose_color,
                    renderer_id,
                )
            else:
                self.mesh_renderer.hide(viewer, renderer_id)

        box_count = runtime.interaction_box_count
        renderer_id = f"scene_{definition.id}_interaction_boxes"
        if definition.show_interaction_boxes and box_count:
            runtime.get_world_interaction_boxes(self.interaction_boxes)
            self.wire_box_renderer.draw_batch(
                viewer,
                self.interaction_boxes,
                box_count,
                _INTERACTION_BOX_COLOR,
                renderer_id,
            )
        else:
            self.wire_box_renderer.hide(viewer, renderer_id)

    def clear(self, viewer) -> None:
        self.mesh_renderer.clear(viewer)
        self.wire_box_renderer.clear(viewer)


class SceneEditor(EditorViewer):
    """Place and edit scene items in a MotionBricks scene."""

    _OUTLINER_WINDOW_TITLE = "Scene"
    _EDITOR_TITLE = "MotionBricks - Scene Editor"
    _DOCUMENT_TYPE = SCENE_DOCUMENT

    def __init__(self, viewer, scene_path: str | Path | None = None) -> None:
        """Open the editor with a new scene, or with the scene at scene_path."""
        super().__init__(viewer)
        self.device = wp.get_device()
        self._rig_assets: RigAssets | None = None
        self._asset_pool: SmartPrimitiveAssetPool | None = None
        self.scene = SceneDefinition()
        self.selected_item_id = self.scene.character_start.id
        self._runtime_views: dict[int, _SceneSmartPrimitiveView] = {}
        self._rename = InlineRename()
        self._reveal_item_ids: set[int] = set()
        self._gizmo_item_id: int | None = None
        self._gizmo_source_transform: wp.transform | None = None
        self._gizmo_transform: wp.transform | None = None
        # Axis locks are editing preferences and are not saved with the scene.
        self._transform_locks = defaultdict(lambda: ([False] * 3, [False] * 3))
        self.character_start_renderer = _CharacterStartRenderer(
            device=self.device
        )
        self.solid_box_renderer = SolidBoxRenderer(
            device=self.device
        )
        self._viewer_ui.on_scene_click = self._select_scene_item

        if scene_path is not None:
            self._open_document(scene_path)

    @property
    def selected(self) -> SceneItem:
        """Selected scene item, or CharacterStart when nothing is selected."""
        return (
            self.scene.find_item(self.selected_item_id)
            or self.scene.character_start
        )

    def _new_document(self) -> None:
        scene = SceneDefinition(rig=self.scene.rig)
        self._replace_scene(
            scene,
            selected_item_id=scene.character_start.id,
            document_path=None,
            runtime_views={},
        )

    def _save_document(self, path: str | Path) -> Path:
        return save_scene_document(self.scene, path, self.selected_item_id, self._capture_camera_state())

    def _open_document(self, path: str | Path) -> None:
        try:
            document_path = Path(path).expanduser().resolve()
            scene, selected_item_id, camera_state = load_scene_document(
                document_path
            )
            check_reference_rigs(scene)
            runtime_views = self._create_runtime_views(scene)
        except ValueError as exc:
            self._status = f"Open failed: {exc}"
            return

        self._replace_scene(
            scene,
            selected_item_id,
            document_path,
            runtime_views,
        )
        self._apply_camera_state(camera_state)

    def _replace_scene(
        self,
        scene: SceneDefinition,
        selected_item_id: int,
        document_path: Path | None,
        runtime_views: dict[int, _SceneSmartPrimitiveView],
    ) -> None:
        self._clear_runtime_views()
        self.solid_box_renderer.clear(self.viewer)
        self.scene = scene
        self.selected_item_id = selected_item_id
        self._runtime_views = runtime_views
        self._rename = InlineRename()
        self._reveal_item_ids = set()
        self._gizmo_item_id = None
        self._gizmo_source_transform = None
        self._gizmo_transform = None
        self._transform_locks.clear()
        self._set_document_path(document_path)

    def _create_runtime_views(
        self,
        scene: SceneDefinition,
    ) -> dict[int, _SceneSmartPrimitiveView]:
        views = {}
        for instance in scene.smart_primitives:
            if instance.reference is not None:
                views[instance.id] = self._create_runtime_view(instance, scene.rig, scene.world_transform(instance))
        return views

    def _create_runtime_view(
        self,
        instance: SmartPrimitiveInstance,
        rig: ModelRig,
        transform: wp.transform,
    ) -> _SceneSmartPrimitiveView:
        model, asset_pool = self._rig_for(rig)
        runtime = asset_pool.instantiate(instance.reference)
        return _SceneSmartPrimitiveView(instance, runtime, model, asset_pool.reference_pose, transform)

    def _rig_for(self, rig: ModelRig) -> tuple[object, SmartPrimitiveAssetPool]:
        """Return the rig's model and asset pool. Another rig replaces the current one."""
        if self._rig_assets is None or self._rig_assets.clip_registry.rig is not rig:
            with wp.ScopedDevice(self.device):
                assets = load_rig_assets(rig)
            self._rig_assets = assets
            self._asset_pool = SmartPrimitiveAssetPool(
                assets.clip_registry, assets.reference_pose, assets.model_to_runtime,
            )
        return self._rig_assets.model, self._asset_pool

    def _assign_reference(
        self,
        instance: SmartPrimitiveInstance,
        path: str | Path,
    ) -> None:
        resolved_path = Path(path).expanduser().resolve()
        # The first reference chooses the rig. Later references must match it.
        character_is_fixed = self._has_references()
        previous_reference = instance.reference
        instance.reference = resolved_path
        try:
            if character_is_fixed:
                check_reference_rigs(self.scene)
                rig = self.scene.rig
            else:
                rig = load_smart_primitive_rig(resolved_path)
            view = self._create_runtime_view(instance, rig, self.scene.world_transform(instance))
        except ValueError as exc:
            instance.reference = previous_reference
            self._status = f"Reference failed: {exc}"
            return

        self.scene.rig = rig
        previous_view = self._runtime_views.get(instance.id)
        if previous_view is not None:
            previous_view.clear(self.viewer)
        self._runtime_views[instance.id] = view
        self._status = None

    def _clear_runtime_views(self) -> None:
        for view in self._runtime_views.values():
            view.clear(self.viewer)
        self._runtime_views.clear()

    def _delete_item(self, item_id: int) -> None:
        """Delete an item and every item under it."""
        removed_ids = self.scene.remove_item(item_id)
        for removed_id in removed_ids:
            view = self._runtime_views.pop(removed_id, None)
            if view is not None:
                view.clear(self.viewer)
            self._transform_locks.pop(removed_id, None)
        if removed_ids:
            self.selected_item_id = self.scene.character_start.id
            self._rename.item_id = None

    def _set_parent(self, item_id: int, parent_id: int | None) -> None:
        item = self.scene.find_item(item_id)
        if item is not None and self.scene.set_parent(item, parent_id):
            self.selected_item_id = item.id

    def _set_item_transform(self, item: SceneItem, transform: wp.transform) -> None:
        """Set the local transform and update the whole branch."""
        item.transform = wp.transform(transform)
        for moved in self.scene.subtree(item):
            world_transform = self.scene.world_transform(moved)
            view = self._runtime_views.get(moved.id)
            if view is not None:
                view.set_transform(world_transform)
            if moved.id == self._gizmo_item_id and self._gizmo_transform is not None:
                self._gizmo_transform[:] = world_transform
                self._gizmo_source_transform[:] = world_transform

    def _select_scene_item(self, origin: wp.vec3, direction: wp.vec3) -> None:
        nearest = self.viewer.camera.far
        for view in self._runtime_views.values():
            distance = view.raycast(origin, direction, nearest)
            if distance < nearest:
                nearest = distance
                self.selected_item_id = view.definition.id
        for cube in self.scene.solid_cubes:
            distance = self.solid_box_renderer.raycast(
                self.scene.world_transform(cube), wp.vec3(cube.width, cube.depth, cube.height),
                origin, direction, nearest,
            )
            if distance < nearest:
                nearest = distance
                self.selected_item_id = cube.id
        start = self.scene.character_start
        distance = self.character_start_renderer.raycast(start.transform, origin, direction, nearest)
        if distance < nearest:
            self.selected_item_id = start.id
        self._reveal_selected()

    def _reveal_selected(self) -> None:
        """Open the tree down to the selected item and scroll to it."""
        item = self.selected
        self._reveal_item_ids = set()
        while item is not None:
            self._reveal_item_ids.add(item.id)
            item = self.scene.find_item(item.parent_id)

    def _draw_editor_scene(self) -> None:
        self.character_start_renderer.draw(
            self.viewer,
            self.scene.character_start.transform,
        )
        for view in self._runtime_views.values():
            view.draw(self.viewer)
        self.solid_box_renderer.draw_batch(
            self.viewer,
            [self.scene.world_transform(cube) for cube in self.scene.solid_cubes],
            [
                wp.vec3(cube.width, cube.depth, cube.height)
                for cube in self.scene.solid_cubes
            ],
            [cube.color for cube in self.scene.solid_cubes],
            "scene",
        )

    def _draw_selection_gizmo(self) -> None:
        item = self.selected
        self._gizmo_item_id = item.id
        self._gizmo_source_transform = self.scene.world_transform(item)
        self._gizmo_transform = wp.transform(self._gizmo_source_transform)
        translation_locks, rotation_locks = self._transform_locks[item.id]
        parent = self.scene.find_item(item.parent_id)
        log_transform_gizmo(
            self.viewer,
            "scene_selection",
            self._gizmo_transform,
            translation_locks=translation_locks,
            rotation_locks=rotation_locks,
            axes_rotation=None if parent is None else self.scene.world_transform(parent).q,
        )

    def _apply_selection_gizmo(self) -> None:
        item = self.scene.find_item(self._gizmo_item_id)
        if (
            item is None
            or self._gizmo_source_transform is None
            or self._gizmo_transform is None
            or transforms_equal(
                self._gizmo_source_transform,
                self._gizmo_transform,
            )
        ):
            return
        transform, item.euler_degrees = apply_transform_locks(
            item.transform, self.scene.local_transform(item, self._gizmo_transform), item.euler_degrees,
            *self._transform_locks[item.id],
        )
        self._set_item_transform(item, transform)

    def _draw_outliner_window(self, ui, dpi_scale: float) -> None:
        self._set_initial_panel_geometry(
            ui,
            dpi_scale,
            _SCENE_WINDOW_WIDTH,
            _SCENE_WINDOW_HEIGHT,
            WINDOW_MARGIN,
        )
        window_visible, _ = ui.begin(self._OUTLINER_WINDOW_TITLE)
        delete_id = None
        if window_visible:
            self._draw_character_selector(ui)
            delete_id = self._draw_scene_toolbar(ui)
            self._draw_status(ui)
            ui.separator()
            for item in (self.scene.character_start, *self.scene.children(None)):
                delete_id = self._draw_scene_item(ui, item) or delete_id
            self._draw_root_drop_target(ui)
        ui.end()
        if delete_id is not None:
            self._delete_item(delete_id)

    def _has_references(self) -> bool:
        return any(instance.reference is not None for instance in self.scene.smart_primitives)

    def _draw_character_selector(self, ui) -> None:
        self.scene.rig = draw_character_selector(
            ui,
            self.scene.rig,
            "Smart Primitive references lock the Character selector. Delete those Smart Primitives or start a "
            "new scene to change it."
            if self._has_references() else None,
        )

    def _draw_scene_toolbar(self, ui) -> int | None:
        selected = self.selected
        add, remove_selected = draw_add_remove_buttons(ui, selected is not self.scene.character_start)
        if add:
            ui.open_popup("##AddSceneItem")
        if ui.begin_popup("##AddSceneItem"):
            add_smart_primitive, _ = ui.menu_item(
                "Smart Primitive",
                "",
                False,
            )
            add_solid_cube, _ = ui.menu_item(
                "Solid Cube",
                "",
                False,
            )
            ui.end_popup()
            if add_smart_primitive:
                item = self.scene.add_smart_primitive()
                self.selected_item_id = item.id
            elif add_solid_cube:
                item = self.scene.add_solid_cube()
                self.selected_item_id = item.id
        return selected.id if remove_selected else None

    def _draw_scene_item(self, ui, item: SceneItem) -> int | None:
        """Draw a branch and return the item requested for deletion."""
        ui.push_id(item.id)
        children = self.scene.children(item.id)
        renaming = item.id == self._rename.item_id
        is_start = item is self.scene.character_start
        node_open, clicked, name = draw_tree_row(
            ui, item.name, item.id, self._rename, self._reveal_item_ids,
            selected=item.id == self.selected_item_id, leaf=not children, renamable=not is_start,
        )
        if clicked:
            self.selected_item_id = item.id
        if name:
            item.name = name

        delete_id = None
        if not renaming and not is_start:
            # Drop an item onto another to place it under that item.
            drag_source(ui, _SCENE_ITEM_PAYLOAD, item.id, item.name)
            dropped_id = accept_drop(ui, _SCENE_ITEM_PAYLOAD)
            if dropped_id is not None:
                self._set_parent(dropped_id, item.id)
            delete_id = self._draw_item_context_menu(ui, item)

        if node_open:
            for child in children:
                delete_id = self._draw_scene_item(ui, child) or delete_id
            ui.tree_pop()
        ui.pop_id()
        return delete_id

    def _draw_root_drop_target(self, ui) -> None:
        """Accept drops below the tree to move an item to the top level."""
        space = ui.get_content_region_avail()
        size = ui.ImVec2(max(space.x, 1.0), max(space.y, ui.get_frame_height()))
        ui.invisible_button("##SceneRoot", size)
        dropped_id = accept_drop(ui, _SCENE_ITEM_PAYLOAD)
        if dropped_id is not None:
            self._set_parent(dropped_id, None)

    def _draw_item_context_menu(self, ui, item: SmartPrimitiveInstance | SolidCube) -> int | None:
        if not ui.begin_popup_context_item("##SceneItemContext"):
            return None
        self.selected_item_id = item.id
        move_to_root, _ = ui.menu_item("Move to Root", "", False, item.parent_id is not None)
        delete_requested, _ = ui.menu_item("Delete", "", False)
        ui.end_popup()
        if move_to_root:
            self._set_parent(item.id, None)
        return item.id if delete_requested else None

    def _draw_properties_window(self, ui, dpi_scale: float) -> None:
        self._set_initial_panel_geometry(
            ui,
            dpi_scale,
            PROPERTIES_WINDOW_WIDTH,
            PROPERTIES_WINDOW_HEIGHT,
            _SCENE_WINDOW_HEIGHT + 2.0 * WINDOW_MARGIN,
        )
        item = self.selected
        window_visible, _ = ui.begin("Properties")
        reference_requested = False
        if window_visible:
            ui.push_id(item.id)
            try:
                ui.text_colored(ui.ImVec4(*NAME_TEXT_COLOR), item.name)
                if ui.collapsing_header(
                    "Transform",
                    flags=ui.TreeNodeFlags_.default_open,
                ):
                    self._draw_transform_details(ui, item, dpi_scale)
                if isinstance(item, SmartPrimitiveInstance):
                    reference_requested = self._draw_smart_primitive_details(ui, item)
                elif isinstance(item, SolidCube):
                    if ui.collapsing_header(
                        "Solid Cube",
                        flags=ui.TreeNodeFlags_.default_open,
                    ):
                        self._draw_solid_cube_details(ui, item, dpi_scale)
            finally:
                ui.pop_id()
        ui.end()

        if reference_requested:
            path = choose_file(
                "Reference Smart Primitive", "Smart Primitive documents", SMART_PRIMITIVE_DOCUMENT.suffix,
                item.reference,
            )
            if path:
                self._assign_reference(item, path)

    def _draw_smart_primitive_details(self, ui, item: SmartPrimitiveInstance) -> bool:
        """Draw the Smart Primitive, Visibility, and Runtime sections.

        Returns:
            Whether the button that picks the referenced document was clicked.
        """
        reference_requested = False
        if ui.collapsing_header(
            "Smart Primitive",
            flags=ui.TreeNodeFlags_.default_open,
        ):
            reference = item.reference
            reference_requested = draw_clip_selector(
                ui, reference.name if reference else "None", reference, item.id, label="Reference",
            )
        if ui.collapsing_header(
            "Visibility",
            flags=ui.TreeNodeFlags_.default_open,
        ):
            _, item.show_target_poses = ui.checkbox(
                "Show Target Poses",
                item.show_target_poses,
            )
            ui.same_line()
            view = self._runtime_views.get(item.id)
            sequence_names = (
                tuple(
                    view.runtime.sequence_name(index)
                    for index in range(view.runtime.sequence_count)
                )
                if view is not None
                else ()
            )
            selected_sequence = (
                0
                if view is None
                or view.target_pose_sequence_index is None
                else view.target_pose_sequence_index + 1
            )
            ui.begin_disabled(
                not item.show_target_poses or not sequence_names
            )
            ui.set_next_item_width(max(
                1.0,
                ui.get_content_region_avail().x
                - ui.get_frame_height()
                - ui.get_style().item_spacing.x,
            ))
            changed, selected_sequence = ui.combo(
                "##TargetPoseSequence",
                selected_sequence,
                ("All",) + sequence_names,
            )
            ui.end_disabled()
            if changed and view is not None:
                view.target_pose_sequence_index = (
                    None
                    if selected_sequence == 0
                    else selected_sequence - 1
                )
            ui.same_line()
            changed, color = ui.color_edit3(
                "Target Pose Color",
                item.target_pose_color,
                ui.ColorEditFlags_.no_inputs | ui.ColorEditFlags_.no_label,
            )
            if changed:
                item.target_pose_color = wp.vec3(*color)
            _, item.show_interaction_boxes = ui.checkbox(
                "Show Interaction Boxes",
                item.show_interaction_boxes,
            )
        if ui.collapsing_header(
            "Runtime",
            flags=ui.TreeNodeFlags_.default_open,
        ):
            if ui.begin_combo("Behavior", item.behavior):
                for name in behavior_types():
                    if ui.selectable(name, name == item.behavior)[0]:
                        item.behavior = name
                ui.end_combo()
        return reference_requested

    def _draw_transform_details(
        self,
        ui,
        item: SceneItem,
        dpi_scale: float,
    ) -> None:
        changed, transform, item.euler_degrees = draw_transform_editor(
            ui, item.transform, item.euler_degrees, dpi_scale,
            *self._transform_locks[item.id],
        )
        if changed:
            self._set_item_transform(item, transform)

    @staticmethod
    def _draw_solid_cube_details(
        ui,
        cube: SolidCube,
        dpi_scale: float,
    ) -> None:
        _, (cube.width, cube.depth, cube.height) = draw_box_dimensions(
            ui, (cube.width, cube.depth, cube.height), _MIN_SOLID_CUBE_DIMENSION, dpi_scale,
        )
        # The color fields accept typed values, so keep each channel from 0 to 1.
        color_changed, color = ui.color_edit3("Color", cube.color)
        if color_changed:
            cube.color = wp.vec3(*(min(max(channel, 0.0), 1.0) for channel in color))


def main() -> None:
    """Parse the command line and open the Scene Editor."""
    import newton.examples

    parser = create_gl_parser()
    parser.add_argument(
        "scene",
        type=Path,
        nargs="?",
        help="Scene document to open at startup.",
    )
    viewer, args = newton.examples.init(parser)
    configure_logging(args)
    viewer.hide_loading_splash()

    SceneEditor(viewer, args.scene).run(args.num_frames if args.test else None)


if __name__ == "__main__":
    main()
