# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Build Smart Primitives by placing clip poses, pivots, and interaction boxes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import newton
import warp as wp

from motionbricks.animation.skeleton import Skeleton
from motionbricks.model.contract import MODEL_POSES_PER_TOKEN, ModelRig
from motionbricks.renderers.base_renderer import BaseRenderer
from motionbricks.renderers.wire_box_renderer import WireBoxRenderer
from motionbricks.renderers.vector_arrow_renderer import VectorArrowRenderer
from motionbricks.runtime.pivot import Axis, ConeSide, ConstraintType, OperationOrder
from motionbricks.runtime.smart_primitive import InteractionBox
from motionbricks.runtime.target_pose import (
    IKPin,
    IK_PINS,
    MAX_TRANSITION_EARLY_BY_FRAMES,
    create_soma_target_pose_ik_rig,
)

from .utils.cli import configure_logging, create_gl_parser
from .runtime.common.clip_registry import ClipLoadError, ClipRegistry
from .runtime.common.rig import create_mesh_renderer, create_skeleton_renderer, load_rig_assets
from .authoring.common import (
    EditorViewer,
    InlineRename,
    NAME_TEXT_COLOR,
    POSES_TO_USE_TOOLTIP,
    PROPERTIES_WINDOW_HEIGHT,
    PROPERTIES_WINDOW_WIDTH,
    RIGHT_DOCK_RATIO,
    WINDOW_MARGIN,
    accept_drop,
    choose_clip,
    apply_transform_locks,
    drag_source,
    draw_add_remove_buttons,
    draw_box_dimensions,
    draw_character_selector,
    draw_clip_scrubber,
    draw_clip_selector,
    draw_int_slider_property,
    draw_label,
    draw_token_range_property,
    draw_transform_editor,
    draw_tree_row,
    log_transform_gizmo,
    transforms_equal,
)
from .runtime.common.io.smart_primitive_document import (
    MAX_TRAJECTORY_SCALE,
    MIN_TRAJECTORY_SCALE,
    SMART_PRIMITIVE_DOCUMENT,
    TRAJECTORY_POINT_COUNT,
    InteractionBoxComponent,
    PivotComponent,
    PlaybackMode,
    SceneNode,
    SmartPrimitiveDocument,
    TargetPoseComponent,
    TargetPoseSequence,
    load_smart_primitive_document,
    load_smart_primitive_rig,
    save_smart_primitive_document,
)
from .runtime.common.io.document_io import DocumentError


_TRAJECTORY_COLOR = wp.vec3(1.0, 0.8, 0.0)
_IK_PIN_COLOR = wp.vec3(1.0, 0.0, 0.0)
_IK_PIN_SPHERE_RADIUS = 0.05
_TRAJECTORY_FRAME_SCALE = 0.075
_MIN_INTERACTION_BOX_SIZE = 0.0
_PIVOT_CONE_SEGMENTS = 64
_PIVOT_CONE_RADIUS = 1.5
_PIVOT_ARROW_LIFT = 0.03


_HIERARCHY_WINDOW_WIDTH = 320
_HIERARCHY_WINDOW_HEIGHT = 380
_SEQUENCES_WINDOW_WIDTH = 360
_SEQUENCES_WINDOW_HEIGHT = 300
_PROPERTIES_DOCK_RATIO = 0.38
_SEQUENCES_DOCK_RATIO = 0.48
_TREE_PAYLOAD = "SMART_PRIMITIVE_NODE"
_SEQUENCE_STEP_PAYLOAD = "SMART_PRIMITIVE_SEQUENCE_STEP"


@wp.kernel
def _update_ik_pin_transforms_kernel(
    pin_joint_indices: wp.array(dtype=wp.int32),
    world_transforms: wp.array(dtype=wp.transform),
    output_transforms: wp.array(dtype=wp.transform),
):
    pin_index = wp.tid()
    output_transforms[pin_index] = world_transforms[
        pin_joint_indices[pin_index]
    ]


@wp.kernel
def _update_hips_trajectory_kernel(
    world_xform: wp.transform,
    hips_transforms: wp.array(dtype=wp.transform),
    frame_scale: float,
    trajectory_color: wp.vec3,
    trajectory_starts: wp.array(dtype=wp.vec3),
    trajectory_ends: wp.array(dtype=wp.vec3),
    trajectory_colors: wp.array(dtype=wp.vec3),
    frame_starts: wp.array(dtype=wp.vec3),
    frame_ends: wp.array(dtype=wp.vec3),
    frame_colors: wp.array(dtype=wp.vec3),
):
    point_index = wp.tid()
    hips_world = wp.mul(world_xform, hips_transforms[point_index])
    frame_line = point_index * 3

    frame_starts[frame_line] = hips_world.p
    frame_starts[frame_line + 1] = hips_world.p
    frame_starts[frame_line + 2] = hips_world.p
    frame_ends[frame_line] = wp.transform_point(hips_world, wp.vec3(frame_scale, 0.0, 0.0))
    frame_ends[frame_line + 1] = wp.transform_point(hips_world, wp.vec3(0.0, frame_scale, 0.0))
    frame_ends[frame_line + 2] = wp.transform_point(hips_world, wp.vec3(0.0, 0.0, frame_scale))
    frame_colors[frame_line] = wp.vec3(1.0, 0.0, 0.0)
    frame_colors[frame_line + 1] = wp.vec3(0.0, 1.0, 0.0)
    frame_colors[frame_line + 2] = wp.vec3(0.0, 0.0, 1.0)

    if point_index + 1 < TRAJECTORY_POINT_COUNT:
        next_hips_world = wp.mul(
            world_xform,
            hips_transforms[point_index + 1],
        )
        trajectory_starts[point_index] = hips_world.p
        trajectory_ends[point_index] = next_hips_world.p
        trajectory_colors[point_index] = trajectory_color


def _ik_pin_label(pin: IKPin) -> str:
    return "".join(part.title() for part in pin.name.split("_"))


@wp.kernel
def _update_pivot_cone_kernel(
    world_xform: wp.transform,
    motion_direction: wp.vec3,
    rotation_axis: wp.vec3,
    half_angle: float,
    first_center: float,
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
    colors: wp.array(dtype=wp.vec3),
):
    line = wp.tid()
    cone = line // (_PIVOT_CONE_SEGMENTS + 2)
    segment = line % (_PIVOT_CONE_SEGMENTS + 2)
    center = first_center + float(cone) * wp.pi
    direction = _PIVOT_CONE_RADIUS * wp.transform_vector(world_xform, motion_direction)
    origin = world_xform.p
    axis = wp.transform_vector(world_xform, rotation_axis)
    if segment < _PIVOT_CONE_SEGMENTS:
        angle = center - half_angle + 2.0 * half_angle * float(segment) / float(_PIVOT_CONE_SEGMENTS)
        next_angle = angle + 2.0 * half_angle / float(_PIVOT_CONE_SEGMENTS)
        starts[line] = origin + wp.quat_rotate(wp.quat_from_axis_angle(axis, angle), direction)
        ends[line] = origin + wp.quat_rotate(wp.quat_from_axis_angle(axis, next_angle), direction)
    else:
        # Join the arc's ends to the pivot. A full circle needs no closing lines.
        angle = center - half_angle
        if segment == _PIVOT_CONE_SEGMENTS + 1:
            angle = center + half_angle
        starts[line] = origin
        ends[line] = origin
        if half_angle < wp.pi:
            ends[line] = origin + wp.quat_rotate(wp.quat_from_axis_angle(axis, angle), direction)
    colors[line] = wp.vec3(1.0, 0.0, 0.0)


class _PivotRenderer(BaseRenderer):
    """Draw the pivot's direction and constraints using its placement in the editor."""

    def __init__(self) -> None:
        super().__init__()
        self.direction = VectorArrowRenderer("pivot_motion_direction", wp.vec3(1.0, 0.8, 0.0))
        self._cone_buffers = {}
        self._line_buffers = {}

    def draw(self, viewer, component: PivotComponent, world_xform: wp.transform, visible: bool) -> None:
        pivot = component.pivot
        direction = wp.vec3(*pivot.motion_direction)
        if visible and component.show_motion_direction:
            lifted = wp.transform(world_xform.p + wp.vec3(0.0, 0.0, _PIVOT_ARROW_LIFT), world_xform.q)
            self.direction.draw(viewer, component.id, lifted, wp.vec3(0.0), direction)
        else:
            self.direction.hide(viewer, component.id)

        self._draw_line(viewer, component, world_xform, visible)
        name = f"/pivot_rotation_constraint_{component.id}"
        buffers = self._cone_buffers.get(component.id)
        if not (
            visible and component.show_rotation_constraint
            and pivot.rotation_axis is not Axis.NONE
            and pivot.rotation_constraint is ConstraintType.CONE
        ):
            if buffers is not None:
                self._log_hidden_once(viewer.log_lines, name, *buffers)
            return

        if buffers is None:
            buffers = tuple(wp.empty(2 * (_PIVOT_CONE_SEGMENTS + 2), dtype=wp.vec3) for _ in range(3))
            self._cone_buffers[component.id] = buffers
        cone_count = 2 if pivot.cone_side is ConeSide.BOTH else 1
        line_count = cone_count * (_PIVOT_CONE_SEGMENTS + 2)
        first_center = math.pi if pivot.cone_side is ConeSide.MIRRORED else 0.0
        local_axis = newton.Axis.from_string(pivot.rotation_axis.value).to_vec3()
        wp.launch(
            _update_pivot_cone_kernel,
            dim=line_count,
            inputs=[world_xform, direction, local_axis, math.radians(pivot.cone_half_angle_degrees), first_center],
            outputs=list(buffers),
        )
        self._log_visible(viewer.log_lines, name, *(buffer[:line_count] for buffer in buffers))

    def _draw_line(self, viewer, component: PivotComponent, world_xform: wp.transform, visible: bool) -> None:
        pivot = component.pivot
        name = f"/pivot_translation_constraint_{component.id}"
        buffers = self._line_buffers.get(component.id)
        if not (
            visible and component.show_translation_constraint
            and pivot.translation_axis is not Axis.NONE
            and pivot.translation_constraint is ConstraintType.LINE_PROJECTION
        ):
            if buffers is not None:
                self._log_hidden_once(viewer.log_lines, name, *buffers)
            return

        if buffers is None:
            buffers = (
                wp.empty(1, dtype=wp.vec3),
                wp.empty(1, dtype=wp.vec3),
                wp.full(1, wp.vec3(1.0, 0.0, 0.0), dtype=wp.vec3),
            )
            self._line_buffers[component.id] = buffers
        start, end = pivot.line_endpoints(world_xform)
        buffers[0].fill_(start)
        buffers[1].fill_(end)
        self._log_visible(viewer.log_lines, name, *buffers)

    def clear(self, viewer) -> None:
        self.direction.clear(viewer)
        self._clear(viewer.lines)
        self._cone_buffers.clear()
        self._line_buffers.clear()


class _InteractionBoxRenderer(WireBoxRenderer):
    """Draw the 12 edges of each Interaction Box."""

    _NAME_PREFIX = "/interaction_box_"

    def __init__(self) -> None:
        super().__init__(max_boxes=1)
        self._box = wp.empty(1, dtype=InteractionBox)

    def draw(
        self,
        viewer,
        component: InteractionBoxComponent,
        world_xform: wp.transform,
    ) -> None:
        box = InteractionBox()
        box.transform = world_xform
        box.half_extents = wp.vec3(0.5 * component.width, 0.5 * component.depth, 0.5 * component.height)
        self._box.fill_(box)
        self.draw_batch(
            viewer,
            self._box,
            1,
            component.color,
            component.id,
        )


class _IKPinRenderer(BaseRenderer):
    """Mark pinned hands and feet on the first sampled pose with spheres."""

    def __init__(self, skeleton: Skeleton) -> None:
        super().__init__()
        rig = create_soma_target_pose_ik_rig(skeleton)
        self._skeleton = skeleton
        self._tip_indices = {chain.pin: chain.tip_index for chain in rig}
        self._world_transforms = wp.empty(
            skeleton.num_joints,
            dtype=wp.transform,
        )
        self._pin_transforms = wp.empty(len(rig), dtype=wp.transform)
        self._pin_colors = wp.array(
            [_IK_PIN_COLOR] * len(rig),
            dtype=wp.vec3,
        )
        self._indices_by_mask: dict[IKPin, wp.array] = {}
        self._pin_counts: dict[int, int] = {}

    @staticmethod
    def _name(component_id: int | str) -> str:
        return f"/target_pose_ik_pins_{component_id}"

    def draw(
        self,
        viewer,
        component: TargetPoseComponent,
        world_xform: wp.transform,
    ) -> None:
        mask = component.ik_pin_mask
        joint_indices = self._indices_by_mask.get(mask)
        if joint_indices is None:
            selected_indices = [
                tip_index
                for pin, tip_index in self._tip_indices.items()
                if mask & pin
            ]
            joint_indices = wp.array(selected_indices, dtype=wp.int32)
            self._indices_by_mask[mask] = joint_indices

        self._skeleton.compute_world_transforms(
            component.pose,
            self._world_transforms,
            root_tx=world_xform,
        )
        num_pins = joint_indices.shape[0]
        wp.launch(
            _update_ik_pin_transforms_kernel,
            dim=num_pins,
            inputs=[
                joint_indices,
                self._world_transforms,
            ],
            outputs=[self._pin_transforms],
        )
        self._pin_counts[component.id] = num_pins
        self._log_visible(
            viewer.log_shapes,
            self._name(component.id),
            newton.GeoType.SPHERE,
            _IK_PIN_SPHERE_RADIUS,
            self._pin_transforms[:num_pins],
            self._pin_colors[:num_pins],
            geo_is_solid=True,
        )

    def hide(self, viewer, component_id: int) -> None:
        pin_count = self._pin_counts.get(component_id)
        if pin_count is None:
            return
        self._log_hidden_once(
            viewer.log_shapes,
            self._name(component_id),
            newton.GeoType.SPHERE,
            _IK_PIN_SPHERE_RADIUS,
            self._pin_transforms[:pin_count],
            self._pin_colors[:pin_count],
            geo_is_solid=True,
        )

    def clear(self, viewer) -> None:
        self._clear(viewer.objects)
        self._indices_by_mask.clear()
        self._pin_counts.clear()


class _HipsTrajectoryRenderer(BaseRenderer):
    """Draw the root path and small axis markers along it."""

    def __init__(self) -> None:
        super().__init__()
        self._buffers = {}

    @staticmethod
    def _trajectory_name(component_id: int) -> str:
        return f"/target_pose_trajectory_{component_id}"

    @staticmethod
    def _frames_name(component_id: int) -> str:
        return f"/target_pose_trajectory_frames_{component_id}"

    def draw(
        self,
        viewer,
        component_id: int,
        world_xform: wp.transform,
        hips_transforms: wp.array,
    ) -> None:
        buffers = self._buffers.get(component_id)
        if buffers is None:
            # The path and the axis markers each have line starts, ends, and colors.
            buffers = (
                tuple(wp.empty(TRAJECTORY_POINT_COUNT - 1, dtype=wp.vec3) for _ in range(3)),
                tuple(wp.empty(TRAJECTORY_POINT_COUNT * 3, dtype=wp.vec3) for _ in range(3)),
            )
            self._buffers[component_id] = buffers

        path, frames = buffers
        wp.launch(
            _update_hips_trajectory_kernel,
            dim=TRAJECTORY_POINT_COUNT,
            inputs=[world_xform, hips_transforms, _TRAJECTORY_FRAME_SCALE, _TRAJECTORY_COLOR],
            outputs=[*path, *frames],
        )
        self._log_visible(viewer.log_lines, self._trajectory_name(component_id), *path)
        self._log_visible(viewer.log_lines, self._frames_name(component_id), *frames)

    def hide(self, viewer, component_id: int) -> None:
        buffers = self._buffers.get(component_id)
        if buffers is None:
            return

        path, frames = buffers
        self._log_hidden_once(viewer.log_lines, self._trajectory_name(component_id), *path)
        self._log_hidden_once(viewer.log_lines, self._frames_name(component_id), *frames)

    def clear(self, viewer) -> None:
        self._clear(viewer.lines)
        self._buffers.clear()


class _TargetPoseRenderers:
    """Draw a target pose's mesh, skeletons, velocity, and other previews.

    The mesh and skeleton renderers match the rig's poses: local joint transforms
    for SOMA or joint coordinates for G1. IK pins apply only to SOMA poses.
    """

    def __init__(self, mesh, skeleton, ik_pins: _IKPinRenderer | None = None) -> None:
        self.mesh = mesh
        self.skeleton = skeleton
        self.ik_pins = ik_pins
        self.velocity = VectorArrowRenderer("target_pose_velocity", wp.vec3(1.0, 0.0, 0.0))
        self.trajectory = _HipsTrajectoryRenderer()
        self._skeleton_colors = wp.empty(MODEL_POSES_PER_TOKEN, dtype=wp.vec3)

    def draw(
        self,
        viewer,
        component: TargetPoseComponent,
        world_xform: wp.transform,
        visible: bool,
    ) -> None:
        self._draw_mesh(viewer, component, world_xform, visible)
        self._draw_velocity(viewer, component, world_xform, visible)
        self._draw_trajectory(viewer, component, world_xform, visible)
        self._draw_skeletons(viewer, component, world_xform, visible)
        self._draw_ik_pins(viewer, component, world_xform, visible)

    def clear(self, viewer) -> None:
        self.mesh.clear(viewer)
        self.velocity.clear(viewer)
        self.trajectory.clear(viewer)
        self.skeleton.clear(viewer)
        if self.ik_pins is not None:
            self.ik_pins.clear(viewer)

    def _draw_mesh(
        self,
        viewer,
        component: TargetPoseComponent,
        world_xform: wp.transform,
        visible: bool,
    ) -> None:
        if visible and component.show_mesh:
            self.mesh.draw_pose(
                viewer,
                component.pose,
                world_xform,
                component.color,
                component.id,
            )
        else:
            self.mesh.hide(viewer, component.id)

    def _draw_velocity(
        self,
        viewer,
        component: TargetPoseComponent,
        world_xform: wp.transform,
        visible: bool,
    ) -> None:
        if visible and component.show_velocity and component.has_velocity:
            self.velocity.draw(
                viewer,
                component.id,
                world_xform,
                component.hips_position,
                component.velocity,
            )
        else:
            self.velocity.hide(viewer, component.id)

    def _draw_trajectory(
        self,
        viewer,
        component: TargetPoseComponent,
        world_xform: wp.transform,
        visible: bool,
    ) -> None:
        if visible and component.show_trajectory and component.has_trajectory:
            self.trajectory.draw(
                viewer,
                component.id,
                world_xform,
                component.trajectory_transforms,
            )
        else:
            self.trajectory.hide(viewer, component.id)

    def _draw_skeletons(
        self,
        viewer,
        component: TargetPoseComponent,
        world_xform: wp.transform,
        visible: bool,
    ) -> None:
        name = f"target_pose_{component.id}"
        if not (visible and component.show_skeletons):
            self.skeleton.hide(viewer, name)
        else:
            self._skeleton_colors.fill_(component.color)
            self.skeleton.draw_batch(
                viewer, component.sampled_poses, component.skeleton_count, world_xform, self._skeleton_colors, name,
            )

    def _draw_ik_pins(
        self,
        viewer,
        component: TargetPoseComponent,
        world_xform: wp.transform,
        visible: bool,
    ) -> None:
        if self.ik_pins is None:
            return
        if visible and component.ik_pin_mask != IKPin.NONE:
            self.ik_pins.draw(viewer, component, world_xform)
        else:
            self.ik_pins.hide(viewer, component.id)


@dataclass(frozen=True)
class _Rig:
    """The clips, reference pose, and target pose previews of the current rig."""

    clip_registry: ClipRegistry
    reference_pose: wp.array
    model_to_runtime: wp.transform
    target_pose_renderers: _TargetPoseRenderers


class SmartPrimitiveEditor(EditorViewer):
    """Edit a Smart Primitive document.

    A document holds the node hierarchy with its Target Poses, Interaction Boxes, and
    Pivots, plus the sequences.
    """

    _EDITOR_TITLE = "MotionBricks - Smart Primitive Editor"
    _OUTLINER_WINDOW_TITLE = "Hierarchy"
    _DOCUMENT_TYPE = SMART_PRIMITIVE_DOCUMENT

    def __init__(self, viewer) -> None:
        """Open the editor with a new SOMA document."""
        super().__init__(viewer)
        self.viewer.renderer.arrow_scale = 1.5
        self.rig = self._load_rig(ModelRig.SOMA)
        self.interaction_box_renderer = _InteractionBoxRenderer()
        self.pivot_renderer = _PivotRenderer()
        self.authoring = SmartPrimitiveDocument(
            self.rig.clip_registry.rig, self.rig.reference_pose, self.rig.model_to_runtime,
        )
        self.selected_id = self.authoring.root.id
        self._reveal_node_ids: set[int] = set()
        self.selected_sequence_id: int | None = None
        self._node_rename = InlineRename()
        self._sequence_rename = InlineRename()
        self._dragged_sequence_step: tuple[int, int] | None = None
        self._pivot_direction_edit: tuple[int, list[float]] | None = None
        self._gizmo_item_id: int | None = None
        self._gizmo_xform: wp.transform | None = None
        self._viewer_ui.on_scene_click = self._select_scene_item

    @property
    def selected(self) -> SceneNode:
        """Selected node, or the root node when nothing is selected."""
        return self.authoring.find_node(self.selected_id) or self.authoring.root

    def _rig_for(self, name: ModelRig) -> _Rig:
        """Return the current rig when it matches name. Otherwise, load the named rig."""
        if self.rig.clip_registry.rig is name:
            return self.rig
        return self._load_rig(name)

    def _load_rig(self, name: ModelRig) -> _Rig:
        """Load a rig with its preview renderers. The current rig stays until it is replaced."""
        assets = load_rig_assets(name)
        renderers = _TargetPoseRenderers(
            create_mesh_renderer(name, assets.model),
            create_skeleton_renderer(name, assets.model, max_instances=MODEL_POSES_PER_TOKEN),
            # Only SOMA has target pose IK.
            _IKPinRenderer(assets.model.skeleton) if name is ModelRig.SOMA else None,
        )
        return _Rig(assets.clip_registry, assets.reference_pose, assets.model_to_runtime, renderers)

    def _use_rig(self, rig: _Rig) -> None:
        """Switch a document whose target poses have no clips to another rig."""
        self.authoring.set_rig(rig.clip_registry.rig, rig.reference_pose, rig.model_to_runtime)
        self.rig.target_pose_renderers.clear(self.viewer)
        self.rig = rig

    def _build_initial_dock_layout(self, ui, dockspace_id: int) -> None:
        dock_builder = ui.internal
        _, right_dock_id, _ = dock_builder.dock_builder_split_node_py(
            dockspace_id,
            ui.Dir.right,
            RIGHT_DOCK_RATIO,
        )
        _, properties_dock_id, upper_dock_id = (
            dock_builder.dock_builder_split_node_py(
                right_dock_id,
                ui.Dir.down,
                _PROPERTIES_DOCK_RATIO,
            )
        )
        _, sequences_dock_id, hierarchy_dock_id = (
            dock_builder.dock_builder_split_node_py(
                upper_dock_id,
                ui.Dir.down,
                _SEQUENCES_DOCK_RATIO,
            )
        )
        dock_builder.dock_builder_dock_window(self._OUTLINER_WINDOW_TITLE, hierarchy_dock_id)
        dock_builder.dock_builder_dock_window("Sequences", sequences_dock_id)
        dock_builder.dock_builder_dock_window("Properties", properties_dock_id)
        dock_builder.dock_builder_finish(dockspace_id)

    def _draw_additional_windows(self, ui, dpi_scale: float) -> None:
        self._draw_sequences_window(ui, dpi_scale)

    def _new_document(self) -> None:
        authoring = SmartPrimitiveDocument(
            self.authoring.rig,
            self.authoring.reference_pose,
            self.authoring.model_to_runtime,
        )
        self._replace_scene(authoring, authoring.root.id, None)

    def _save_document(self, path: str | Path) -> Path:
        return save_smart_primitive_document(self.authoring, path, self.selected_id, self._capture_camera_state())

    def _open_document(self, path: str | Path) -> None:
        try:
            document_path = Path(path).expanduser().resolve()
            rig = self._rig_for(load_smart_primitive_rig(document_path))
            authoring, selected_id, camera_state = load_smart_primitive_document(
                rig.clip_registry,
                rig.reference_pose,
                rig.model_to_runtime,
                document_path,
            )
        except DocumentError as exc:
            self._status = f"Open failed: {exc}"
            return

        # Replacing the scene clears the previous rig's previews.
        self._replace_scene(authoring, selected_id, document_path)
        self.rig = rig
        self._apply_camera_state(camera_state)

    def _replace_scene(
        self,
        authoring: SmartPrimitiveDocument,
        selected_id: int,
        document_path: Path | None,
    ) -> None:
        """Switch documents and clear any unfinished edits from the previous one."""
        self._clear_component_renderers()
        self.authoring = authoring
        self.selected_id = selected_id
        self._reveal_node_ids = set()
        self.selected_sequence_id = (
            authoring.sequences[0].id if authoring.sequences else None
        )
        self._node_rename = InlineRename()
        self._sequence_rename = InlineRename()
        self._dragged_sequence_step = None
        self._pivot_direction_edit = None
        self._gizmo_item_id = None
        self._gizmo_xform = None
        self._set_document_path(document_path)

    def _draw_editor_scene(self) -> None:
        self._draw_interaction_boxes()
        self._draw_target_poses()
        self._draw_pivots()

    def _draw_pivots(self) -> None:
        for node, component in self.authoring.iter_pivots():
            self.pivot_renderer.draw(
                self.viewer, component, self.authoring.world_xform(node), self.authoring.is_visible(node),
            )

    def _draw_interaction_boxes(self) -> None:
        for node, component in self.authoring.iter_interaction_boxes():
            if self.authoring.is_visible(node):
                self.interaction_box_renderer.draw(
                    self.viewer,
                    component,
                    self.authoring.world_xform(node),
                )
            else:
                self.interaction_box_renderer.hide(
                    self.viewer,
                    component.id,
                )

    def _select_and_reveal_node(self, node: SceneNode) -> None:
        self.selected_id = node.id
        self._reveal_node_ids = {
            ancestor.id for ancestor in self.authoring.iter_ancestors(node)
        }

    def _select_scene_item(self, origin: wp.vec3, direction: wp.vec3) -> None:
        nearest = self.viewer.camera.far
        picked_node = None
        for node, component in self.authoring.iter_target_poses():
            if not self.authoring.is_visible(node) or not component.show_mesh:
                continue
            distance = self.rig.target_pose_renderers.mesh.raycast_pose(
                component.pose,
                self.authoring.target_pose_world_xform(node),
                origin, direction, nearest,
            )
            if distance < nearest:
                nearest = distance
                picked_node = node
        for node, component in self.authoring.iter_interaction_boxes():
            if not self.authoring.is_visible(node):
                continue
            distance = self.interaction_box_renderer.raycast(
                self.authoring.world_xform(node),
                wp.vec3(component.width, component.depth, component.height),
                origin, direction, nearest,
            )
            if distance < nearest:
                nearest = distance
                picked_node = node
        if picked_node is not None:
            self._select_and_reveal_node(picked_node)

    def _draw_target_poses(self) -> None:
        for node, component in self.authoring.iter_target_poses():
            self.rig.target_pose_renderers.draw(
                self.viewer,
                component,
                self.authoring.target_pose_world_xform(node),
                self.authoring.is_visible(node),
            )

    def _clear_component_renderers(self) -> None:
        self.rig.target_pose_renderers.clear(self.viewer)
        self.interaction_box_renderer.clear(self.viewer)
        self.pivot_renderer.clear(self.viewer)

    def _clear_component_renderer(
        self,
        node: SceneNode,
        component: TargetPoseComponent | InteractionBoxComponent | PivotComponent,
    ) -> None:
        if component is node.target_pose:
            self.rig.target_pose_renderers.clear(self.viewer)
        elif component is node.pivot:
            self.pivot_renderer.clear(self.viewer)
        else:
            self.interaction_box_renderer.clear(self.viewer)

    def _draw_selection_gizmo(self) -> None:
        selected = self.selected
        if not self.authoring.is_visible(selected):
            self._gizmo_item_id = None
            self._gizmo_xform = None
            return

        self._gizmo_item_id = selected.id
        self._gizmo_xform = self.authoring.world_xform(selected)
        log_transform_gizmo(
            self.viewer,
            "selection",
            self._gizmo_xform,
            translation_locks=selected.translation_locks,
            rotation_locks=selected.rotation_locks,
            axes_rotation=None if selected.parent is None else self.authoring.world_xform(selected.parent).q,
        )

    def _apply_selection_gizmo(self) -> None:
        gizmo_node = self.authoring.find_node(self._gizmo_item_id)
        if (
            gizmo_node is None
            or self._gizmo_xform is None
            or transforms_equal(self.authoring.world_xform(gizmo_node), self._gizmo_xform)
        ):
            return

        gizmo_local, gizmo_node.euler_degrees = apply_transform_locks(
            self._local_xform(gizmo_node), self._local_xform(gizmo_node, self._gizmo_xform),
            gizmo_node.euler_degrees, gizmo_node.translation_locks, gizmo_node.rotation_locks,
        )
        self._set_local_xform(gizmo_node, gizmo_local)

    def _draw_sequences_window(self, ui, dpi_scale: float) -> None:
        self._set_initial_panel_geometry(
            ui,
            dpi_scale,
            _SEQUENCES_WINDOW_WIDTH,
            _SEQUENCES_WINDOW_HEIGHT,
            _HIERARCHY_WINDOW_HEIGHT + 2 * WINDOW_MARGIN,
        )
        window_visible, _ = ui.begin("Sequences")
        sequence_to_remove = None
        if window_visible:
            selected_sequence = self.authoring.find_sequence(self.selected_sequence_id)
            add, remove_selected = draw_add_remove_buttons(ui, selected_sequence is not None)
            if add:
                sequence = self.authoring.add_sequence()
                self.selected_sequence_id = sequence.id
                self._sequence_rename.begin(sequence.id, sequence.name)
            if remove_selected:
                sequence_to_remove = selected_sequence

            ui.separator()
            if not self.authoring.sequences:
                ui.text_disabled("No target pose sequences.")
            for sequence in list(self.authoring.sequences):
                if self._draw_sequence(ui, sequence):
                    sequence_to_remove = sequence
        ui.end()

        if sequence_to_remove is not None:
            self.authoring.remove_sequence(sequence_to_remove)
            self.selected_sequence_id = None
            self._sequence_rename.item_id = None

    def _draw_sequence(self, ui, sequence: TargetPoseSequence) -> bool:
        ui.push_id(sequence.id)
        try:
            flags = ui.TreeNodeFlags_.default_open
            if sequence.id == self.selected_sequence_id:
                flags |= ui.TreeNodeFlags_.selected
            sequence_open = ui.collapsing_header(
                f"{sequence.name}###Sequence",
                flags=flags,
            )
            header_clicked = ui.is_item_clicked()
            if header_clicked:
                self.selected_sequence_id = sequence.id
            self._accept_target_pose_node_drop(ui, sequence)
            delete_sequence = self._draw_sequence_context_menu(ui, sequence)

            if self._sequence_rename.item_id == sequence.id:
                if name := self._sequence_rename.draw(ui):
                    if any(
                        other is not sequence and other.name == name
                        for other in self.authoring.sequences
                    ):
                        self._status = f"Cannot rename the sequence. {name!r} is already in use."
                    else:
                        sequence.name = name
                        self._status = None

            if sequence_open:
                self._draw_sequence_pivot(ui, sequence)
                remove_index = None
                for index, node_id in enumerate(list(sequence.node_ids)):
                    if self._draw_sequence_step(ui, sequence, index, node_id):
                        remove_index = index
                if remove_index is not None:
                    self.authoring.remove_sequence_node(sequence, remove_index)

                if ui.button("+ Add Target Pose"):
                    ui.open_popup("##AddTargetPose")
                self._draw_add_target_pose_popup(ui, sequence)
            return delete_sequence
        finally:
            ui.pop_id()

    def _draw_sequence_pivot(self, ui, sequence: TargetPoseSequence) -> None:
        node = self.authoring.find_node(sequence.pivot_node_id)
        draw_label(ui, "Pivot")
        opened = ui.begin_combo("##Pivot", "None" if node is None else node.name)
        if ui.is_item_clicked() and node is not None:
            self._select_and_reveal_node(node)
            self.selected_sequence_id = sequence.id
        if opened:
            choices = [(None, "None")] + [
                (pivot_node, pivot_node.name) for pivot_node, _ in self.authoring.iter_pivots()
            ]
            for choice, label in choices:
                choice_id = None if choice is None else choice.id
                selected, _ = ui.selectable(f"{label}##{choice_id}", choice_id == sequence.pivot_node_id)
                if selected:
                    self.authoring.set_sequence_pivot(sequence, choice)
                    self.selected_sequence_id = sequence.id
                    if choice is not None:
                        self._select_and_reveal_node(choice)
            ui.end_combo()
        else:
            choice = self.authoring.find_node(accept_drop(ui, _TREE_PAYLOAD))
            if choice is not None and choice.pivot is not None:
                self.authoring.set_sequence_pivot(sequence, choice)
                self._select_and_reveal_node(choice)
                self.selected_sequence_id = sequence.id

    def _draw_sequence_context_menu(
        self,
        ui,
        sequence: TargetPoseSequence,
    ) -> bool:
        if not ui.begin_popup_context_item("##SequenceContext"):
            return False
        self.selected_sequence_id = sequence.id
        rename_sequence, _ = ui.menu_item("Rename", "", False)
        delete_sequence, _ = ui.menu_item("Delete", "", False)
        ui.end_popup()
        if rename_sequence:
            self._sequence_rename.begin(sequence.id, sequence.name)
        return delete_sequence

    def _draw_sequence_step(
        self,
        ui,
        sequence: TargetPoseSequence,
        index: int,
        node_id: int,
    ) -> bool:
        node = self.authoring.find_node(node_id)
        ui.push_id(index)
        try:
            available_width = ui.get_content_region_avail().x
            remove_width = ui.calc_text_size("x").x + 20.0
            clicked, _ = ui.selectable(
                f"{index + 1} - {node.name}##TargetPose",
                node_id == self.selected_id,
                size=ui.ImVec2(max(1.0, available_width - remove_width), 0.0),
            )
            if clicked:
                self._select_and_reveal_node(node)
                self.selected_sequence_id = sequence.id
            if drag_source(ui, _SEQUENCE_STEP_PAYLOAD, node_id, f"{index + 1} - {node.name}"):
                self._dragged_sequence_step = (sequence.id, index)
            self._accept_sequence_step_drop(ui, sequence, index)

            ui.same_line()
            return ui.small_button("x")
        finally:
            ui.pop_id()

    def _accept_target_pose_node_drop(self, ui, sequence: TargetPoseSequence) -> None:
        node = self.authoring.find_node(accept_drop(ui, _TREE_PAYLOAD))
        if node is not None and node.target_pose is not None:
            self.authoring.add_sequence_node(sequence, node)
            self._select_and_reveal_node(node)
            self.selected_sequence_id = sequence.id

    def _accept_sequence_step_drop(
        self,
        ui,
        sequence: TargetPoseSequence,
        destination_index: int,
    ) -> None:
        dropped = accept_drop(ui, _SEQUENCE_STEP_PAYLOAD) is not None
        dragged = self._dragged_sequence_step
        if dropped and dragged is not None and dragged[0] == sequence.id:
            source_index = dragged[1]
            if source_index != destination_index:
                self.authoring.move_sequence_node(sequence, source_index, destination_index)
            self._dragged_sequence_step = None

    def _draw_add_target_pose_popup(
        self,
        ui,
        sequence: TargetPoseSequence,
    ) -> None:
        if not ui.begin_popup("##AddTargetPose"):
            return
        targets = list(self.authoring.iter_target_poses())
        if not targets:
            ui.text_disabled("No hierarchy nodes own a Target Pose.")
        for node, _component in targets:
            selected, _ = ui.selectable(f"{node.name}###{node.id}", False)
            if selected:
                self.authoring.add_sequence_node(sequence, node)
                self._select_and_reveal_node(node)
                self.selected_sequence_id = sequence.id
                ui.close_current_popup()
        ui.end_popup()

    def _draw_outliner_window(self, ui, dpi_scale: float) -> None:
        self._set_initial_panel_geometry(
            ui,
            dpi_scale,
            _HIERARCHY_WINDOW_WIDTH,
            _HIERARCHY_WINDOW_HEIGHT,
            WINDOW_MARGIN,
        )
        window_visible, _ = ui.begin(self._OUTLINER_WINDOW_TITLE)

        delete_id = None
        if window_visible:
            self._draw_character_selector(ui)
            delete_id = self._draw_hierarchy_toolbar(ui)
            self._draw_status(ui)
            ui.separator()
            tree_delete_id = self._draw_tree_node(ui, self.authoring.root)
            if tree_delete_id is not None:
                delete_id = tree_delete_id
        ui.end()

        node = self.authoring.find_node(delete_id)
        if node is not None:
            self._clear_component_renderers()
            self.authoring.remove_node(node)
            self._node_rename.item_id = None
            self.selected_id = self.authoring.root.id

    def _draw_hierarchy_toolbar(self, ui) -> int | None:
        add, remove_selected = draw_add_remove_buttons(ui, self.selected is not self.authoring.root)
        if add:
            self.selected_id = self.authoring.add_node(self.selected).id
        return self.selected.id if remove_selected else None

    def _draw_character_selector(self, ui) -> None:
        has_clips = any(target.clip is not None for _, target in self.authoring.iter_target_poses())
        rig = draw_character_selector(
            ui,
            self.authoring.rig,
            "Target Poses with clips lock the Character selector. Remove those Target Poses or start a new "
            "document to change it."
            if has_clips else None,
        )
        if rig is not self.authoring.rig:
            self._use_rig(self._rig_for(rig))

    def _draw_tree_node(self, ui, node: SceneNode) -> int | None:
        ui.push_id(node.id)
        _, node.visible = ui.checkbox("##Visible", node.visible)
        ui.same_line()
        renaming = node.id == self._node_rename.item_id
        node_open, clicked, name = draw_tree_row(
            ui, node.name, node.id, self._node_rename, self._reveal_node_ids,
            selected=node.id == self.selected_id, hidden=not node.visible,
        )
        if clicked:
            self.selected_id = node.id
        if name:
            node.name = name

        delete_id = None
        if not renaming:
            self._draw_tree_node_drag_drop(ui, node)
            delete_id = self._draw_tree_node_context_menu(ui, node)

        if node_open:
            for child in list(node.children):
                child_delete_id = self._draw_tree_node(ui, child)
                if child_delete_id is not None:
                    delete_id = child_delete_id
            ui.tree_pop()
        ui.pop_id()
        return delete_id

    def _draw_tree_node_drag_drop(self, ui, node: SceneNode) -> None:
        if node is not self.authoring.root:
            drag_source(ui, _TREE_PAYLOAD, node.id, node.name)
        dragged = self.authoring.find_node(accept_drop(ui, _TREE_PAYLOAD))
        if dragged is not None and self.authoring.reparent(dragged, node):
            self.selected_id = dragged.id

    def _draw_tree_node_context_menu(self, ui, node: SceneNode) -> int | None:
        if not ui.begin_popup_context_item("##NodeContext"):
            return None

        self.selected_id = node.id
        delete_node, _ = ui.menu_item(
            "Delete",
            "",
            False,
            node is not self.authoring.root,
        )
        add_child, _ = ui.menu_item("Add Child", "", False)
        ui.end_popup()

        if add_child:
            self.selected_id = self.authoring.add_node(node).id
        return node.id if delete_node else None

    def _draw_properties_window(self, ui, dpi_scale: float) -> None:
        self._set_initial_panel_geometry(
            ui,
            dpi_scale,
            PROPERTIES_WINDOW_WIDTH,
            PROPERTIES_WINDOW_HEIGHT,
            _HIERARCHY_WINDOW_HEIGHT
            + _SEQUENCES_WINDOW_HEIGHT
            + 3 * WINDOW_MARGIN,
        )
        node = self.selected
        remove_component: TargetPoseComponent | InteractionBoxComponent | PivotComponent | None = None
        window_visible, _ = ui.begin("Properties")
        if window_visible:
            # Give each node its own widget IDs so switching nodes cannot carry
            # an unfinished text edit into the new selection.
            ui.push_id(node.id)
            try:
                ui.text_colored(ui.ImVec4(*NAME_TEXT_COLOR), node.name)
                if ui.collapsing_header(
                    "Transform",
                    flags=ui.TreeNodeFlags_.default_open,
                ):
                    self._draw_transform_details(ui, node, dpi_scale)

                if node.target_pose is not None and self._draw_target_pose_component(
                    ui, node.target_pose, dpi_scale,
                ):
                    remove_component = node.target_pose
                if node.interaction_box is not None and self._draw_interaction_box_component(
                    ui, node.interaction_box, dpi_scale,
                ):
                    remove_component = node.interaction_box
                if node.pivot is not None and self._draw_pivot_component(ui, node.pivot, dpi_scale):
                    remove_component = node.pivot

                if not node.components:
                    ui.text_disabled("Right click empty space to add a component.")

                self._draw_sequence_references(ui, node)
                self._draw_add_component_popup(ui, node)
            finally:
                ui.pop_id()

        ui.end()
        if remove_component is not None:
            self._clear_component_renderer(node, remove_component)
            self.authoring.remove_component(node, remove_component)

    def _draw_sequence_references(self, ui, node: SceneNode) -> None:
        sequences = list(self.authoring.sequences_referencing(node))
        if not sequences:
            return
        if not ui.collapsing_header(
            "Referenced by",
            flags=ui.TreeNodeFlags_.default_open,
        ):
            return
        for sequence in sequences:
            occurrence_count = sequence.node_ids.count(node.id)
            suffix = "" if occurrence_count == 1 else f" ({occurrence_count} steps)"
            if sequence.pivot_node_id == node.id:
                suffix = " (Pivot)"
            selected, _ = ui.selectable(
                f"{sequence.name}{suffix}###{sequence.id}",
                sequence.id == self.selected_sequence_id,
            )
            if selected:
                self.selected_sequence_id = sequence.id

    def _draw_add_component_popup(self, ui, node: SceneNode) -> None:
        popup_flags = (
            ui.PopupFlags_.mouse_button_right | ui.PopupFlags_.no_open_over_items
        )
        if not ui.begin_popup_context_window(
            "##PropertiesContext",
            popup_flags,
        ):
            return

        if ui.begin_menu("Add"):
            if ui.begin_menu("Components"):
                add_target_pose, _ = ui.menu_item(
                    "Target Pose",
                    "",
                    False,
                    node.target_pose is None,
                )
                if add_target_pose:
                    self.authoring.add_target_pose(node)
                add_interaction_box, _ = ui.menu_item(
                    "Interaction Box",
                    "",
                    False,
                    node.interaction_box is None,
                )
                if add_interaction_box:
                    self.authoring.add_interaction_box(node)
                add_pivot, _ = ui.menu_item("Pivot", "", False, node.pivot is None)
                if add_pivot:
                    self.authoring.add_pivot(node)
                ui.end_menu()
            ui.end_menu()
        ui.end_popup()

    def _draw_transform_details(
        self,
        ui,
        node: SceneNode,
        dpi_scale: float,
    ) -> None:
        changed, transform, node.euler_degrees = draw_transform_editor(
            ui, self._local_xform(node), node.euler_degrees, dpi_scale,
            node.translation_locks, node.rotation_locks,
        )
        if changed:
            self._set_local_xform(node, transform)

    def _draw_pivot_component(self, ui, component: PivotComponent, dpi_scale: float) -> bool:
        opened, remove = self._draw_component_header(ui, "Pivot", component.id)
        if not opened or remove:
            return remove

        pivot = component.pivot
        direction = list(pivot.motion_direction)
        if self._pivot_direction_edit is not None and self._pivot_direction_edit[0] == component.id:
            direction = self._pivot_direction_edit[1]
        draw_label(ui, "Motion Direction")
        changed, direction = ui.input_float3(f"##MotionDirection{component.id}", direction, "%.3f")
        if changed:
            # Keep the typed numbers in the fields, but use a unit direction for previews and saving.
            self._pivot_direction_edit = (component.id, list(direction))
            try:
                pivot.set_motion_direction(direction)
                self._status = None
            except ValueError as exc:
                self._status = str(exc)
        if ui.is_item_deactivated_after_edit():
            self._pivot_direction_edit = None

        draw_label(
            ui, "Operation Order",
            tooltip="Rotating first also turns the axis the pivot slides along.",
        )
        orders = list(OperationOrder)
        _, index = ui.combo(
            f"##OperationOrder{component.id}", orders.index(pivot.operation_order),
            ["Translation -> Rotation", "Rotation -> Translation"],
        )
        pivot.operation_order = orders[index]

        def draw_rule_row(label, axis, constraint, constraints, axis_labels):
            draw_label(
                ui, label, column=100.0 * dpi_scale, width=115.0 * dpi_scale,
                tooltip="Axes are local to the pivot.",
            )
            axes = list(Axis)
            _, index = ui.combo(
                f"##{label.lower()}_axis{component.id}", axes.index(axis),
                axis_labels,
            )
            axis = axes[index]
            ui.same_line()
            ui.set_next_item_width(-1.0)
            ui.begin_disabled(axis is Axis.NONE)
            _, index = ui.combo(
                f"##{label.lower()}_constraint{component.id}", constraints.index(constraint),
                [choice.value.replace("_", " ").title() for choice in constraints],
            )
            ui.end_disabled()
            return axis, constraints[index]

        pivot.translation_axis, pivot.translation_constraint = draw_rule_row(
            "Translation", pivot.translation_axis, pivot.translation_constraint,
            [ConstraintType.UNCONSTRAINED, ConstraintType.LINE_PROJECTION],
            ["None", "TranslateX", "TranslateY", "TranslateZ"],
        )
        if pivot.translation_axis is not Axis.NONE and pivot.translation_constraint is ConstraintType.LINE_PROJECTION:
            ui.indent(16.0 * dpi_scale)
            draw_label(
                ui, "Offsets (m)", tooltip="Signed distances from the authored pivot along its selected local axis.",
            )
            _, pivot.line_min_offset, pivot.line_max_offset = ui.drag_float_range2(
                f"##LineOffsets{component.id}", pivot.line_min_offset, pivot.line_max_offset,
                v_speed=0.01, format="Min: %.2f", format_max="Max: %.2f",
                flags=ui.SliderFlags_.always_clamp,
            )
            ui.unindent(16.0 * dpi_scale)

        pivot.rotation_axis, pivot.rotation_constraint = draw_rule_row(
            "Rotation", pivot.rotation_axis, pivot.rotation_constraint,
            [ConstraintType.UNCONSTRAINED, ConstraintType.CONE],
            ["None", "RotateX", "RotateY", "RotateZ"],
        )

        show_cone = pivot.rotation_axis is not Axis.NONE and pivot.rotation_constraint is ConstraintType.CONE
        if show_cone:
            ui.indent(16.0 * dpi_scale)
            draw_label(
                ui, "Half Angle (\u00b0)",
                tooltip="Maximum turn on each side of the center. 45 degrees makes a 90-degree cone.",
            )
            _, pivot.cone_half_angle_degrees = ui.slider_float(
                f"##ConeHalfAngle{component.id}", pivot.cone_half_angle_degrees, 0.0, 180.0,
                "%.1f", flags=ui.SliderFlags_.always_clamp,
            )
            draw_label(
                ui, "Cone Side",
                tooltip="Mirrored centers the cone 180 degrees from Motion Direction around the selected local axis.",
            )
            sides = list(ConeSide)
            changed, index = ui.combo(
                f"##ConeSide{component.id}", sides.index(pivot.cone_side),
                ["Motion Direction", "Mirrored", "Both"],
            )
            if changed:
                pivot.cone_side = sides[index]
            ui.unindent(16.0 * dpi_scale)
        ui.separator()
        _, component.show_motion_direction = ui.checkbox(
            f"Show Motion Direction##{component.id}", component.show_motion_direction,
        )
        _, component.show_translation_constraint = ui.checkbox(
            f"Show Translation Constraint##{component.id}", component.show_translation_constraint,
        )
        _, component.show_rotation_constraint = ui.checkbox(
            f"Show Rotation Constraint##{component.id}", component.show_rotation_constraint,
        )
        return False

    def _draw_target_pose_component(
        self,
        ui,
        component: TargetPoseComponent,
        dpi_scale: float,
    ) -> bool:
        component_open, remove_component = self._draw_component_header(
            ui,
            "Target Pose",
            component.id,
        )
        if not component_open or remove_component:
            return remove_component

        self._draw_clip_selector(ui, component)
        self._draw_target_pose_timeline(ui, component, dpi_scale)
        changed, hips_constrained = ui.checkbox(
            "Constrain Root",
            component.hips_constrained,
        )
        if changed:
            component.constrain_hips(hips_constrained)
        if component.clip is not None and self.authoring.rig is ModelRig.SOMA:
            self._draw_target_pose_ik_pins(ui, component, dpi_scale)
        ui.text_unformatted(
            f"Speed: {component.velocity_magnitude:.3f} m/s"
        )

        ui.separator()
        self._draw_target_pose_sampling_options(ui, component)
        ui.separator()
        self._draw_target_pose_visualization_options(ui, component)
        return False

    @staticmethod
    def _draw_target_pose_ik_pins(
        ui,
        component: TargetPoseComponent,
        dpi_scale: float,
    ) -> None:
        selected_pins = [pin for pin in IK_PINS if component.ik_pin_mask & pin]
        if not selected_pins:
            preview = "None"
        elif len(selected_pins) == len(IK_PINS):
            preview = "All"
        else:
            preview = ", ".join(_ik_pin_label(pin) for pin in selected_pins)

        draw_label(ui, "IK Pins", column=80.0 * dpi_scale)
        if not ui.begin_combo(f"##IKPins{component.id}", preview):
            return

        for pin in IK_PINS:
            selected = bool(component.ik_pin_mask & pin)
            changed, selected = ui.checkbox(
                f"{_ik_pin_label(pin)}##IKPin{int(pin)}_{component.id}",
                selected,
            )
            if changed:
                component.set_ik_pin_enabled(pin, selected)
        ui.end_combo()

    @staticmethod
    def _draw_component_header(
        ui,
        label: str,
        component_id: int,
    ) -> tuple[bool, bool]:
        component_open = ui.collapsing_header(
            f"{label}###{component_id}",
            flags=ui.TreeNodeFlags_.default_open,
        )

        remove_component = False
        if ui.begin_popup_context_item(f"##{label}Context{component_id}"):
            remove_component, _ = ui.menu_item(
                "Remove Component",
                "",
                False,
            )
            ui.end_popup()
        return component_open, remove_component

    def _draw_interaction_box_component(
        self,
        ui,
        component: InteractionBoxComponent,
        dpi_scale: float,
    ) -> bool:
        component_open, remove_component = self._draw_component_header(
            ui,
            "Interaction Box",
            component.id,
        )
        if not component_open or remove_component:
            return remove_component

        changed, dimensions = draw_box_dimensions(
            ui, (component.width, component.depth, component.height), _MIN_INTERACTION_BOX_SIZE, dpi_scale,
        )
        if changed:
            component.set_dimensions(*dimensions)

        color_changed, color = ui.color_edit3(
            f"Color##InteractionBox{component.id}",
            component.color,
            ui.ColorEditFlags_.no_inputs,
        )
        if color_changed:
            component.color = wp.vec3(*color)
        return False

    @staticmethod
    def _draw_target_pose_timeline(
        ui,
        component: TargetPoseComponent,
        dpi_scale: float,
    ) -> None:
        changed, sample_time, show_frames = draw_clip_scrubber(
            ui,
            component.clip.animation if component.clip is not None else None,
            component.sample_time,
            component.playback_mode is PlaybackMode.FRAMES,
            dpi_scale,
        )
        component.playback_mode = PlaybackMode.FRAMES if show_frames else PlaybackMode.TIME
        if changed:
            component.set_time(sample_time)

    def _draw_target_pose_sampling_options(
        self,
        ui,
        component: TargetPoseComponent,
    ) -> None:
        num_poses_to_use = draw_int_slider_property(
            ui,
            "Poses to Use",
            component.num_poses_to_use,
            1,
            MODEL_POSES_PER_TOKEN,
            component.id,
            tooltip=POSES_TO_USE_TOOLTIP,
        )
        if num_poses_to_use != component.num_poses_to_use:
            component.set_num_poses_to_use(num_poses_to_use)
        component.transition_early_by_frames = draw_int_slider_property(
            ui,
            "Transition Early By (frames)",
            component.transition_early_by_frames,
            0,
            MAX_TRANSITION_EARLY_BY_FRAMES,
            component.id,
            tooltip="Move on to the next step this many frames before the generated motion ends.",
        )
        min_tokens, max_tokens = draw_token_range_property(
            ui, component.min_tokens, component.max_tokens, component.id,
        )
        component.set_token_range(min_tokens, max_tokens)

    @staticmethod
    def _draw_target_pose_visualization_options(
        ui,
        component: TargetPoseComponent,
    ) -> None:
        color_changed, color = ui.color_edit3(
            "Color",
            component.color,
            ui.ColorEditFlags_.no_inputs,
        )
        if color_changed:
            component.color = wp.vec3(*color)

        _, component.show_mesh = ui.checkbox("Show Mesh", component.show_mesh)
        _, component.show_skeletons = ui.checkbox("Show Skeleton(s)", component.show_skeletons)
        _, component.show_velocity = ui.checkbox("Show Velocity", component.show_velocity)
        _, component.show_trajectory = ui.checkbox("Show Trajectory", component.show_trajectory)
        ui.same_line()
        ui.set_next_item_width(-1.0)
        changed, trajectory_scale = ui.slider_float(
            f"##TrajectoryScale{component.id}",
            component.trajectory_scale,
            MIN_TRAJECTORY_SCALE,
            MAX_TRAJECTORY_SCALE,
            "x %.1f",
            flags=ui.SliderFlags_.always_clamp,
        )
        if changed:
            component.set_trajectory_scale(trajectory_scale)

    def _draw_clip_selector(self, ui, component: TargetPoseComponent) -> None:
        clip = component.clip
        if draw_clip_selector(
            ui, clip.name if clip else "None", clip.path if clip else None,
            component.id,
        ):
            path = self._choose_clip()
            if path:
                try:
                    component.set_clip(self.rig.clip_registry.load(path))
                    self._status = None
                except ClipLoadError as exc:
                    self._status = f"Open failed: {exc}"

    def _local_xform(
        self,
        node: SceneNode,
        world_xform=None,
    ) -> wp.transform:
        if world_xform is None:
            return wp.transform(node.local_xform)
        if node.parent is None:
            return wp.transform(world_xform)

        return wp.mul(
            wp.transform_inverse(self.authoring.world_xform(node.parent)),
            world_xform,
        )

    def _set_local_xform(
        self,
        node: SceneNode,
        local_xform: wp.transform,
    ) -> None:
        node.local_xform = wp.transform(local_xform)
        if node.id == self._gizmo_item_id and self._gizmo_xform is not None:
            self._gizmo_xform[:] = self.authoring.world_xform(node)

    def _choose_clip(self) -> str:
        return choose_clip(self.rig.clip_registry)


def main() -> None:
    """Parse the command line and open the Smart Primitive Editor."""
    import newton.examples

    parser = create_gl_parser()

    viewer, args = newton.examples.init(parser)
    configure_logging(args)
    viewer.hide_loading_splash()

    SmartPrimitiveEditor(viewer).run(
        args.num_frames if args.test else None
    )


if __name__ == "__main__":
    main()
