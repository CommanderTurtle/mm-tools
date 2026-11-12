# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Viewer controls and UI helpers shared by the editors and the Inbetweener."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path

import newton
import numpy as np
from pyglet.math import Vec3
import warp as wp

from motionbricks.model.contract import (
    MODEL_MAX_TOKEN_COUNT,
    MODEL_MIN_TOKEN_COUNT,
    MODEL_POSES_PER_TOKEN,
    ModelRig,
)

from ..runtime.common.clip_registry import ClipRegistry
from ..runtime.common.io.document_io import DocumentError, DocumentType
from ..runtime.common.rig import RIG_LABELS
from ..utils.file_dialog import choose_file
from ..utils.tooltips import show_delayed_tooltip



EDITOR_FRAME_RATE = 60.0
WINDOW_MARGIN = 10.0
RIGHT_DOCK_RATIO = 0.30
PROPERTIES_WINDOW_WIDTH = 440.0
PROPERTIES_WINDOW_HEIGHT = 420.0
NAME_TEXT_COLOR = (0.60, 0.80, 1.0, 1.0)
POSES_TO_USE_TOOLTIP = (
    "How many of the target's four poses inform the model. One pose makes the motion stop at "
    "the target. More poses add velocity, acceleration, and momentum."
)

_MENU_BAR_ALPHA = 0.8
_PANEL_ALPHA = 0.9
_PROPERTIES_DOCK_RATIO = 0.52
_FULL_ROTATION = 360.0
_HALF_ROTATION = 0.5 * _FULL_ROTATION
_DISPLAY_ZERO_EPSILON = 5.0e-4
_TRANSFORM_AXES = (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
_TRANSFORM_AXIS_NAMES = ("X", "Y", "Z")
_LOCKED_AXIS_COLOR = (1.0, 0.3, 0.3, 1.0)
_HIDDEN_NAME_ALPHA = 0.45


class FileAction(Enum):
    """File menu commands that an editor can request."""

    NEW = "new"
    OPEN = "open"
    SAVE = "save"
    SAVE_AS = "save_as"
    EXIT = "exit"


@dataclass
class InlineRename:
    """Rename state for one item, drawn as a text field in place of its name."""

    item_id: int | None = None
    text: str = ""
    focus: bool = False

    def begin(self, item_id: int, name: str) -> None:
        """Start renaming an item, with its current name in the field."""
        self.item_id, self.text, self.focus = item_id, name, True

    def draw(self, ui) -> str | None:
        """Draw the rename field, and return the new name once it is confirmed.

        Enter or leaving the field confirms it. Escape cancels it.
        """
        ui.set_next_item_width(-1.0)
        if self.focus:
            ui.set_keyboard_focus_here()
            self.focus = False
        submitted, self.text = ui.input_text(
            f"##rename_{self.item_id}", self.text,
            ui.InputTextFlags_.enter_returns_true | ui.InputTextFlags_.auto_select_all,
        )
        if ui.is_key_pressed(ui.Key.escape, False):
            self.item_id = None
        elif submitted or ui.is_item_deactivated():
            self.item_id = None
            return self.text.strip() or None
        return None


def draw_label(ui, label: str, *, tooltip: str | None = None, column: float = 0.0, width: float = -1.0) -> None:
    """Draw a label, then place and size the next widget on the same row.

    Args:
        ui: The ImGui module.
        label: Text before the widget. A colon follows it.
        tooltip: Text shown when the pointer rests on the label.
        column: Where the widget starts, in pixels from the window's left edge. Zero places
            the widget right after the label.
        width: Widget width in pixels. -1 fills the rest of the row.
    """
    ui.align_text_to_frame_padding()
    ui.text_unformatted(f"{label}:")
    if tooltip:
        show_delayed_tooltip(ui, tooltip)
    ui.same_line(column)
    ui.set_next_item_width(width)


def draw_add_remove_buttons(ui, can_remove: bool) -> tuple[bool, bool]:
    """Draw the + and - buttons of a panel, and return whether each one was clicked."""
    add = ui.button("+")
    ui.same_line()
    ui.begin_disabled(not can_remove)
    remove = ui.button("-")
    ui.end_disabled()
    return add, remove


def draw_tree_row(
    ui,
    name: str,
    item_id: int,
    rename: InlineRename,
    reveal_ids: set[int],
    *,
    selected: bool,
    leaf: bool = False,
    renamable: bool = True,
    hidden: bool = False,
) -> tuple[bool, bool, str | None]:
    """Draw one row of an outliner tree.

    A click selects the row. A double click selects it and starts renaming it. When the
    row is open, the caller draws its children and then calls ui.tree_pop().

    Args:
        ui: The ImGui module.
        name: Name shown in the row.
        item_id: ID of the row's item.
        rename: Rename state shared by the tree's rows.
        reveal_ids: IDs of rows to open so a new selection shows. The selected row scrolls
            into view and clears the set.
        selected: Highlight the row.
        leaf: Draw the row without an arrow.
        renamable: Let a double click rename the item.
        hidden: Dim the name of an item that the view hides.

    Returns:
        Whether the row is open, whether the user selected it, and the new name once a
        rename is confirmed.
    """
    renaming = item_id == rename.item_id
    flags = ui.TreeNodeFlags_.open_on_arrow | ui.TreeNodeFlags_.default_open
    if not renaming:
        flags |= ui.TreeNodeFlags_.span_avail_width
    if leaf:
        flags |= ui.TreeNodeFlags_.leaf
    if selected:
        flags |= ui.TreeNodeFlags_.selected
    reveal = item_id in reveal_ids
    if reveal and not selected:
        ui.set_next_item_open(True)
    text_color = (*NAME_TEXT_COLOR[:3], _HIDDEN_NAME_ALPHA) if hidden else NAME_TEXT_COLOR
    ui.push_style_color(ui.Col_.text, ui.ImVec4(*text_color))
    row_open = ui.tree_node_ex("###TreeRow" if renaming else f"{name}###TreeRow", flags=flags)
    ui.pop_style_color()
    if reveal and selected:
        ui.set_scroll_here_y()
        reveal_ids.clear()

    if renaming:
        ui.same_line()
        return row_open, False, rename.draw(ui)
    toggled = ui.is_item_toggled_open()
    if renamable and ui.is_item_hovered() and ui.is_mouse_double_clicked(0) and not toggled:
        rename.begin(item_id, name)
        return row_open, True, None
    return row_open, ui.is_item_clicked() and not toggled, None


def drag_source(ui, payload_type: str, item_id: int, text: str) -> bool:
    """Let the last widget be dragged with item_id as its payload.

    Returns:
        True while the widget is dragged. text follows the pointer during the drag.
    """
    if not ui.begin_drag_drop_source():
        return False
    ui.set_drag_drop_payload_py_id(payload_type, item_id)
    ui.text(text)
    ui.end_drag_drop_source()
    return True


def accept_drop(ui, payload_type: str) -> int | None:
    """Return the item ID dropped on the last widget in this frame, or None."""
    if not ui.begin_drag_drop_target():
        return None
    payload = ui.accept_drag_drop_payload_py_id(payload_type)
    ui.end_drag_drop_target()
    return None if payload is None else payload.data_id


def draw_int_slider_property(
    ui, label: str, value: int, minimum: int, maximum: int, item_id: int,
    *, tooltip: str | None = None,
) -> int:
    """Draw a labeled integer slider that fills the available width."""
    draw_label(ui, label, tooltip=tooltip)
    _, value = ui.slider_int(f"##{label}{item_id}", value, minimum, maximum, flags=ui.SliderFlags_.always_clamp)
    return value


def draw_token_range_property(ui, min_tokens: int, max_tokens: int, item_id: int) -> tuple[int, int]:
    """Draw the Tokens to Generate range and its raw frames, then return the range."""
    draw_label(ui, "Tokens to Generate")
    _, min_tokens, max_tokens = ui.drag_int_range2(
        f"##tokens{item_id}", min_tokens, max_tokens, 1.0, MODEL_MIN_TOKEN_COUNT, MODEL_MAX_TOKEN_COUNT,
        "Min: %d", "Max: %d", flags=ui.SliderFlags_.always_clamp,
    )
    min_frames = min_tokens * MODEL_POSES_PER_TOKEN
    max_frames = max_tokens * MODEL_POSES_PER_TOKEN
    frame_range = str(min_frames) if min_frames == max_frames else f"{min_frames} to {max_frames}"
    ui.text_unformatted(f"Raw frames: {frame_range}")
    return min_tokens, max_tokens


def draw_box_dimensions(
    ui, dimensions: tuple[float, float, float], minimum: float, dpi_scale: float,
) -> tuple[bool, tuple[float, float, float]]:
    """Draw Width, Depth, and Height fields in meters.

    Returns:
        Whether a value changed, and the width, depth, and height. Edited values are at
        least minimum. A value that is not finite becomes minimum.
    """
    values = list(dimensions)
    changed = False
    for index, label in enumerate(("Width", "Depth", "Height")):
        draw_label(ui, label, column=80.0 * dpi_scale, width=100.0 * dpi_scale)
        value_changed, values[index] = ui.input_float(f"##{label}", values[index])
        changed |= value_changed
    if changed:
        values = [max(value, minimum) if math.isfinite(value) else minimum for value in values]
    return changed, tuple(values)


def clean_display_value(value: float) -> float:
    """Return 0 for a value too small to show."""

    return 0.0 if abs(value) < _DISPLAY_ZERO_EPSILON else value


def stable_rotation_degrees(rotation: wp.quat, reference: list[float] | None) -> list[float]:
    """Keep displayed rotation angles close to their previous values to avoid jumps."""

    rpy = wp.quat_to_rpy(rotation)
    primary = [float(wp.degrees(rpy[index])) for index in range(3)]
    if reference is None:
        return [clean_display_value(value) for value in primary]

    primary = [
        _nearest_angle(primary[index], reference[index])
        for index in range(3)
    ]
    alternate = [
        _nearest_angle(primary[0] + _HALF_ROTATION, reference[0]),
        _nearest_angle(_HALF_ROTATION - primary[1], reference[1]),
        _nearest_angle(primary[2] + _HALF_ROTATION, reference[2]),
    ]
    primary_distance = sum(
        (primary[index] - reference[index]) ** 2 for index in range(3)
    )
    alternate_distance = sum(
        (alternate[index] - reference[index]) ** 2 for index in range(3)
    )
    if alternate_distance < primary_distance:
        primary = alternate
    return [clean_display_value(value) for value in primary]


def transforms_equal(lhs: wp.transform, rhs: wp.transform) -> bool:
    """Compare two transforms, ignoring tiny differences and treating q and -q as the same rotation."""

    position_equal = all(abs(a - b) <= 1e-6 for a, b in zip(lhs.p, rhs.p))
    rotation_equal = (
        all(abs(a - b) <= 1e-6 for a, b in zip(lhs.q, rhs.q))
        or all(abs(a + b) <= 1e-6 for a, b in zip(lhs.q, rhs.q))
    )
    return position_equal and rotation_equal


def ellipsize_text(ui, text: str, max_width: float) -> str:
    """Shorten text with ... to fit max_width (in pixels)."""

    if max_width <= 0.0:
        return ""
    if ui.calc_text_size(text).x <= max_width:
        return text

    ellipsis = "..."
    if ui.calc_text_size(ellipsis).x > max_width:
        return ""
    lower = 0
    upper = len(text)
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if ui.calc_text_size(text[:middle] + ellipsis).x <= max_width:
            lower = middle
        else:
            upper = middle - 1
    return text[:lower] + ellipsis


def draw_character_selector(ui, rig: ModelRig, locked_reason: str | None = None) -> ModelRig:
    """Show the Character combo and return the chosen rig.

    When locked_reason is given, the combo is disabled and its tooltip
    shows the reason.
    """
    rigs = list(RIG_LABELS)
    draw_label(ui, "Character")
    ui.begin_disabled(locked_reason is not None)
    _, index = ui.combo("##Character", rigs.index(rig), list(RIG_LABELS.values()))
    ui.end_disabled()
    if locked_reason is not None and ui.is_item_hovered(ui.HoveredFlags_.allow_when_disabled):
        ui.set_tooltip(locked_reason)
    return rigs[index]


def draw_clip_selector(
    ui, name: str, path: Path | None, item_id: int, *, label: str = "Clip",
) -> bool:
    """Show a file name after the label, with its path in a tooltip. Return True when the load button is clicked."""
    style = ui.get_style()
    button_width = ui.calc_text_size("...").x + 2 * style.frame_padding.x
    ui.align_text_to_frame_padding()
    ui.text_unformatted(f"{label}:")
    ui.same_line()
    name_width = ui.get_content_region_avail().x - button_width - style.item_spacing.x
    ui.text_unformatted(ellipsize_text(ui, name, name_width))
    if path is not None:
        show_delayed_tooltip(ui, str(path))
    ui.same_line()
    return ui.button(f"...##LoadClip{item_id}", ui.ImVec2(button_width, 0))


def choose_clip(clip_registry: ClipRegistry, current_path: Path | None = None) -> str:
    """Show the Open dialog for the clips of the registry's rig, BVH for SOMA and CSV for G1.

    Returns:
        The chosen path, or an empty string when the dialog is canceled.
    """
    suffix = clip_registry.suffix
    kind = suffix[1:].upper()
    return choose_file(f"Open {kind} Clip", f"{kind} clips", suffix, current_path)


def draw_transform_editor(
    ui, transform: wp.transform, euler_degrees: list[float] | None, scale: float,
    translation_locks: list[bool], rotation_locks: list[bool],
) -> tuple[bool, wp.transform, list[float]]:
    """Edit position and rotation, with reset menus and axis locks.

    Args:
        ui: The ImGui module.
        transform: Transform shown in the fields.
        euler_degrees: Angles from the previous call, in degrees, or None. They keep the
            shown angles from jumping.
        scale: DPI scale for the field layout.
        translation_locks: Lock flag for each axis. The lock menus edit it in place.
        rotation_locks: Lock flag for each axis. The lock menus edit it in place.

    Returns:
        Whether a value changed, the edited transform, and the angles to pass back on the
        next call.
    """
    position = [clean_display_value(float(value)) for value in transform.p]
    rotation = stable_rotation_degrees(transform.q, euler_degrees)
    moved = draw_transform_row(ui, "Translation", position, scale, translation_locks)
    rotated = draw_transform_row(ui, "Rotation", rotation, scale, rotation_locks)
    position = [
        value if moved and not translation_locks[axis] else transform.p[axis]
        for axis, value in enumerate(position)
    ]
    quaternion = wp.quat_rpy(*(wp.radians(value) for value in rotation)) if rotated else transform.q
    return moved or rotated, wp.transform(wp.vec3(*position), quaternion), rotation


def apply_transform_locks(
    current: wp.transform, edited: wp.transform, euler_degrees: list[float] | None,
    translation_locks: list[bool], rotation_locks: list[bool],
) -> tuple[wp.transform, list[float]]:
    """Keep locked axes unchanged when applying a gizmo edit."""
    position = wp.vec3(*(
        current.p[axis] if translation_locks[axis] else edited.p[axis]
        for axis in range(3)
    ))
    current_euler = stable_rotation_degrees(current.q, euler_degrees)
    edited_euler = stable_rotation_degrees(edited.q, current_euler)
    rotation = [
        current_euler[axis] if rotation_locks[axis] else edited_euler[axis]
        for axis in range(3)
    ]
    if all(rotation_locks):
        quaternion = current.q
    elif any(rotation_locks):
        quaternion = wp.quat_rpy(*(wp.radians(value) for value in rotation))
    else:
        quaternion = edited.q
    return wp.transform(position, quaternion), rotation


def draw_transform_row(
    ui,
    label: str,
    values: list[float],
    dpi_scale: float,
    locks: list[bool],
) -> bool:
    """Draw X, Y, and Z fields with a reset menu and axis locks."""

    ui.begin_group()
    changed = _draw_transform_reset(ui, label, values, locks)
    ui.same_line(110.0 * dpi_scale)
    for index, axis_name in enumerate(_TRANSFORM_AXIS_NAMES):
        if index:
            ui.same_line()
        changed |= _draw_transform_axis(
            ui,
            label,
            axis_name,
            index,
            values,
            locks,
            dpi_scale,
        )
    ui.end_group()
    return changed


def draw_clip_scrubber(
    ui,
    animation,
    sample_time: float,
    show_frames: bool,
    dpi_scale: float,
) -> tuple[bool, float, bool]:
    """Edit the sample time and return (changed, time_in_seconds, show_frames).

    Frame numbers use the clip's sample rate. Switching to Frames snaps to the
    nearest frame, and switching to Seconds keeps the same time. The controls
    stay visible but disabled until a clip is loaded.
    """
    original_time = sample_time
    num_frames = animation.num_frames if animation is not None else 0
    sample_rate = animation.sample_rate if animation is not None else 1.0
    last_frame = max(0, num_frames - 1)
    duration = last_frame / sample_rate
    frame = min(max(round(sample_time * sample_rate), 0), last_frame)

    ui.begin_disabled(animation is None)
    if ui.button("Frames" if show_frames else "Seconds", ui.ImVec2(65.0 * dpi_scale, 0.0)):
        show_frames = not show_frames
        if show_frames:
            sample_time = frame / sample_rate
    ui.same_line()
    ui.set_next_item_width(250.0 * dpi_scale)
    if show_frames:
        changed, frame = ui.slider_int("##SamplePosition", frame, 0, last_frame, flags=ui.SliderFlags_.always_clamp)
        if changed:
            sample_time = frame / sample_rate
        ui.same_line()
        ui.text(str(num_frames))
    else:
        changed, sample_time = ui.slider_float(
            "##SamplePosition", sample_time, 0.0, duration, "%.2f", flags=ui.SliderFlags_.always_clamp,
        )
        ui.same_line()
        ui.text(f"{duration:.2f} s")
    ui.end_disabled()
    return changed or sample_time != original_time, sample_time, show_frames


def log_transform_gizmo(
    viewer,
    name: str,
    transform: wp.transform,
    *,
    translation_locks: list[bool],
    rotation_locks: list[bool],
    axes_rotation: wp.quat | None = None,
) -> None:
    """Draw handles for the unlocked axes.

    axes_rotation sets their orientation. Omit it to use the world axes.
    """

    viewer.log_gizmo(
        name,
        transform,
        translate=tuple(
            axis
            for axis, locked in zip(_TRANSFORM_AXES, translation_locks)
            if not locked
        ),
        rotate=tuple(
            axis
            for axis, locked in zip(_TRANSFORM_AXES, rotation_locks)
            if not locked
        ),
    )
    if axes_rotation is not None:
        viewer._gizmo_log[name]["axes_rotation"] = wp.quat(axes_rotation)


def _read_gizmo_transform(matrix) -> wp.transform:
    """Read an ImGuizmo matrix as a Warp transform."""
    rows = np.asarray(matrix.values, dtype=np.float32).reshape(4, 4, order="F")
    return wp.transform_from_matrix(wp.mat44(*rows.flatten()))


def _gizmo_matrix_values(transform: wp.transform) -> list[float]:
    """Return matrix values in ImGuizmo's column order."""
    matrix = np.asarray(wp.transform_to_matrix(transform), dtype=np.float32).reshape(4, 4)
    return matrix.flatten(order="F").tolist()


class _TransformGizmo:
    """Draw handles in the chosen axes and keep movement within the unlocked axes."""

    def __init__(self, giz, gizmo_log) -> None:
        self._giz = giz
        self._gizmo_log = gizmo_log or {}

    def __getattr__(self, name):
        return getattr(self._giz, name)

    def push_id(self, name) -> None:
        entry = self._gizmo_log.get(name, {})
        self._translation_axes = entry.get("translate", _TRANSFORM_AXES)
        self._axes_rotation = entry.get("axes_rotation", wp.quat_identity())
        self._translation_drawn = False
        self._giz.push_id(name)

    def manipulate(self, view, projection, operation, mode, matrix, *args):
        giz = self._giz
        translating = bool(operation & giz.OPERATION.translate)
        if translating:
            if self._translation_drawn:
                return False
            self._translation_drawn = True
            giz.set_axis_mask(
                *(axis not in self._translation_axes for axis in _TRANSFORM_AXES)
            )
            if len(self._translation_axes) == 2:
                # Keep the square handle between the two unlocked axes.
                operation = giz.OPERATION.translate
        else:
            giz.set_axis_mask(False, False, False)

        # The handles have their own orientation at the item's position.
        current = _read_gizmo_transform(matrix)
        axes_rotation = self._axes_rotation
        handle_matrix = giz.Matrix16(_gizmo_matrix_values(wp.transform(current.p, axes_rotation)))
        if not giz.manipulate(view, projection, operation, giz.MODE.local, handle_matrix, *args):
            return False

        edited = _read_gizmo_transform(handle_matrix)
        offset = wp.quat_rotate_inv(axes_rotation, edited.p - current.p)
        if translating:
            # The center handle can also move along locked axes. Remove that movement.
            offset = wp.vec3(*(
                offset[index] if axis in self._translation_axes else 0.0
                for index, axis in enumerate(_TRANSFORM_AXES)
            ))
        rotation_delta = wp.mul(edited.q, wp.quat_inverse(axes_rotation))
        rotation = wp.normalize(wp.mul(rotation_delta, current.q))
        position = current.p + wp.quat_rotate(axes_rotation, offset)
        matrix.values[:] = _gizmo_matrix_values(wp.transform(position, rotation))
        return True


class NewtonViewerUIBridge:
    """Manage Newton's viewer panels, gizmos with axis locks, and clicks in the view."""

    def __init__(self, viewer: newton.viewer.ViewerGL) -> None:
        """Start with the viewer's options and stats panels hidden.

        Set on_scene_click to receive the world ray of each click in the view.
        """
        self.viewer = viewer
        self.show_options = False
        self.show_stats = False
        self.dockspace_id: int | None = None
        self._viewer_gui = None
        self._options_callback = None
        self._stats_callback = None
        self._gizmos_callback = None
        self.on_scene_click = None
        self._scene_click_pending = False

    def install(self, ui) -> None:
        """Enable docking, and route the viewer's options, stats, and gizmo panels here."""
        ui.get_io().config_flags |= ui.ConfigFlags_.docking_enable
        viewer_gui = self.viewer.gui
        self._viewer_gui = viewer_gui
        self._options_callback = viewer_gui._render_left_panel
        self._stats_callback = viewer_gui._render_stats_overlay
        self._gizmos_callback = viewer_gui._render_gizmos
        viewer_gui._render_left_panel = self._render_options
        viewer_gui._render_stats_overlay = self._render_stats
        viewer_gui._render_gizmos = self._render_gizmos

    def _render_options(self) -> None:
        if self.show_options:
            self._options_callback()

    def _render_stats(self) -> None:
        if self.show_stats:
            self._stats_callback()

    def _render_gizmos(self) -> None:
        viewer_ui = self._viewer_gui.ui
        ui = viewer_ui.imgui
        draw_list = ui.get_background_draw_list(ui.get_main_viewport())
        viewer_ui.giz.set_drawlist(draw_list)

        clip_rect_pushed = False
        if self.dockspace_id is not None:
            central_node = ui.internal.dock_builder_get_central_node(
                self.dockspace_id
            )
            if central_node is not None:
                clip_min = central_node.pos
                clip_max = ui.ImVec2(
                    central_node.pos.x + central_node.size.x,
                    central_node.pos.y + central_node.size.y,
                )
                draw_list.push_clip_rect(clip_min, clip_max, True)
                clip_rect_pushed = True
        giz = viewer_ui.giz
        try:
            viewer_ui.giz = _TransformGizmo(
                giz, self.viewer._gizmo_log
            )
            self._gizmos_callback()
        finally:
            viewer_ui.giz = giz
            giz.set_axis_mask(False, False, False)
            if clip_rect_pushed:
                draw_list.pop_clip_rect()
        self._handle_scene_click(ui, giz)

    def _handle_scene_click(self, ui, giz) -> None:
        if self.on_scene_click is None:
            return
        io = ui.get_io()
        blocked = io.want_capture_mouse or giz.is_over() or giz.is_using_any()
        if ui.is_mouse_clicked(0):
            self._scene_click_pending = not blocked
        if blocked or ui.is_mouse_dragging(0):
            self._scene_click_pending = False
        if ui.is_mouse_released(0):
            clicked = self._scene_click_pending
            self._scene_click_pending = False
            if clicked and io.display_size.x > 0 and io.display_size.y > 0:
                # Convert mouse coordinates to camera pixels, with Y measured from the bottom.
                mouse = ui.get_mouse_pos()
                viewport = ui.get_main_viewport()
                camera = self.viewer.camera
                x = (mouse.x - viewport.pos.x) * camera.width / io.display_size.x
                y = (1.0 - (mouse.y - viewport.pos.y) / io.display_size.y) * camera.height
                origin, direction = camera.get_world_ray(x, y)
                self.on_scene_click(wp.vec3(*origin), wp.vec3(*direction))


class EditorViewer:
    """Shared frame loop, menus, panels, and file handling for the editors."""

    # Each editor sets its window title, its outliner window's title, and the kind of
    # document it edits.
    _EDITOR_TITLE: str
    _OUTLINER_WINDOW_TITLE: str
    _DOCUMENT_TYPE: DocumentType

    def __init__(self, viewer: newton.viewer.ViewerGL) -> None:
        """Set up the editor on a graphical viewer.

        A null viewer raises TypeError.
        """
        if isinstance(viewer, newton.viewer.ViewerNull):
            raise TypeError("This editor requires a graphical viewer.")

        self.viewer = viewer
        self._viewer_time = 0.0
        self._viewer_ui = NewtonViewerUIBridge(viewer)
        self._dock_layout_pending = True
        # An error message to show in the editor.
        self._status: str | None = None
        self._exit_requested = False
        self._document_path: Path | None = None

        viewer_gui = viewer.gui
        if viewer_gui is not None:
            self._viewer_ui.install(viewer_gui.ui.imgui)

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        self.model = builder.finalize()
        self.state = self.model.state()
        viewer.set_model(self.model)
        viewer.set_world_offsets([0, 0, 0])
        viewer.vsync = True
        viewer.register_ui_callback(self.gui, position="free")
        self._update_window_title()

    def _capture_camera_state(self) -> dict:
        camera = self.viewer.camera
        return {
            "pos": [float(camera.pos[index]) for index in range(3)],
            "pivot": [float(camera.pivot[index]) for index in range(3)],
            "pitch": float(camera.pitch),
            "yaw": float(camera.yaw),
            "fov": float(camera.fov),
            "near": float(camera.near),
            "far": float(camera.far),
        }

    def _apply_camera_state(self, camera_state: dict | None) -> None:
        if camera_state is None:
            return

        camera = self.viewer.camera
        camera.pos = Vec3(*camera_state["pos"])
        camera.pivot = Vec3(*camera_state["pivot"])
        camera.pitch = camera_state["pitch"]
        camera.yaw = camera_state["yaw"]
        camera.fov = camera_state["fov"]
        camera.near = camera_state["near"]
        camera.far = camera_state["far"]

    def render(self) -> None:
        """Draw one editor frame, then apply the selection gizmo's changes."""
        self.viewer.begin_frame(self._viewer_time)
        self._draw_editor_scene()
        self._draw_selection_gizmo()
        self.viewer.log_state(self.state)
        self.viewer.end_frame()
        self._apply_selection_gizmo()
        self._viewer_time += 1.0 / EDITOR_FRAME_RATE

    def run(self, max_frames: int | None = None) -> None:
        """Draw frames until the window closes, Exit is chosen, or max_frames is reached."""
        frame = 0
        while not self._exit_requested and self.viewer.is_running() and (
            max_frames is None or frame < max_frames
        ):
            self.render()
            frame += 1
        self.viewer.close()

    def gui(self, ui) -> None:
        """Draw the menu bar, the dock space, and the editor windows."""
        dpi_scale = max(1.0, float(ui.get_style().font_scale_dpi))
        self._draw_main_menu_bar(ui)
        self._draw_dockspace(ui)
        self._draw_outliner_window(ui, dpi_scale)
        self._draw_additional_windows(ui, dpi_scale)
        self._draw_properties_window(ui, dpi_scale)
        self._draw_exit_confirmation(ui)

    def _draw_additional_windows(self, ui, dpi_scale: float) -> None:
        """Draw any extra panels provided by this editor."""

    def _draw_main_menu_bar(self, ui) -> None:
        file_action = None
        style = ui.get_style()
        menu_bar_color = style.color_(ui.Col_.menu_bar_bg)
        ui.push_style_color(
            ui.Col_.menu_bar_bg,
            ui.ImVec4(
                menu_bar_color.x,
                menu_bar_color.y,
                menu_bar_color.z,
                _MENU_BAR_ALPHA,
            ),
        )
        try:
            if ui.begin_main_menu_bar():
                file_action = self._draw_file_menu(ui)
                self._draw_tools_menu(ui)
                ui.end_main_menu_bar()
        finally:
            ui.pop_style_color()

        # Handle the choice after the menu bar ends, so the menu closes before a file dialog opens.
        if file_action is FileAction.EXIT:
            ui.open_popup("Exit")
        else:
            self._handle_file_action(file_action)

    def _draw_exit_confirmation(self, ui) -> None:
        ui.set_next_window_pos(
            ui.get_main_viewport().get_center(), ui.Cond_.appearing,
            pivot=ui.ImVec2(0.5, 0.5),
        )
        if not ui.begin_popup_modal("Exit", flags=ui.WindowFlags_.always_auto_resize)[0]:
            return

        ui.text_unformatted("Save the current document before exiting?")
        if ui.button("Save"):
            if self._handle_file_action(FileAction.SAVE):
                self._exit_requested = True
                ui.close_current_popup()
        ui.same_line()
        if ui.button("Don't Save"):
            self._exit_requested = True
            ui.close_current_popup()
        ui.same_line()
        if ui.button("Cancel"):
            ui.close_current_popup()
        self._draw_status(ui)
        ui.end_popup()

    def _draw_tools_menu(self, ui) -> None:
        if not ui.begin_menu("Tools"):
            return
        viewer_ui = self._viewer_ui
        _, viewer_ui.show_options = ui.menu_item("Viewer Options", "", viewer_ui.show_options)
        _, viewer_ui.show_stats = ui.menu_item("Viewer Stats", "", viewer_ui.show_stats)
        ui.end_menu()

    def _draw_dockspace(self, ui) -> None:
        dockspace_flags = (
            ui.DockNodeFlags_.passthru_central_node
            | ui.DockNodeFlags_.no_docking_over_central_node
        )
        dockspace_id = ui.dock_space_over_viewport(
            0,
            ui.get_main_viewport(),
            dockspace_flags,
        )
        self._viewer_ui.dockspace_id = dockspace_id
        if not self._dock_layout_pending:
            return

        dock_builder = ui.internal
        root_node = dock_builder.dock_builder_get_node(dockspace_id)
        if root_node is None:
            return
        if root_node.is_split_node() or not root_node.is_empty():
            self._dock_layout_pending = False
            return

        self._build_initial_dock_layout(ui, dockspace_id)
        self._dock_layout_pending = False

    def _build_initial_dock_layout(self, ui, dockspace_id: int) -> None:
        """Set up the tree panel and the Properties panel when no layout has been saved."""

        dock_builder = ui.internal
        _, right_dock_id, _ = dock_builder.dock_builder_split_node_py(
            dockspace_id,
            ui.Dir.right,
            RIGHT_DOCK_RATIO,
        )
        _, properties_dock_id, outliner_dock_id = (
            dock_builder.dock_builder_split_node_py(
                right_dock_id,
                ui.Dir.down,
                _PROPERTIES_DOCK_RATIO,
            )
        )
        dock_builder.dock_builder_dock_window(
            self._OUTLINER_WINDOW_TITLE,
            outliner_dock_id,
        )
        dock_builder.dock_builder_dock_window(
            "Properties",
            properties_dock_id,
        )
        dock_builder.dock_builder_finish(dockspace_id)

    @staticmethod
    def _set_initial_panel_geometry(
        ui,
        dpi_scale: float,
        width: float,
        height: float,
        top_offset: float,
    ) -> None:
        """Place a panel before ImGui has a saved layout for it."""

        viewport = ui.get_main_viewport()
        margin = WINDOW_MARGIN * dpi_scale
        ui.set_next_window_pos(
            ui.ImVec2(
                viewport.work_pos.x + viewport.work_size.x - margin,
                viewport.work_pos.y + top_offset * dpi_scale,
            ),
            ui.Cond_.first_use_ever,
            pivot=ui.ImVec2(1.0, 0.0),
        )
        ui.set_next_window_size(
            ui.ImVec2(width * dpi_scale, height * dpi_scale),
            ui.Cond_.first_use_ever,
        )
        ui.set_next_window_bg_alpha(_PANEL_ALPHA)

    def _draw_status(self, ui) -> None:
        if self._status is None:
            return

        ui.push_style_color(
            ui.Col_.text,
            ui.ImVec4(1.0, 0.35, 0.35, 1.0),
        )
        ui.text_wrapped(self._status)
        ui.pop_style_color()

    def _draw_editor_scene(self) -> None:
        raise NotImplementedError

    def _draw_selection_gizmo(self) -> None:
        raise NotImplementedError

    def _apply_selection_gizmo(self) -> None:
        raise NotImplementedError

    def _draw_file_menu(self, ui) -> FileAction | None:
        """Draw the File menu and return the chosen action."""
        if not ui.begin_menu("File"):
            return None

        new_requested, _ = ui.menu_item("New", "", False)
        ui.separator()
        open_requested, _ = ui.menu_item("Open...", "", False)
        save_requested, _ = ui.menu_item("Save", "", False)
        save_as_requested, _ = ui.menu_item("Save As...", "", False)
        ui.separator()
        exit_requested, _ = ui.menu_item("Exit", "", False)
        ui.end_menu()

        if new_requested:
            return FileAction.NEW
        if open_requested:
            return FileAction.OPEN
        if save_requested:
            return FileAction.SAVE
        if save_as_requested:
            return FileAction.SAVE_AS
        if exit_requested:
            return FileAction.EXIT
        return None

    def _handle_file_action(self, file_action: FileAction | None) -> bool | None:
        """Handle a File menu choice. Save and Save As return whether the save succeeded."""
        if file_action is FileAction.NEW:
            self._new_document()
        elif file_action is FileAction.OPEN:
            self._open()
        elif file_action is FileAction.SAVE:
            return self._save()
        elif file_action is FileAction.SAVE_AS:
            return self._save_as()
        return None

    def _open(self) -> None:
        kind = self._DOCUMENT_TYPE.kind
        path = choose_file(f"Open {kind}", f"{kind} documents", self._DOCUMENT_TYPE.suffix, self._document_path)
        if path:
            self._open_document(path)

    def _save(self) -> bool:
        if self._document_path is None:
            return self._save_as()
        return self._save_to(self._document_path)

    def _save_as(self) -> bool:
        kind = self._DOCUMENT_TYPE.kind
        path = choose_file(
            f"Save {kind}", f"{kind} documents", self._DOCUMENT_TYPE.suffix, self._document_path, save=True,
        )
        if not path:
            return False
        return self._save_to(path)

    def _save_to(self, path: str | Path) -> bool:
        try:
            document_path = self._save_document(path)
        except DocumentError as exc:
            self._status = f"Save failed: {exc}"
            return False
        self._set_document_path(document_path)
        return True

    def _set_document_path(self, document_path: Path | None) -> None:
        """Make document_path the current file and show its name in the window title."""
        self._document_path = document_path
        self._status = None
        self._update_window_title()

    def _update_window_title(self) -> None:
        title = self._EDITOR_TITLE
        if self._document_path is not None:
            title = f"{title} - {self._DOCUMENT_TYPE.name_from_path(self._document_path)}"
        self.viewer.renderer.set_title(title)

    def _new_document(self) -> None:
        raise NotImplementedError

    def _open_document(self, path: str | Path) -> None:
        """Open the document at path, or show why it failed in the status line."""
        raise NotImplementedError

    def _save_document(self, path: str | Path) -> Path:
        """Save the document to path and return the saved file, or raise DocumentError."""
        raise NotImplementedError

    def _draw_outliner_window(self, ui, dpi_scale: float) -> None:
        raise NotImplementedError

    def _draw_properties_window(self, ui, dpi_scale: float) -> None:
        raise NotImplementedError


def _nearest_angle(angle: float, reference: float) -> float:
    delta = (angle - reference + _HALF_ROTATION) % _FULL_ROTATION
    return reference + delta - _HALF_ROTATION


def _draw_transform_reset(ui, label: str, values: list[float], locks: list[bool]) -> bool:
    ui.align_text_to_frame_padding()
    label_text = f"{label}:"
    ui.selectable(
        f"{label_text}##{label}ResetLabel",
        False,
        ui.SelectableFlags_.none,
        ui.ImVec2(ui.calc_text_size(label_text).x, 0.0),
    )
    changed = False
    if ui.begin_popup_context_item(f"##{label}Reset"):
        reset, _ = ui.menu_item("Reset", "", False)
        if reset:
            for axis in range(3):
                if not locks[axis]:
                    changed |= values[axis] != 0.0
                    values[axis] = 0.0
        ui.end_popup()
    return changed


def _draw_transform_axis(
    ui,
    label: str,
    axis_name: str,
    index: int,
    values: list[float],
    locks: list[bool],
    dpi_scale: float,
) -> bool:
    ui.begin_group()
    ui.align_text_to_frame_padding()
    if locks[index]:
        ui.push_style_color(
            ui.Col_.text,
            ui.ImVec4(*_LOCKED_AXIS_COLOR),
        )
    ui.selectable(
        f"{axis_name}##{label}{axis_name}LockLabel",
        False,
        ui.SelectableFlags_.none,
        ui.ImVec2(ui.calc_text_size(axis_name).x, 0.0),
    )
    if locks[index]:
        ui.pop_style_color()
    if ui.begin_popup_context_item(f"##{label}{axis_name}Lock"):
        action = "Unlock" if locks[index] else "Lock"
        toggle_lock, _ = ui.menu_item(
            f"{action} {axis_name}",
            "",
            False,
        )
        if toggle_lock:
            locks[index] = not locks[index]
        ui.end_popup()
    ui.same_line()
    ui.begin_disabled(locks[index])
    ui.set_next_item_width(60.0 * dpi_scale)
    changed, values[index] = ui.input_float(
        f"##{label}{axis_name}",
        values[index],
    )
    ui.end_disabled()
    ui.end_group()
    return changed


__all__ = [
    "NAME_TEXT_COLOR",
    "POSES_TO_USE_TOOLTIP",
    "PROPERTIES_WINDOW_HEIGHT",
    "PROPERTIES_WINDOW_WIDTH",
    "RIGHT_DOCK_RATIO",
    "WINDOW_MARGIN",
    "EditorViewer",
    "FileAction",
    "InlineRename",
    "NewtonViewerUIBridge",
    "apply_transform_locks",
    "draw_character_selector",
    "draw_clip_scrubber",
    "draw_clip_selector",
    "draw_int_slider_property",
    "draw_token_range_property",
    "draw_transform_editor",
    "log_transform_gizmo",
    "transforms_equal",
]
