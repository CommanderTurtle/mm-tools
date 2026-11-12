# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Draw interaction boxes and the probe box as wireframes."""

import warp as wp

from motionbricks.renderers.line_renderer import LineRenderer
from motionbricks.renderers.picking import raycast_lines
from motionbricks.runtime.smart_primitive import InteractionBox


_BOX_EDGE_COUNT = 12


@wp.func
def _box_edge(half_extents: wp.vec3, edge_index: int) -> tuple[wp.vec3, wp.vec3]:
    axis = edge_index // 4
    corner = edge_index % 4
    sign0 = float((corner % 2) * 2 - 1)
    sign1 = float((corner // 2) * 2 - 1)

    local_start = wp.vec3(0.0)
    local_end = wp.vec3(0.0)
    if axis == 0:
        v0 = sign0 * half_extents[1]
        v1 = sign1 * half_extents[2]
        local_start = wp.vec3(-half_extents[0], v0, v1)
        local_end = wp.vec3(half_extents[0], v0, v1)
    elif axis == 1:
        v0 = sign0 * half_extents[0]
        v1 = sign1 * half_extents[2]
        local_start = wp.vec3(v0, -half_extents[1], v1)
        local_end = wp.vec3(v0, half_extents[1], v1)
    else:
        v0 = sign0 * half_extents[0]
        v1 = sign1 * half_extents[1]
        local_start = wp.vec3(v0, v1, -half_extents[2])
        local_end = wp.vec3(v0, v1, half_extents[2])
    return local_start, local_end


@wp.kernel
def _update_box_lines_kernel(
    boxes: wp.array(dtype=InteractionBox),
    color: wp.vec3,
    line_starts: wp.array(dtype=wp.vec3),
    line_ends: wp.array(dtype=wp.vec3),
    line_colors: wp.array(dtype=wp.vec3),
):
    box_index, edge_index = wp.tid()
    box = boxes[box_index]
    local_start, local_end = _box_edge(box.half_extents, edge_index)

    output_index = box_index * _BOX_EDGE_COUNT + edge_index
    line_starts[output_index] = wp.transform_point(box.transform, local_start)
    line_ends[output_index] = wp.transform_point(box.transform, local_end)
    line_colors[output_index] = color


class WireBoxRenderer(LineRenderer):
    """Draw batches of InteractionBox values as 12 wireframe edges."""

    _NAME_PREFIX = "/wire_boxes_"

    def __init__(self, max_boxes: int = 1) -> None:
        """Allocate line buffers for the boxes.

        Args:
            max_boxes: Number of boxes the buffers hold at first. They grow when a batch
                needs more.
        """
        super().__init__(_BOX_EDGE_COUNT, max_boxes)

    @staticmethod
    def raycast(
        transform: wp.transform,
        dimensions: wp.vec3,
        origin: wp.vec3,
        direction: wp.vec3,
        max_distance: float = 1.0e6,
    ) -> float:
        """Return the distance where a ray hits a box's edges.

        Only the edges count, not the faces.

        Args:
            transform: World transform of the box.
            dimensions: Full size of the box along each of its axes, in meters.
            origin: Ray origin in world space.
            direction: Unit ray direction in world space.
            max_distance: Longest distance to test, in meters. A miss returns it.
        """
        half_extents = dimensions * 0.5
        edges = (_box_edge(half_extents, edge_index) for edge_index in range(_BOX_EDGE_COUNT))
        return raycast_lines(transform, edges, origin, direction, max_distance)

    def raycast_batch(
        self, boxes: wp.array, num_boxes: int, origin: wp.vec3, direction: wp.vec3, max_distance: float,
    ) -> float:
        """Return the distance to the nearest box edge along a ray.

        The boxes are read back from the device on each call.

        Args:
            boxes: Boxes in world space.
            num_boxes: Number of boxes to test.
            origin: Ray origin in world space.
            direction: Unit ray direction in world space.
            max_distance: Longest distance to test, in meters. A miss returns it.
        """
        for box in boxes[:num_boxes].numpy():
            max_distance = self.raycast(
                wp.transform(*box["transform"]), wp.vec3(*box["half_extents"]) * 2.0,
                origin, direction, max_distance,
            )
        return max_distance

    def draw_batch(
        self,
        viewer,
        boxes: wp.array,
        num_boxes: int,
        color: wp.vec3,
        instance_id: int | str,
    ) -> None:
        """Draw the first num_boxes boxes as one batch.

        Args:
            viewer: Newton viewer to draw into.
            boxes: Boxes in world space.
            num_boxes: Number of boxes to draw, at least 1.
            color: RGB color of every edge.
            instance_id: Identifier of the drawn batch. Drawing the same identifier again
                updates it.
        """
        if boxes.dtype != InteractionBox or boxes.ndim != 1:
            raise ValueError("boxes must be a 1D InteractionBox Warp array.")
        if not 1 <= num_boxes <= boxes.shape[0]:
            raise ValueError(f"num_boxes must be in [1, {boxes.shape[0]}].")
        self._ensure_capacity(num_boxes)

        wp.launch(
            _update_box_lines_kernel,
            dim=(num_boxes, _BOX_EDGE_COUNT),
            inputs=[boxes, color],
            outputs=[self.line_starts, self.line_ends, self.line_colors],
            device=boxes.device,
        )
        self._log_batch(viewer, instance_id, num_boxes)
