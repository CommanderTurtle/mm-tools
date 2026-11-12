# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Draw solid boxes that cast shadows in the Newton viewer."""

from collections.abc import Sequence

import newton
import warp as wp
from newton._src.geometry.raycast import map_ray_to_local, ray_intersect_box

from motionbricks.renderers.base_renderer import BaseRenderer


_UNIT_CUBE_HALF_EXTENTS = (0.5, 0.5, 0.5)
_DEFAULT_MATERIAL = wp.vec4(0.5, 0.0, 0.0, 0.0)


class SolidBoxRenderer(BaseRenderer):
    """Draw movable colored boxes as solid Newton mesh instances."""

    _PROTOTYPE_NAME = "/geometry/motionbricks_unit_cube"

    def __init__(self, max_boxes: int = 1, device=None) -> None:
        """Allocate buffers for the boxes.

        Args:
            max_boxes: Number of boxes the buffers hold at first. They grow when a batch
                needs more.
            device: Device for new arrays.
        """
        super().__init__()
        if max_boxes <= 0:
            raise ValueError("max_boxes must be positive.")
        self.device = wp.get_device(device)
        self._prototype_registered = False
        self._capacity = 0
        self._allocate_buffers(max_boxes)

    @staticmethod
    def _name(instance_id: int | str) -> str:
        return f"/solid_boxes_{instance_id}"

    @staticmethod
    def raycast(
        transform: wp.transform,
        dimensions: wp.vec3,
        origin: wp.vec3,
        direction: wp.vec3,
        max_distance: float = 1.0e6,
    ) -> float:
        """Return the distance where a ray hits a box.

        Args:
            transform: World transform of the box.
            dimensions: Full size of the box along each of its axes, in meters.
            origin: Ray origin in world space.
            direction: Unit ray direction in world space.
            max_distance: Longest distance to test, in meters. A miss returns it.
        """
        origin, direction = map_ray_to_local(transform, origin, direction)
        distance, _ = ray_intersect_box(origin, direction, dimensions * 0.5)
        return distance if 0 <= distance < max_distance else max_distance

    def draw_batch(
        self,
        viewer,
        transforms: Sequence[wp.transform],
        dimensions: Sequence[wp.vec3],
        colors: Sequence[wp.vec3],
        instance_id: int | str,
    ) -> None:
        """Draw one batch of boxes. An empty batch hides it.

        Args:
            viewer: Newton viewer to draw into.
            transforms: World transform of each box.
            dimensions: Full size of each box along its axes, in meters.
            colors: RGB color of each box.
            instance_id: Identifier of the drawn batch. Drawing the same identifier again
                updates it.
        """
        box_count = len(transforms)
        if box_count == 0:
            self.hide(viewer, instance_id)
            return
        if len(dimensions) != box_count or len(colors) != box_count:
            raise ValueError(
                "transforms, dimensions, and colors must have equal lengths."
            )

        self._ensure_capacity(box_count)
        self.transforms[:box_count].assign(transforms)
        self.dimensions[:box_count].assign(dimensions)
        self.colors[:box_count].assign(colors)

        if not self._prototype_registered:
            viewer.log_geo(
                self._PROTOTYPE_NAME,
                newton.GeoType.BOX,
                _UNIT_CUBE_HALF_EXTENTS,
                0.0,
                True,
                hidden=True,
            )
            self._prototype_registered = True

        self._log_visible(
            viewer.log_instances,
            self._name(instance_id),
            self._PROTOTYPE_NAME,
            self.transforms[:box_count],
            self.dimensions[:box_count],
            self.colors[:box_count],
            self.materials[:box_count],
        )

    def hide(self, viewer, instance_id: int | str) -> None:
        """Hide one drawn batch, keeping its data for the next draw.

        Args:
            viewer: Newton viewer that shows the drawing.
            instance_id: Identifier of the batch to hide.
        """
        self._log_hidden_once(
            viewer.log_instances,
            self._name(instance_id),
            self._PROTOTYPE_NAME,
            None,
            None,
            None,
            None,
        )

    def clear(self, viewer) -> None:
        """Remove every drawn batch from the viewer.

        Args:
            viewer: Newton viewer that shows the drawing.
        """
        self._clear(viewer.objects)

    def _ensure_capacity(self, box_count: int) -> None:
        if box_count > self._capacity:
            self._allocate_buffers(max(box_count, self._capacity * 2))

    def _allocate_buffers(self, capacity: int) -> None:
        self._capacity = capacity
        self.transforms = wp.empty(
            capacity,
            dtype=wp.transform,
            device=self.device,
        )
        self.dimensions = wp.empty(
            capacity,
            dtype=wp.vec3,
            device=self.device,
        )
        self.colors = wp.empty(
            capacity,
            dtype=wp.vec3,
            device=self.device,
        )
        self.materials = wp.array(
            [_DEFAULT_MATERIAL] * capacity,
            dtype=wp.vec4,
            device=self.device,
        )


__all__ = ["SolidBoxRenderer"]
