# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Draw directions and velocities as arrows in the Newton viewer."""

import warp as wp

from motionbricks.renderers.base_renderer import BaseRenderer


@wp.kernel
def _update_vector_arrow_kernel(
    world_xform: wp.transform,
    local_origin: wp.vec3,
    vector: wp.vec3,
    color: wp.vec3,
    arrow_starts: wp.array(dtype=wp.vec3),
    arrow_ends: wp.array(dtype=wp.vec3),
    arrow_colors: wp.array(dtype=wp.vec3),
):
    start = wp.transform_point(world_xform, local_origin)
    arrow_starts[0] = start
    arrow_ends[0] = start + wp.transform_vector(world_xform, vector)
    arrow_colors[0] = color


class VectorArrowRenderer(BaseRenderer):
    """Draw a direction or velocity as an arrow."""

    def __init__(self, name_prefix: str, color: wp.vec3) -> None:
        """Set the arrow color and the prefix of the viewer names.

        Args:
            name_prefix: Prefix of each drawn arrow's viewer name.
            color: RGB color of every arrow.
        """
        super().__init__()
        self._name_prefix = name_prefix
        self._color = color
        self._buffers = {}

    def _name(self, instance_id: int | str) -> str:
        return f"/{self._name_prefix}_{instance_id}"

    def draw(
        self,
        viewer,
        instance_id: int | str,
        world_xform: wp.transform,
        local_origin: wp.vec3,
        vector: wp.vec3,
    ) -> None:
        """Draw one arrow from a point along a vector, both in a local space.

        Args:
            viewer: Newton viewer to draw into.
            instance_id: Identifier of the drawn arrow. Drawing the same identifier
                again updates it.
            world_xform: World transform of the local space.
            local_origin: Start of the arrow in the local space.
            vector: Direction and length of the arrow in the local space.
        """
        buffers = self._buffers.get(instance_id)
        if buffers is None:
            buffers = (
                wp.empty(1, dtype=wp.vec3),
                wp.empty(1, dtype=wp.vec3),
                wp.empty(1, dtype=wp.vec3),
            )
            self._buffers[instance_id] = buffers

        arrow_starts, arrow_ends, arrow_colors = buffers
        wp.launch(
            _update_vector_arrow_kernel,
            dim=1,
            inputs=[
                world_xform,
                local_origin,
                vector,
                self._color,
            ],
            outputs=[arrow_starts, arrow_ends, arrow_colors],
        )

        self._log_visible(
            viewer.log_arrows,
            self._name(instance_id),
            arrow_starts,
            arrow_ends,
            arrow_colors,
        )

    def hide(self, viewer, instance_id: int | str) -> None:
        """Hide one drawn arrow, keeping its data for the next draw.

        Args:
            viewer: Newton viewer that shows the drawing.
            instance_id: Identifier of the arrow to hide.
        """
        buffers = self._buffers.get(instance_id)
        if buffers is None:
            return

        self._log_hidden_once(
            viewer.log_arrows,
            self._name(instance_id),
            *buffers,
        )

    def clear(self, viewer) -> None:
        """Remove every drawn arrow from the viewer.

        Args:
            viewer: Newton viewer that shows the drawing.
        """
        self._clear(viewer.arrows)
        self._buffers.clear()
