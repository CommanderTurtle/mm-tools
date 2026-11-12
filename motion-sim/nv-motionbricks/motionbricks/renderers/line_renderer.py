# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Draw named batches of lines from buffers that grow to fit the largest batch."""

import warp as wp

from motionbricks.renderers.base_renderer import BaseRenderer


class LineRenderer(BaseRenderer):
    """Draw named batches of lines in the Newton viewer.

    An item is one thing a batch draws, such as a pose or a box. Every item uses the
    same number of lines. A subclass sets _NAME_PREFIX. It draws a batch in three steps:

    1. Call _ensure_capacity() with the batch's item count.
    2. Fill line_starts, line_ends, and line_colors.
    3. Call _log_batch().

    hide() logs the batch's last lines as hidden.
    """

    _NAME_PREFIX: str

    def __init__(self, lines_per_item: int, capacity: int, device=None) -> None:
        """Allocate line buffers for the first batches.

        Args:
            lines_per_item: Number of lines each item uses.
            capacity: Number of items the buffers hold at first. They grow when a batch
                needs more. It must be positive.
            device: Device of the buffers. None uses the current device.
        """
        super().__init__()
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self._lines_per_item = lines_per_item
        self._device = device
        self._capacity = 0
        self._line_counts: dict[str, int] = {}
        self._allocate_buffers(capacity)

    def hide(self, viewer, instance_id: int | str) -> None:
        """Hide one drawn batch, keeping its data for the next draw.

        Args:
            viewer: Newton viewer that shows the drawing.
            instance_id: Identifier of the batch to hide.
        """
        name = self._name(instance_id)
        line_count = self._line_counts.get(name)
        if line_count is not None:
            self._log_hidden_once(viewer.log_lines, name, *self._lines(line_count))

    def clear(self, viewer) -> None:
        """Remove every drawn batch from the viewer.

        Args:
            viewer: Newton viewer that shows the drawing.
        """
        self._clear(viewer.lines)
        self._line_counts.clear()

    def _name(self, instance_id: int | str) -> str:
        return f"{self._NAME_PREFIX}{instance_id}"

    def _log_batch(self, viewer, instance_id: int | str, item_count: int) -> None:
        """Show the lines of the first item_count items as the batch named by instance_id."""
        name = self._name(instance_id)
        line_count = item_count * self._lines_per_item
        self._line_counts[name] = line_count
        self._log_visible(viewer.log_lines, name, *self._lines(line_count))

    def _lines(self, line_count: int) -> tuple[wp.array, wp.array, wp.array]:
        return self.line_starts[:line_count], self.line_ends[:line_count], self.line_colors[:line_count]

    def _ensure_capacity(self, item_count: int) -> None:
        if item_count > self._capacity:
            self._allocate_buffers(max(item_count, self._capacity * 2))

    def _allocate_buffers(self, capacity: int) -> None:
        """Allocate the line buffers for capacity items. Subclasses extend it with their own buffers."""
        self._capacity = capacity
        line_count = capacity * self._lines_per_item
        self.line_starts = wp.empty(line_count, dtype=wp.vec3, device=self._device)
        self.line_ends = wp.empty_like(self.line_starts)
        self.line_colors = wp.empty_like(self.line_starts)


__all__ = ["LineRenderer"]
