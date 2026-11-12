# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tooltips shared by the MotionBricks apps."""

from __future__ import annotations

import time

TOOLTIP_DELAY_SECONDS = 1.0
_WRAP_WIDTH_EM = 28.0
# The hovered item, by text and position, and when the mouse arrived on it.
_hover = {"item": None, "since": 0.0}


def show_delayed_tooltip(ui, text: str, delay: float = TOOLTIP_DELAY_SECONDS) -> None:
    """Show text in a tooltip once the mouse has rested on the last item for delay seconds.

    The delay uses the wall clock, because the viewer does not give ImGui real frame times.
    The text wraps into a short paragraph.
    """
    corner = ui.get_item_rect_min()
    item = (text, corner.x, corner.y)
    if not ui.is_item_hovered():
        if _hover["item"] == item:
            _hover["item"] = None
        return
    now = time.perf_counter()
    if _hover["item"] != item:
        _hover["item"], _hover["since"] = item, now
    if now - _hover["since"] < delay:
        return
    ui.begin_tooltip()
    ui.push_text_wrap_pos(ui.get_font_size() * _WRAP_WIDTH_EM)
    ui.text_unformatted(text)
    ui.pop_text_wrap_pos()
    ui.end_tooltip()
