# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""File dialogs shared by the MotionBricks apps."""

from __future__ import annotations

from pathlib import Path


def choose_file(
    title: str,
    label: str,
    suffix: str,
    current_path: Path | None = None,
    *,
    save: bool = False,
) -> str:
    """Show a native Open or Save dialog that lists one kind of file.

    Args:
        title: Dialog title.
        label: Name of the file filter, such as "Scene documents".
        suffix: File suffix that the dialog lists and adds, such as ".scene.json".
        current_path: File whose folder the dialog opens in. The Save dialog also
            starts with its name.
        save: Show the Save dialog instead of the Open dialog.

    Returns:
        The chosen path, or an empty string when the dialog is canceled.
    """

    import tkinter as tk
    from tkinter import filedialog

    options = {"title": title, "defaultextension": suffix, "filetypes": ((label, f"*{suffix}"),)}
    if current_path is not None:
        options["initialdir"] = str(current_path.parent)
        if save:
            options["initialfile"] = current_path.name
    root = tk.Tk()
    root.withdraw()
    try:
        dialog = filedialog.asksaveasfilename if save else filedialog.askopenfilename
        return dialog(**options)
    finally:
        root.destroy()


__all__ = ["choose_file"]
