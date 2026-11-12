# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Find the downloaded checkpoint files, and load them after checking their checksums."""

import hashlib
import logging
from pathlib import Path

import numpy as np
from safetensors import SafetensorError
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


def sha256(path):
    """Return the SHA-256 digest of a file in lowercase hexadecimal."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_file(path, expected):
    """Check a file's size and SHA-256 digest against its checksum entry."""
    if not path.is_file():
        raise ValueError(f"Missing checkpoint file: {path}.")
    if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
        raise ValueError(
            f"Checksum mismatch: {path}. Use the unmodified checkpoint files."
        )
    logger.debug("Verified %s", path)


def checkpoint_root(source, release):
    """Return the folder that holds the checkpoint files.

    source is a downloaded model folder, such as checkpoints/MotionBricks/soma, with the
    files inside its root_directory. That inner folder works too.
    """
    source = source.expanduser().resolve()
    if source.is_file():
        raise ValueError(f"Checkpoints must be a folder, not a file: {source}.")
    if (source / release["files"][0]["path"]).is_file():
        return source
    return source / release["root_directory"]


def load_sources(root, release):
    """Verify every file, then load the weights and normalization arrays as plain data."""
    for entry in release["files"]:
        verify_file(root / entry["path"], entry)
    sources = {}
    for entry in release["files"]:
        filename = entry["path"]
        if filename.endswith(".safetensors"):
            logger.debug("Loading %s", filename)
            try:
                sources[filename] = load_file(root / filename, device="cpu")
            except SafetensorError as error:
                raise ValueError(f"Invalid SafeTensors weights: {filename}.") from error
        elif filename.endswith(".npy"):
            sources[filename] = np.load(root / filename, allow_pickle=False)
        else:
            raise ValueError(
                f"Unsupported checkpoint file format: {filename}. Use the released checkpoint files."
            )
    return sources
