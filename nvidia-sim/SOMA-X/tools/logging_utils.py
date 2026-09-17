# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logging helpers for SOMA command-line tools."""

import argparse
import logging

LOG_LEVEL_CHOICES = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET")


def add_logging_args(
    parser: argparse.ArgumentParser,
    *,
    default_level: str = "INFO",
) -> None:
    """Add standard verbosity controls to a CLI parser."""
    group = parser.add_argument_group("logging")
    verbosity = group.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--log-level",
        choices=LOG_LEVEL_CHOICES,
        default=None,
        help=f"Set log verbosity (default: {default_level}).",
    )
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Show warnings and errors only.",
    )


def log_level_from_args(
    args: argparse.Namespace,
    *,
    default_level: int | str = logging.INFO,
) -> int:
    """Resolve an argparse namespace to a numeric logging level."""
    if getattr(args, "log_level", None):
        level = logging.getLevelName(args.log_level)
    elif getattr(args, "verbose", False):
        level = logging.DEBUG
    elif getattr(args, "quiet", False):
        level = logging.WARNING
    else:
        level = default_level

    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {level!r}")
    return level


def configure_logging(
    args: argparse.Namespace,
    *,
    default_level: int | str = logging.INFO,
) -> int:
    """Configure root logging for a CLI and return the selected level."""
    level = log_level_from_args(args, default_level=default_level)
    logging.basicConfig(level=level, format="%(message)s")
    return level
