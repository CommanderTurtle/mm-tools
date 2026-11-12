# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install the SOMA procedural-control reference Blender add-on."""

import argparse
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.logging_utils import add_logging_args, configure_logging  # noqa: E402

ADDON_NAME = "soma_procedural_blender"
ADDON_VERSION = "0.1"

logger = logging.getLogger(__name__)


def blender_version_from_binary(blender_binary: str = "blender") -> str:
    """Return Blender's major.minor version string from a Blender executable."""

    completed = subprocess.run(
        [blender_binary, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"Blender\s+(\d+\.\d+)", completed.stdout)
    if match is None:
        raise ValueError(f"Could not parse Blender version from {blender_binary!r}")
    return match.group(1)


def default_blender_addon_dir(version: str | None = None) -> Path:
    """Return the current user's Blender add-ons directory."""

    if version is None:
        version = blender_version_from_binary()

    system = platform.system()
    home = Path.home()
    if system == "Windows":
        appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
        return appdata / "Blender Foundation" / "Blender" / version / "scripts" / "addons"
    if system == "Darwin":
        return home / "Library" / "Application Support" / "Blender" / version / "scripts" / "addons"
    return home / ".config" / "blender" / version / "scripts" / "addons"


def install_addon(
    plugin_root: Path,
    addon_dir: Path,
    force: bool = False,
    copy: bool = False,
    dry_run: bool = False,
) -> Path:
    """Install the Blender add-on as a symlink or copied package directory."""

    source = plugin_root.resolve() / ADDON_NAME
    if not (source / "__init__.py").exists():
        raise FileNotFoundError(f"Missing Blender add-on package under {plugin_root}")

    target = addon_dir / ADDON_NAME
    if target.exists() or target.is_symlink():
        if not force:
            raise FileExistsError(f"{target} already exists. Pass --force to overwrite.")
        if not dry_run:
            if target.is_symlink() or target.is_file():
                target.unlink()
            else:
                shutil.rmtree(target)

    if dry_run:
        return target

    addon_dir.mkdir(parents=True, exist_ok=True)
    if copy or platform.system() == "Windows":
        shutil.copytree(source, target)
    else:
        target.symlink_to(source, target_is_directory=True)
    return target


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plugin-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to tools/soma_procedural_blender.",
    )
    parser.add_argument(
        "--addon-dir",
        type=Path,
        default=None,
        help="Blender scripts/addons directory. Defaults to the detected user config path.",
    )
    parser.add_argument(
        "--blender-binary",
        default="blender",
        help="Blender executable used to detect the user add-on version directory.",
    )
    parser.add_argument(
        "--blender-version",
        default=None,
        help="Blender version directory, for example 4.0. Overrides --blender-binary detection.",
    )
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing add-on path.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing.")
    add_logging_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args)
    version = args.blender_version
    if args.addon_dir is None:
        if version is None:
            version = blender_version_from_binary(args.blender_binary)
        addon_dir = default_blender_addon_dir(version)
    else:
        addon_dir = args.addon_dir

    target = install_addon(
        plugin_root=args.plugin_root,
        addon_dir=addon_dir,
        force=args.force,
        copy=args.copy,
        dry_run=args.dry_run,
    )
    action = "Would install" if args.dry_run else "Installed"
    mode = "copy" if args.copy or platform.system() == "Windows" else "symlink"
    logger.info(f"{action} Blender add-on ({mode}): {target}")
    logger.info("Then in Blender: enable add-on 'SOMA Procedural Transforms'")
    logger.info("Reference panel: Properties > Object > SOMA Procedural")


if __name__ == "__main__":
    main()
