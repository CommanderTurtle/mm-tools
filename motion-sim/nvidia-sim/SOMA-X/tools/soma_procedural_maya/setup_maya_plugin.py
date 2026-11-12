# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install the SOMA procedural-control reference Maya module file."""

import argparse
import logging
import os
import platform
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.logging_utils import add_logging_args, configure_logging  # noqa: E402

MODULE_NAME = "SOMAProceduralMaya"
MODULE_VERSION = "0.1"
PLUGIN_FILE = "soma_procedural_maya_plugin.py"

logger = logging.getLogger(__name__)


def default_maya_module_dir() -> Path:
    system = platform.system()
    home = Path.home()
    if system == "Windows":
        documents = Path(os.environ.get("USERPROFILE", str(home))) / "Documents"
        return documents / "maya" / "modules"
    if system == "Darwin":
        return home / "Library" / "Preferences" / "Autodesk" / "maya" / "modules"
    return home / "maya" / "modules"


def build_module_file(plugin_root: Path) -> str:
    plugin_root = plugin_root.resolve()
    return "\n".join(
        [
            f"+ {MODULE_NAME} {MODULE_VERSION} {plugin_root.as_posix()}",
            "PYTHONPATH +:= .",
            "MAYA_PLUG_IN_PATH +:= plug-ins",
            "",
        ]
    )


def install_module_file(
    plugin_root: Path,
    module_dir: Path,
    force: bool = False,
    dry_run: bool = False,
) -> Path:
    plugin_root = plugin_root.resolve()
    if not (plugin_root / "plug-ins" / PLUGIN_FILE).exists():
        raise FileNotFoundError(f"Missing Maya plug-in file under {plugin_root}")

    module_path = module_dir / f"{MODULE_NAME}.mod"
    if module_path.exists() and not force:
        raise FileExistsError(f"{module_path} already exists. Pass --force to overwrite.")

    if not dry_run:
        module_dir.mkdir(parents=True, exist_ok=True)
        module_path.write_text(build_module_file(plugin_root), encoding="utf-8")
    return module_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plugin-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to tools/soma_procedural_maya.",
    )
    parser.add_argument(
        "--module-dir",
        type=Path,
        default=default_maya_module_dir(),
        help="Maya modules directory where the .mod file should be written.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing .mod file.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing.")
    add_logging_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args)
    module_path = install_module_file(
        plugin_root=args.plugin_root,
        module_dir=args.module_dir,
        force=args.force,
        dry_run=args.dry_run,
    )
    action = "Would write" if args.dry_run else "Wrote"
    logger.info(f"{action} Maya module file: {module_path}")
    logger.info(f"Then in Maya: loadPlugin {PLUGIN_FILE}")
    logger.info("Reference node: somaProceduralTransforms")
    logger.info("Reference command: somaCreateProceduralRigReference -definitionPath <json>")


if __name__ == "__main__":
    main()
