#!/usr/bin/env python3

"""Validate package version metadata for release workflows."""

import argparse
import ast
import configparser
import re
import sys
from pathlib import Path

_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _read_setup_version(repo_root: Path) -> str:
    config = configparser.ConfigParser()
    config.read(repo_root / "setup.cfg")
    return config["metadata"]["version"].strip()


def _read_package_version(repo_root: Path) -> str:
    init_path = repo_root / "soma" / "__init__.py"
    module = ast.parse(init_path.read_text(), filename=str(init_path))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        is_version_assignment = any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
        if not is_version_assignment:
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    raise RuntimeError("Could not find string assignment to soma.__version__")


def _normalize_expected(value: str) -> str:
    expected = value[1:] if value.startswith("v") else value
    if not _VERSION_RE.fullmatch(expected):
        raise ValueError(f"Expected release version must be MAJOR.MINOR.PATCH, got: {value}")
    return expected


def check_versions(repo_root: Path, expected: str | None = None) -> str:
    setup_version = _read_setup_version(repo_root)
    package_version = _read_package_version(repo_root)

    if setup_version != package_version:
        raise ValueError(
            f"Version mismatch: setup.cfg has {setup_version}, "
            f"soma.__version__ has {package_version}"
        )

    if expected:
        normalized_expected = _normalize_expected(expected)
        if setup_version != normalized_expected:
            raise ValueError(
                f"Version mismatch: package metadata has {setup_version}, "
                f"release tag expects {normalized_expected}"
            )

    return setup_version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", help="Expected version, optionally prefixed with v.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root to inspect.",
    )
    args = parser.parse_args()

    try:
        version = check_versions(args.root.resolve(), args.expected)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
