#!/usr/bin/env python
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Build the VBench scorer package used for the paper into ``VBench/vbench``.

    python setup_scorer.py                      # download vbench==0.1.5 with pip, build, verify
    python setup_scorer.py --wheel vbench-0.1.5-py3-none-any.whl
    python setup_scorer.py --verify             # only check an existing build

The package is the published PyPI wheel ``vbench==0.1.5`` (pinned by SHA-256) with
``patches/upstream_post_0.1.5.diff`` and ``patches/vbench_package.diff`` applied.
Only the files the 16 text-to-video dimensions and the RAFT static filter use are kept.
Every file is checked against ``scorer_manifest.json``. The package bundles third-party
code, so this repository builds it locally instead of including it. See docs/inference.md#scorer-provenance.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "VBench"
WHEEL = "vbench-0.1.5-py3-none-any.whl"
WHEEL_SHA256 = "13266ed4715d91a7669ab58484b9e010bc91bb96e129d414fb512d4937890023"
PATCHES = ("upstream_post_0.1.5.diff", "vbench_package.diff")

sys.path.insert(0, str(HERE))
from protocol import sha256_file  # noqa: E402


def manifest_files():
    return json.loads((HERE / "scorer_manifest.json").read_text(encoding="utf-8"))["vendored_files"]


def verify(root=ROOT):
    """Raise unless ``root`` holds exactly the manifest's files with their SHA-256."""
    if not (root / "vbench").is_dir():
        raise FileNotFoundError(f"{root / 'vbench'} is missing; build it with setup_scorer.py")
    expected = manifest_files()
    on_disk = {str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts}
    missing = sorted(set(expected) - set(on_disk))
    extra = sorted(set(on_disk) - set(expected))
    changed = sorted(rel for rel in set(expected) & set(on_disk) if expected[rel] != on_disk[rel])
    if missing or extra or changed:
        raise RuntimeError(f"scorer tree differs from scorer_manifest.json: missing={missing} extra={extra} "
                           f"changed={changed}")
    return len(on_disk)


def build(wheel):
    if sha256_file(wheel) != WHEEL_SHA256:
        raise RuntimeError(f"{wheel} is not the vbench 0.1.5 wheel (SHA-256 mismatch)")
    dest = ROOT / "vbench"
    if dest.exists():
        raise FileExistsError(f"{dest} exists; remove it to rebuild")
    wanted = {rel for rel in manifest_files() if rel.startswith("vbench/")}
    with zipfile.ZipFile(wheel) as archive:
        absent = wanted - set(archive.namelist())
        if absent:
            raise RuntimeError(f"wheel lacks {sorted(absent)}")
        for rel in sorted(wanted):
            target = ROOT / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(rel))
    for name in PATCHES:
        subprocess.run(["patch", "-p1", "--forward", "--no-backup-if-mismatch", "--quiet", "-d", str(ROOT),
                        "-i", str(HERE / "patches" / name)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wheel", type=Path, help=f"a local copy of {WHEEL} (default: download with pip)")
    parser.add_argument("--verify", action="store_true", help="only verify an existing build")
    args = parser.parse_args()
    if not args.verify:
        with tempfile.TemporaryDirectory() as tmp:
            wheel = args.wheel
            if wheel is None:
                subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary", ":all:",
                                "-d", tmp, "vbench==0.1.5"], check=True)
                wheel = Path(tmp) / WHEEL
            build(wheel)
    print(f"VBench scorer OK: {verify()} files match scorer_manifest.json")


if __name__ == "__main__":
    main()
