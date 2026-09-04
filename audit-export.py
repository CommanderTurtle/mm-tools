#!/usr/bin/env python3
"""Audit the tracked runtime payload; optionally export it without touching production."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import tomllib
from collections import defaultdict


def tracked(root: Path) -> list[str]:
    return subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().rstrip("\0").split("\0")


def audit(root: Path, files: list[str]) -> dict:
    problems = []
    sizes = defaultdict(lambda: [0, 0])
    included = set(files)
    for name in files:
        path = root / name
        if path.is_symlink() or not path.is_file():
            problems.append(f"Missing/non-regular tracked file: {name}")
            continue
        if any(p in {".venv", "node_modules", "__pycache__", ".git", "outputs"} for p in path.relative_to(root).parts):
            problems.append(f"Runtime artifact is tracked: {name}")
        if path.name in {".env", "id_rsa", "id_ed25519"} or path.suffix in {".pyc", ".safetensors", ".gguf", ".pt", ".ckpt"}:
            problems.append(f"Local configuration/checkpoint is tracked: {name}")
        entry = sizes[name.split('/')[0]]
        entry[0] += 1
        entry[1] += path.stat().st_size
        if path.name != "pyproject.toml":
            continue
        project = tomllib.loads(path.read_text()).get("project", {})
        required = []
        readme = project.get("readme")
        if isinstance(readme, str):
            required.append(readme)
        elif isinstance(readme, dict) and "file" in readme:
            required.append(readme["file"])
        license_spec = project.get("license")
        if isinstance(license_spec, dict) and "file" in license_spec:
            required.append(license_spec["file"])
        for pattern in project.get("license-files", []):
            matches = list(path.parent.glob(pattern))
            if not matches:
                problems.append(f"Missing license glob: {name}: {pattern}")
            required.extend(str(p.relative_to(path.parent)) for p in matches)
        for dependency in required:
            target = (path.parent / dependency).relative_to(root).as_posix()
            if target not in included:
                problems.append(f"Build metadata needs untracked file: {name} -> {target}")
    return {"files": len(files), "bytes": sum(v[1] for v in sizes.values()),
            "projects": dict(sorted(sizes.items())), "problems": problems}


def export(root: Path, files: list[str]) -> str:
    # The source list is the index; bytes are the current working tree, as with
    # git ls-files | tar. No .git archive, secrets, models, or production writes.
    destination = Path(tempfile.mkdtemp(prefix="mm-tools-export-"))
    manifest = {}
    for name in files:
        source = root / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Cannot safely export non-regular file: {name}")
        target = destination / name
        if not target.resolve().is_relative_to(destination):
            raise ValueError(f"Invalid tracked path: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        target.write_bytes(data)
        target.chmod(source.stat().st_mode & 0o777)
        manifest[name] = hashlib.sha256(data).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != manifest[name]:
            raise ValueError(f"Export verification failed: {name}")
    (destination / "EXPORT-MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return str(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true", help="Create a fresh temporary tracked-file copy and SHA256 manifest")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    files = tracked(root)
    report = audit(root, files)
    if args.export and not report["problems"]:
        report["export"] = export(root, files)
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report["problems"]))


if __name__ == "__main__":
    main()
