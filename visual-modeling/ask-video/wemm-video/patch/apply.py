"""Apply a version-matched WeMM overlay to an installed package or source tree."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from tempfile import NamedTemporaryFile


PATCH_ROOT = Path(__file__).resolve().parent
VERSIONS = {"vllm": ("0.18.0", "0.31.0"), "sglang": ("0.5.16", "0.5.19")}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def target_root(engine, version, checkout):
    if checkout is not None:
        return (checkout / "python" if engine == "sglang" else checkout).resolve()

    installed = importlib.metadata.version(engine)
    if installed.split("+", 1)[0] != version:
        raise ValueError(f"Expected {engine} {version}, found {installed}")
    spec = importlib.util.find_spec(engine)
    if spec is None or spec.origin is None:
        raise ValueError(f"Cannot locate the installed {engine} package")
    return Path(spec.origin).resolve().parent.parent


def file_path(root, name, engine):
    relative = Path(name)
    if len(relative.parts) < 2 or relative.parts[0] != engine or ".." in relative.parts:
        raise ValueError(f"Invalid manifest path: {name}")
    path = root / relative
    if path.resolve() != path or (path.exists() and not path.is_file()):
        raise ValueError(f"Expected a regular file or a new path: {path}")
    return path


def build_plan(root, bundle, engine):
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest["upstream_tag"] != f"v{bundle.name}":
        raise ValueError("Patch directory and manifest versions differ")
    for name, expected in manifest["compatibility"].items():
        path = file_path(root, name, engine)
        if not path.is_file() or sha256(path.read_bytes()) != expected:
            raise ValueError(f"Incompatible {engine} source file: {path}")

    plan = []
    for name, record in manifest["files"].items():
        source = file_path(bundle / "overlay", name, engine)
        path = file_path(root, name, engine)
        data = source.read_bytes()
        if sha256(data) != record["sha256"]:
            raise ValueError(f"Patch file does not match its manifest: {source}")
        compile(data, str(source), "exec")
        exists = path.is_file()
        current = sha256(path.read_bytes()) if exists else None
        if current == record["sha256"]:
            continue
        if current != record["upstream_sha256"]:
            raise ValueError(f"Unexpected base content in {path}; use a clean {engine} {bundle.name}")
        mode = stat.S_IMODE(path.stat().st_mode) if exists else 0o644
        plan.append((path, data, mode))
    return plan


def write_source(path, data, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".wemm-", delete=False) as stream:
        pending = Path(stream.name)
        try:
            stream.write(data)
            stream.close()
            pending.chmod(mode)
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
    try:
        # Keep imports correct when a pre-existing bytecode file has a matching timestamp.
        Path(importlib.util.cache_from_source(str(path))).unlink(missing_ok=True)
    except NotImplementedError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=VERSIONS)
    parser.add_argument("version", help="Exact upstream package version")
    parser.add_argument("--target", type=Path, help="Apply to an upstream repository checkout")
    parser.add_argument("--dry-run", action="store_true", help="Check all hashes without writing")
    args = parser.parse_args()
    if args.version not in VERSIONS[args.engine]:
        parser.error(f"Choose {args.engine} " + " or ".join(VERSIONS[args.engine]))

    try:
        root = target_root(args.engine, args.version, args.target)
        bundle = PATCH_ROOT / args.engine / args.version
        package = root / args.engine
        if package.resolve() != package or not package.is_dir():
            raise ValueError(f"Package directory does not exist or is a symlink: {package}")
        directory = os.open(package, os.O_RDONLY | os.O_DIRECTORY)
        try:
            lock = fcntl.LOCK_SH if args.dry_run else fcntl.LOCK_EX
            fcntl.flock(directory, lock)
            plan = build_plan(root, bundle, args.engine)
            if not args.dry_run:
                for path, data, mode in plan:
                    write_source(path, data, mode)
        finally:
            os.close(directory)
    except (OSError, SyntaxError, ValueError, importlib.metadata.PackageNotFoundError) as error:
        print(f"Patch error: {error}", file=sys.stderr)
        return 1
    action = "would update" if args.dry_run else "updated"
    print(f"{args.engine} {args.version}: {action} {len(plan)} files in {package}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
