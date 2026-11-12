"""Portable document references and a private, persistent workspace."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import uuid

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT
WORK = ROOT / "workspace"
PIN = {
    'cache': 'motionbricks/54fc563aeaaf819a753d72cc9fdc6713989cdd43',
}



def tree_stamp(tree: Path) -> str:
    """Identity of the source this lane runs, derived from the files on disk."""
    generated = {"__pycache__", ".venv", "node_modules", "workspace", ".work",
                 "runs", ".git-archives"}
    scratch = {".env", ".physics-ready", ".snapshot-members", "MUJOCO_LOG.TXT"}
    digest = hashlib.sha256()
    for path in sorted(p for p in tree.rglob("*")
                       if p.is_file() and p.suffix not in (".pyc", ".lst")
                       and p.name not in scratch
                       and not generated & set(p.relative_to(tree).parts)):
        digest.update(path.relative_to(tree).as_posix().encode())
        digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()[:40]


LANE_STAMP = tree_stamp(ROOT)

def resolve(name: str) -> Path:
    if not isinstance(name, str):
        raise ValueError("File path must be text.")
    parts = PurePosixPath(name).parts
    if not parts or parts[0] not in ("assets", "workspace") or "\\" in name:
        raise ValueError("Choose an asset or workspace file.")
    base = SOURCE / "assets" if parts[0] == "assets" else WORK
    path = base.joinpath(*parts[1:]).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("File leaves the studio workspace.")
    return path


def virtual(path: Path) -> str:
    for prefix, base in (("assets", SOURCE / "assets"), ("workspace", WORK)):
        if path.resolve().is_relative_to(base.resolve()):
            return prefix + "/" + path.resolve().relative_to(base.resolve()).as_posix()
    raise ValueError(
        "Document references a file outside assets/workspace: " + str(path)
    )


def refs(value, convert):
    if isinstance(value, dict):
        return {
            key: convert(item)
            if key in ("clip", "reference") and isinstance(item, str) and item
            else refs(item, convert)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [refs(item, convert) for item in value]
    return value


def read_document(name):
    path = resolve(name)
    if not path.is_file() or path.suffix.lower() != ".json":
        raise ValueError("Choose an existing JSON document.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return refs(data, lambda ref: virtual(path.parent / ref))


def write_document(name, data):
    path = resolve(name)
    if not path.is_relative_to(WORK.resolve()) or path.suffix != ".json":
        raise ValueError("Save JSON documents in workspace/.")
    portable = refs(
        data, lambda ref: Path(os.path.relpath(resolve(ref), path.parent)).as_posix()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + uuid.uuid4().hex + ".partial")
    try:
        temporary.write_text(
            json.dumps(portable, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def catalog():
    result = []
    for prefix, base in (("assets", SOURCE / "assets"), ("workspace", WORK)):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if (
                path.is_file()
                and path.suffix.lower() in (".json", ".bvh", ".csv", ".npz")
                and not any(p.startswith(".") for p in path.relative_to(base).parts)
            ):
                result.append(
                    {
                        "path": prefix + "/" + path.relative_to(base).as_posix(),
                        "name": path.name,
                        "size": path.stat().st_size,
                    }
                )
    return result
