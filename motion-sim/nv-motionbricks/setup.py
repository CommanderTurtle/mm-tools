"""Verify the picker's cache and expose it at native asset paths; no network."""

import hashlib
import json
from pathlib import Path

from storage import PIN, ROOT

REPO = next(
    (p for p in ROOT.parents if (p / "models/download_models.py").is_file()), None
)
if REPO is None:
    raise SystemExit("Place this studio inside the mm-tools checkout.")
MANIFEST = json.loads((REPO / "models/manifests/motionbricks.json").read_text())
CACHE = REPO / "models" / PIN["cache"]


def verify() -> None:
    for entry in MANIFEST["files"]:
        file = CACHE / entry["path"]
        if not file.is_file() or file.stat().st_size != entry["size"]:
            raise SystemExit(
                "Select MotionBricks in models/download_models.py first: " + str(file)
            )
        with file.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                raise SystemExit("Cached asset checksum differs: " + str(file))


def link(path: Path, target: Path) -> None:
    if path.is_symlink() and path.resolve() == target.resolve():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise SystemExit(
            "Preserving existing path; expected a model-cache link: " + str(path)
        )
    path.symlink_to(target, target_is_directory=target.is_dir())


def main() -> None:
    verify()
    link(ROOT / "models", CACHE)
    for relative in ("assets", "motionbricks/assets"):
        link(ROOT / relative, CACHE / relative)


if __name__ == "__main__":
    main()
