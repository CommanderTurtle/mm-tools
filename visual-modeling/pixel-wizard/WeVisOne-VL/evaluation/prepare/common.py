"""Helper functions shared by the conversion scripts in ``evaluation/prepare``."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    os.replace(temporary, path)
    print(f"wrote {count} rows -> {path}", flush=True)
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def iter_parquet(paths: Iterable[Path], columns: list[str] | None = None) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq

    for path in sorted(paths):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=columns):
            yield from batch.to_pylist()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_suffix(image: dict[str, Any], default: str = ".jpg") -> str:
    suffix = Path(str(image.get("path") or "")).suffix.lower()
    return suffix if suffix in IMAGE_SUFFIXES else default


def write_bytes(payload: bytes, destination: Path) -> None:
    """Write ``payload`` unless an identical-size file is already there."""
    if destination.is_file() and destination.stat().st_size == len(payload):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)


def save_jpeg(payload: bytes, destination: Path) -> None:
    """Re-encode an embedded image as RGB JPEG (quality 95)."""
    from PIL import Image

    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.open(io.BytesIO(payload)).convert("RGB").save(destination, format="JPEG", quality=95)


def json_rle(mask: np.ndarray) -> dict[str, Any]:
    from pycocotools import mask as mask_utils

    return rle_to_json(mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))))


def rle_to_json(rle: dict[str, Any]) -> dict[str, Any]:
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(value) for value in rle["size"]], "counts": counts}


def _member_path(name: str) -> PurePosixPath:
    relative = PurePosixPath(name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe archive member: {name!r}")
    return relative


def extract_members(archive: Path, destination: Path, wanted: set[str] | None = None) -> None:
    """Extract the regular files in ``wanted`` (archive-relative paths; all if None) from a zip or tar.

    Each file keeps its archive path below ``destination``; files already there are skipped.
    """

    def target_of(relative: PurePosixPath) -> Path:
        return destination / Path(*relative.parts)

    if wanted is None:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as bundle:
                wanted = {_member_path(m.filename).as_posix() for m in bundle.infolist() if not m.is_dir()}
        else:
            with tarfile.open(archive) as bundle:
                wanted = {_member_path(m.name).as_posix() for m in bundle if m.isfile()}
    remaining = {name for name in wanted if not target_of(PurePosixPath(name)).is_file()}
    if not remaining:
        return
    print(f"extracting {len(remaining)} files from {archive}", flush=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                relative = _member_path(member.filename)
                if member.is_dir() or relative.as_posix() not in remaining:
                    continue
                target = target_of(relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1 << 20)
                remaining.discard(relative.as_posix())
    else:
        with tarfile.open(archive) as bundle:
            for member in bundle:
                relative = _member_path(member.name)
                if not member.isfile() or relative.as_posix() not in remaining:
                    continue
                target = target_of(relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1 << 20)
                remaining.discard(relative.as_posix())
                if not remaining:
                    break
    if remaining:
        raise FileNotFoundError(f"{len(remaining)} files absent from {archive}: {sorted(remaining)[:5]}")


def link_directory(link: Path, target: Path) -> None:
    """Point ``link`` at ``target`` unless it already resolves there."""
    target = target.resolve(strict=True)
    if link.is_symlink() or link.exists():
        if link.resolve() != target:
            raise FileExistsError(f"{link} exists and does not point to {target}")
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
