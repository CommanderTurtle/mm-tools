# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Cache TensorRT plans by a hash of the model, TensorRT version, and GPU."""

import hashlib
import json
import os
from pathlib import Path
import platform
import re
from typing import Any, Mapping
import uuid

from .errors import EngineCacheError


_CACHE_SCHEMA_VERSION = 1
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file in lowercase hexadecimal.

    The file is read in chunks, so a large ONNX model never loads into memory at once.

    Args:
        path: File to hash.
    """

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def make_cache_identity(
    *,
    onnx_sha256: str,
    tensorrt_version: str,
    build_options: Mapping[str, Any],
    device_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the values that determine TensorRT plan compatibility.

    The identity also records the operating system and CPU architecture.

    Args:
        onnx_sha256: SHA-256 digest of the ONNX file, as sha256_file() returns it.
        tensorrt_version: TensorRT version string.
        build_options: Build options that change the plan, as
            TensorRTBuildOptions.cache_payload() returns them.
        device_fingerprint: Values that identify the GPU, such as its name and
            architecture.
    """

    return {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "onnx_sha256": onnx_sha256,
        "tensorrt_version": tensorrt_version,
        "build_options": dict(build_options),
        "device": dict(device_fingerprint),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
    }


def cache_key_for(identity: Mapping[str, Any]) -> str:
    """Hash a cache identity into the key in its plan's file name, a SHA-256 digest.

    Args:
        identity: Values from make_cache_identity().
    """

    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class TensorRTEngineCache:
    """A directory of atomically written TensorRT plan files.

    A JSON file next to each plan describes how it was built. Failed reads and
    writes raise EngineCacheError.
    """

    def __init__(self, directory: str | Path) -> None:
        """Store plans in directory.

        Args:
            directory: Directory for the cache. The first write creates it.
        """
        self.directory = Path(directory).expanduser().resolve()

    def path_for(self, model_name: str, key: str) -> Path:
        """Return the path of the plan file.

        Characters that are unsafe in file names become dashes.

        Args:
            model_name: Name of the model, usually the ONNX file's stem.
            key: Cache key of the plan.
        """
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name).strip(".-") or "model"
        return self.directory / f"{safe_name}-{key}.plan"

    def metadata_path_for(self, model_name: str, key: str) -> Path:
        """Return the path of the JSON file that describes the plan.

        Args:
            model_name: Name of the model, usually the ONNX file's stem.
            key: Cache key of the plan.
        """
        return self.path_for(model_name, key).with_suffix(".json")

    def load(self, model_name: str, key: str) -> bytes | None:
        """Return the cached plan, or None when the cache has no entry.

        An empty plan file counts as no entry, and the cache removes it.

        Args:
            model_name: Name of the model, usually the ONNX file's stem.
            key: Cache key of the plan.
        """
        path = self.path_for(model_name, key)
        if not path.is_file() or not self.metadata_path_for(model_name, key).is_file():
            return None
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise EngineCacheError(f"Could not read plan cache entry {path}: {exc}") from exc
        if not data:
            self.invalidate(model_name, key)
            return None
        return data

    def store(
        self,
        model_name: str,
        key: str,
        plan: bytes,
        *,
        metadata: Mapping[str, Any],
    ) -> None:
        """Write the plan and its JSON file. Both files are written atomically.

        Args:
            model_name: Name of the model, usually the ONNX file's stem.
            key: Cache key of the plan.
            plan: Serialized TensorRT plan. It must not be empty.
            metadata: JSON values saved next to the plan.
        """
        if not plan:
            raise EngineCacheError("Cannot cache an empty TensorRT plan.")

        path = self.path_for(model_name, key)
        try:
            _write_atomic(path, plan)
            self._store_metadata(model_name, key, metadata)
        except OSError as exc:
            raise EngineCacheError(f"Could not write plan cache entry {path}: {exc}") from exc

    def _store_metadata(
        self,
        model_name: str,
        key: str,
        metadata: Mapping[str, Any],
    ) -> None:
        """Write a readable file next to the plan that describes how it was built."""

        plan_path = self.path_for(model_name, key)
        path = self.metadata_path_for(model_name, key)
        record = dict(metadata)
        record["cache_key"] = key
        record["plan_file"] = plan_path.name

        try:
            data = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
            _write_atomic(path, data)
        except (OSError, TypeError, ValueError) as exc:
            raise EngineCacheError(f"Could not write plan cache metadata {path}: {exc}") from exc

    def invalidate(self, model_name: str, key: str) -> None:
        """Remove the plan and its JSON file, if they exist.

        Args:
            model_name: Name of the model, usually the ONNX file's stem.
            key: Cache key of the plan.
        """
        for path in (
            self.path_for(model_name, key),
            self.metadata_path_for(model_name, key),
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                raise EngineCacheError(
                    f"Could not remove invalid plan cache entry {path}: {exc}"
                ) from exc


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary_path.open("xb") as stream:
            stream.write(data)
            stream.flush()
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
