"""Strict on-disk contract for LC64 sparse-structure VAE X2 latents."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_NAME = "fire3d.ss_hcvae_latents.v1"
LEGACY_SCHEMA_NAMES = {"ff_holoscene.ss_vae_x2_latents.v1"}
PROVENANCE_FILENAME = "provenance.json"
ROTATIONS = ("000", "090", "180", "270")


@dataclass(frozen=True)
class SsX2LatentMetadata:
    input_resolution: int
    latent_resolution: int
    latent_channels: int
    occupied_count: int


def normalize_rotation_label(rotation: str | int) -> str:
    text = str(rotation).strip().lower().replace("rot", "")
    try:
        degrees = int(text) % 360
    except ValueError as exc:
        raise ValueError(f"Invalid SS latent rotation {rotation!r}") from exc
    label = f"{degrees:03d}"
    if label not in ROTATIONS:
        raise ValueError(
            f"Unsupported SS latent rotation {rotation!r}; expected one of {ROTATIONS}"
        )
    return label


def safe_latent_stem(name: str) -> str:
    """Return a flat filename-safe form without changing ordinary asset IDs."""
    text = str(name).strip()
    if not text:
        raise ValueError("SS latent stem cannot be empty")
    return text.replace("/", "__").replace("\\", "__")


def latent_filename(name: str, rotation: str | int) -> str:
    return f"{safe_latent_stem(name)}__rot{normalize_rotation_label(rotation)}.npz"


def resolve_latent_path(root: str | Path, name: str, rotation: str | int) -> Path:
    """Resolve exactly the requested rotation; never fall back to another view."""
    root = Path(root)
    rotation = normalize_rotation_label(rotation)
    candidates: list[Path] = []
    raw_name = str(name)
    if Path(raw_name).name == raw_name:
        candidates.append(root / f"{raw_name}__rot{rotation}.npz")
    safe_path = root / latent_filename(raw_name, rotation)
    if safe_path not in candidates:
        candidates.append(safe_path)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Missing SS VAE X2 latent for {name!r}, rotation={rotation}, root={root}; "
        f"checked={[str(path) for path in candidates]}"
    )


def _scalar_int(archive: Any, key: str, path: Path) -> int:
    if key not in archive:
        raise ValueError(f"Missing {key!r} in SS latent {path}")
    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f"Expected scalar {key!r} in {path}, got shape {value.shape}")
    return int(value.reshape(()).item())


def load_ss_x2_latent(
    path: str | Path,
    *,
    expected_input_resolution: int = 8,
    expected_latent_resolution: int = 2,
    expected_latent_channels: int = 8,
) -> tuple[np.ndarray, SsX2LatentMetadata]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        if "ss_latent" not in archive:
            raise ValueError(f"Missing 'ss_latent' in {path}")
        latent = np.asarray(archive["ss_latent"], dtype=np.float32)
        metadata = SsX2LatentMetadata(
            input_resolution=_scalar_int(archive, "input_resolution", path),
            latent_resolution=_scalar_int(archive, "latent_resolution", path),
            latent_channels=_scalar_int(archive, "latent_channels", path),
            occupied_count=_scalar_int(archive, "occupied_count", path),
        )

    expected_shape = (
        int(expected_latent_channels),
        int(expected_latent_resolution),
        int(expected_latent_resolution),
        int(expected_latent_resolution),
    )
    if latent.shape != expected_shape:
        raise ValueError(f"Expected SS latent {expected_shape}, got {latent.shape} at {path}")
    if metadata.input_resolution != int(expected_input_resolution):
        raise ValueError(
            f"SS input resolution mismatch at {path}: {metadata.input_resolution} != "
            f"{expected_input_resolution}"
        )
    if metadata.latent_resolution != int(expected_latent_resolution):
        raise ValueError(
            f"SS latent resolution mismatch at {path}: {metadata.latent_resolution} != "
            f"{expected_latent_resolution}"
        )
    if metadata.latent_channels != int(expected_latent_channels):
        raise ValueError(
            f"SS latent channel mismatch at {path}: {metadata.latent_channels} != "
            f"{expected_latent_channels}"
        )
    max_occupied = int(expected_input_resolution) ** 3
    if not 0 <= metadata.occupied_count <= max_occupied:
        raise ValueError(
            f"Invalid occupied_count={metadata.occupied_count} at {path}; expected [0,{max_occupied}]"
        )
    if not np.isfinite(latent).all():
        raise ValueError(f"Non-finite SS latent at {path}")
    return latent, metadata


def save_ss_x2_latent_atomic(
    path: str | Path,
    latent: np.ndarray,
    *,
    input_resolution: int,
    occupied_count: int,
) -> None:
    path = Path(path)
    latent = np.asarray(latent, dtype=np.float32)
    if latent.ndim != 4:
        raise ValueError(f"Expected SS latent [C,R,R,R], got {latent.shape}")
    if latent.shape[1:] != (latent.shape[1],) * 3:
        raise ValueError(f"SS latent spatial dimensions must be cubic, got {latent.shape}")
    if not np.isfinite(latent).all():
        raise ValueError("Refusing to save non-finite SS latent")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            ss_latent=latent,
            input_resolution=np.asarray(input_resolution, dtype=np.int16),
            latent_resolution=np.asarray(latent.shape[1], dtype=np.int16),
            latent_channels=np.asarray(latent.shape[0], dtype=np.int16),
            occupied_count=np.asarray(occupied_count, dtype=np.int32),
        )
    os.replace(tmp, path)


def load_and_validate_provenance(
    root: str | Path,
    *,
    expected_ss_artifact_id: str | None = None,
    expected_ss_step: int | None = None,
    expected_ss_encoder_sha256: str | None = None,
    filename: str = PROVENANCE_FILENAME,
) -> dict[str, Any]:
    path = Path(root) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing SS latent provenance manifest: {path}")
    payload = json.loads(path.read_text())
    if payload.get("schema") not in {SCHEMA_NAME, *LEGACY_SCHEMA_NAMES}:
        raise ValueError(
            f"Unexpected SS latent provenance schema at {path}: {payload.get('schema')!r}"
        )
    ss_vae = payload.get("ss_vae") or {}
    if expected_ss_artifact_id is not None:
        actual = str(ss_vae.get("artifact_id", ""))
        if actual != str(expected_ss_artifact_id):
            raise ValueError(
                f"SS VAE artifact mismatch at {path}: {actual!r} != "
                f"{expected_ss_artifact_id!r}"
            )
    if expected_ss_step is not None and int(ss_vae.get("step", -1)) != int(expected_ss_step):
        raise ValueError(
            f"SS VAE step mismatch at {path}: {ss_vae.get('step')} != {expected_ss_step}"
        )
    if expected_ss_encoder_sha256 is not None:
        actual = str(ss_vae.get("encoder_sha256", "")).lower()
        expected = str(expected_ss_encoder_sha256).lower()
        if actual != expected:
            raise ValueError(f"SS encoder SHA-256 mismatch at {path}: {actual} != {expected}")
    return payload
