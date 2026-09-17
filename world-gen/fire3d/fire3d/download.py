"""Hugging Face artifact downloads for Fire3D."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import Iterable

MODEL_REPO = "hongchi/Fire3D"
DATA_REPO = "hongchi/Fire3D"
DATASET_SUBDIRS = {
    "ithor": "ithor",
    "imaginarium": "Imaginarium",
    "scannetpp": "scannetpp",
    "single_image": "single_image",
}
EVALUATION_DATASETS = ("shaper",)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_download(**kwargs):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required; run scripts/install.sh first"
        ) from error
    return snapshot_download(**kwargs)


def download_models(destination: Path, *, revision: str | None = None) -> Path:
    destination = destination.expanduser().resolve()
    _snapshot_download(
        repo_id=MODEL_REPO,
        repo_type="model",
        revision=revision,
        local_dir=destination,
    )
    hub_config_path = destination / "config.json"
    if not hub_config_path.is_file():
        raise FileNotFoundError(
            "The Fire3D model snapshot is missing its root config.json query file"
        )
    hub_config = json.loads(hub_config_path.read_text(encoding="utf-8"))
    if hub_config.get("schema") != "fire3d_model_bundle_v1":
        raise ValueError("Unsupported Fire3D Hub model config")
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "fire3d_model_bundle_v1":
        raise ValueError("Unsupported Fire3D model manifest")
    for record in manifest.get("files", []):
        path = destination / record["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing downloaded model file: {path}")
        digest = sha256_file(path)
        if digest != record["sha256"]:
            raise ValueError(
                f"Checksum mismatch for {record['path']}: "
                f"{digest} != {record['sha256']}"
            )
    return destination


def download_data(
    destination: Path,
    datasets: Iterable[str],
    *,
    scene_ids: Iterable[str] = (),
    revision: str | None = None,
    keep_archives: bool = False,
) -> Path:
    destination = destination.expanduser().resolve()
    selected = tuple(dict.fromkeys(datasets))
    requested_scenes = tuple(dict.fromkeys(scene_ids))
    unknown = sorted(set(selected) - set(DATASET_SUBDIRS))
    if unknown:
        raise ValueError(f"Unknown Fire3D datasets: {unknown}")
    if requested_scenes and len(selected) != 1:
        raise ValueError("--scene-id requires exactly one --dataset")

    archive_root = destination / ".fire3d_archives"
    metadata_patterns = ["README.md", "manifest.json", "checksums.sha256", "licenses/**"]
    _snapshot_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        revision=revision,
        local_dir=archive_root,
        allow_patterns=metadata_patterns,
    )
    manifest = json.loads((archive_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "fire3d_inference_data_v1":
        raise ValueError("Unsupported Fire3D dataset manifest")

    archive_paths: list[str] = []
    for dataset in selected:
        record = manifest["datasets"][dataset]
        archive_paths.extend(record.get("common_archives", []))
        scenes = record["scenes"]
        chosen = requested_scenes or tuple(scenes)
        missing = sorted(set(chosen) - set(scenes))
        if missing:
            raise ValueError(f"Scenes are not in the {dataset} release whitelist: {missing}")
        archive_paths.extend(scenes[scene]["archive"] for scene in chosen)

    _snapshot_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        revision=revision,
        local_dir=archive_root,
        allow_patterns=archive_paths,
    )
    for relative in dict.fromkeys(archive_paths):
        archive = archive_root / relative
        expected = manifest["archives"][relative]["sha256"]
        digest = sha256_file(archive)
        if digest != expected:
            raise ValueError(f"Checksum mismatch for {relative}: {digest} != {expected}")
        with tarfile.open(archive, "r") as handle:
            target = destination.resolve()
            for member in handle.getmembers():
                member_path = (destination / member.name).resolve()
                if target != member_path and target not in member_path.parents:
                    raise ValueError(f"Unsafe archive member in {relative}: {member.name}")
            handle.extractall(destination)
    if not keep_archives:
        shutil.rmtree(archive_root)
    return destination


def download_evaluation_data(
    destination: Path,
    evaluations: Iterable[str],
    *,
    revision: str | None = None,
    keep_archives: bool = False,
) -> Path:
    destination = destination.expanduser().resolve()
    selected = tuple(dict.fromkeys(evaluations))
    unknown = sorted(set(selected) - set(EVALUATION_DATASETS))
    if unknown:
        raise ValueError(f"Unknown Fire3D evaluation datasets: {unknown}")
    archive_root = destination / ".fire3d_archives"
    metadata_patterns = ["README.md", "manifest.json", "checksums.sha256", "licenses/**"]
    _snapshot_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        revision=revision,
        local_dir=archive_root,
        allow_patterns=metadata_patterns,
    )
    manifest = json.loads((archive_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "fire3d_inference_data_v1":
        raise ValueError("Unsupported Fire3D dataset manifest")
    archive_paths = [
        path
        for name in selected
        for path in manifest.get("evaluations", {}).get(name, {}).get("archives", [])
    ]
    if len(archive_paths) != len(selected):
        raise ValueError("Selected evaluation data is absent from the release manifest")
    _snapshot_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        revision=revision,
        local_dir=archive_root,
        allow_patterns=archive_paths,
    )
    for relative in archive_paths:
        archive = archive_root / relative
        expected = manifest["archives"][relative]["sha256"]
        if sha256_file(archive) != expected:
            raise ValueError(f"Checksum mismatch for {relative}")
        with tarfile.open(archive, "r") as handle:
            target = destination.resolve()
            for member in handle.getmembers():
                member_path = (destination / member.name).resolve()
                if target != member_path and target not in member_path.parents:
                    raise ValueError(f"Unsafe archive member in {relative}: {member.name}")
            handle.extractall(destination)
    if not keep_archives:
        shutil.rmtree(archive_root)
    return destination
