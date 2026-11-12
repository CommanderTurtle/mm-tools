"""Download the pinned runtime assets used by 4DAnyone."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from fdanyone.assets import (
    BIREFNET_DIR,
    BIREFNET_FILES,
    BIREFNET_REPO_ID,
    BIREFNET_REVISION,
    DPVO_GVHMR_TARGET,
    EXAMPLE_FILES,
    GVHMR_LINKS,
    HF_REPO_ID,
    HF_REVISION,
    MODEL_FILES,
    PERCEPTUAL_VGG19,
    SMPLX_MODEL,
    resolve_dpvo_checkpoint,
    resolve_perceptual_vgg19,
)
from fdanyone.errors import AssetError

LOGGER = logging.getLogger("fdanyone")

SMPLX_NEUTRAL_URL = "https://huggingface.co/lilpotat/pytorch3d/resolve/main/models/SMPLX_NEUTRAL.npz?download=true"
SMPLX_NEUTRAL_SHA256 = "376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992"
SMPLX_ARCHIVE_MEMBER = ("models", "smplx", "SMPLX_NEUTRAL.npz")


def _snapshot(
    allow_patterns: list[str],
    local_dir: Path,
    *,
    repo_id: str = HF_REPO_ID,
    revision: str = HF_REVISION,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise AssetError("Install requirements.txt before downloading assets.") from exc

    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=allow_patterns,
            local_dir=local_dir,
        )
    except Exception as exc:
        raise AssetError(
            f"Could not download {repo_id}@{revision}. Check the network connection and Hugging Face access."
        ) from exc


def ensure_foreground_model(model_dir: str | Path = "models") -> Path:
    root = Path(model_dir).expanduser().resolve() / BIREFNET_DIR
    missing = [relative for relative in BIREFNET_FILES if not (root / relative).is_file()]
    if missing:
        LOGGER.info("Downloading BiRefNet foreground model (first run only)")
        _snapshot(
            missing,
            root,
            repo_id=BIREFNET_REPO_ID,
            revision=BIREFNET_REVISION,
        )
    return root


def require_gvhmr_checkout(gvhmr_root: str | Path) -> Path:
    root = Path(gvhmr_root).expanduser().resolve()
    if not (root / "hmr4d/__init__.py").is_file():
        raise AssetError(
            f"The integrated GVHMR runtime is missing at {root}. Re-run this project's setup."
        )
    return root


def _ensure_link(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        try:
            if destination.resolve(strict=True).samefile(source):
                return
        except FileNotFoundError:
            pass
        destination.unlink()
    elif destination.exists():
        if destination.samefile(source):
            return
        raise AssetError(
            f"GVHMR asset location is occupied by an unrelated file: {destination}. "
            f"Move it away so the downloaded {source.name} can be linked."
        )
    relative = os.path.relpath(source, start=destination.parent)
    destination.symlink_to(relative)


def create_classic_gvhmr_links(
    model_dir: str | Path = "models",
    gvhmr_root: str | Path = ".",
    *,
    require_models: bool = True,
    require_smplx: bool = True,
) -> Path:
    """Create the ignored compatibility links expected by upstream GVHMR."""

    root = require_gvhmr_checkout(gvhmr_root)
    models = Path(model_dir).expanduser().resolve()
    for relative, target in GVHMR_LINKS:
        source = models / relative
        if not source.is_file():
            required = require_smplx if relative == SMPLX_MODEL else require_models
            if required:
                command = "scripts/download_smplx.py" if relative == SMPLX_MODEL else "scripts/download_model.py"
                raise AssetError(f"Model file is missing: {source}. Run `python {command}` first.")
            continue
        _ensure_link(source, root / target)
    return root


def create_dpvo_gvhmr_link(
    model_dir: str | Path = "models",
    gvhmr_root: str | Path = ".",
    *,
    required: bool = True,
) -> Path | None:
    """Link the optional DPVO weights into the path expected by GVHMR."""

    root = require_gvhmr_checkout(gvhmr_root)
    try:
        source = resolve_dpvo_checkpoint(model_dir)
    except AssetError:
        if required:
            raise
        return None
    destination = root / DPVO_GVHMR_TARGET
    _ensure_link(source, destination)
    return destination


def ensure_models(
    model_dir: str | Path = "models",
    gvhmr_root: str | Path = ".",
) -> Path:
    """Download any missing published model file and refresh the GVHMR links."""

    require_gvhmr_checkout(gvhmr_root)
    models = Path(model_dir).expanduser().resolve()
    missing = [relative for relative in MODEL_FILES if not (models / relative).is_file()]
    if missing:
        LOGGER.info("Downloading %d model files from %s (first run only)", len(missing), HF_REPO_ID)
        # Repository paths match the local layout, so download straight into
        # place; huggingface_hub stages and resumes partial files itself.
        _snapshot(missing, models)
    ensure_foreground_model(models)
    create_classic_gvhmr_links(models, gvhmr_root, require_smplx=False)
    return models


def ensure_perceptual_vgg19(model_dir: str | Path = "models") -> Path:
    """Download only the optional VGG-19 reconstruction asset when missing."""

    models = Path(model_dir).expanduser().resolve()
    destination = models / PERCEPTUAL_VGG19
    if not destination.is_file():
        LOGGER.info("Downloading the perceptual VGG-19 model (first use only)")
        _snapshot([PERCEPTUAL_VGG19], models)
    return resolve_perceptual_vgg19(models)


def download_model(
    model_dir: str = "models",
    gvhmr_root: str = ".",
) -> dict[str, str]:
    """Download the published model checkpoints."""

    models = ensure_models(model_dir, gvhmr_root)
    return {
        "models": str(models),
        "revision": HF_REVISION,
        "foreground_revision": BIREFNET_REVISION,
    }


def download_example(data_dir: str = "data") -> dict[str, str]:
    """Download the bundled example clips."""

    data = Path(data_dir).expanduser().resolve()
    destinations = {relative: data / Path(relative).relative_to("data") for relative in EXAMPLE_FILES}
    missing = [relative for relative, destination in destinations.items() if not destination.is_file()]
    if missing:
        # Repository paths carry a leading ``data/`` prefix while --data_dir is
        # the local root itself, so stage the snapshot and move each file.
        data.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".download-", dir=data) as temporary:
            staging = Path(temporary)
            _snapshot(missing, staging)
            for relative in missing:
                destination = destinations[relative]
                destination.parent.mkdir(parents=True, exist_ok=True)
                (staging / relative).replace(destination)
    return {"examples": str(data / "source/pexels"), "revision": HF_REVISION}


def ensure_example_video(video_path: str | Path) -> Path:
    """Fetch a bundled example clip when its expected file is missing."""

    path = Path(video_path).expanduser()
    if path.is_file():
        return path
    matches = [relative for relative in EXAMPLE_FILES if PurePosixPath(relative).name == path.name]
    if not matches:
        raise AssetError(f"Input video does not exist: {path.resolve()}")
    LOGGER.info("Downloading the bundled example clip %s", path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination so the final rename stays on one filesystem.
    # Each invocation owns its staging directory. Concurrent first-use downloads
    # must not move or remove another invocation's files.
    with tempfile.TemporaryDirectory(prefix=".download-", dir=path.parent) as temporary:
        staging = Path(temporary)
        _snapshot(matches[:1], staging)
        (staging / matches[0]).replace(path)
    return path


def _copy_model_from_source(source: Path, destination: Path) -> None:
    if source.name == "SMPLX_NEUTRAL.npz":
        shutil.copyfile(source, destination)
        return
    if zipfile.is_zipfile(source):
        try:
            with zipfile.ZipFile(source) as archive:
                candidates = [
                    info
                    for info in archive.infolist()
                    if not info.is_dir() and PurePosixPath(info.filename).parts[-3:] == SMPLX_ARCHIVE_MEMBER
                ]
                if len(candidates) != 1:
                    raise AssetError(
                        "The archive must contain exactly one models/smplx/SMPLX_NEUTRAL.npz file. "
                        "Download models_smplx_v1_1.zip from the official SMPL-X website."
                    )
                with archive.open(candidates[0]) as model, destination.open("wb") as output:
                    shutil.copyfileobj(model, output, length=8 * 1024 * 1024)
        except zipfile.BadZipFile:
            raise AssetError(f"SMPL-X archive is invalid: {source}") from None
        return
    raise AssetError("Select models_smplx_v1_1.zip or SMPLX_NEUTRAL.npz.")


def install_smplx(
    source_path: str | Path,
    model_dir: str | Path = "models",
    gvhmr_root: str | Path = ".",
) -> Path:
    """Install a user-provided official ZIP or neutral NPZ."""

    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise AssetError(f"SMPL-X source does not exist: {source}")

    target = Path(model_dir).expanduser().resolve() / SMPLX_MODEL
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.download-{os.getpid()}"
    try:
        _copy_model_from_source(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    create_classic_gvhmr_links(model_dir, gvhmr_root, require_models=False, require_smplx=True)
    return target


def _download_pinned_smplx(destination: Path) -> None:
    """Fetch and checksum the neutral parameter file used by GVHMR."""

    digest = hashlib.sha256()
    request = urllib.request.Request(SMPLX_NEUTRAL_URL, headers={"User-Agent": "mm-tools/4d-ify"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response, destination.open("wb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
    except (OSError, urllib.error.URLError) as exc:
        destination.unlink(missing_ok=True)
        raise AssetError(f"Could not fetch the pinned SMPL-X neutral parameter: {exc}") from None
    if digest.hexdigest() != SMPLX_NEUTRAL_SHA256:
        destination.unlink(missing_ok=True)
        raise AssetError("The downloaded SMPL-X neutral parameter failed its SHA-256 check.")


def download_smplx(
    archive_path: str | None = None,
    model_dir: str = "models",
    gvhmr_root: str = ".",
) -> dict[str, str] | None:
    """Install the pinned SMPL-X neutral body model inside this project."""

    target = Path(model_dir).expanduser().resolve() / SMPLX_MODEL
    if target.is_file():
        create_classic_gvhmr_links(model_dir, gvhmr_root, require_models=False, require_smplx=True)
        return {"installed": str(target)}

    if archive_path is not None:
        return {"installed": str(install_smplx(archive_path, model_dir, gvhmr_root))}

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.download-{os.getpid()}"
    try:
        _download_pinned_smplx(temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    create_classic_gvhmr_links(model_dir, gvhmr_root, require_models=False, require_smplx=True)
    return {"installed": str(target)}


def ensure_smplx(
    model_dir: str | Path = "models",
    gvhmr_root: str | Path = ".",
) -> Path:
    """Ensure setup has installed the project-local neutral body parameters."""

    require_gvhmr_checkout(gvhmr_root)
    target = Path(model_dir).expanduser().resolve() / SMPLX_MODEL
    if target.is_file():
        create_classic_gvhmr_links(model_dir, gvhmr_root, require_models=False, require_smplx=True)
        return target

    result = download_smplx(model_dir=str(model_dir), gvhmr_root=str(gvhmr_root))
    if result is None or not target.is_file():
        raise AssetError("SMPL-X setup was cancelled; inference cannot continue.")
    LOGGER.info("SMPL-X installed; continuing inference")
    return target
