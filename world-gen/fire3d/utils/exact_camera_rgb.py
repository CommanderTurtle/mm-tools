"""Per-scene resolution of exact-camera (corrected) RGB frames.

The original validation renders carry a false-alpha bug: source GLBs declare
`alphaMode=BLEND`
on materials that are physically opaque (every alpha texel 255, base-color alpha
factor 1.0), so Blender 4.5 imports them as `BLENDED` and routes them through
Eevee's transparent face-sorting path. Self-overlapping faces then render as
missing or mislayered -- see
`design/20260828_150444-0500_imaginarium_rgb_false_blend_culling_fix.md`.

`eval/rendering/rerender_imaginarium_exact_camera_rgb.py` fixes that by promoting only
provably-opaque materials `BLENDED -> DITHERED`. Corrected frames can be stored
either in the canonical sibling `renders_updated/<scene>/<video>_frames` or in
a historical run-local overlay. Camera JSON, depth, masks, transforms and
meshes always continue to resolve from the original shared root.

This module makes the corrected frames the **default** and the original renders
the **fallback**, resolved per scene and per video. Canonical frames are used
only when their completion marker binds them to the current source camera and
frame inventory. Historical overlays still require a complete frame count and
must not be byte-identical passthrough copies, so a partial rerender can never
silently shrink a scene.

Controls:
  `FF_EXACT_RGB=0`             disable entirely, always use the original renders
  `FF_EXACT_RGB_ROOTS=a:b:c`   override the search roots (highest priority first)
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIRECTORY = "renders_updated"
CANONICAL_VALIDATION_FILENAME = "exact_camera_rgb_validation.json"
CANONICAL_VALIDATION_SCHEMA = "ff_exact_camera_rgb_validation_v1"

DEFAULT_SEARCH_ROOTS: tuple[Path, ...] = ()

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_resolution_log: dict[tuple[str, str, int], str] = {}
_resolution_kind: dict[tuple[str, str, int], str] = {}
_passthrough_cache: dict[Path, bool] = {}
_canonical_validation_cache: dict[tuple[Path, Path], bool] = {}


def enabled() -> bool:
    return os.environ.get("FF_EXACT_RGB", "1") not in {"0", "false", "False"}


def override_search_roots() -> list[Path]:
    override = os.environ.get("FF_EXACT_RGB_ROOTS")
    if override:
        return [Path(p) for p in override.split(":") if p]
    return []


def search_roots() -> list[Path]:
    """Optional caller-provided corrected-frame overlays."""

    return [*override_search_roots(), *DEFAULT_SEARCH_ROOTS]


def _frame_count(directory: Path) -> int:
    try:
        return sum(1 for p in directory.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
    except OSError:
        return 0


def _sample_images(directory: Path, limit: int = 2) -> list[Path]:
    try:
        names = sorted(
            p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
        )
    except OSError:
        return []
    return names[:limit]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_frames_dir(original: Path, scene_id: str) -> Path:
    """Return the canonical sibling path corresponding to an original RGB dir."""

    try:
        dataset_root = original.parents[2]
    except IndexError as error:
        raise ValueError(f"Cannot derive dataset root from frames path: {original}") from error
    return dataset_root / CANONICAL_DIRECTORY / scene_id / original.name


def canonical_is_complete(
    candidate: Path,
    original: Path,
    *,
    dataset_subdir: str,
    scene_id: str,
    video_id: int,
) -> bool:
    """Validate the cheap, immutable completion contract for a canonical scene."""

    marker = candidate.parent / CANONICAL_VALIDATION_FILENAME
    cache_key = (candidate, original)
    cached = _canonical_validation_cache.get(cache_key)
    if cached is not None:
        return cached

    valid = False
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
        actual_names = sorted(
            path.name
            for path in candidate.iterdir()
            if path.suffix.lower() in _IMAGE_SUFFIXES
        )
        frame_records = record.get("frames", [])
        recorded_sizes = {
            item["name"]: int(item["bytes"])
            for item in frame_records
            if isinstance(item, dict) and "name" in item and "bytes" in item
        }
        actual_sizes = {
            name: (candidate / name).stat().st_size for name in actual_names
        }
        camera_path = original.parent / f"{int(video_id)}.json"
        valid = bool(
            candidate.is_dir()
            and record.get("schema") == CANONICAL_VALIDATION_SCHEMA
            and record.get("status") == "complete"
            and record.get("dataset_subdir") == dataset_subdir
            and record.get("scene_id") == scene_id
            and int(record.get("video_id", -1)) == int(video_id)
            and record.get("frames_dir_name") == original.name
            and int(record.get("frame_count", -1)) == len(actual_names)
            and record.get("frame_names") == actual_names
            and recorded_sizes == actual_sizes
            and len(actual_names) == _frame_count(original) > 0
            and record.get("source_camera_sha256") == _sha256(camera_path)
            and record.get("probe_differs_from_original") is True
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        valid = False
    # Cache only successful immutable publications. A missing/incomplete scene
    # may be atomically published later in the same long-lived process.
    if valid:
        _canonical_validation_cache[cache_key] = True
    return valid


def _is_passthrough_copy(candidate: Path, original: Path) -> bool:
    """True when a candidate overlay is byte-identical to the original frames.

    Every percept+recon run writes a full ``dataset/`` tree, but only re-renders
    the scenes it actually processed; the rest are copied straight from the
    deprecated originals. Such a directory passes the frame-count check while
    carrying none of the exact-camera correction, so matching on count alone
    silently serves deprecated RGBs (observed on iTHOR FloorPlan312/328, whose
    genuine re-render lives in a later root). A real re-render is a separate
    Blender pass and differs from the original by 15-45 mean grey levels, so
    byte equality is a reliable passthrough signal.
    """

    cached = _passthrough_cache.get(candidate)
    if cached is not None:
        return cached
    samples = _sample_images(candidate)
    if not samples:
        _passthrough_cache[candidate] = False
        return False
    compared = 0
    for sample in samples:
        counterpart = original / sample.name
        if not counterpart.is_file():
            continue
        try:
            if sample.read_bytes() != counterpart.read_bytes():
                _passthrough_cache[candidate] = False
                return False
        except OSError:
            _passthrough_cache[candidate] = False
            return False
        compared += 1
    # No shared filenames means we cannot prove it is a copy; keep it.
    result = compared > 0
    _passthrough_cache[candidate] = result
    return result


def resolve_frames_dir(
    frames_dir: str | Path,
    *,
    dataset_subdir: str,
    scene_id: str,
    video_id: int,
) -> str:
    """Corrected frames for this scene/video if available, else the original.

    `frames_dir` is the original path; only the RGB directory is redirected, so
    cameras, depth, masks and transforms continue to resolve against the
    original root exactly as before.
    """

    original = Path(frames_dir)
    if not enabled():
        return str(original)
    baseline = _frame_count(original)
    name = original.name  # e.g. "0_frames"

    canonical = canonical_frames_dir(original, scene_id)
    if canonical_is_complete(
        canonical,
        original,
        dataset_subdir=dataset_subdir,
        scene_id=scene_id,
        video_id=video_id,
    ):
        _record_resolution(
            dataset_subdir,
            scene_id,
            video_id,
            canonical,
            "canonical",
            _frame_count(canonical),
        )
        return str(canonical)

    for root in search_roots():
        candidate = root / dataset_subdir / "renders" / scene_id / name
        if not candidate.is_dir():
            continue
        count = _frame_count(candidate)
        # A partial or mismatched rerender must never change the camera/RGB
        # inventory associated with this scene.
        if count == 0 or (baseline and count != baseline):
            continue
        # ... nor may a passthrough copy of the deprecated frames win the race.
        if _is_passthrough_copy(candidate, original):
            continue
        kind = "override" if root in override_search_roots() else "historical"
        _record_resolution(
            dataset_subdir,
            scene_id,
            video_id,
            candidate,
            kind,
            count,
        )
        return str(candidate)

    _record_resolution(
        dataset_subdir,
        scene_id,
        video_id,
        original,
        "original_fallback",
        baseline,
    )
    return str(original)


def _record_resolution(
    dataset_subdir: str,
    scene_id: str,
    video_id: int,
    path: Path,
    kind: str,
    frame_count: int,
) -> None:
    key = (dataset_subdir, scene_id, int(video_id))
    if key in _resolution_log:
        return
    _resolution_log[key] = str(path)
    _resolution_kind[key] = kind
    print(
        f"[exact-rgb] {dataset_subdir}/{scene_id} v{video_id}: "
        f"{frame_count} frames from {kind}: {path}",
        flush=True,
    )


def resolution_report() -> dict[str, str]:
    """What was redirected this process, for run provenance."""

    return {f"{d}/{s}/v{v}": path for (d, s, v), path in _resolution_log.items()}


def resolution_provenance_report() -> dict[str, dict[str, str]]:
    """Resolved path plus canonical/override/history/fallback source kind."""

    return {
        f"{d}/{s}/v{v}": {
            "path": path,
            "kind": _resolution_kind[(d, s, v)],
        }
        for (d, s, v), path in _resolution_log.items()
    }
