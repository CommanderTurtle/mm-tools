"""Shared WeMM video policy v2; mirrored unchanged into the vLLM overlay."""

import math
import os

POLICY_VERSION = "wemm_video_v2"
VIDEO_OPTION_KEYS = (
    "fps", "num_frames", "min_frames", "max_frames", "total_pixels",
    "min_pixels", "max_pixels",
)
VIDEO_CAP_KEYS = ("max_allowed_frames", "max_allowed_total_pixels")
VIDEO_ENV = {
    "fps": "WEMM_VIDEO_FPS",
    "min_frames": "WEMM_VIDEO_MIN_FRAMES",
    "max_frames": "WEMM_VIDEO_MAX_FRAMES",
    "total_pixels": "WEMM_VIDEO_LONGEST_EDGE",
    "min_pixels": "WEMM_VIDEO_SHORTEST_EDGE",
}


def video_environment_defaults():
    """Legacy environment variables supply defaults, never request locks."""
    return {key: os.environ[name] for key, name in VIDEO_ENV.items()
            if os.environ.get(name, "").strip()}


def _number(name, value, integer=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0 or (integer and not number.is_integer()):
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'finite number'}")
    return int(number) if integer else number


def video_metadata_fps(metadata, default=None):
    """Prefer header precision only when it agrees with the effective FPS.

    Older TorchCodec serializes header FPS with six decimal places, while
    newer versions expose the full rational float. Keep that precision for
    stable one-decimal timestamp tokens at half-decimal boundaries.
    A discrepancy above 1% is not rounding noise: use the duration-derived
    average instead, including streams whose header advertises twice the FPS.
    """
    rates = {}
    for name in ('average_fps_from_header', 'average_fps'):
        value = getattr(metadata, name, None)
        if value is not None:
            try:
                rates[name] = _number('video FPS', value)
            except ValueError:
                pass
    header = rates.get('average_fps_from_header')
    average = rates.get('average_fps')
    fps = header if header is not None else average
    if header is not None and average is not None:
        if not math.isclose(header, average, rel_tol=0.01, abs_tol=1e-6):
            fps = average
    if fps is not None:
        return round(fps, 6) or fps
    return _number('video FPS', default)


def resolve_video_options(defaults=None, overrides=None):
    """Request values override defaults; explicit service caps stay immutable."""
    defaults, overrides = dict(defaults or {}), dict(overrides or {})
    for values in (defaults, overrides):
        if values.get("nframes") is not None:
            if values.get("num_frames") is not None and values["num_frames"] != values["nframes"]:
                raise ValueError("Conflicting num_frames and nframes")
            values["num_frames"] = values.pop("nframes")
    result = {key: value for key, value in defaults.items() if value is not None}
    result.update({key: value for key, value in overrides.items() if value is not None})
    for key in VIDEO_CAP_KEYS:
        if key in defaults:
            result[key] = defaults[key]
    for key, value in {
        "fps": 2.0, "min_frames": 4, "max_frames": 128,
        "total_pixels": 117964800, "min_pixels": 4096,
    }.items():
        result.setdefault(key, value)
    for key in VIDEO_OPTION_KEYS + VIDEO_CAP_KEYS:
        if result.get(key) is not None:
            result[key] = _number(key, result[key], integer=key != "fps")
    if result["min_frames"] < 2 or result["max_frames"] < result["min_frames"]:
        raise ValueError("Video frame limits require 2 <= min_frames <= max_frames")
    if result.get("max_pixels", result["min_pixels"]) < result["min_pixels"]:
        raise ValueError("max_pixels must be at least min_pixels")
    frame_cap = result.get("max_allowed_frames")
    if frame_cap is not None and max(result["max_frames"], result.get("num_frames", 0)) > frame_cap:
        raise ValueError("Requested video frames exceed max_allowed_frames")
    pixel_cap = result.get("max_allowed_total_pixels")
    if pixel_cap is not None and result["total_pixels"] > pixel_cap:
        raise ValueError("Requested total_pixels exceeds max_allowed_total_pixels")
    return result


def video_frame_count(total_frames, original_fps, options):
    """Preserve temporal pairs and the existing num_frames-before-fps recipe."""
    options = resolve_video_options(options)
    original_fps = _number("original_fps", original_fps)
    if total_frames < 2:
        raise ValueError("Video needs at least two frames")
    lower = math.ceil(options["min_frames"] / 2) * 2
    upper = min(options["max_frames"], total_frames) // 2 * 2
    desired = options.get("num_frames", total_frames / original_fps * options["fps"])
    return int(min(max(desired, lower), upper, total_frames)) // 2 * 2


def video_resize_size(num_frames, options):
    """Keep the legacy factor 2; longest/shortest_edge are video pixel volumes."""
    options = resolve_video_options(options)
    if num_frames < 2 or num_frames % 2:
        raise ValueError("Decoded video frames must form temporal pairs")
    maximum = int(max(
        min(768 * 32**2, options["total_pixels"] / num_frames * 2),
        int(options["min_pixels"] * 1.05),
    ))
    if options.get("max_pixels") is not None:
        maximum = min(maximum, options["max_pixels"])
    return {"longest_edge": num_frames * maximum,
            "shortest_edge": num_frames * options["min_pixels"]}
