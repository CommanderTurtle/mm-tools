"""Dependency-light helpers for separate camera and text CFG."""


def camera_guidance_is_active(camera_scale, has_camera_condition):
    """Return whether a separate camera guidance branch is needed."""
    return bool(has_camera_condition) and float(camera_scale) != 1.0


def combine_camera_text_cfg(
    base,
    camera,
    text,
    camera_scale,
    text_scale,
):
    """Combine camera-only and text-on-camera prediction differences."""
    return (
        base
        + float(camera_scale) * (camera - base)
        + float(text_scale) * (text - camera)
    )
