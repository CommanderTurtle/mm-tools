"""Camera helpers used by the pruned SimpleVO runtime."""


def focal_length_from_mm(width, height, mm=24):
    """Convert a full-frame-equivalent focal length to image pixels."""

    diagonal_full_frame = (24**2 + 36**2) ** 0.5
    diagonal_image = (width**2 + height**2) ** 0.5
    return diagonal_image / diagonal_full_frame * mm
