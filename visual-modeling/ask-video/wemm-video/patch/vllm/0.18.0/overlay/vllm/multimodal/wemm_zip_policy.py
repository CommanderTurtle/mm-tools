"""Frame-count policy for pre-sampled WeMM ZIP videos."""
import math


def positive_frame_count(value, name="num_frames"):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not math.isfinite(number) or number <= 0 or not number.is_integer():
        raise ValueError(f"{name} must be a positive integer; omit it for the bounded ZIP default")
    return int(number)


def zip_frame_indices(total, num_frames=None, **options):
    """Keep all frames up to the limit; sample the whole timeline above it."""
    total = positive_frame_count(total, "ZIP image count")
    limits = [positive_frame_count(options.get("max_frames", 128), "max_frames")]
    for key in ("max_allowed_frames", "wemm_zip_max_frames"):
        if options.get(key) is not None:
            limits.append(positive_frame_count(options[key], key))
    limit = min(limits)
    count = positive_frame_count(num_frames)
    if count is None:
        count = min(total, limit)
    elif count > limit:
        raise ValueError(f"Requested ZIP num_frames={count} exceeds frame limit {limit}")
    elif count > total:
        raise ValueError(f"Requested ZIP num_frames={count} exceeds available frames {total}")
    if count == 1:
        return [0]
    # Integer form of truncated linspace; include both ends of the archive.
    return [index * (total - 1) // (count - 1) for index in range(count)]
