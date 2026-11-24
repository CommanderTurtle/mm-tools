"""WeMM audio timestamp policy, matching the training renderer.

Seconds per timestamp are independent of encoder windows.
"""
import math
import os


def resolve_timestamp_interval(value=None, default=1.0):
    """Seconds per timestamp: explicit value, environment, processor default."""
    if value is None:
        value = os.environ.get('TIMESTAMP_INTERVAL_SEC', default)
    if isinstance(value, bool):
        raise ValueError('timestamp_interval_sec must be a finite positive number')
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('timestamp_interval_sec must be a finite positive number') from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError('timestamp_interval_sec must be a finite positive number')
    return value


def timestamp_interval_option(options, default=None):
    nested = options.get('audio_url', {})
    sources = [options, nested] if isinstance(nested, dict) else [options]
    if any('TIMESTAMP_INTERVAL_SEC' in source for source in sources):
        raise ValueError('Use lowercase timestamp_interval_sec in audio options')
    values = [source['timestamp_interval_sec'] for source in sources
              if source.get('timestamp_interval_sec') is not None]
    if values and any(resolve_timestamp_interval(value) != resolve_timestamp_interval(values[0])
                      for value in values[1:]):
        raise ValueError('Conflicting timestamp_interval_sec values')
    return values[0] if values else default


def audio_timestamp_groups(count, window, duration=None, timestamp_interval_sec=None):
    """Training renderer: ceil real duration, balanced effective pads, padded tail."""
    interval = resolve_timestamp_interval(timestamp_interval_sec)
    count = int(count)
    if count <= 0:
        return [], [], 0
    padded_duration = count * float(window)
    if duration is None:
        duration, effective = padded_duration, count
    else:
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Audio duration must be finite and positive')
        duration = min(duration, padded_duration)
        effective = max(1, min(count, math.ceil(duration / float(window) - 1e-9)))
    sample_fps = 1 / interval
    groups = max(1, math.ceil(duration * sample_fps - 1e-9))
    base, remainder = divmod(effective, groups)
    sizes = [base] * groups
    step = float(groups) / float(remainder) if remainder else 0
    for i in range(1, remainder + 1):
        sizes[min(groups - 1, math.floor(i * step) - 1)] += 1
    sizes[-1] += count - effective
    return sizes, [index / sample_fps for index in range(groups)], effective
