"""Bounded segment-AR history; current-segment attention is unchanged.

The budget includes cached text. Eviction rounds up to a complete video frame,
or removes an entire record (including text), so sparse routing stays aligned.
"""

from dataclasses import dataclass
from hymm.utils.inference_metrics import profiled


@dataclass
class DenseVideo:
    """Frame geometry only: dense SWA never constructs sparse router prototypes."""

    start: int
    image_tokens: int
    end: int
    time_start: int
    height: int
    width: int


def append_dense_record(cache, valid_len, image_tokens, text_tokens, tokens_per_frame):
    if tokens_per_frame <= 0 or image_tokens % tokens_per_frame:
        raise ValueError("Dense SWA requires complete video frames and tokens_per_frame > 0")
    records = cache.setdefault("swa_video", [])
    if sum(r.end - r.start for r in records) != valid_len:
        raise ValueError("Dense SWA metadata does not cover the existing KV cache")
    start_frame = cache.get("swa_next_frame", 0)
    records.append(DenseVideo(
        valid_len, image_tokens, valid_len + image_tokens + text_tokens,
        start_frame, 1, tokens_per_frame,
    ))
    cache["swa_next_frame"] = start_frame + image_tokens // tokens_per_frame


def _validated_frame_tokens(record):
    if record.height <= 0 or record.width <= 0:
        raise ValueError("SWA video frame height and width must be positive")
    frame_tokens = record.height * record.width
    if record.image_tokens <= 0 or record.image_tokens % frame_tokens:
        raise ValueError("SWA video records must contain a positive whole number of frames")
    return frame_tokens


def eviction_cut(records, total, capacity):
    """Smallest frame/record-aligned prefix whose removal meets the budget."""
    excess = total - capacity
    if excess <= 0:
        return 0
    for record in records:
        if record.end < excess:
            continue
        frame_tokens = _validated_frame_tokens(record)
        frames = max(0, (excess - record.start + frame_tokens - 1) // frame_tokens)
        cut = record.start + frames * frame_tokens
        # Do not leave orphaned text without any video in a sparse record.
        return record.end if cut >= record.start + record.image_tokens else cut
    raise ValueError("SWA frame records do not cover the cache")


def trim_records(records, cut):
    from dataclasses import replace

    kept = []
    for record in records:
        if record.end <= cut:
            continue
        removed = max(0, cut - record.start)
        frame_tokens = _validated_frame_tokens(record)
        if removed % frame_tokens or removed >= record.image_tokens:
            raise ValueError("SWA eviction must retain complete video frames")
        frames = removed // frame_tokens
        extra = {}
        if getattr(record, "prototypes", None) is not None:
            nframes = record.image_tokens // frame_tokens
            blocks_per_frame = record.prototypes.shape[2] // nframes
            # clone releases the evicted prototype backing allocation.
            extra["prototypes"] = record.prototypes[:, :, frames * blocks_per_frame:].clone()
        kept.append(replace(
            record,
            start=max(0, record.start - cut),
            end=record.end - cut,
            image_tokens=record.image_tokens - removed,
            time_start=record.time_start + frames,
            **extra,
        ))
    return kept


@profiled("kv_rebuild")
def commit_window(cache, mask, current_k, current_v, current_mask, valid_len):
    """Commit after attention, never allocate an over-budget persistent KV."""
    import torch

    if cache.get("storage_format") == "qvg":
        from .qvg_kv_cache import commit_qvg_window

        return commit_qvg_window(cache, mask, current_k, current_v, current_mask, valid_len)
    capacity = cache["swa_capacity"]
    total = valid_len + current_k.shape[1]
    record_key = "swa_video" if "swa_video" in cache else "sparse_video"
    records = cache[record_key]
    cut = eviction_cut(records, total, capacity)
    history_start = min(cut, valid_len)
    remaining_history = valid_len - history_start
    current_start = max(0, cut - valid_len)
    int8_storage = cache["k"].dtype == torch.int8
    keys = ("k", "v", "k_scale", "v_scale") if int8_storage else ("k", "v")
    for key in keys:
        if remaining_history and history_start:
            cache[key][:, :remaining_history].copy_(
                cache[key][:, history_start:valid_len].clone()
            )
    if remaining_history and history_start:
        mask[:, :remaining_history].copy_(mask[:, history_start:valid_len].clone())
    if current_start < current_k.shape[1]:
        if int8_storage:
            from .int8_kv_cache import quantize_kv_pair_into

            quantize_kv_pair_into(
                cache, current_k[:, current_start:].contiguous(),
                current_v[:, current_start:].contiguous(), remaining_history,
            )
        else:
            end = total - cut
            cache["k"][:, remaining_history:end].copy_(current_k[:, current_start:])
            cache["v"][:, remaining_history:end].copy_(current_v[:, current_start:])
    new_len = total - cut
    mask[:, remaining_history:new_len].copy_(current_mask[:, current_start:])
    mask[:, new_len:] = False
    mask._action_tail_ranges = []
    cache[record_key] = trim_records(records, cut)
    cache["swa_valid_len"] = new_len
    cache["swa_evicted"] = cut
    if int8_storage and current_start < current_k.shape[1]:
        stats = cache.setdefault("kv_codec_stats", {})
        stats["encode_calls"] = stats.get("encode_calls", 0) + 1
