"""WeMM-Video Processor - Token arrangement logic for vLLM inference.

This module implements the core token arrangement logic for the WeMM-Video
model in vLLM, aligning with the training-time processor behavior. It handles the
construction of interleaved video/audio token sequences with timestamps.
"""

from math import ceil

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "get_wemm_mrope_input_positions",
    "split_into_groups",
    "build_video_audio_input_ids",
    "build_audio_input_ids",
    "get_special_token_ids",
    "compute_audio_tokens_per_segment",
]


def get_special_token_ids(tokenizer) -> dict[str, int]:
    """Get special token IDs used by WeMM-Video.

    Args:
        tokenizer: HuggingFace tokenizer instance.

    Returns:
        Dictionary mapping token names to their integer IDs.
    """
    return {
        "vision_start": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        "vision_end": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        "video_pad": tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        "audio_pad": tokenizer.convert_tokens_to_ids("<|audio_pad|>"),
        "audio_start": tokenizer.convert_tokens_to_ids("<|audio_start|>"),
        "audio_end": tokenizer.convert_tokens_to_ids("<|audio_end|>"),
    }


def compute_audio_tokens_per_segment(
    second_per_window: float = 0.2,
    second_stride: float | None = None,
    query_length: int = 1,
    whisper_time_steps: int = 1500,
) -> int:
    """Compute audio tokens per 30s segment from WhisperEncoder config.

    The WhisperEncoder unfolds 1500 time steps with a sliding window:
      kernel = round(1500 * second_per_window / 30)
      stride = round(1500 * second_stride / 30)
      L = (1500 - kernel) // stride + 1
      tokens_per_segment = L * query_length

    For second_per_window=0.2: L=150, tokens=150
    For second_per_window=0.5: L=60, tokens=60

    Args:
        second_per_window: Whisper encoder unfold window duration.
        second_stride: Whisper encoder unfold stride duration (defaults to
        second_per_window).
        query_length: Q-Former query token count.
        whisper_time_steps: Whisper encoder output time steps (1500 for 30s).

    Returns:
        Number of audio tokens produced per 30s segment.
    """
    if second_stride is None:
        second_stride = second_per_window
    kernel = round(whisper_time_steps * second_per_window / 30.0)
    stride = round(whisper_time_steps * second_stride / 30.0)
    if stride <= 0:
        stride = 1
    L = (whisper_time_steps - kernel) // stride + 1
    return L * query_length


def split_into_groups(
    count: int,
    group_count: int,
    second_per_grid_t: float | None = None,
) -> list[int]:
    """Evenly distribute `count` items into `group_count` groups.

    When `second_per_grid_t` is provided, uses boundary-based splitting that
    computes boundaries at regular time intervals. Otherwise, uses arithmetic
    distribution with evenly-spaced remainder allocation.

    Args:
        count: Total number of items to distribute.
        group_count: Number of groups to split into.
        second_per_grid_t: If provided, the time interval per grid step used
            for boundary-based splitting.

    Returns:
        A list of integers representing the number of items in each group.
    """
    count = int(count)
    group_count = int(group_count)

    if group_count <= 0:
        return []

    if second_per_grid_t is None:
        # Arithmetic distribution with evenly-spaced remainder allocation
        base = count // group_count
        remainder = count % group_count

        if remainder == 0:
            return [base] * group_count

        group_list = [base] * group_count
        step = float(group_count) / float(remainder)
        for i in range(1, remainder + 1):
            position = i * step
            index = int(np.floor(position)) - 1
            if index >= group_count:
                index = group_count - 1
            group_list[index] += 1

        return group_list
    else:
        # Boundary-based splitting using time intervals
        boundary_positions = torch.floor(
            torch.arange(1, group_count, dtype=torch.float32) * float(second_per_grid_t)
        ).long()
        boundary_positions = torch.clamp(boundary_positions, min=0, max=count)

        boundaries = [0, *boundary_positions.tolist(), count]
        group_list = [
            int(boundaries[idx + 1]) - int(boundaries[idx])
            for idx in range(group_count)
        ]

        return group_list


def _format_timestamp_tokens(
    tokenizer,
    seconds: float,
    timestamp_format: str = "{:.1f} seconds",
) -> list[int]:
    """Encode a timestamp string like '<0.5 seconds>' into token IDs.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        seconds: The time value in seconds.
        timestamp_format: Format string for the timestamp text.

    Returns:
        List of token IDs representing the formatted timestamp.
    """
    timestamp_text = "<" + timestamp_format.format(seconds) + ">"
    token_ids = tokenizer.encode(timestamp_text, add_special_tokens=False)
    return token_ids


def _split_audio_length_for_video(
    audio_length: int,
    grid_t: int,
    video_fps: float,
    temporal_patch_size: int = 2,
    second_per_window: float = 0.2,
) -> list[int]:
    """Split total audio tokens across temporal grid steps for video+audio.

    Computes the expected audio tokens per temporal step based on the video
    frame rate and temporal patch size, then distributes audio tokens
    accordingly using boundary-based splitting.

    Args:
        audio_length: Total number of audio tokens.
        grid_t: Number of temporal grid steps in the video.
        video_fps: Actual FPS of the sampled video frames.
        temporal_patch_size: Temporal merging factor.
        second_per_window: Seconds per audio token.

    Returns:
        List of audio token counts for each temporal step.
    """
    # Time duration per temporal grid step
    second_per_grid_t = temporal_patch_size / video_fps
    # Expected audio tokens per grid step
    audio_tokens_per_grid_t = second_per_grid_t / second_per_window

    # Use boundary-based splitting
    audio_groups = split_into_groups(
        count=audio_length,
        group_count=grid_t,
        second_per_grid_t=audio_tokens_per_grid_t,
    )

    return audio_groups


def build_video_audio_input_ids(
    tokenizer,
    video_grid_thw: torch.Tensor,
    audio_length: int,
    video_fps: float,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    second_per_window: float = 0.2,
    enable_timestamps: bool = True,
    timestamp_format: str = "{:.1f} seconds",
    start_seconds: float = 0.0,
    frame_indices: list[int] | None = None,
    origin_video_fps: float | None = None,
) -> list[int]:
    """Generate token IDs for a video_audio block matching the training format.

    Produces the interleaved token sequence:
        <timestamp_0><|vision_start|><|video_pad|>*N <|audio_pad|>*M <|vision_end|>
        <timestamp_1><|vision_start|><|video_pad|>*N <|audio_pad|>*M <|vision_end|>
        ...

    Args:
        tokenizer: HuggingFace tokenizer instance.
        video_grid_thw: Tensor of shape (3,) containing [T, H, W] of the video
            after Qwen3VL processing.
        audio_length: Total number of audio tokens from Whisper feature extraction.
        video_fps: Actual sampled FPS used for audio token distribution.
        temporal_patch_size: Temporal merging factor (default: 2).
        merge_size: Spatial merge size (default: 2).
        second_per_window: Seconds per audio token (default: 0.2).
        enable_timestamps: Whether to include timestamp tokens (default: True).
        timestamp_format: Format string for timestamps (default: "{:.1f} seconds").
        start_seconds: Base time offset for continuous timestamps (default: 0.0).
        frame_indices: Original video frame indices selected by the loader.
        origin_video_fps: FPS of the original video, used with frame_indices for
            exact timestamp generation.

    Returns:
        List of token IDs representing the full video+audio block.
    """
    special_tokens = get_special_token_ids(tokenizer)
    vision_start_id = special_tokens["vision_start"]
    vision_end_id = special_tokens["vision_end"]
    video_pad_id = special_tokens["video_pad"]
    audio_pad_id = special_tokens["audio_pad"]

    if video_fps is None or video_fps <= 0:
        video_fps = 1.0

    # Extract grid dimensions
    grid_t = int(video_grid_thw[0].item())
    grid_h = int(video_grid_thw[1].item())
    grid_w = int(video_grid_thw[2].item())

    # Number of video_pad tokens per temporal step
    block_repeat = (grid_h * grid_w) // (merge_size * merge_size)

    # Split audio tokens across temporal steps. This intentionally uses the
    # sampled FPS aligned to audio duration (same as lmms-eval), not origin FPS.
    audio_groups = _split_audio_length_for_video(
        audio_length=audio_length,
        grid_t=grid_t,
        video_fps=video_fps,
        temporal_patch_size=temporal_patch_size,
        second_per_window=second_per_window,
    )

    # Prefer exact timestamp generation from real frame indices, matching
    # Qwen3VLProcessor._calculate_timestamps / lmms-eval video_metadata path.
    exact_timestamps: list[float] | None = None
    if (
        frame_indices is not None
        and origin_video_fps is not None
        and origin_video_fps > 0
    ):
        indices = list(frame_indices)
        if indices:
            required = grid_t * temporal_patch_size
            if len(indices) < required:
                indices = indices + [indices[-1]] * (required - len(indices))
            elif len(indices) % temporal_patch_size != 0:
                pad = temporal_patch_size - len(indices) % temporal_patch_size
                indices = indices + [indices[-1]] * pad

            exact_timestamps = []
            for i in range(0, min(len(indices), required), temporal_patch_size):
                left = indices[i]
                right = indices[min(i + temporal_patch_size - 1, len(indices) - 1)]
                exact_timestamps.append(
                    start_seconds + (left + right) / (2.0 * origin_video_fps)
                )

    # Build the token sequence
    input_ids: list[int] = []

    for t_idx in range(grid_t):
        if exact_timestamps is not None and t_idx < len(exact_timestamps):
            frame_center_time = exact_timestamps[t_idx]
        else:
            # Fallback for callers without metadata: uniform sampled frames.
            frame_center_time = (
                start_seconds
                + (t_idx * temporal_patch_size + (temporal_patch_size - 1) / 2.0)
                / video_fps
            )

        # Add timestamp tokens
        if enable_timestamps:
            timestamp_tokens = _format_timestamp_tokens(
                tokenizer, frame_center_time, timestamp_format
            )
            input_ids.extend(timestamp_tokens)

        # Add vision_start
        input_ids.append(vision_start_id)

        # Add video_pad tokens
        input_ids.extend([video_pad_id] * block_repeat)

        # Add audio_pad tokens for this temporal step
        audio_count = audio_groups[t_idx] if t_idx < len(audio_groups) else 0
        input_ids.extend([audio_pad_id] * audio_count)

        # Add vision_end
        input_ids.append(vision_end_id)

    logger.debug(
        "Built video_audio input_ids: grid_t=%d, block_repeat=%d, "
        "audio_length=%d, total_tokens=%d, exact_timestamps=%s",
        grid_t,
        block_repeat,
        audio_length,
        len(input_ids),
        exact_timestamps is not None,
    )

    return input_ids


def build_audio_input_ids(
    tokenizer,
    audio_length: int,
    audio_sample_fps: float | None = None,
    second_per_window: float = 0.2,
    enable_timestamps: bool = True,
    timestamp_format: str = "{:.1f} seconds",
    start_seconds: float = 0.0,
    actual_duration_seconds: float | None = None,
    timestamp_interval_sec: float | None = None,
) -> list[int]:
    """Training audio layout, with explicit seconds per timestamp (default 1s).

    The legacy audio_sample_fps argument remains supported when explicitly set.
    Real duration determines effective tokens and timestamp count; padded audio
    embeddings are retained in the last group.
    """
    import math
    from vllm.multimodal.wemm_timestamps import audio_timestamp_groups, resolve_timestamp_interval

    if audio_sample_fps is not None:
        fps = resolve_timestamp_interval(audio_sample_fps)
        if timestamp_interval_sec is not None and not math.isclose(float(timestamp_interval_sec), 1 / fps):
            raise ValueError("Conflicting audio_sample_fps and timestamp_interval_sec")
        timestamp_interval_sec = 1 / fps
    interval = resolve_timestamp_interval(timestamp_interval_sec)
    sizes, times, _ = audio_timestamp_groups(
        audio_length, second_per_window, actual_duration_seconds, interval)
    special = get_special_token_ids(tokenizer)
    ids = [special["audio_start"]]
    for size, time in zip(sizes, times):
        if enable_timestamps:
            ids.extend(_format_timestamp_tokens(tokenizer, start_seconds + time, timestamp_format))
        ids.extend([special["audio_pad"]] * size)
    return ids + [special["audio_end"]]


def get_wemm_mrope_input_positions(
    input_tokens: list[int], mm_features: list, config
) -> tuple[torch.Tensor, int]:
    """Build visual grid positions while keeping bundled audio one-dimensional.

    Qwen3-VL counts the complete span up to vision_end as visual tokens.
    WeMM inserts audio_pad after video_pad in that span, so only consecutive
    video_pad tokens participate in the grid. Audio and subsequent text start
    at the preceding positions' maximum plus one. Keep the parent's image,
    empty-frame and lumped-frame EVS behavior without changing its iterator.
    """
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

    if not input_tokens:
        return torch.empty((3, 0), dtype=torch.long), 0

    vision_features = [f for f in mm_features if f.modality != "audio"]
    chunks = []
    consumed = 0
    for (
        offset,
        height,
        width,
        count,
    ) in Qwen3VLForConditionalGeneration._iter_mm_grid_hw(
        input_tokens,
        vision_features,
        video_token_id=config.video_token_id,
        vision_start_token_id=config.vision_start_token_id,
        vision_end_token_id=config.vision_end_token_id,
        spatial_merge_size=config.vision_config.spatial_merge_size,
    ):
        if count and input_tokens[offset] == config.video_token_id:
            video_end = offset
            while (
                video_end < len(input_tokens)
                and input_tokens[video_end] == config.video_token_id
            ):
                video_end += 1
            count = video_end - offset
        elif count:
            # Full bypass image spans may include wrapper or prefix tokens.
            offset = input_tokens.index(config.image_token_id, offset)
        if count == 0:
            continue

        text_length = offset - consumed
        start = int(chunks[-1].max()) + 1 if chunks else 0
        chunks.append(np.broadcast_to(np.arange(text_length), (3, text_length)) + start)
        grid = np.indices((1, height, width)).reshape(3, -1)
        expected = height * width
        if count > expected:
            full_frames, remainder = divmod(count, expected)
            for _ in range(full_frames):
                chunks.append(grid + text_length + start)
                start = int(chunks[-1].max()) + 1
                text_length = 0
            if remainder:
                chunks.append(grid[:, :remainder] + text_length + start)
        else:
            chunks.append(grid + text_length + start)
        consumed = offset + count

    if consumed < len(input_tokens):
        start = int(chunks[-1].max()) + 1 if chunks else 0
        length = len(input_tokens) - consumed
        chunks.append(np.broadcast_to(np.arange(length), (3, length)) + start)
    positions = np.concatenate(chunks, axis=1).reshape(3, -1)
    delta = int(positions.max()) + 1 - len(input_tokens)
    return torch.from_numpy(positions), delta
