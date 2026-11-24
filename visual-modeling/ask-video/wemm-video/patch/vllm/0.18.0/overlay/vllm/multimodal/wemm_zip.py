"""Shared ZIP frame decoder; preserves the release vLLM resize and time policy."""
from __future__ import annotations

import logging
import math
import os
import time
import zipfile
from io import BytesIO
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .wemm_zip_policy import zip_frame_indices

logger = logging.getLogger(__name__)
MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2
IMAGE_PATCH_SIZE = 16
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768
FRAME_FACTOR = 2
MODEL_SEQ_LEN = int(float(os.environ.get("MODEL_SEQ_LEN", 128000)))


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer >= 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer <= 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    max_pixels = (
        max_pixels if max_pixels is not None else (IMAGE_MAX_TOKEN_NUM * factor**2)
    )
    min_pixels = (
        min_pixels if min_pixels is not None else (IMAGE_MIN_TOKEN_NUM * factor**2)
    )
    assert max_pixels >= min_pixels, (
        "The max_pixels of image must be greater than or equal to min_pixels."
    )
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def _resize_frames_impl(
    frames_np: np.ndarray,
    sample_fps: float = 1.0,
    **kwargs,
) -> tuple[npt.NDArray, dict[str, Any]]:
    """Resize pre-loaded frames with token budget constraints and return metadata.

    This is the shared resize logic used by both load_frames_from_zip and
    load_frames_from_images.

    Args:
        frames_np: (T, H, W, 3) uint8 numpy array of raw frames
        sample_fps: Effective FPS for temporal alignment in prompts
        **kwargs: Token budget args (min_pixels, max_pixels, total_pixels,
                  resized_height, resized_width)

    Returns:
        Tuple of (frames_array, metadata_dict)
        where frames_array has shape (T, H, W, 3) as float32 [0, 1]
    """
    actual_nframes = frames_np.shape[0]
    video = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # (T, C, H, W)

    # Apply resizing with token budget constraints
    ele = kwargs.copy()
    image_factor = IMAGE_PATCH_SIZE * SPATIAL_MERGE_SIZE  # 32
    VIDEO_FRAME_MIN_PIXELS = VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
    VIDEO_FRAME_MAX_PIXELS = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor

    nframes_actual, _, height, width = video.shape
    min_pixels = ele.get("min_pixels", VIDEO_FRAME_MIN_PIXELS)
    total_pixels = ele.get(
        "total_pixels",
        MODEL_SEQ_LEN * image_factor * image_factor * 0.9,
    )
    max_pixels = max(
        min(
            VIDEO_FRAME_MAX_PIXELS,
            total_pixels / nframes_actual * FRAME_FACTOR,
        ),
        int(min_pixels * 1.05),
    )
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    if max_pixels_supposed > max_pixels:
        logger.warning(
            "The given max_pixels[%s] exceeds limit[%s].",
            max_pixels_supposed,
            max_pixels,
        )
    max_pixels = min(max_pixels_supposed, max_pixels)

    # Resize frames
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()  # Normalize to [0, 1]

    # Convert to (T, H, W, C) numpy
    frames_array = video.permute(0, 2, 3, 1).numpy()

    # Frame indices: all frames sequential
    frame_indices = list(range(actual_nframes))

    metadata = {
        "fps": sample_fps,
        "frames_indices": frame_indices,
        "total_num_frames": actual_nframes,
        "video_backend": "frames",
        "do_sample_frames": False,
    }

    logger.info(
        "Frames: %s frames @ %s fps, resized to %sx%s",
        actual_nframes,
        sample_fps,
        resized_height,
        resized_width,
    )

    return frames_array, metadata


def load_zip_video(
    zip_bytes: bytes,
    sample_fps: float = 2.0,
    num_frames: int | None = None,
    **kwargs,
) -> tuple[npt.NDArray, dict[str, Any]]:
    """Load video frames from a ZIP archive containing image files.

    ZIP should contain frames named in alphabetical order: 00000.jpg, 00001.jpg, etc.
    Frames are resized to uniform dimensions before stacking.

    Args:
        zip_bytes: Raw bytes of ZIP file containing frame images
        sample_fps: Effective frames per second for temporal alignment in prompts
        num_frames: Positive requested count; None uses all frames up to the
            effective deployment/request frame limit. Excess frames are sampled
            uniformly, rather than keeping a prefix. Negative counts are errors.
        **kwargs: Additional args for token budget (min_pixels, max_pixels, etc)

    Returns:
        Tuple of (frames_array, metadata_dict)
        where frames_array has shape (num_frames, height, width, 3) as float32 [0,1]
        and metadata_dict contains fps, total_num_frames, frames_indices, etc.

    Raises:
        RuntimeError: If ZIP is invalid or contains no image files
    """
    import time

    st = time.time()
    if isinstance(sample_fps, bool) or not math.isfinite(float(sample_fps)) or float(sample_fps) <= 0:
        raise ValueError("ZIP fps must be a positive finite number")
    sample_fps = float(sample_fps)

    # Extract frames from ZIP in sorted order
    frames_list = []
    frame_names = []
    frame_sizes = []  # Track dimensions for normalization

    try:
        with zipfile.ZipFile(BytesIO(zip_bytes), "r") as zf:
            # Get all file names, filter to image extensions
            all_files = zf.namelist()
            image_files = [
                f
                for f in sorted(all_files)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))
                and not f.startswith("__MACOSX/")  # Skip Mac OS metadata
            ]

            # Also extract audio files (WAV/MP3/FLAC) if bundled in ZIP
            audio_files = [
                f
                for f in all_files
                if f.lower().endswith((".wav", ".mp3", ".flac"))
                and not f.startswith("__MACOSX/")
            ]
            wav_bytes = None
            if audio_files:
                wav_bytes = zf.read(audio_files[0])
                logger.info(
                    "Found bundled audio in ZIP: %s (%s bytes)",
                    audio_files[0],
                    len(wav_bytes),
                )

            if not image_files:
                raise RuntimeError(
                    f"ZIP archive contains no image files. Files: {all_files[:10]}"
                )

            original_count = len(image_files)
            selected_indices = zip_frame_indices(original_count, num_frames, **kwargs)
            if len(selected_indices) < original_count:
                logger.info("ZIP frame selection: %s -> %s, uniform across full archive",
                            original_count, len(selected_indices))
            image_files = [image_files[index] for index in selected_indices]

            # Load each frame and track sizes
            for frame_file in image_files:
                try:
                    frame_bytes = zf.read(frame_file)
                    img = Image.open(BytesIO(frame_bytes)).convert("RGB")
                    frame_array = np.array(img, dtype=np.uint8)
                    frames_list.append(frame_array)
                    frame_names.append(frame_file)
                    frame_sizes.append(frame_array.shape[:2])  # (H, W)
                except Exception as e:
                    raise RuntimeError(f"Failed to decode selected ZIP frame {frame_file}: {e}") from e

    except Exception as e:
        raise RuntimeError(f"Failed to process ZIP: {e}") from e

    if not frames_list:
        raise RuntimeError("No valid frames could be extracted from ZIP")

    actual_nframes = len(frames_list)
    logger.info(
        "Loaded %s frames from ZIP in %ss", actual_nframes, f"{time.time() - st:.3f}"
    )

    # CRITICAL FIX: Normalize frame sizes before stacking
    # Multi-image ZIPs may have frames with different dimensions
    # Calculate a common target size from the first frame
    first_h, first_w = frame_sizes[0]

    if not all(h == first_h and w == first_w for h, w in frame_sizes):
        logger.warning(
            (
                "Frame size mismatch detected: %s. Resizing all frames to match first"
                " frame: %sx%s"
            ),
            set(frame_sizes),
            first_h,
            first_w,
        )
        # Resize mismatched frames to common size
        normalized_frames = []
        for i, frame_array in enumerate(frames_list):
            if frame_array.shape[:2] != (first_h, first_w):
                # Convert to PIL, resize, convert back
                img = Image.fromarray(frame_array)
                img_resized = img.resize((first_w, first_h), Image.BICUBIC)
                frame_array = np.array(img_resized, dtype=np.uint8)
            normalized_frames.append(frame_array)
        frames_list = normalized_frames

    # Now stack frames - all should have uniform shape
    frames_np = np.stack(frames_list, axis=0)  # (T, H, W, 3)
    frames_array, metadata = _resize_frames_impl(
        frames_np, sample_fps=sample_fps, **kwargs
    )
    metadata["video_backend"] = "zip"
    metadata["wemm_pre_resized"] = True
    metadata["width"] = int(frames_array.shape[2])
    metadata["height"] = int(frames_array.shape[1])
    metadata["frame_names"] = frame_names
    metadata["frames_indices"] = selected_indices
    metadata["total_num_frames"] = original_count
    metadata["duration"] = original_count / sample_fps
    metadata["zip_original_num_frames"] = original_count
    metadata["zip_decoded_num_frames"] = actual_nframes
    metadata["zip_frame_policy"] = "all" if actual_nframes == original_count else "uniform"
    # Include bundled audio bytes if found in ZIP
    if wav_bytes is not None:
        metadata["audio_bytes"] = wav_bytes

    return frames_array, metadata
