import contextlib

# SPDX-License-Identifier: Apache-2.0
import logging

#   import math
#   import torch
#   from torchvision import transforms
#   from torchvision.transforms import InterpolationMode
#
# The code below assumes numpy, math, and the VideoLoader/VIDEO_LOADER_REGISTRY
# are already defined in video.py.
# ============================================================
import math
import os
import tempfile
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

from .video import VIDEO_LOADER_REGISTRY, VideoLoader
from .wemm_video_policy import resolve_video_options, video_frame_count, video_resize_size, video_metadata_fps
from .wemm_zip_policy import positive_frame_count, zip_frame_indices
from .wemm_zip import load_zip_video

logger = logging.getLogger(__name__)

# ---- WeMM video processing constants ----

MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768

FPS = 2.0
FRAME_FACTOR = 2
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
MAX_NUM_WORKERS_FETCH_VIDEO = 8

IMAGE_PATCH_SIZE = 16

MODEL_SEQ_LEN = int(float(os.environ.get("MODEL_SEQ_LEN", 128000)))


# ============================================================
# Per-video pixel budget (longest_edge / shortest_edge)
# ============================================================
# These come from the model's `video_preprocessor_config.json`
# (`size.longest_edge` / `size.shortest_edge`). They cap, respectively,
# the total pixel volume across the whole video and the minimum
# per-frame pixel count after smart_resize.
#
# Resolution priority (high → low):
#   1. env var WEMM_VIDEO_LONGEST_EDGE / WEMM_VIDEO_SHORTEST_EDGE
#   2. value injected by wemm_omni at processor-load time (read from
#      `processor.video_processor.size`)
#   3. legacy formula `MODEL_SEQ_LEN * image_factor**2 * 0.9` for
#      total_pixels and `VIDEO_MIN_TOKEN_NUM * image_factor**2` for
#      min_pixels (kept as a tail-end fallback)
# ============================================================


def _wemm_int_env_local(name):
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        logger.warning("WeMMVideoLoader: bad value for %s=%r, ignoring", name, v)
        return None


_WEMM_VIDEO_LONGEST_EDGE_ENV = _wemm_int_env_local("WEMM_VIDEO_LONGEST_EDGE")
_WEMM_VIDEO_SHORTEST_EDGE_ENV = _wemm_int_env_local("WEMM_VIDEO_SHORTEST_EDGE")
# wemm_omni populates these once it has loaded the HF processor.


def _wemm_resolve_total_pixels(image_factor):
    """Resolve total per-video pixel budget. None → use legacy formula."""
    if _WEMM_VIDEO_LONGEST_EDGE_ENV is not None:
        return _WEMM_VIDEO_LONGEST_EDGE_ENV, "env"
    return MODEL_SEQ_LEN * image_factor * image_factor * 0.9, "legacy"


def _wemm_resolve_min_pixels(image_factor):
    """Resolve per-frame minimum pixel budget."""
    if _WEMM_VIDEO_SHORTEST_EDGE_ENV is not None:
        return _WEMM_VIDEO_SHORTEST_EDGE_ENV, "env"
    return VIDEO_MIN_TOKEN_NUM * image_factor * image_factor, "legacy"


# ---- WeMM video helper functions ----


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


def smart_nframes(
    ele: dict[str, Any],
    total_frames: int,
    video_fps: int | float,
) -> int:
    """Calculate the number of frames for video used for model inputs.

    Args:
        ele: A dict contains the configuration of video.
            Supports either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video,
                      only used when fps is provided.
                    - max_frames: the maximum number of frames of the video,
                      only used when fps is provided.
        total_frames: The original total number of frames of the video.
        video_fps: The original fps of the video.

    Raises:
        ValueError: nframes should be in interval [FRAME_FACTOR, total_frames].

    Returns:
        The number of frames for video used for model inputs.
    """
    return video_frame_count(total_frames, video_fps, ele)


def calculate_video_frame_range(
    ele: dict[str, Any],
    total_frames: int,
    video_fps: float,
) -> tuple[int, int, int]:
    """
    Calculate the start and end frame indices based on the given time range.

    Args:
        ele: A dictionary containing optional 'video_start' and 'video_end'
             keys (in seconds).
        total_frames: Total number of frames in the video.
        video_fps: Frames per second of the video.

    Returns:
        A tuple containing (start_frame, end_frame, frame_count).

    Raises:
        ValueError: If input parameters are invalid or the time range is
                    inconsistent.
    """
    # Validate essential parameters
    if video_fps <= 0:
        raise ValueError("video_fps must be a positive number")
    if total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")

    # Get start and end time in seconds
    video_start = ele.get("video_start")
    video_end = ele.get("video_end")

    if video_start is None and video_end is None:
        return 0, total_frames - 1, total_frames

    max_duration = total_frames / video_fps

    # Process start frame
    if video_start is not None:
        video_start_clamped = max(0.0, min(video_start, max_duration))
        start_frame = math.ceil(video_start_clamped * video_fps)
    else:
        start_frame = 0

    # Process end frame
    if video_end is not None:
        video_end_clamped = max(0.0, min(video_end, max_duration))
        end_frame = math.floor(video_end_clamped * video_fps)
        end_frame = min(end_frame, total_frames - 1)
    else:
        end_frame = total_frames - 1

    # Validate frame order
    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid time range: Start frame {start_frame} "
            f"(at {video_start_clamped if video_start is not None else 0}s) "
            f"exceeds end frame {end_frame} "
            f"(at {video_end_clamped if video_end is not None else max_duration}s). "
            f"Video duration: {max_duration:.2f}s "
            f"({total_frames} frames @ {video_fps}fps)"
        )

    logger.info(
        (
            "calculate video frame range: start_frame=%s, end_frame=%s, total_frames="
            "%s from video_start=%s, video_end=%s, video_fps=%s"
        ),
        start_frame,
        end_frame,
        total_frames,
        video_start,
        video_end,
        f"{video_fps:.3f}",
    )
    return start_frame, end_frame, end_frame - start_frame + 1


# ============================================================
# Video Backend Dispatch (torchcodec > decord > opencv)
# ============================================================
#
# Why three backends:
#   - HF training stack (data_qwen.py) uses **decord** to decode video.
#     OpenCV produces pixel values that diverge by ~1-3% per frame from
#     decord on the same indices, which is enough to flip greedy answers
#     on long videos under bf16 attention.
#   - **torchcodec** is the modern PyTorch-native decoder; on the test
#     videos it returns pixel values bit-identical to decord but is
#     packaged as a wheel and is the preferred path going forward.
#   - **opencv** is kept as the final fallback so the loader still works
#     on environments without either modern decoder.
#
# Backend selection:
#   Default chain (auto): torchcodec -> decord -> opencv
#   Override via env var WEMM_VIDEO_BACKEND=torchcodec|decord|opencv|auto
# ============================================================

_WEMM_TORCHCODEC_AVAILABLE = False
_WEMM_DECORD_AVAILABLE = False
try:
    import torchcodec.decoders as _wemm_torchcodec_probe  # noqa: F401

    _WEMM_TORCHCODEC_AVAILABLE = True
    logger.info("WeMMVideoLoader: torchcodec backend available")
except Exception as _wemm_e:  # noqa: BLE001
    logger.info("WeMMVideoLoader: torchcodec NOT available (%s)", _wemm_e)
try:
    import decord as _wemm_decord_probe  # noqa: F401

    _WEMM_DECORD_AVAILABLE = True
    logger.info("WeMMVideoLoader: decord backend available")
except Exception as _wemm_e:  # noqa: BLE001
    logger.info("WeMMVideoLoader: decord NOT available (%s)", _wemm_e)


def _wemm_resolve_video_backend_chain():
    """Return ordered backend list based on env override + availability.

    Priority (auto): torchcodec > decord > opencv.
    """
    pref = os.environ.get("WEMM_VIDEO_BACKEND", "auto").lower().strip()
    if pref == "auto":
        chain = []
        if _WEMM_TORCHCODEC_AVAILABLE:
            chain.append("torchcodec")
        if _WEMM_DECORD_AVAILABLE:
            chain.append("decord")
        chain.append("opencv")
        return chain
    if pref in ("torchcodec", "decord", "opencv"):
        # When explicit, still allow opencv as last-ditch fallback so a
        # missing wheel doesn't crash inference.
        return [pref] + ([] if pref == "opencv" else ["opencv"])
    logger.warning("WeMMVideoLoader: unknown WEMM_VIDEO_BACKEND=%r, using auto", pref)
    chain = []
    if _WEMM_TORCHCODEC_AVAILABLE:
        chain.append("torchcodec")
    if _WEMM_DECORD_AVAILABLE:
        chain.append("decord")
    chain.append("opencv")
    return chain


def _wemm_decode_video_bytes_torchcodec(data, ele):
    """Decode video bytes using torchcodec.

    Returns:
        (frames_thwc_uint8: np.ndarray, video_fps: float,
         total_frames: int, idx: list[int], t_decode: float)
    """
    from torchcodec.decoders import VideoDecoder

    st = time.time()
    decoder = VideoDecoder(data)
    md = decoder.metadata
    total_frames_orig = int(md.num_frames or 0)
    video_fps = video_metadata_fps(md, default=30.0)
    if total_frames_orig <= 0:
        raise RuntimeError(f"torchcodec: invalid total_frames={total_frames_orig}")

    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele, total_frames_orig, video_fps
    )
    nframes = smart_nframes(
        ele,
        total_frames=total_frames,
        video_fps=video_fps,
    )
    # Use HF-compatible truncated linspace (matches data_qwen.py decord
    # path), NOT the legacy round-based torch.linspace().round().long().
    idx = np.linspace(start_frame, end_frame, nframes, dtype=int).tolist()
    batch = decoder.get_frames_at(indices=idx)
    # batch.data : (N, C, H, W) uint8 torch tensor
    frames_thwc = batch.data.permute(0, 2, 3, 1).contiguous().numpy()
    return frames_thwc, video_fps, total_frames, idx, time.time() - st


def _wemm_decode_video_bytes_decord(data, ele):
    """Decode video bytes using decord (matches HF data_qwen.py)."""
    import decord

    with contextlib.suppress(Exception):
        decord.bridge.set_bridge("native")

    st = time.time()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_file.write(data)
        tmp_path = tmp_file.name
    try:
        vr = decord.VideoReader(tmp_path, num_threads=1)
        total_frames_orig = int(len(vr))
        video_fps = float(vr.get_avg_fps()) or 30.0
        if total_frames_orig <= 0:
            raise RuntimeError(f"decord: invalid total_frames={total_frames_orig}")

        start_frame, end_frame, total_frames = calculate_video_frame_range(
            ele, total_frames_orig, video_fps
        )
        nframes = smart_nframes(
            ele,
            total_frames=total_frames,
            video_fps=video_fps,
        )
        idx = np.linspace(
            start_frame,
            end_frame,
            nframes,
            dtype=int,
        ).tolist()
        # decord get_batch returns NDArray (N, H, W, C) uint8
        frames_thwc = vr.get_batch(idx).asnumpy()
        return frames_thwc, video_fps, total_frames, idx, time.time() - st
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _wemm_decode_video_bytes_opencv(data, ele):
    """Decode video bytes using OpenCV (legacy fallback)."""
    import cv2

    st = time.time()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
        tmp_file.write(data)
        tmp_path = tmp_file.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError(f"cv2: failed to open video: {tmp_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_orig = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps <= 0:
            video_fps = 30.0
        if total_frames_orig <= 0:
            total_frames_orig = 1

        start_frame, end_frame, total_frames = calculate_video_frame_range(
            ele, total_frames_orig, video_fps
        )
        nframes = smart_nframes(
            ele,
            total_frames=total_frames,
            video_fps=video_fps,
        )
        # Switched to HF-compatible truncated linspace so even when this
        # fallback is hit the indices match the decord/torchcodec paths.
        idx = np.linspace(
            start_frame,
            end_frame,
            nframes,
            dtype=int,
        ).tolist()

        frames_list = []
        for frame_idx in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(frame)
            else:
                if frames_list:
                    frames_list.append(frames_list[-1].copy())
                else:
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                    frames_list.append(np.zeros((h, w, 3), dtype=np.uint8))
        cap.release()
        frames_thwc = np.stack(frames_list)
        return (
            frames_thwc,
            float(video_fps),
            total_frames,
            idx,
            time.time() - st,
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


_WEMM_BACKEND_DECODERS = {
    "torchcodec": _wemm_decode_video_bytes_torchcodec,
    "decord": _wemm_decode_video_bytes_decord,
    "opencv": _wemm_decode_video_bytes_opencv,
}


# ---- WeMM Video Loader ----


@VIDEO_LOADER_REGISTRY.register("wemm")
class WeMMVideoLoader(VideoLoader):
    @staticmethod
    def load_frames_from_zip(*args, **kwargs):
        return load_frames_from_zip(*args, **kwargs)

    @staticmethod
    def load_frames_from_images(*args, **kwargs):
        return load_frames_from_images(*args, **kwargs)

    @staticmethod
    def _resize_frames(*args, **kwargs):
        return _resize_frames_impl(*args, **kwargs)

    """Video loader for WeMM model using torchcodec backend."""

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int | None = None,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        """Load video from bytes data.

        Automatically detects whether the input is a ZIP archive (containing
        pre-extracted frames) or a video file (MP4). ZIP files are identified
        by their magic number (PK\\x03\\x04).

        Args:
            data: Bytes data of the video (MP4) or ZIP (frames)
            num_frames: Explicit positive count; ZIP default keeps frames up to
                deployment max_frames, sampling the whole archive above it.
            **kwargs: Additional arguments including:
                - sample_fps: FPS for temporal alignment (ZIP mode, default 1.0)
                - video_start: Start time in seconds for video clipping
                - video_end: End time in seconds for video clipping
                - fps: FPS for frame sampling
                - nframes: Number of frames to sample
                - min_frames: Minimum number of frames
                - max_frames: Maximum number of frames

        Returns:
            Tuple of (frames_array, metadata_dict)
        """
        # The WeMM connector passes None explicitly instead of the generic
        # VideoMediaIO default of 32. Both count aliases remain supported.
        num_frames = positive_frame_count(kwargs.pop("nframes", num_frames))

        # Auto-detect ZIP format by magic number
        if len(data) >= 4 and data[:4] == b"PK\x03\x04":
            # Prefer explicit user-provided fps/sample_fps over env default
            sample_fps = kwargs.pop("sample_fps", None)
            if sample_fps is None:
                sample_fps = kwargs.pop("fps", None)
            if sample_fps is None:
                sample_fps = float(os.environ.get("WEMM_SAMPLE_FPS", "1.0"))
            # Call module-level function directly (not via cls.) to avoid
            # staticmethod resolution issues across different Python versions
            return load_frames_from_zip(
                data, sample_fps=sample_fps, num_frames=num_frames, **kwargs
            )
        if num_frames is not None:
            kwargs["nframes"] = num_frames
        return cls._load_from_video_bytes(
            data, num_frames=64 if num_frames is None else num_frames, **kwargs
        )

    @classmethod
    def _load_from_video_bytes(
        cls,
        data: bytes,
        num_frames: int = 64,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        """Load video from bytes via the configured backend chain.

        Backend priority is torchcodec > decord > opencv (override with
        env var WEMM_VIDEO_BACKEND). Each backend produces a uint8
        (T, H, W, C) numpy array; this method then runs the WeMM
        token-budget resize step (shared across backends).
        """
        # Element dict for sampling configuration
        ele = kwargs.copy()
        if "nframes" not in ele and "fps" not in ele:
            ele["nframes"] = num_frames

        chain = _wemm_resolve_video_backend_chain()
        last_err: Exception | None = None
        actual_backend: str | None = None
        frames_thwc = None
        video_fps = 0.0
        total_frames = 0
        idx: list = []
        t_decode = 0.0
        for backend in chain:
            try:
                if backend == "torchcodec" and not _WEMM_TORCHCODEC_AVAILABLE:
                    raise ImportError("torchcodec not available at import time")
                if backend == "decord" and not _WEMM_DECORD_AVAILABLE:
                    raise ImportError("decord not available at import time")
                fn = _WEMM_BACKEND_DECODERS[backend]
                frames_thwc, video_fps, total_frames, idx, t_decode = fn(
                    data,
                    ele,
                )
                actual_backend = backend
                break
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "WeMMVideoLoader: backend %s failed: %s; trying next", backend, e
                )
                last_err = e
                continue
        if actual_backend is None:
            raise RuntimeError(f"All video backends failed: {last_err}")

        logger.info(
            "video decode [%s]: total_frames=%s, video_fps=%s, nframes=%s, time=%ss",
            actual_backend,
            total_frames,
            f"{video_fps:.3f}",
            len(idx),
            f"{t_decode:.3f}",
        )

        # ---- Compute pixel budget (used either for resize OR forwarded
        #      to the HF processor when we skip resize) ----
        image_factor = IMAGE_PATCH_SIZE * SPATIAL_MERGE_SIZE
        # Per-frame ABSOLUTE upper bound; intentionally kept as the legacy
        # ceiling so a single tiny-nframes video can't blow up. The model
        # spec budget (longest_edge) is the dominant constraint via
        # `total_pixels / nframes * FRAME_FACTOR` below.
        VIDEO_FRAME_MAX_PIXELS = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor
        nframes_actual = int(frames_thwc.shape[0])
        height = int(frames_thwc.shape[1])
        width = int(frames_thwc.shape[2])

        # Resolve min_pixels: ele > env > model spec > legacy default.
        if "min_pixels" in ele:
            min_pixels = int(ele["min_pixels"])
            min_pixels_src = "client"
        else:
            min_pixels, min_pixels_src = _wemm_resolve_min_pixels(image_factor)

        # Resolve total_pixels: ele > env > model spec > legacy default.
        if "total_pixels" in ele:
            total_pixels = float(ele["total_pixels"])
            total_pixels_src = "client"
        else:
            total_pixels, total_pixels_src = _wemm_resolve_total_pixels(image_factor)

        options = resolve_video_options(
            {"total_pixels": total_pixels, "min_pixels": min_pixels}, ele
        )
        resize_size = video_resize_size(nframes_actual, options)
        max_pixels = resize_size["longest_edge"] // nframes_actual
        logger.info(
            "WeMMVideoLoader budget: total_pixels=%.0f (%s), "
            "min_pixels=%d (%s), max_pixels=%d (per-frame)",
            total_pixels,
            total_pixels_src,
            min_pixels,
            min_pixels_src,
            max_pixels,
        )

        # ---- Skip pre-resize by default (HF-equivalent path) ----
        # WHY: vLLM's old behavior was to torchvision-bicubic-resize uint8
        # frames here, then pass float32 frames to the HF processor with
        # `do_resize=False`. But vLLM's smart_resize and HF processor's
        # smart_resize end up with different target (h, w) (mainly due to
        # how max_pixels propagates), and torchvision bicubic differs from
        # HF processor's bicubic. The combined effect was that
        # `pixel_values_videos` and even `video_grid_thw` differed from
        # what the HF training stack would produce on the SAME video bytes,
        # which flipped Q2's greedy answer from `fox` (correct) to `rabbit`.
        # Set WEMM_SKIP_PRE_RESIZE=0 to fall back to the legacy buggy path.
        skip_pre_resize = os.environ.get("WEMM_SKIP_PRE_RESIZE", "1").strip() != "0"

        if skip_pre_resize:
            # Hand uint8 frames straight to the HF processor; rely on its
            # own smart_resize to honour min_pixels/max_pixels. We pass our
            # computed budget through metadata so wemm_omni._call_hf_processor
            # can forward it as kwargs (and so do_resize stays at its
            # default of True, matching HF training).
            frames_array = np.ascontiguousarray(frames_thwc)  # uint8 (T,H,W,C)

            metadata = {
                "fps": float(video_fps),
                "frames_indices": idx,
                "total_num_frames": total_frames,
                "video_backend": actual_backend,
                "do_sample_frames": False,
                # Hint to wemm_omni._call_hf_processor:
                "wemm_pre_resized": False,
                "wemm_max_pixels": int(max_pixels),
                "wemm_min_pixels": int(min_pixels),
                "wemm_video_size": resize_size,
            }
            return frames_array, metadata

        # ---- Legacy path: pre-resize via torchvision bicubic ----
        # frames_thwc is uint8 (T, H, W, C); convert to (T, C, H, W) for
        # torchvision.transforms.functional.resize.
        video = torch.from_numpy(np.ascontiguousarray(frames_thwc)).permute(
            0,
            3,
            1,
            2,
        )

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
        ).float()

        frames_array = video.permute(0, 2, 3, 1).numpy()

        metadata = {
            "fps": float(video_fps),
            "frames_indices": idx,
            "total_num_frames": total_frames,
            "video_backend": actual_backend,
            "do_sample_frames": False,
            "wemm_pre_resized": True,
            "wemm_max_pixels": int(max_pixels),
            "wemm_min_pixels": int(min_pixels),
        }
        return frames_array, metadata


# ============================================================
# PHASE 1: Frame Loading from ZIP Archives / Image Lists
# ============================================================


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


def load_frames_from_zip(zip_bytes, sample_fps=2.0, num_frames=None, **kwargs):
    """Use the same ZIP decoding/resize policy as HF and SGLang."""
    return load_zip_video(zip_bytes, sample_fps=sample_fps, num_frames=num_frames, **kwargs)


def load_frames_from_images(
    image_arrays: list[np.ndarray],
    sample_fps: float = 1.0,
    **kwargs,
) -> tuple[npt.NDArray, dict[str, Any]]:
    """Load video from a list of pre-loaded image arrays.

    Args:
        image_arrays: List of (H, W, 3) uint8 numpy arrays (RGB)
        sample_fps: Effective FPS for temporal alignment in prompts
        **kwargs: Additional args for token budget (min_pixels, max_pixels, etc)

    Returns:
        Tuple of (frames_array, metadata_dict)
    """
    if not image_arrays:
        raise RuntimeError("Empty image list provided")

    frames_np = np.stack(image_arrays, axis=0)  # (T, H, W, 3)
    frames_array, metadata = _resize_frames_impl(
        frames_np, sample_fps=sample_fps, **kwargs
    )
    metadata["video_backend"] = "image_list"

    return frames_array, metadata


# Add methods to WeMMVideoLoader class
