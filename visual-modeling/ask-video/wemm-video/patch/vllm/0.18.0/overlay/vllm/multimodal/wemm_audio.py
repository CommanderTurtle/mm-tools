# SPDX-License-Identifier: Apache-2.0
import copy

#   import math
#   from io import BytesIO
#   from pathlib import Path
#   from typing import Any, Optional, Union
#   import torch
#   from transformers import WhisperFeatureExtractor
#
# The code below assumes numpy and librosa are already imported in audio.py.
# ============================================================
import math
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from transformers import WhisperFeatureExtractor

from vllm.logger import init_logger

try:
    import librosa
except ImportError:
    from vllm.utils.import_utils import PlaceholderModule

    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

try:
    from torchcodec.decoders import AudioDecoder as TorchcodecAudioDecoder
except (ImportError, RuntimeError):
    TorchcodecAudioDecoder = None

logger = init_logger(__name__)

# ---- WeMM audio constants and helpers ----

SAMPLE_RATE = 16000

# Whisper-style processors: 7b uses feature_size=128, 2b uses 80
_audio_processor_7b: Any | None = None
_audio_processor_2b: Any | None = None


def _get_whisper_processor(audio_feature_size: int = 128) -> Any:
    """Lazy-init WhisperFeatureExtractor (feature_size 128 or 80)."""
    global _audio_processor_7b, _audio_processor_2b
    if audio_feature_size == 128:
        if _audio_processor_7b is None:
            _audio_processor_7b = WhisperFeatureExtractor(
                feature_size=128,
                sampling_rate=16000,
                hop_length=160,
                chunk_length=30,
            )
        return copy.deepcopy(_audio_processor_7b)
    else:
        if _audio_processor_2b is None:
            _audio_processor_2b = WhisperFeatureExtractor(
                feature_size=80,
                sampling_rate=16000,
                hop_length=160,
                chunk_length=30,
            )
        return copy.deepcopy(_audio_processor_2b)


def _video_audio_waveform(samples):
    """Validate the decoded waveform without clipping or peak normalization."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("Video audio must be finite, nonempty mono float32 samples")
    return samples


def _decode_video_audio_ffmpeg_float32(audio_file):
    """Float-only fallback: never serialize video audio as PCM16 WAV."""
    import subprocess
    import tempfile
    from contextlib import ExitStack

    # MP4 with a trailing moov atom needs seekable input. Store the original
    # encoded bytes unchanged; this is not waveform serialization.
    with ExitStack() as stack:
        if isinstance(audio_file, bytes):
            media = stack.enter_context(tempfile.NamedTemporaryFile(suffix=".media"))
            media.write(audio_file)
            media.flush()
            source = media.name
        else:
            source = str(audio_file)
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", source, "-vn",
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-acodec", "pcm_f32le",
             "-f", "f32le", "pipe:1"],
            capture_output=True, timeout=60, check=True,
        )
    return _video_audio_waveform(np.frombuffer(result.stdout, dtype="<f4").copy())


def _decode_audio_to_16k_mono(
    audio_file: bytes | Path | BytesIO,
    *,
    preserve_float32: bool = False,
) -> list[npt.NDArray]:
    """Decode audio to 16kHz mono using torchcodec, librosa, or ffmpeg fallback."""
    # Ensure we have raw bytes for potential reuse across fallbacks
    raw_bytes: bytes | None = None
    if isinstance(audio_file, bytes):
        raw_bytes = audio_file
    elif isinstance(audio_file, BytesIO):
        audio_file.seek(0)
        raw_bytes = audio_file.read()

    if TorchcodecAudioDecoder is not None:
        if isinstance(audio_file, Path):
            source = str(audio_file)
        else:
            source = BytesIO(raw_bytes) if raw_bytes else BytesIO(audio_file)
        try:
            decoder = TorchcodecAudioDecoder(
                source,
                sample_rate=SAMPLE_RATE,
                num_channels=1,
            )
            audio = decoder.get_all_samples()
            samples = audio.data.numpy().squeeze(0)
            return [_video_audio_waveform(samples) if preserve_float32 else samples]
        except Exception:
            pass  # Fall through to librosa/ffmpeg

    if preserve_float32:
        # Match TorchCodec's mono/resample semantics with FFmpeg float output.
        # Standalone audio retains the existing fallback/PCM16 contract below.
        return [_decode_video_audio_ffmpeg_float32(
            raw_bytes if raw_bytes is not None else audio_file)]

    # Try soundfile (handles WAV/FLAC from BytesIO reliably)
    if raw_bytes is not None:
        try:
            import soundfile as sf

            audio, sr = sf.read(BytesIO(raw_bytes), dtype="float32")
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            return [audio.astype(np.float32)]
        except Exception:
            pass

    # Try librosa (reset BytesIO position)
    try:
        if raw_bytes is not None:
            audio_file_io = BytesIO(raw_bytes)
        elif isinstance(audio_file, Path):
            audio_file_io = audio_file
        else:
            audio_file_io = audio_file
        audio, _ = librosa.load(audio_file_io, sr=SAMPLE_RATE, mono=True)
        return [audio.astype(np.float32)]
    except Exception:
        pass

    # Fallback: ffmpeg
    import os
    import subprocess
    import tempfile

    if raw_bytes is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(raw_bytes)
            input_path = tmp.name
    elif isinstance(audio_file, Path):
        input_path = str(audio_file)
    else:
        raise ValueError(
            "Cannot decode audio: all backends failed and no raw bytes available"
        )

    output_path = input_path + ".pcm.wav"
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, timeout=60, check=True)

        try:
            import soundfile

            audio, _ = soundfile.read(output_path, dtype="float32")
        except ImportError:
            audio = (
                np.fromfile(output_path, dtype=np.int16)[22:].astype(np.float32)
                / 32768.0
            )

        return [audio.astype(np.float32)]
    finally:
        if raw_bytes is not None and os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


def process_audio(
    audio_file: bytes | Path | BytesIO,
    audio_feature_size: int | None = 128,
    *,
    preserve_float32: bool = False,
) -> tuple[list[Any] | None, list[int] | None, list[float] | None]:
    """
    Process audio for WeMM model inference.

    Fully aligned with reference process_audio:
    - Decode to 16k single channel (AudioDecoder or librosa)
    - Pad to at least 1s if shorter
    - Split into 30s segments, run WhisperFeatureExtractor per segment
    - Return audio_inputs (list of tensors shape [segment_num, feature_size, 3000]),
      audio_lengths (list of segment_num*60), audio_durations (list of duration in s).
    """
    audio_kwargs = {
        "sampling_rate": SAMPLE_RATE,
        "padding": "max_length",
        "return_attention_mask": False,
    }
    processor = _get_whisper_processor(audio_feature_size)

    audio_data = (_decode_audio_to_16k_mono(audio_file, preserve_float32=True)
                  if preserve_float32 else _decode_audio_to_16k_mono(audio_file))

    audio_inputs: list[Any] = []
    audio_lengths: list[int] = []
    audio_durations: list[float] = []

    for idx in range(len(audio_data)):
        duration = audio_data[idx].shape[0] / audio_kwargs["sampling_rate"]
        audio_durations.append(duration)

        if audio_data[idx].shape[0] < audio_kwargs["sampling_rate"]:
            padding = audio_kwargs["sampling_rate"] - audio_data[idx].shape[0]
            audio_data[idx] = np.pad(
                audio_data[idx],
                (0, padding),
                mode="constant",
                constant_values=0,
            )

        segment_len = 30 * audio_kwargs["sampling_rate"]
        audio_lst = [
            audio_data[idx][k : k + segment_len]
            for k in range(0, len(audio_data[idx]), segment_len)
        ]

        spectrogram_lst = []
        for a in audio_lst:
            out = processor(
                a,
                sampling_rate=audio_kwargs["sampling_rate"],
                return_tensors="pt",
            )
            feat = out["input_features"].squeeze(0)
            spectrogram_lst.append(feat)

        audio_inputs.append(torch.stack(spectrogram_lst, dim=0))
        audio_lengths.append(
            math.ceil(len(audio_data[idx]) / (30 * audio_kwargs["sampling_rate"])) * 60
        )

    return audio_inputs, audio_lengths, audio_durations


# ============================================================
# PHASE 2: WAV File Loading Helper
# ============================================================


def load_wav_bytes(
    wav_bytes: bytes,
    audio_feature_size: int | None = 128,
) -> tuple[list[torch.Tensor] | None, list[int] | None, list[float] | None]:
    """Load and process WAV file bytes for WeMM audio.

    Wraps the existing process_audio() function to accept WAV bytes directly.
    Handles decoding via TorchCodec/Librosa/FFmpeg with automatic fallback.

    Args:
        wav_bytes: Raw WAV file bytes
        audio_feature_size: 128 for 7B model, 80 for 2B model

    Returns:
        Tuple of (audio_feature, audio_lengths, audio_durations)
        - audio_feature: list of torch.Tensor with shape (segments, feature_size, 3000)
        - audio_lengths: list of segment counts (typically [60], [120], etc.)
        - audio_durations: list of actual durations in seconds

    Raises:
        ValueError: If WAV is invalid or cannot be decoded
    """
    return process_audio(
        BytesIO(wav_bytes),
        audio_feature_size=audio_feature_size,
    )
