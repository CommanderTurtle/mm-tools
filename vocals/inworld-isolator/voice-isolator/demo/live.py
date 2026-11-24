# Copyright 2026 Inworld AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One bounded microphone recording through a persistent realtime stream."""

import dataclasses
import threading
import time
import typing

import numpy as np

from inworld import voice_isolation

SAMPLE_RATE = 48000
BLOCK_FRAMES = 4
LOOKAHEAD_FRAMES = 2
WAVEFORM_BINS = 480
FREQUENCY_BINS = 96
TIME_BINS = 80
FFT_SAMPLES = 2048
MIN_DECIBELS = -100.0


@dataclasses.dataclass(frozen=True)
class LiveUpdate:
    """Progress and bounded, JSON-compatible plots for one recording.

    ``visualization`` contains ``sample_rate``, ``start_seconds``,
    ``end_seconds``, ``frequency_hz`` and ``decibels``. Its ``input`` and
    ``output`` each contain a ``waveform`` list of [minimum, maximum] pairs
    and a ``spectrogram`` matrix indexed [frequency][time]. Frequencies run
    low to high and times run oldest to newest; intensity 0 means -100 dBFS
    and 255 means 0 dBFS. Both signals share the same finalized time window.
    """

    input_seconds: float
    output_seconds: float
    processing_seconds: float
    finished: bool
    limit_reached: bool
    visualization: dict[str, typing.Any]


def _mono_float(samples: np.ndarray) -> np.ndarray:
    """Converts browser PCM to mono float32 without adjusting its level."""
    if samples.ndim not in (1, 2) or (samples.ndim == 2 and
                                      samples.shape[1] == 0):
        raise ValueError("Microphone audio must have samples and channels.")
    if samples.dtype.kind not in "iuf":
        raise ValueError("Microphone audio must contain numeric PCM samples.")
    if samples.dtype.kind in "iu":
        info = np.iinfo(samples.dtype)
        midpoint = float(info.max) + 1.0
        if samples.dtype.kind == "u":
            midpoint /= 2.0
            samples = (samples.astype(np.float32) - midpoint) / midpoint
        else:
            samples = samples.astype(np.float32) / midpoint
    else:
        samples = samples.astype(np.float32, copy=False)
    if not np.isfinite(samples).all():
        raise ValueError("Microphone audio must contain finite samples.")
    if samples.ndim == 2:
        samples = samples.mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(samples, dtype=np.float32)


def _plot(samples: np.ndarray) -> dict[str, list]:
    """Returns fixed-scale amplitude and spectral previews of one window."""
    if samples.size == 0:
        return {"waveform": [], "spectrogram": []}
    boundaries = np.linspace(0,
                             samples.size,
                             min(samples.size, WAVEFORM_BINS) + 1,
                             dtype=np.int64)
    waveform = [[float(samples[a:b].min()),
                 float(samples[a:b].max())]
                for a, b in zip(boundaries[:-1], boundaries[1:])]
    centers = np.linspace(0,
                          samples.size - 1,
                          min(TIME_BINS, samples.size),
                          dtype=np.int64)
    padded = np.pad(samples, (FFT_SAMPLES // 2, FFT_SAMPLES // 2))
    frames = padded[centers[:, None] + np.arange(FFT_SAMPLES)[None]]
    window = np.hanning(FFT_SAMPLES).astype(np.float32)
    magnitude = np.abs(np.fft.rfft(frames * window, axis=1))
    magnitude *= 2.0 / window.sum()
    edges = np.linspace(0,
                        magnitude.shape[1],
                        FREQUENCY_BINS + 1,
                        dtype=np.int64)
    bands = np.stack(
        [magnitude[:, a:b].max(axis=1) for a, b in zip(edges[:-1], edges[1:])])
    decibels = 20.0 * np.log10(np.maximum(bands, 1e-8))
    intensity = np.rint(
        np.clip((decibels - MIN_DECIBELS) / -MIN_DECIBELS, 0, 1) * 255).astype(
            np.uint8)
    return {"waveform": waveform, "spectrogram": intensity.tolist()}


class LiveSession:
    """Owns one browser's resampler, model stream and bounded recording.

    The injected isolator must already use streaming mode and the released
    two-frame look-ahead. Its mode must remain fixed while sessions exist.
    Pass the same ``model_lock`` to sessions sharing a model to serialize
    device inference. Each session independently serializes its own events.

    ``push`` accepts only newly recorded chunks, never a growing recording.
    ``finish`` flushes once and returns mono float32 audio at 48 kHz, and
    ``recorded`` returns the unprocessed input it was isolated from. At the
    recording limit, the session finishes automatically; later chunks are
    ignored. ``reset`` releases the recording and stream; start a new instance
    for the next recording. No audio is written to disk by this class.
    """

    def __init__(
        self,
        model: voice_isolation.VoiceIsolator,
        *,
        max_seconds: float = 60.0,
        window_seconds: float = 6.0,
        model_lock: typing.ContextManager | None = None,
    ) -> None:
        setting = (model.mode, model.lookahead_frames)
        if setting != ("streaming", LOOKAHEAD_FRAMES):
            raise ValueError("Live audio needs the realtime model in streaming"
                             " mode with two look-ahead frames.")
        if model.sample_rate != SAMPLE_RATE:
            raise ValueError("Live audio needs a 48 kHz model.")
        if not np.isfinite(max_seconds) or not 0 < max_seconds <= 60:
            raise ValueError("The recording limit must be between 0 and 60 s.")
        if not np.isfinite(window_seconds) or not 0 < window_seconds <= 8:
            raise ValueError("The visualization window must be at most 8 s.")
        self._lock = threading.RLock()
        self._model_lock = (model_lock
                            if model_lock is not None else threading.RLock())
        with self._model_lock:
            self._stream = model.stream(block_frames=BLOCK_FRAMES)
        self._max_seconds = float(max_seconds)
        self._capacity = max(1, int(max_seconds * SAMPLE_RATE))
        self._window = max(1, int(window_seconds * SAMPLE_RATE))
        self._input = np.empty(self._capacity, dtype=np.float32)
        self._output = np.empty(self._capacity, dtype=np.float32)
        self._input_count = 0
        self._output_count = 0
        self._source_rate = None
        self._source_count = 0
        self._resampler = None
        self._processing_seconds = 0.0
        self._finished = False
        self._limit_reached = False

    def _append_output(self, samples: np.ndarray) -> None:
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise RuntimeError("The live model returned invalid audio.")
        count = samples.size
        if self._output_count + count > self._input_count:
            raise RuntimeError(
                "The live model returned more audio than received.")
        end = self._output_count + count
        self._output[self._output_count:end] = samples
        self._output_count = end

    def _feed(self, samples: np.ndarray) -> None:
        samples = samples[:self._capacity - self._input_count]
        if samples.size == 0:
            return
        end = self._input_count + samples.size
        self._input[self._input_count:end] = samples
        self._input_count = end
        with self._model_lock:
            start = time.perf_counter()
            output = self._stream.push(samples)
            self._processing_seconds += time.perf_counter() - start
        self._append_output(output)

    def push(self, audio: tuple[int, np.ndarray] | None) -> LiveUpdate:
        """Processes new browser samples and returns a rolling visualization.

        Integer PCM is converted at its fixed full-scale level. The browser
        rate may differ from 48 kHz but must stay constant for the recording;
        a stateful resampler retains fractional positions across chunks.
        """
        with self._lock:
            if self._finished or audio is None:
                return self.snapshot()
            rate, raw = audio
            valid_rate = (isinstance(rate, (int, np.integer)) and
                          not isinstance(rate, bool))
            if not valid_rate or rate <= 0:
                raise ValueError("The microphone sample rate must be positive.")
            if self._source_rate is not None and rate != self._source_rate:
                raise ValueError("The microphone sample rate changed; start a"
                                 " new recording.")
            raw = np.asarray(raw)
            if raw.ndim not in (1, 2):
                raise ValueError(
                    "Microphone audio must have samples and channels.")
            remaining = max(0,
                            int(self._max_seconds * rate) - self._source_count)
            samples = _mono_float(raw[:remaining])
            if samples.size == 0:
                return self.snapshot()
            if self._source_rate is None:
                if rate != SAMPLE_RATE:
                    try:
                        # Resampling belongs to the optional demo extra.
                        # pylint: disable-next=import-outside-toplevel
                        import soxr
                    except ImportError as error:
                        raise ImportError("Live microphone resampling needs"
                                          " the demo extra (soxr).") from error
                    self._resampler = soxr.ResampleStream(rate,
                                                          SAMPLE_RATE,
                                                          1,
                                                          dtype="float32")
                self._source_rate = int(rate)
            self._source_count += samples.size
            converted = (samples if self._resampler is None else
                         self._resampler.resample_chunk(samples))
            self._feed(converted)
            if self._source_count >= int(self._max_seconds * rate):
                self._limit_reached = True
                self._finish()
            return self.snapshot()

    def _finish(self) -> None:
        if self._finished:
            return
        if self._resampler is not None:
            self._feed(
                self._resampler.resample_chunk(np.empty(0, dtype=np.float32),
                                               last=True))
            self._resampler = None
        if self._input_count:
            with self._model_lock:
                start = time.perf_counter()
                output = self._stream.flush(as_numpy=True)
                self._processing_seconds += time.perf_counter() - start
            self._append_output(output)
        self._stream = None
        self._finished = True
        if self._output_count != self._input_count:
            raise RuntimeError(
                "The live model did not return the full recording.")

    def finish(self) -> tuple[int, np.ndarray]:
        """Flushes once and returns the complete isolated recording."""
        with self._lock:
            self._finish()
            return SAMPLE_RATE, self._output[:self._output_count].copy()

    @property
    def finished(self) -> bool:
        """Whether the take has ended: finished, capped at the limit or reset.

        Cheap, unlike ``snapshot``, which also builds the plots.
        """
        return self._finished

    def recorded(self) -> tuple[int, np.ndarray]:
        """Returns the microphone input so far, mono float32 at 48 kHz.

        This is the audio the stream received, sample-aligned with the
        isolated recording ``finish`` returns: after ``finish`` both have the
        same length, and the input minus the voice is the removed background.
        """
        with self._lock:
            return SAMPLE_RATE, self._input[:self._input_count].copy()

    def snapshot(self) -> LiveUpdate:
        """Returns progress and plots of the latest finalized audio window."""
        with self._lock:
            end = self._output_count
            start = max(0, end - self._window)
            visualization = {
                "sample_rate": SAMPLE_RATE,
                "start_seconds": start / SAMPLE_RATE,
                "end_seconds": end / SAMPLE_RATE,
                "frequency_hz": [0, SAMPLE_RATE // 2],
                "decibels": [MIN_DECIBELS, 0],
                "input": _plot(self._input[start:end]),
                "output": _plot(self._output[start:end]),
            }
            return LiveUpdate(
                input_seconds=(self._source_count /
                               self._source_rate if self._source_rate else 0.0),
                output_seconds=end / SAMPLE_RATE,
                processing_seconds=self._processing_seconds,
                finished=self._finished,
                limit_reached=self._limit_reached,
                visualization=visualization)

    def reset(self) -> None:
        """Discards the recording and releases the session's inference state."""
        with self._lock:
            self._stream = None
            self._resampler = None
            self._input = np.empty(0, dtype=np.float32)
            self._output = np.empty(0, dtype=np.float32)
            self._input_count = self._output_count = self._source_count = 0
            self._source_rate = None
            self._processing_seconds = 0.0
            self._finished = True
