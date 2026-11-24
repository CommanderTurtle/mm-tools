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
"""Per-browser callbacks for the live microphone tab.

The browser sends each piece of microphone audio as ``pcm``: base64 of
little-endian 16-bit PCM, at most one second per event, with its sample
rate. Every answer is a value for the panel; a failure is an "error" phase
naming the take it belongs to, which the browser acts on at once.
"""

import base64
import dataclasses
import math
import threading
import time

import gradio as gr
import numpy as np

import live

# Live takes running at once: each holds about 1.3 GB on the device, and on
# an Apple M4 Max a third concurrent take already falls behind.
MAX_LIVE_TAKES = 2
# A take whose browser sent nothing for this long has failed on its side (a
# lost connection, a closed microphone): the next take to start releases it.
IDLE_SECONDS = 10.0
MIN_RATE, MAX_RATE = 8000, 192000
_EXPIRED = (f"This recording stopped after {IDLE_SECONDS:g} seconds without"
            " audio. Please start a new recording.")


def _now() -> float:
    """The clock that times takes, looked up when called."""
    return time.monotonic()


@dataclasses.dataclass
class Recording:
    """One browser take, its ordered audio and its streaming state."""

    take: str
    session: live.LiveSession
    sequence: int = 0
    touched: float = dataclasses.field(default_factory=_now)
    pushing: int = 0
    expired: bool = False

    @property
    def idle(self) -> bool:
        """Unfinished, with no audio in flight and none for IDLE_SECONDS."""
        return (not self.session.finished and not self.pushing and
                time.monotonic() - self.touched >= IDLE_SECONDS)

    @property
    def active(self) -> bool:
        """Whether the take holds a place: unfinished and not idle."""
        return not self.session.finished and not self.idle


def close_recording(recording):
    """Releases a browser's stream when its state expires."""
    if recording is not None:
        recording.session.reset()


def _player(rate, audio):
    """A player value at the recording's own level, or None when empty."""
    # Integer audio prevents Gradio from raising quiet output to full scale.
    preview = np.round(np.clip(audio, -1, 1) * 32767).astype(np.int16)
    return gr.Audio(value=(rate, preview) if len(preview) else None)


def _error(take, message):
    """An error answer for ``take``; the browser ends that take with it."""
    return dict(take=take if isinstance(take, str) else None,
                phase="error",
                message=message)


def _decode(pcm, rate) -> tuple[int, np.ndarray]:
    """Checks one browser event's audio; raises ValueError when invalid."""
    if (not isinstance(rate, (int, float)) or isinstance(rate, bool) or
            not math.isfinite(rate) or rate != int(rate) or
            not MIN_RATE <= rate <= MAX_RATE):
        raise ValueError("Invalid microphone rate.")
    rate = int(rate)
    # Base64 of one second of 16-bit samples, rounded up to whole groups.
    if not isinstance(pcm, str) or len(pcm) > 4 * math.ceil(2 * rate / 3):
        raise ValueError("Invalid microphone audio.")
    raw = base64.b64decode(pcm, validate=True)
    if len(raw) % 2:
        raise ValueError("Invalid microphone audio.")
    return rate, np.frombuffer(raw, dtype="<i2")


class Online:
    """Shares model weights while giving each browser its own audio state.

    ``device`` names where the model runs (for example "Metal GPU"); every
    answer carries it for the panel's speed readout. At most ``max_takes``
    browsers record at once. A browser's take is released when its state
    expires, when it starts another, and when its tab closes (``release``,
    registered as the page's unload event). A take starts only if the model
    is free within ``start_wait`` seconds (on an Apple GPU an Offline
    isolation holds it for its whole length).
    """

    def __init__(self,
                 model,
                 model_lock: threading.RLock,
                 device: str = "",
                 max_takes: int = MAX_LIVE_TAKES,
                 start_wait: float = 2.0):
        self.model = model
        self.model_lock = model_lock
        self.device = device
        self.max_takes = max_takes
        self.start_wait = start_wait
        self._takes = {}
        self._starting = 0
        # browsers with a start in flight, and those that closed meanwhile
        self._pending = {}
        self._closed = set()
        self._takes_lock = threading.Lock()

    def _response(self, recording, phase, message="", update=None):
        if update is None:
            update = recording.session.snapshot()
        return dict(take=recording.take,
                    sequence=recording.sequence,
                    phase=phase,
                    message=message,
                    device=self.device,
                    input_seconds=update.input_seconds,
                    output_seconds=update.output_seconds,
                    processing_seconds=update.processing_seconds,
                    limit_reached=update.limit_reached,
                    visualization=update.visualization)

    def start(self,
              recording,
              event: gr.EventData,
              request: gr.Request | None = None):
        """Starts a fresh take after the browser grants microphone access.

        Returns the state, the panel's value and both players cleared.
        """
        take = getattr(event, "take", None)
        if not isinstance(take, str) or not 1 <= len(take) <= 80:
            return recording, _error(
                None, "Invalid recording. Please start again."), None, None
        browser = getattr(request, "session_hash", None)
        with self._takes_lock:
            released = [self._takes.pop(browser, None)]
            # Idle takes give their place up for good: an expired take is
            # refused if its browser comes back.
            for key, other in list(self._takes.items()):
                if other.idle:
                    other.expired = True
                    released.append(self._takes.pop(key))
        # Outside the registry's lock: a session's own lock may be held by a
        # push that waits for the model.
        for old in (recording, *released):
            close_recording(old)
        with self._takes_lock:
            busy = self._starting + sum(
                other.active for other in self._takes.values())
            if busy >= self.max_takes:
                return None, _error(
                    take, "The live demo is busy with other recordings."
                    " Please try again in a minute."), None, None
            self._starting += 1
            self._pending[browser] = self._pending.get(browser, 0) + 1
        closed = False
        try:
            if not self.model_lock.acquire(timeout=self.start_wait):
                return None, _error(
                    take, "The model is busy with other work."
                    " Please try again in a moment."), None, None
            try:
                session = live.LiveSession(self.model,
                                           model_lock=self.model_lock)
            except (RuntimeError, ValueError) as error:
                return None, _error(
                    take, f"The recording could not start: {error}"), None, None
            finally:
                self.model_lock.release()
            recording = Recording(take, session)
            with self._takes_lock:
                closed = browser in self._closed
                if browser is not None and not closed:
                    self._takes[browser] = recording
        finally:
            with self._takes_lock:
                self._starting -= 1
                self._pending[browser] -= 1
                if not self._pending[browser]:
                    del self._pending[browser]
                    self._closed.discard(browser)
        if closed:
            # The tab closed while the take was starting: nobody will use it.
            close_recording(recording)
            return None, _error(take, "The page was closed."), None, None
        return recording, self._response(recording, "ready"), None, None

    def release(self, request: gr.Request | None = None) -> None:
        """Releases a closed or reloaded tab's take at once.

        Gradio keeps a closed tab's state for an hour; the stream it holds
        (about 1.3 GB on the device) should not outlive the page.
        """
        browser = getattr(request, "session_hash", None)
        with self._takes_lock:
            recording = self._takes.pop(browser, None)
            if browser in self._pending:
                self._closed.add(browser)
        close_recording(recording)

    def push(self, recording, event: gr.EventData):
        """Consumes each browser event's audio once, in order."""
        take = getattr(event, "take", None)
        if recording is None or take != recording.take:
            # A take this server does not hold (a restart, an expired state
            # or a late answer for an older take): the browser ends it, and
            # ignores the answer if it has moved on.
            return recording, _error(
                take, "This recording is no longer on the server."
                " Please start a new recording.")
        if recording.expired:
            return recording, _error(recording.take, _EXPIRED)
        recording.touched = time.monotonic()
        recording.pushing += 1
        try:
            return self._push(recording, event)
        finally:
            recording.pushing -= 1
            recording.touched = time.monotonic()

    def _push(self, recording, event: gr.EventData):
        sequence = getattr(event, "sequence", None)
        if (not isinstance(sequence, int) or isinstance(sequence, bool) or
                sequence != recording.sequence + 1):
            recording.session.reset()
            return recording, _error(
                recording.take,
                "Audio arrived out of order. Start a new recording.")
        try:
            audio = _decode(getattr(event, "pcm", None),
                            getattr(event, "rate", None))
            update = recording.session.push(audio)
        except (ValueError, RuntimeError, TypeError, OverflowError):
            recording.session.reset()
            return recording, _error(
                recording.take,
                "Microphone processing stopped. Please start a new recording.")
        recording.sequence = sequence
        return recording, self._response(recording, "active", update=update)

    def finish(self, recording, event: gr.EventData):
        """Flushes the final samples after capture and queued audio stop.

        Returns the state, the panel's value, the isolated voice and the
        unprocessed microphone input it came from, sample-aligned.
        """
        take = getattr(event, "take", None)
        if recording is None or take != recording.take:
            return recording, _error(
                take, "This recording is no longer on the server."
                " Please start a new recording."), gr.skip(), gr.skip()
        if recording.expired:
            return recording, _error(recording.take,
                                     _EXPIRED), gr.skip(), gr.skip()
        if getattr(event, "sequence", None) != recording.sequence:
            return recording, _error(
                recording.take,
                "Some microphone audio did not reach the server."
                " Please start a new recording."), gr.skip(), gr.skip()
        try:
            rate, audio = recording.session.finish()
        except (RuntimeError, ValueError) as error:
            recording.session.reset()
            return recording, _error(
                recording.take, f"The recording could not be finished: {error}"
            ), gr.skip(), gr.skip()
        _, original = recording.session.recorded()
        return (recording, self._response(recording, "finished"),
                _player(rate, audio), _player(rate, original))
