"""Adaptive control for the network server (pure Python, no MLX): how far ahead of playback reply audio is sent, how much of it is held before a reply starts, and how loud the incoming microphone audio is made. Each is derived from what was just measured, so a different link, a busier machine or a quieter microphone changes the behaviour without anyone tuning a number.

Playout: the link's jitter comes from round trips to the client, the model's speed and a typical reply's length come from recent replies, and past stalls from the gaps in what was sent. Two limits are policy and one is a fact about the client:
- capacity_s: how much audio the client can hold, a property of the client (its playback buffer plus its queue);
- max_hold_s: the longest a reply may be held back before it starts, a policy about latency.
The safety factors below widen an estimate that is a percentile, not a bound.

InputGain: scales the microphone so that speech reaches the level the engine's thresholds were set for."""
from __future__ import annotations

import collections

FRAME_S = 0.02
JITTER_MARGIN = 1.5            # the slowest recent round trip is not the slowest to come
CAPACITY_USE = 0.75            # of the client's buffer, so a burst does not overflow it
PRIOR_LEAD_FRACTION = 0.5      # of capacity: the send-ahead before any round trip has been measured
PRIOR_HOLD_S = 0.8             # the hold before any reply has been timed
SPEECH_RMS = 0.1               # the speech level (rms of a 20 ms frame) the engine's thresholds assume


def percentile(values, q: float) -> float:
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))] if v else 0.0


class Playout:
    def __init__(self, capacity_s: float, max_hold_s: float):
        self.capacity, self.max_hold = capacity_s, max_hold_s
        self.rtt: collections.deque[float] = collections.deque(maxlen=50)         # this link, recent
        self.speed: collections.deque[float] = collections.deque(maxlen=8)        # seconds of speech made per second of generating, recent replies
        self.length: collections.deque[float] = collections.deque(maxlen=8)       # seconds of speech per reply, recent replies
        self.stalls: collections.deque[float] = collections.deque(maxlen=8)       # the longest stall inside each recent reply (0 for none)
        self.worst = 0.0                                                          # the longest stall in the reply in progress

    def new_link(self) -> None:
        self.rtt.clear()

    def observe_rtt(self, s: float) -> None:
        self.rtt.append(s)

    def observe_stall(self, s: float) -> None:
        self.worst = max(self.worst, s)

    def observe_reply(self, audio_s: float, gen_s: float, cancelled: bool = False) -> None:
        """A reply finished: its length, how fast it was made, and the worst stall in it."""
        worst, self.worst = self.worst, 0.0
        if cancelled or audio_s <= 0 or gen_s <= 0: return
        self.speed.append(audio_s / gen_s); self.length.append(audio_s); self.stalls.append(worst)

    def jitter(self) -> float:
        """The spread of the one-way delay, from the spread of the round trips: how far the slowest recent one runs beyond the fastest, halved. A single late packet is a gap in the sound, so the worst case counts, not a percentile."""
        if len(self.rtt) < 5: return 0.0
        return max(0.0, max(self.rtt) - min(self.rtt)) / 2

    def lead(self) -> float:
        """How far ahead of the moment the client plays it a frame may be sent: enough to ride out the link's jitter, within what the client can hold."""
        if len(self.rtt) < 5: return self.capacity * PRIOR_LEAD_FRACTION
        return min(self.capacity * CAPACITY_USE, max(3 * FRAME_S, JITTER_MARGIN * self.jitter() + 2 * FRAME_S))

    def hold(self) -> float:
        """How much reply audio to have ready before it starts playing. If the model makes speech slower than it is spoken the buffer drains by the difference, so a reply of the usual length needs length x (1 - speed) up front; after that, enough to cover the longest recent stall; plus the link's
        jitter. Never more than max_hold_s."""
        if not self.speed: return min(PRIOR_HOLD_S, self.max_hold)
        speed = sorted(self.speed)[len(self.speed) // 2]
        usual = percentile(self.length, 0.75)
        drain = usual * max(0.0, 1.0 - speed)
        stall = max(percentile(self.stalls, 0.95), self.worst)
        return min(self.max_hold, max(2 * FRAME_S, max(drain, stall) + JITTER_MARGIN * self.jitter()))


class InputGain:
    """Automatic gain for the incoming microphone: after speech, the gain moves toward the one that would make the speech SPEECH_RMS."""

    def __init__(self, max_gain: float = 16.0, target: float = SPEECH_RMS):
        self.max_gain, self.target = max_gain, target
        self.gain = 1.0
        self.window: collections.deque[float] = collections.deque(maxlen=1500)    # the last 30 s of frame levels (before the gain)

    def observe(self, rms: float) -> None:
        self.window.append(rms)

    def update(self) -> float:
        """Call about once a second. Speech is the frames well above the room's floor; with under a second of speech in the window the gain stays."""
        if len(self.window) < 100: return self.gain
        floor = percentile(self.window, 0.2)
        voiced = [x for x in self.window if x > max(0.002, 4 * floor)]
        if len(voiced) < 50: return self.gain
        wanted = min(self.max_gain, max(1.0, self.target / percentile(voiced, 0.9)))
        self.gain += 0.3 * (wanted - self.gain)
        return self.gain
