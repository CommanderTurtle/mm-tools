"""Local echo cancellation (WebRTC AEC3 via livekit's AudioProcessingModule): lets the user talk over the reply on speakers
without the reply's own sound being heard as the user speaking. Works on 20 ms frames at 16 kHz."""
import collections
import numpy as np
from livekit import rtc

SR, F = 16000, 160                     # APM works on 10 ms frames
APM_MAX_DELAY_MS = 500                 # WebRTC accepts 0..500; above that it returns a warning that livekit raises as an error


def _i16(x):
    return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()


class Echo:
    def __init__(self, delay_ms=40, ns=True):
        self.apm = rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=ns, high_pass_filter=True, auto_gain_control=False)
        delay_ms = max(0, int(delay_ms))
        # Bluetooth / AirPlay / HDMI outputs can exceed APM's 500 ms limit: delay the reference ourselves by the excess
        # and give APM only the remainder, so the canceller still lines up.
        self.extra_ms = max(0, delay_ms - APM_MAX_DELAY_MS)
        self.delay_ms = min(delay_ms, APM_MAX_DELAY_MS)
        n = self.extra_ms * SR // 1000
        self.ref = collections.deque([0.0] * n) if n else None

    def process(self, mic, far):
        """mic, far: 320 float32 samples each (far = what the speaker played in the same 20 ms). Returns the cleaned mic frame."""
        out = np.empty(len(mic), np.float32)
        if self.ref is not None:
            self.ref.extend(far.tolist())
            far = np.fromiter((self.ref.popleft() for _ in range(len(mic))), np.float32, len(mic))
        for s in range(0, len(mic), F):
            fr = rtc.AudioFrame(_i16(far[s:s + F]), SR, 1, F); self.apm.process_reverse_stream(fr)
            try:
                self.apm.set_stream_delay_ms(self.delay_ms)
            except RuntimeError:
                pass                       # out-of-range warning only; never fatal
            nf = rtc.AudioFrame(_i16(mic[s:s + F]), SR, 1, F); self.apm.process_stream(nf)
            out[s:s + F] = np.frombuffer(nf.data, np.int16).astype(np.float32) / 32767
        return out
