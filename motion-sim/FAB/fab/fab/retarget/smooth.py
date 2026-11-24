"""Fixed-lag smoothing of per-frame motion outputs (G1 joint angles / root rotation, SMPL pose, joints) for online use.

At frame t the output for frame t - lag is released: a Gaussian-weighted local-linear fit over the window
[t - lag - W, t] (W = ceil(3 sigma) frames of history, `lag` frames of look-ahead), evaluated at t - lag. The
local-linear fit removes the lag bias of a one-sided weighted mean. Recommended at 30 fps: lag 3, sigma 2.5 (100 ms;
lag 2, sigma 1.5 when 67 ms matters).

Rotations are filtered as unit quaternions (sign kept continuous with the previous frame, renormalised); joint angles
and positions as plain vectors.

    sm = LagSmoother(lag=3, sigma=2.5, kinds={"g1_q": "vec", "g1_root": "rot"})
    for frame in stream:
        out = sm.push({"g1_q": q, "g1_root": R})   # None for the first `lag` frames, then frame t - lag's values
    tail = sm.flush()                             # the last `lag` frames (window padded with the final frame)
"""

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation


def lag_weights(lag, sigma):
    """Offsets k (frames relative to the output frame) and local-linear weights c_k: y_t = sum_k c_k x_{t+k}."""
    W = int(np.ceil(3 * sigma))
    ks = np.arange(-W, lag + 1, dtype=np.float64)
    w = np.exp(-0.5 * (ks / sigma) ** 2)
    if len(ks) < 3:
        return ks.astype(int), w / w.sum()
    S0, S1, S2 = w.sum(), (w * ks).sum(), (w * ks * ks).sum()
    return ks.astype(int), w * (S2 - S1 * ks) / (S0 * S2 - S1 ** 2)


def lag_filter(x, lag=3, sigma=2.5):
    """Batch version over a whole sequence x (T, ...): identical to streaming push / flush (edges repeat the ends)."""
    ks, c = lag_weights(lag, sigma)
    T = len(x)
    idx = np.clip(np.arange(T)[:, None] + ks[None], 0, T - 1)
    return np.tensordot(c, x[idx].swapaxes(0, 1), axes=1)


def _to_quat(R, prev):
    q = Rotation.from_matrix(np.asarray(R, np.float64).reshape(-1, 3, 3)).as_quat()
    if prev is not None:
        q = np.where(((q * prev).sum(-1) < 0)[:, None], -q, q)
    return q


class LagSmoother:
    """Streaming fixed-lag smoother of a dict of per-frame arrays. kinds: name -> "vec" (any shape), "rot" (... 3 x 3
    rotation matrices) or "aa" (... x 3 axis-angle vectors, e.g. SMPL pose reshaped (24, 3) or flat (72,))."""

    def __init__(self, lag=3, sigma=2.5, kinds=None):
        self.lag, self.kinds = lag, dict(kinds or {})
        self.ks, self.c = lag_weights(lag, sigma)
        self.W = -int(self.ks[0])
        self.reset()

    def _encode(self, frame):
        enc = {}
        for k, v in frame.items():
            kind = self.kinds.get(k, "vec")
            v = np.asarray(v, np.float64)
            if kind == "rot":
                enc[k] = (_to_quat(v, self.prev.get(k)), v.shape)
            elif kind == "aa":
                R = Rotation.from_rotvec(v.reshape(-1, 3)).as_matrix()
                enc[k] = (_to_quat(R, self.prev.get(k)), v.shape)
            else:
                enc[k] = (v, v.shape)
            if kind in ("rot", "aa"):
                self.prev[k] = enc[k][0]
        return enc

    def _decode(self, k, x, shape):
        kind = self.kinds.get(k, "vec")
        if kind == "vec":
            return x.reshape(shape)
        q = x / np.linalg.norm(x, axis=-1, keepdims=True)
        if kind == "rot":
            return Rotation.from_quat(q).as_matrix().reshape(shape)
        return Rotation.from_quat(q).as_rotvec().reshape(shape)

    def _frame(self, i):  # encoded frame i, indices clipped to the frames seen (edges repeat the ends)
        i = min(max(i, 0), self.seen - 1)
        return self.buf[i - (self.seen - len(self.buf))]

    def _emit(self, center):
        window = [self._frame(center + int(k)) for k in self.ks]
        out = {}
        for k in window[0]:
            x = sum(c * f[k][0] for c, f in zip(self.c, window))
            out[k] = self._decode(k, x, window[0][k][1])
        self.n_out += 1
        return out

    def push(self, frame):
        """Add frame t; returns the smoothed frame t - lag (dict) or None while fewer than lag + 1 frames were seen."""
        self.buf.append(self._encode(frame))
        self.seen += 1
        if self.seen <= self.lag:
            return None
        return self._emit(self.seen - 1 - self.lag)

    def flush(self):
        """The frames still pending (the last lag), their look-ahead padded with the final frame."""
        return [self._emit(c) for c in range(self.n_out, self.seen)]

    def reset(self):
        self.buf = deque(maxlen=self.W + self.lag + 1)
        self.prev, self.seen, self.n_out = {}, 0, 0
