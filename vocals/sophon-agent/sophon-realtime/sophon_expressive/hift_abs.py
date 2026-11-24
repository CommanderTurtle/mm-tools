# Copyright 2026 Adventists AI. Apache-2.0.
"""HiFT vocoder with phase carried over absolute sample positions (seamless streaming windows)."""
import torch, numpy as np, torch.nn.functional as F
def _f02sine_abs(self, f0_values):
    ctx = getattr(self, "_abs", None)
    if ctx is None: return self._f02sine_orig(f0_values)
    rad = (f0_values / self.sampling_rate) % 1
    for b, (k_, tg, ini, off) in enumerate(ctx):
        if ini: rad[b, 0, :] = rad[b, 0, :] + self.rand_ini[0].to(rad.device)
    rad = F.interpolate(rad.transpose(1, 2), scale_factor=1 / self.upsample_scale, mode="linear").transpose(1, 2)
    cum = torch.cumsum(rad.double(), 1)
    for b, (k_, tg, ini, off) in enumerate(ctx):
        if k_ is not None: cum[b] += tg - cum[b, k_]
    self._cum_out = cum
    ph = (torch.remainder(cum * self.upsample_scale, 1.0) * (2 * np.pi)).float()
    ph = F.interpolate(ph.transpose(1, 2), scale_factor=self.upsample_scale, mode="nearest").transpose(1, 2)
    return torch.sin(ph)
def _fwd_abs(self, f0):
    ctx = getattr(self, "_abs", None)
    if ctx is None: return self._forward_orig(f0)
    fn = f0 * torch.arange(1, self.harmonic_num + 2, device=f0.device, dtype=f0.dtype)[None, None]
    sw = self._f02sine(fn) * self.sine_amp; uv = self._f02uv(f0)
    na = uv * self.noise_std + (1 - uv) * self.sine_amp / 3; n = sw.shape[1]
    if getattr(self, "_swg", None) is None or self._swg.device != sw.device: self._swg = self.sine_waves[0].to(sw.device)
    U = int(self.upsample_scale); L = self._swg.shape[0]       # the noise table holds 300 s: past it, wrap around (it is random noise)
    buf = torch.stack([self._swg[(off * U) % max(1, L - n): (off * U) % max(1, L - n) + n] for _, _, _, off in ctx])
    noise = na * buf; return sw * uv + noise, uv, noise
def patch(hift):
    sg = hift.m_source.l_sin_gen; C = type(sg)
    if not hasattr(C, "_f02sine_orig"): C._f02sine_orig = C._f02sine; C._f02sine = _f02sine_abs; C._forward_orig = C.forward; C.forward = _fwd_abs
    return sg
def plan(s, ws_):
    st = getattr(s, "habs", None)
    if st is None: return (None, None, ws_ == 0, ws_)
    pws, pcum = st; d = ws_ - pws; K = int(__import__("os").environ.get("E2V2_HIFT_ABSK", "24"))
    k = min(K, pcum.shape[0] - 1 - d)
    if k < 0: return (None, None, ws_ == 0, ws_)
    return (k, pcum[d + k], ws_ == 0, ws_)
def commit(s, ctx_row, cum_row):
    s.habs = (ctx_row[3], cum_row)
@torch.no_grad()
def infer_one(hift, s, mel_win, ws_, finalize):
    sg = patch(hift); c = plan(s, ws_); sg._abs = [c]
    try: w, _ = hift.inference(speech_feat=mel_win, finalize=finalize); commit(s, c, sg._cum_out[0])
    finally: sg._abs = None
    return w
