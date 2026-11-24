# Copyright 2026 Adventists AI. Apache-2.0.
"""CUDA-Graph wrappers for the vocoder backbone and other row-independent modules (fixed shapes replay a captured graph; other shapes fall back to the eager path; same outputs)."""
import os, torch
BK = [1, 2, 4, 8, 16, 24, 32]
import torch.nn.functional as F
def _cc_forward(self, x, cache=None):
    """Same as CausalConv1d.forward, with the zero padding done on the GPU."""
    if cache is None or cache.size(2) == 0: x = F.pad(x, (self.causal_padding, 0) if self.causal_type == 'left' else (0, self.causal_padding))
    else: x = torch.concat([cache, x], dim=2) if self.causal_type == 'left' else torch.concat([x, cache], dim=2)
    return torch.nn.Conv1d.forward(self, x)
def patch_conv(hift):
    for C in {type(m) for m in hift.modules() if type(m).__name__ == 'CausalConv1d'}:
        if not getattr(C, '_fastpad', False): C.forward = _cc_forward; C._fastpad = True
class HiftGraph:
    def __init__(s, hift, max_b=16):
        patch_conv(hift); s.h = hift; s.orig = hift.decode; s.g = {}; s.seen = {}; s.max_b = max_b; s.on = True; s.hit = 0; s.miss = 0
        hift.stft_window = hift.stft_window.to(next(hift.parameters()).device); hift.decode = s.decode
    def _core(s, x, ss):
        h = s.h; L = h.conv_pre_look_right; x = h.conv_pre(x[:, :, :-L], x[:, :, -L:])
        for i in range(h.num_upsamples):
            x = F.leaky_relu(x, h.lrelu_slope); x = h.ups[i](x)
            if i == h.num_upsamples - 1: x = h.reflection_pad(x)
            x = x + h.source_resblocks[i](h.source_downs[i](ss)); xs = None
            for j in range(h.num_kernels):
                r = h.resblocks[i * h.num_kernels + j](x); xs = r if xs is None else xs + r
            x = xs / h.num_kernels
        return h.conv_post(F.leaky_relu(x))
    def _cap(s, B, x, ss):
        xi = x[:1].repeat(B, 1, 1).clone(); si = ss[:1].repeat(B, 1, 1).clone()
        for _ in range(3): s._core(xi, si)
        torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): out = s._core(xi, si)
        return (g, xi, si, out)
    def decode(self, x, s=None, finalize=True):
        src = s; s = self; h = s.h
        if (not s.on) or finalize or x.shape[0] > s.max_b: s.miss += 1; return s.orig(x=x, s=src, finalize=finalize)
        B0 = x.shape[0]; B = next(b for b in BK if b >= B0); key = (B, x.shape[2], src.shape[2])
        e = s.g.get(key)
        if e is None:
            n = s.seen[key] = s.seen.get(key, 0) + 1
            if n < int(os.environ.get("S220_HG_AFTER", "3")) or len(s.g) >= 40: s.miss += 1; return s.orig(x=x, s=src, finalize=finalize)
        import numpy as np
        up = int(np.prod(h.upsample_rates)); re_, im_ = h._stft(src.squeeze(1)); cut = up * h.conv_pre_look_right
        ss = torch.cat([re_[:, :, :-cut], im_[:, :, :-cut]], dim=1)
        if e is None: e = s.g[key] = s._cap(B, x, ss)
        g, xi, si, out = e; xi[:B0].copy_(x); si[:B0].copy_(ss); g.replay(); s.hit += 1
        y = out[:B0]; nf = h.istft_params["n_fft"] // 2 + 1
        w = h._istft(torch.exp(y[:, :nf, :]), torch.sin(y[:, nf:, :]))
        w = w[:, :-up * h.istft_params["hop_len"]]
        return torch.clamp(w, -h.audio_limit, h.audio_limit)
    def warm(s, mk, Tws=(56,), Bs=(1, 2, 4, 8, 16)):
        import engine as EN
        for Tw in Tws:
            for B in Bs:
                if B > s.max_b: continue
                rs = mk(B, Tw)
                for _ in range(int(os.environ.get("S220_HG_AFTER", "3")) + 1): EN._hift_batch(s.h, rs)

class GraphFn:
    """Capture fn (tensor in, tensor out, rows independent) per (batch bucket, shape); unseen or failing shapes use fn directly."""
    def __init__(s, fn, buckets=(1, 2, 4, 8, 16), after=3, max_graphs=120):
        s.fn = fn; s.bk = buckets; s.after = after; s.maxg = max_graphs; s.g = {}; s.seen = {}; s.bad = set(); s.hit = 0; s.miss = 0
    def __call__(s, x):
        B0 = x.shape[0]
        if B0 > s.bk[-1]: s.miss += 1; return s.fn(x)
        B = next(b for b in s.bk if b >= B0); key = (B,) + tuple(x.shape[1:]); e = s.g.get(key)
        if e is None:
            n = s.seen[key] = s.seen.get(key, 0) + 1
            if key in s.bad or n < s.after or len(s.g) >= s.maxg: s.miss += 1; return s.fn(x)
            try:
                xi = torch.cat([x] + [x[:1]] * (B - B0), 0).clone()
                for _ in range(2): s.fn(xi)
                torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g): out = s.fn(xi)
                e = s.g[key] = (g, xi, out)
            except Exception as ex:
                s.bad.add(key); print("GraphFn capture failed", key, repr(ex)[:160], flush=True); torch.cuda.synchronize(); return s.fn(x)
        g, xi, out = e; xi[:B0].copy_(x); g.replay(); s.hit += 1
        return out[:B0].clone()
