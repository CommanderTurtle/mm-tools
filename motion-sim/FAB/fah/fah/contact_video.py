"""FAH-Video contact: the causal per-hand-track model on top of the image contact head.

Per hand side the highest-score hand of each frame forms the track (as the training tracks); every frame the model sees the
last T frames of that track (zero rows where the side had no hand, the window zero-padded at the start), so the output is
online and the first frames see what the trained model saw. Inputs are the image head's *uncalibrated* dense logits (the
head's contact_bias added back: the video model was trained on them), so a calibrated image head and this module give the
held-out numbers of the paper. Output per hand:
  contact_video (778,)   (1 - a) p_frame + a p_video, mapped to the 778 vertices, then calibrated so that 0.5 is the
                         val-picked threshold (logit(p) - logit(thr_blend)); same units as the image head
  contact_score_video    the video flag probability; contact_flag_video = its hysteresis state (on > h_on, off < h_off;
                         smooth=2: on the mean of this and the previous frame's probability when that frame had the hand)
"""
import numpy as np
import torch
import torch.nn as nn


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def frame_feats(p, n_vert, bias=0.0):
    """(3 + n_vert + 66,) features of one hand (left hands mirrored into the right-hand frame); bias = the image head's
    contact_bias (its dense logits back to the uncalibrated scale the video model was trained on)."""
    j = np.asarray(p["joints3d"], np.float64)
    s = 1.0 if p["is_right"] else -1.0
    rel = (j - j[:1]) * 10.0
    rel[:, 0] *= s
    z = max(j[0, 2], 1e-3)
    root = [s * j[0, 0] / z, j[0, 1] / z, np.log(z)]
    return np.concatenate([[1.0, p.get("score", 1.0), logit(p["contact_score"])], np.clip(logit(p["contact_dense"]) + bias, -9, 9)[:n_vert],
                           rel.ravel(), root]).astype(np.float32)


class ContactVideo(nn.Module):
    def __init__(self, n_vert=400, dim=256, layers=4, heads=4, max_len=64, tok_dim=0):
        super().__init__()
        self.n_vert = n_vert
        d = 3 + n_vert + 66
        self.inp = nn.Sequential(nn.Linear(d + 132, dim), nn.GELU(), nn.Linear(dim, dim))
        self.tok = nn.Sequential(nn.LayerNorm(tok_dim), nn.Linear(tok_dim, dim)) if tok_dim else None
        self.pos = nn.Parameter(torch.randn(max_len, dim) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, heads, 4 * dim, 0.1, batch_first=True, norm_first=True), layers)
        self.out_v = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_vert))
        self.out_f = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        for m in (self.out_v[-1], self.out_f[-1]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, tok=None):
        """x (B, T, D) -> dense logits (B, T, n_vert), flag logit (B, T): per-frame logits + causal residuals."""
        B, T, _ = x.shape
        m = x[..., :1]
        kin = x[..., 3 + self.n_vert:]
        mp = torch.cat([torch.zeros_like(m[:, :1]), m[:, :-1]], 1) * m
        v = torch.cat([torch.zeros_like(kin[:, :1]), kin[:, 1:] - kin[:, :-1]], 1) * mp
        a = torch.cat([torch.zeros_like(v[:, :1]), v[:, 1:] - v[:, :-1]], 1) * torch.cat([torch.zeros_like(mp[:, :1]), mp[:, :-1]], 1) * mp
        h = self.inp(torch.cat([x, v * 10, a * 10], -1)) + self.pos[:T]
        if self.tok is not None and tok is not None:
            h = h + self.tok(tok.float()) * m
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        o = self.enc(h, mask=mask, is_causal=True)
        return x[..., 3:3 + self.n_vert] + self.out_v(o), x[..., 2] + self.out_f(o)[..., 0]


class VideoContact:
    """Online FAH-Video contact over the frames of one video (reset() between videos). ckpt: train.contact_video checkpoint
    (after `pick`); R: the image head's (778, n_vert) basis map (HandEstimator.sub_R); bias: the image head's contact_bias
    (main tolerance; 0 for an uncalibrated head). cuda_graph: the model's forward (always the full T-frame window, zero
    padded) replayed from a captured CUDA graph (same kernels, same outputs; the eager forward is launch-bound). smooth: 1 =
    the flag hysteresis on each frame's probability, 2 = on the mean of this and the previous frame's (fewer flag switches,
    about +1 frame latency); h: (h_on, h_off) instead of the checkpoint's (fah.api.CONTACT_SMOOTH_H for smooth=2)."""

    def __init__(self, ckpt, R, bias=0.0, device="cuda", cuda_graph=False, smooth=1, h=None):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False) if isinstance(ckpt, str) else ckpt
        a = ck["args"]
        self.T, self.nv, self.use_tok, self.dev = a["T"], ck["n_vert"], ck["use_tok"], torch.device(device)
        self.model = ContactVideo(self.nv, a["dim"], a["layers"], tok_dim=ck["tok_dim"]).to(self.dev).eval()
        self.model.load_state_dict(ck["model"])
        self.alpha, self.thr = float(ck.get("blend_alpha", 0.0)), float(ck.get("thr_blend", 0.5))
        self.h_on, self.h_off = float(ck.get("h_on", 0.6)), float(ck.get("h_off", 0.4))
        if h is not None:
            self.h_on, self.h_off = float(h[0]), float(h[1])
        assert smooth in (1, 2), "smooth must be 1 or 2"
        self.smooth = smooth
        self.R, self.bias, self.tok_dim = np.asarray(R, np.float64), float(bias), ck["tok_dim"]
        self.cuda_graph, self.graph = cuda_graph and self.dev.type == "cuda", None
        self.reset()

    def _forward(self, x, tok):
        if not self.cuda_graph:
            return self.model(x, tok)
        if self.graph is None:
            self.static = (x.clone(), None if tok is None else tok.clone())
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.model(*self.static)
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_out = self.model(*self.static)
        self.static[0].copy_(x)
        if tok is not None:
            self.static[1].copy_(tok)
        self.graph.replay()
        return self.static_out

    def reset(self):
        self.buf = {True: [], False: []}
        self.state = {True: False, False: False}
        self.missing = {True: 0, False: 0}  # consecutive frames without a track hand, per side
        self.prev = {True: None, False: None}  # the previous frame's flag probability (None: that frame had no track hand)

    @torch.no_grad()
    def __call__(self, hands):
        """hands: this frame's hand dicts (bench / API format: is_right, score, joints3d, contact_dense, contact_score[,
        token]); adds contact_video / contact_score_video / contact_flag_video to the track hand of each side."""
        for side in (True, False):
            c = [h for h in hands if bool(h["is_right"]) == side and "contact_dense" in h]
            h = max(c, key=lambda q: q.get("score", 0)) if c else None
            f = frame_feats(h, self.nv, self.bias) if h is not None else np.zeros(3 + self.nv + 66, np.float32)
            tk = np.asarray(h["token"], np.float32) if h is not None and h.get("token") is not None else np.zeros(self.tok_dim, np.float32)
            self.buf[side] = (self.buf[side] + [(f, tk)])[-self.T:]
            if h is None:
                self.missing[side] += 1
                self.prev[side] = None
                continue
            if self.missing[side] >= self.T:  # back after >= T frames without the hand: a new track (flag and window restart;
                self.state[side], self.buf[side] = False, self.buf[side][-1:]  # the window held only zero rows: same input)
            self.missing[side] = 0
            pad = self.T - len(self.buf[side])
            x = np.concatenate([np.zeros((pad, len(f)), np.float32), np.stack([b[0] for b in self.buf[side]])])
            x = torch.as_tensor(x)[None].to(self.dev)
            tok = None
            if self.use_tok:
                tok = np.concatenate([np.zeros((pad, self.tok_dim), np.float32), np.stack([b[1] for b in self.buf[side]])])
                tok = torch.as_tensor(tok)[None].to(self.dev)
            lg, fg = self._forward(x, tok)
            p2 = 1 / (1 + np.exp(-lg[0, -1].double().cpu().numpy()))
            p1 = 1 / (1 + np.exp(-(logit(h["contact_dense"]) + self.bias)))  # the image head's uncalibrated dense map
            pb = (1 - self.alpha) * np.clip(self.R @ p1, 0, 1) + self.alpha * np.clip(self.R @ p2, 0, 1)
            pf = float(torch.sigmoid(fg[0, -1]))
            pe = 0.5 * (pf + self.prev[side]) if self.smooth == 2 and self.prev[side] is not None else pf
            self.prev[side] = pf
            st = pe > self.h_on if not self.state[side] else pe >= self.h_off
            self.state[side] = st
            h["contact_video"] = (1 / (1 + np.exp(-(logit(pb) - logit(self.thr))))).astype(np.float32)
            h["contact_score_video"], h["contact_flag_video"] = pf, bool(st)
        return hands
