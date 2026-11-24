# Copyright 2026 Adventists AI. Apache-2.0.
"""Streaming E2 renderer (chunked causal flow matching with left context)."""
import torch
_GR = {}; _PS = {}
class Stream:
    def __init__(s, dec, mur, mr, spk, ch=16, left=12, nt=10, cfg=None):
        s.dec, s.est = dec, dec.estimator; s.ch, s.left, s.nt = ch, left, nt
        import os as _o0
        s.cfg = (float(_o0.environ["E2V2_CFG"]) if _o0.environ.get("E2V2_CFG", "") != "" else dec.inference_cfg_rate) if cfg is None else cfg
        dev = mur.device; s.dt_ = torch.float32
        ts = torch.linspace(0, 1, nt + 1, device=dev); s.ts = 1 - torch.cos(ts * 0.5 * torch.pi)
        s.noise = dec.rand_noise.to(dev); s.spk = spk.float()
        s.P = mur.shape[2]; s.mur, s.mr = mur.float(), mr.float()
        pk = (mur.data_ptr(), tuple(round(float(x), 4) for x in s.spk.flatten()[:8].tolist()), nt, float(s.cfg))
        if pk not in _PS: _PS[pk] = s._run(s.noise[:, :, : s.P].float(), s.mur, s.mr, [], None)
        s.pstates = _PS[pk]
        s.chunks = []; s.n = 0
    def _graph_for(s, T, P, dev):
        NB = 1 if s.cfg == 0 else 2
        key = (T, P, s.left, s.ch, str(dev), NB); s._graphs = _GR
        if key in s._graphs: return s._graphs[key]
        g = dict(x=torch.zeros(NB, 80, T, device=dev), mu=torch.zeros(NB, 80, T, device=dev), c=torch.zeros(NB, 80, T, device=dev), sp=torch.zeros(NB, 80, device=dev), t=torch.zeros(1, device=dev))
        i = torch.arange(T, device=dev); ci = torch.where(i < P, torch.full_like(i, -1), (i - P) // s.ch)
        vis = (ci[None, :] <= ci[:, None]) & (ci[None, :] >= ci[:, None] - s.left); vis[:, :P] = True; g["am"] = vis[None, None].expand(NB, 1, T, T).contiguous()
        m = torch.ones(NB, 1, T, device=dev)
        def body():
            s._t = g["t"]; return s._est(g["x"], m, g["mu"], None, g["sp"], g["c"], g["am"])
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(2): body()
        torch.cuda.current_stream().wait_stream(st)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr): g["out"] = body()
        g["graph"] = gr; s._graphs[key] = g; return g
    def _vel(s, x, mu, cond, P, nctx, L):
        T = x.shape[2]
        if getattr(s, "use_graph", True) and L == s.ch and P > 0 and x.is_cuda:
            g = s._graph_for(T, P, x.device)
            if s.cfg == 0:
                g["x"].copy_(x); g["mu"].copy_(mu); g["c"].copy_(cond); g["sp"].copy_(s.spk); g["t"].copy_(s._t); g["graph"].replay(); return g["out"][:, :, -L:].clone()
            g["x"].copy_(x.expand(2, -1, -1)); g["mu"][0].copy_(mu[0]); g["mu"][1].zero_(); g["c"][0].copy_(cond[0]); g["c"][1].zero_(); g["sp"][0].copy_(s.spk[0]); g["sp"][1].zero_(); g["t"].copy_(s._t)
            g["graph"].replay(); v = g["out"]; v = (1 + s.cfg) * v[:1] - s.cfg * v[1:]
            return v[:, :, -L:].clone()
        """x/mu/cond: [1,80,T]（提示 + 上下文块 + 新块）；只返回新块（最后 L 帧）的 CFG 速度"""
        T = x.shape[2]; dev = x.device
        i = torch.arange(T, device=dev); ci = torch.where(i < P, torch.full_like(i, -1), (i - P) // s.ch)
        vis = (ci[None, :] <= ci[:, None]) & (ci[None, :] >= ci[:, None] - s.left); vis[:, :P] = True
        if s.cfg == 0:
            return s._est(x, torch.ones(1, 1, T, device=dev), mu, None, s.spk, cond, vis[None, None])[:, :, -L:]
        amask = vis[None, None].expand(2, 1, T, T)
        x2 = x.expand(2, -1, -1); mu2 = torch.cat([mu, torch.zeros_like(mu)]); c2 = torch.cat([cond, torch.zeros_like(cond)]); sp2 = torch.cat([s.spk, torch.zeros_like(s.spk)])
        m = torch.ones(2, 1, T, device=dev)
        v = s._est(x2, m, mu2, None, sp2, c2, amask)
        v, vu = v[:1], v[1:]; v = (1 + s.cfg) * v - s.cfg * vu
        return v[:, :, -L:]
    def _est(s, x, mask, mu, t, spks, cond, amask):
        e = s.est; xx = x.transpose(1, 2); mm = mu.transpose(1, 2); cc = cond.transpose(1, 2)
        tt = e.time_embed(s._t.repeat(x.shape[0]))
        h = e.input_embed(xx, cc, mm, spks)
        rope = e.rotary_embed.forward_from_seq_len(h.shape[1])
        res = h
        for b in e.transformer_blocks: h = b(h, tt, mask=amask.bool(), rope=rope)
        if e.long_skip_connection is not None: h = e.long_skip_connection(torch.cat((h, res), dim=-1))
        h = e.norm_out(h, tt); return e.proj_out(h).transpose(1, 2)
    def _run(s, x0, mu, cond_self, ctx, nP):
        states = [x0]; x = x0
        for k in range(1, s.nt + 1):
            s._t = s.ts[k - 1:k]; dt = s.ts[k] - s.ts[k - 1]
            v = s._vel(x, mu, cond_self, x.shape[2], 0, x.shape[2]) if not ctx else None
            x = x + dt * v; states.append(x)
        return states
    @torch.no_grad()
    def gen(s, mu_new):
        L = mu_new.shape[2]; NN = s.noise.shape[2]; off = s.P + (s.n * s.ch + getattr(s, 'noff', 0)) % max(s.ch, NN - s.P - s.ch); s.n += 1
        x = s.noise[:, :, off: off + L].float(); ctx = s.chunks[-s.left:]; states = [x]
        mu_ctx = torch.cat([s.mur] + [m for _, m in ctx] + [mu_new.float()], 2)
        cond = torch.zeros_like(mu_ctx); cond[:, :, : s.P] = s.mr
        nt1 = getattr(s, "nt_once", None); s.nt_once = None
        sd = (s.nt // nt1) if (nt1 and nt1 < s.nt and s.nt % nt1 == 0) else 1
        for kk in range(1, s.nt // sd + 1):
            k0, k = (kk - 1) * sd, kk * sd
            s._t = s.ts[k0:k0 + 1]; dt = s.ts[k] - s.ts[k0]
            xs = torch.cat([s.pstates[k0]] + [st[k0] for st, _ in ctx] + [x], 2)
            v = s._vel(xs, mu_ctx, cond, s.P, len(ctx), L)
            xn = x + dt * v
            for j in range(1, sd): states.append(x + (xn - x) * j / sd)
            x = xn; states.append(x)
        s.chunks.append((states, mu_new.float()))
        if len(s.chunks) == 2 and s.chunks[0][1].shape[2] + s.chunks[1][1].shape[2] == s.ch and s.chunks[0][1].shape[2] < s.ch:
            (s0, m0), (s1, m1) = s.chunks; s.chunks = [([torch.cat([a_, b_], 2) for a_, b_ in zip(s0, s1)], torch.cat([m0, m1], 2))]
        if len(s.chunks) > s.left + 2: s.chunks = s.chunks[-(s.left + 2):]
        return x

    @torch.no_grad()
    def gen_safe(s, mu_new, tries=2, hf_th=-1.5, max_th=2.3):
        import os as _o2
        mel = s._gen_safe0(mu_new, tries, hf_th, max_th)
        if _o2.environ.get("E2V2_HFFLOOR_OUT", "") != "": mel = mel.clone(); mel[:, 72:] = float(_o2.environ["E2V2_HFFLOOR_OUT"])
        return mel
    def _gen_safe0(s, mu_new, tries=2, hf_th=-1.5, max_th=2.3):
        mel = s.gen(mu_new); s.n_retry = getattr(s, "n_retry", 0); s.n_bad = getattr(s, "n_bad", 0)
        import os as _os
        if _os.environ.get("E2V2_HFFLOOR", "") != "":
            fl = float(_os.environ["E2V2_HFFLOOR"])
            for _ in range(tries):
                if float(mel[:, :72].max()) <= max_th: break
                s.n -= 1; s.chunks.pop(); s.noff = getattr(s, "noff", 0) + 7919; s.n_retry += 1; mel = s.gen(mu_new)
            mel = mel.clone(); mel[:, 72:] = fl
            if float(mel.max()) > max_th: s.n_bad += 1; mel = mel.clamp(max=1.8)
            if _os.environ.get("E2V2_HFFLOOR_CACHE", "0") == "1": st, m_ = s.chunks[-1]; st[-1] = mel.clone()
            return mel
        for _ in range(tries):
            if float(mel[:, 72:].max()) <= hf_th and float(mel.max()) <= max_th: return mel
            s.n -= 1; s.chunks.pop(); s.noff = getattr(s, "noff", 0) + 7919; s.n_retry += 1; mel = s.gen(mu_new)
        if float(mel[:, 72:].max()) > hf_th or float(mel.max()) > max_th:
            s.n_bad += 1; mel = mel.clone(); mel[:, 72:] = mel[:, 72:].clamp(max=-6.0); mel = mel.clamp(max=1.8)
            st, m_ = s.chunks[-1]; st[-1] = mel.clone()
        return mel

import os as _okvc
if _okvc.environ.get("E2V2_KVC", "0") == "1":
    import e2v2_kvc as _KV; _KV.enable()
