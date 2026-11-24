# Copyright 2026 Adventists AI. Apache-2.0.
"""Left-context key/value cache for the E2 flow estimator."""
import torch, torch.nn.functional as F
from x_transformers.x_transformers import apply_rotary_pos_emb
class SL(list): pass
_PKV = {}
def est_new(e, xt, mt, ct, sp, t, L, KV, T):
    R = xt.shape[0]; tt = e.time_embed(t.repeat(R))
    h = e.input_embed(xt.transpose(1, 2), ct.transpose(1, 2), mt.transpose(1, 2), sp)[:, -L:]
    fr, _ = e.rotary_embed.forward_from_seq_len(T); res = h; ks, vs = [], []
    Tc = 0 if KV is None else KV.shape[3]; f = fr[:, Tc:Tc + L]; fc = fr[:, :Tc]
    for l, b in enumerate(e.transformer_blocks):
        norm, g_msa, sh, sc, g_mlp = b.attn_norm(h, emb=tt); a = b.attn
        q = a.to_q(norm); k = a.to_k(norm); v = a.to_v(norm); ks.append(k); vs.append(v)
        qr = apply_rotary_pos_emb(q, f); kr = apply_rotary_pos_emb(k, f)
        if Tc: K = torch.cat([apply_rotary_pos_emb(KV[l, 0], fc), kr], 1); V = torch.cat([KV[l, 1], v], 1)
        else: K, V = kr, v
        H = a.heads; hd = K.shape[-1] // H
        o = F.scaled_dot_product_attention(qr.view(R, L, H, hd).transpose(1, 2), K.view(R, -1, H, hd).transpose(1, 2), V.view(R, -1, H, hd).transpose(1, 2))
        o = a.to_out[1](a.to_out[0](o.transpose(1, 2).reshape(R, L, H * hd)))
        h = h + g_msa.unsqueeze(1) * o
        h = h + g_mlp.unsqueeze(1) * b.ff(b.ff_norm(h) * (1 + sc[:, None]) + sh[:, None])
    if e.long_skip_connection is not None: h = e.long_skip_connection(torch.cat((h, res), dim=-1))
    h = e.norm_out(h, tt); return e.proj_out(h).transpose(1, 2), torch.stack([torch.stack(ks), torch.stack(vs)], 1)
def rows(s, x, mu, cond):
    if s.cfg == 0: return x, mu, cond, s.spk
    return torch.cat([x, x]), torch.cat([mu, torch.zeros_like(mu)]), torch.cat([cond, torch.zeros_like(cond)]), torch.cat([s.spk, torch.zeros_like(s.spk)])
def comb(s, v): return v if s.cfg == 0 else (1 + s.cfg) * v[:1] - s.cfg * v[1:]
@torch.no_grad()
def prompt_kv(s):
    key = (id(s.pstates), float(s.cfg), s.nt)
    if key not in _PKV:
        out = []
        for k in range(s.nt):
            x, mu, c, sp = rows(s, s.pstates[k], s.mur, s.mr)
            _, kv = est_new(s.est, x, mu, c, sp, s.ts[k:k + 1], s.P, None, s.P); out.append(kv)
        _PKV[key] = out
    s.pkv = _PKV[key]; return s.pkv
def ok(s, L, ctx):
    nt1 = getattr(s, "nt_once", None)
    return L == s.ch and not (nt1 and nt1 < s.nt) and all(getattr(st, "kv", None) is not None and len(st.kv) >= s.nt for st, _ in ctx)
def start(s, mu_new):
    L = mu_new.shape[2]; ctx = s.chunks[-s.left:]
    NN = s.noise.shape[2]; off = s.P + (s.n * s.ch + getattr(s, 'noff', 0)) % max(s.ch, NN - s.P - s.ch); s.n += 1; s.nt_once = None
    x = s.noise[:, :, off: off + L].float(); states = SL([x]); states.kv = []
    mu_ctx = torch.cat([s.mur] + [m for _, m in ctx] + [mu_new.float()], 2); cond = torch.zeros_like(mu_ctx); cond[:, :, : s.P] = s.mr
    if getattr(s, "pkv", None) is None: prompt_kv(s)
    return dict(L=L, ctx=ctx, x=x, states=states, mu_ctx=mu_ctx, cond=cond, mu=mu_new)
def inputs(s, c, k):
    T = c["mu_ctx"].shape[2]; LT = min(T, c["L"] + 64)
    xs = torch.cat([s.pstates[k]] + [st[k] for st, _ in c["ctx"]] + [c["x"]], 2)
    xr, mr_, cr, sp = rows(s, xs[:, :, -LT:], c["mu_ctx"][:, :, -LT:], c["cond"][:, :, -LT:])
    KV = torch.cat([s.pkv[k]] + [st.kv[k] for st, _ in c["ctx"]], 3)
    return xr, mr_, cr, sp, KV, T
def finish(s, c):
    s.chunks.append((c["states"], c["mu"].float()))
    if len(s.chunks) > s.left + 2: s.chunks = s.chunks[-(s.left + 2):]
    return c["x"]
@torch.no_grad()
def gen_kvc(s, mu_new):
    L = mu_new.shape[2]; ctx = s.chunks[-s.left:]
    if not ok(s, L, ctx):
        r = s.gen0_kvc(mu_new); st, m = s.chunks[-1]
        if not isinstance(st, SL): st = SL(st); s.chunks[-1] = (st, m)
        st.kv = None; return r
    c = start(s, mu_new)
    for k in range(s.nt):
        xr, mr_, cr, sp, KV, T = inputs(s, c, k); dt = s.ts[k + 1] - s.ts[k]
        v, kv = est_new(s.est, xr, mr_, cr, sp, s.ts[k:k + 1], c["L"], KV, T)
        c["states"].kv.append(kv); c["x"] = c["x"] + dt * comb(s, v); c["states"].append(c["x"])
    return finish(s, c)
def enable():
    import e2v2_stream as ES
    if not hasattr(ES.Stream, "gen0_kvc"): ES.Stream.gen0_kvc = ES.Stream.gen; ES.Stream.gen = gen_kvc
