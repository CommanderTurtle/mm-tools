# Copyright 2026 Adventists AI. Apache-2.0.
"""220M talker inference engine: one CUDA-Graph replay per frame for all sessions (decoder + flow-matching head + reading cursor), preallocated KV caches; sessions join / leave at any time."""
import os, torch, torch.nn.functional as F
import talker220_model as S
from talker220_model import sinus

class Engine:
    def __init__(s, m, B, Tmax=2048, Smax=1024, stall_n=20, tail=2, dt=torch.bfloat16):
        s.m = m; s.L = list(m.dec.layers); s.H = S.H; s.d = S.d; s.dh = S.d // S.H; s.B = B; s.Tmax = Tmax; s.Smax = Smax; s.dt = dt; dev = m.bos.device
        s.stall_n, s.tail = stall_n, tail
        nl = len(s.L)
        s.sk = torch.zeros(nl, B, s.H, Tmax, s.dh, device=dev, dtype=dt); s.sv = torch.zeros_like(s.sk)
        s.ck = torch.zeros(nl, B, s.H, Smax, s.dh, device=dev, dtype=dt); s.cv = torch.zeros_like(s.ck)
        s.mpad = torch.ones(B, Smax, dtype=torch.bool, device=dev)
        s.off = torch.zeros(B, dtype=torch.long, device=dev)
        s.n = torch.ones(B, dtype=torch.long, device=dev)
        s.g = torch.zeros(B, s.d, device=dev, dtype=torch.float32)
        s.reg = torch.full((B,), 2, dtype=torch.long, device=dev)
        s.SIN = sinus(Tmax + 1, s.d, dev).float()

        s.pos = torch.zeros(B, dtype=torch.long, device=dev); s.fprev = torch.zeros(B, S.D_V, device=dev); s.first = torch.ones(B, device=dev)
        s.cur = torch.zeros(B, dtype=torch.long, device=dev); s.cf = torch.zeros(B, device=dev); s.stall = torch.zeros(B, device=dev); s.lastc = torch.zeros(B, dtype=torch.long, device=dev)
        s.tend = torch.full((B,), 10 ** 9, dtype=torch.long, device=dev); s.done = torch.zeros(B, dtype=torch.bool, device=dev)
        s.slim = torch.full((B, Smax), float(stall_n), device=dev); s.Smax = Smax
        s.fout = torch.zeros(B, S.D_V, device=dev); s.alive = torch.zeros(B, dtype=torch.bool, device=dev)
        s.arT = torch.arange(Tmax, device=dev); s.arS = torch.arange(Smax, device=dev); s.graph = None
    def _h(s, x): B, T, _ = x.shape; return x.view(B, T, s.H, s.dh).transpose(1, 2)
    @torch.no_grad()
    def load(s, j, mem, ipad_n, off, g=None):
        """Load slot j: mem [Sj,d] (text memory incl. context), off = context length, ipad_n = text tokens."""
        Sj = mem.shape[0]; assert Sj <= s.Smax
        for i, l in enumerate(s.L):
            W, b = l.multihead_attn.in_proj_weight, l.multihead_attn.in_proj_bias
            k = F.linear(mem.float(), W[s.d:2 * s.d], b[s.d:2 * s.d]); v = F.linear(mem.float(), W[2 * s.d:], b[2 * s.d:])
            s.ck[i, j, :, :Sj] = k.view(Sj, s.H, s.dh).transpose(0, 1).to(s.dt); s.cv[i, j, :, :Sj] = v.view(Sj, s.H, s.dh).transpose(0, 1).to(s.dt)
        s.mpad[j] = True; s.mpad[j, :Sj] = False; s.off[j] = off; s.n[j] = ipad_n
        s.g[j] = g.float() if g is not None else 0
    @torch.no_grad()
    def reset_slot(s, j):
        s.pos[j] = 0; s.fprev[j] = 0; s.first[j] = 1; s.cur[j] = 0; s.cf[j] = 0; s.stall[j] = 0; s.lastc[j] = 0; s.tend[j] = 10 ** 9; s.done[j] = False
    @torch.no_grad()
    def reset(s):
        s.pos.zero_(); s.fprev.zero_(); s.first.fill_(1); s.cur.zero_(); s.cf.zero_(); s.stall.zero_(); s.lastc.zero_(); s.tend.fill_(10 ** 9); s.done.zero_()
    @torch.no_grad()
    def _step(s):
        m = s.m; B = s.B; d = s.d

        x0 = (m.bos + m.reg(s.reg)).float(); x1 = m.vin(((s.fprev - m.mu) / m.sd)).float()
        x = torch.where(s.first[:, None] > 0, x0, x1) + s.SIN.index_select(0, s.pos.clamp(max=s.Tmax - 1)) + s.g
        x = x[:, None].to(s.dt)

        sbias = torch.zeros(B, 1, 1, s.Tmax, device=x.device, dtype=s.dt).masked_fill((s.arT[None] > s.pos[:, None])[:, None, None], float("-inf"))
        tix = s.arS[None] - s.off[:, None]
        mt = (tix > (s.cur[:, None] + S.LA)) & (tix > 0); mt = mt | s.mpad
        cbias = torch.zeros(B, 1, 1, s.Smax, device=x.device, dtype=s.dt).masked_fill(mt[:, None, None], float("-inf"))
        for i, l in enumerate(s.L):
            h = l.norm1(x); W, b = l.self_attn.in_proj_weight, l.self_attn.in_proj_bias
            q, k, v = F.linear(h, W.to(s.dt), b.to(s.dt)).chunk(3, -1); q, k, v = s._h(q), s._h(k), s._h(v)
            ix = s.pos.clamp(max=s.Tmax - 1).view(B, 1, 1, 1).expand(B, s.H, 1, s.dh); s.sk[i].scatter_(2, ix, k); s.sv[i].scatter_(2, ix, v)
            a = F.scaled_dot_product_attention(q, s.sk[i], s.sv[i], attn_mask=sbias); x = x + l.self_attn.out_proj(a.transpose(1, 2).reshape(B, 1, d))
            h = l.norm2(x); Wc, bc = l.multihead_attn.in_proj_weight, l.multihead_attn.in_proj_bias
            q = s._h(F.linear(h, Wc[:d].to(s.dt), bc[:d].to(s.dt)))
            a = F.scaled_dot_product_attention(q, s.ck[i], s.cv[i], attn_mask=cbias); x = x + l.multihead_attn.out_proj(a.transpose(1, 2).reshape(B, 1, d))
            h = l.norm3(x); x = x + l.linear2(l.activation(l.linear1(h)))
        y = m.ln(x)[:, 0]; out = m.out(y).float(); st = m.stop(y)[:, 0].float(); av = m.adv(y).float()
        f = out * m.sd + m.mu; f = m.fm.sample(y.float(), 0.0, m.fm.to_z(f))
        t = s.pos
        stop_now = (s.cur >= s.n - 1) & ((torch.sigmoid(st) > 0.5) | (t - s.tend >= s.tail))
        s.done.copy_(s.done | stop_now); s.alive.copy_(~s.done); s.fout.copy_(f)
        pa = torch.softmax(av, -1); cf = s.cf + (pa * torch.arange(S.NADV, device=x.device)).sum(-1)
        c = torch.minimum(s.n - 1, cf.long()); same = c == s.lastc
        stall = torch.where(same, s.stall + 1, torch.zeros_like(s.stall)); lastc = torch.where(same, s.lastc, c)
        force = (stall >= s.slim.gather(1, c.clamp(max=s.Smax - 1)[:, None])[:, 0]) & (c < s.n - 1); c = torch.where(force, c + 1, c); cf = torch.where(force, c.float(), cf)
        stall = torch.where(force, torch.zeros_like(stall), stall); lastc = torch.where(force, c, lastc)
        tend = torch.where((c >= s.n - 1) & (s.tend > 10 ** 8), t, s.tend)
        s.cf.copy_(cf); s.cur.copy_(c); s.stall.copy_(stall); s.lastc.copy_(lastc); s.tend.copy_(tend)
        live = (~s.done).to(s.pos.dtype); s.fprev.copy_(f); s.first.mul_(1 - live.float()); s.pos.add_(live)
    def capture(s):
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        snap = [x.clone() for x in (s.pos, s.fprev, s.first, s.cur, s.cf, s.stall, s.lastc, s.tend, s.done)]; s.done.fill_(False)
        with torch.cuda.stream(st), torch.autocast("cuda", dtype=s.dt):
            for _ in range(2): s._step()
        torch.cuda.current_stream().wait_stream(st)
        for x, y in zip((s.pos, s.fprev, s.first, s.cur, s.cf, s.stall, s.lastc, s.tend, s.done), snap): x.copy_(y)
        s.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(s.graph), torch.autocast("cuda", dtype=s.dt): s._step()
        s.reset()
    def step(s):
        if s.graph is None: s.capture()
        s.graph.replay()

@torch.no_grad()
def prep(m, tok, texts, langs, conds=None):
    """Text -> memory (with context) and global condition. Returns list[(mem[Sj,d], n_text, off, g)]."""
    dev = m.bos.device; IM = tok.convert_tokens_to_ids("<|im_end|>"); res = []
    for j, (t, lg) in enumerate(zip(texts, langs)):
        c = (conds or [None] * len(texts))[j] or {}; it = {k: v for k, v in c.items() if v is not None and k != "ids"}
        ids = torch.tensor([list(c["ids"]) if c.get("ids") is not None else tok(t, add_special_tokens=False).input_ids + [IM]], device=dev); ipad = torch.zeros_like(ids, dtype=torch.bool)
        if "ctxt" in it: it["ctxt"] = torch.tensor(it["ctxt"])
        n_real = ids.shape[1]; PADN = int(os.environ.get("S220_PREP_PAD", "0"))
        if PADN > 0 and n_real % PADN:
            Lp = (n_real // PADN + 1) * PADN; ids = torch.cat([ids, torch.full((1, Lp - n_real), IM, device=dev, dtype=ids.dtype)], 1); ipad = torch.zeros_like(ids, dtype=torch.bool); ipad[:, n_real:] = True
        m.cond = S.make_cond([it], ids.shape[1], dev, train=False) if it else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=os.environ.get("S220_PREP_FP32", "0") != "1"):
            mem = m.memory(ids, ipad, torch.tensor([S.LANG.get(lg, 0)], device=dev)); g = None; cd = m.cond or {}
            if cd.get("para") is not None: g = m.para_up(cd["para"].float()) * cd["pm"][:, None].float()
            if cd.get("lead") is not None: g = (0 if g is None else g) + m.lead_up(cd["lead"].float())
        res.append((mem[0][:m._off + n_real], n_real, m._off, None if g is None else g[0])); m.cond = None
    return res

