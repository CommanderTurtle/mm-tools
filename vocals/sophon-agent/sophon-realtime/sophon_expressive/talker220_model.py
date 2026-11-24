# Copyright 2026 Adventists AI. Apache-2.0.
"""Sophon-Realtime E2E talker: streaming reply text (+ sound-event marks) -> vector frames. Inference only."""
import os, json, math
import torch, torch.nn as nn, torch.nn.functional as F
E_ = os.environ.get
D_TXT, D_V, d, L_DEC, L_ENC, H = 2560, 5120, int(E_("S_D", "1024")), int(E_("S_LDEC", "12")), int(E_("S_LENC", "4")), 16
LA = int(E_("S_LA", "3")); NADV = 4
LANG = {"zh": 0, "en": 1, "ja": 2}

def sinus(n, dd, dev):
    pos = torch.arange(n, device=dev, dtype=torch.float32)[:, None]; i = torch.arange(0, dd, 2, device=dev, dtype=torch.float32)
    a = pos / (10000 ** (i / dd)); pe = torch.zeros(n, dd, device=dev); pe[:, 0::2] = a.sin(); pe[:, 1::2] = a.cos(); return pe

FM_K = int(E_("S_FMK", "1024")); FW = int(E_("S_FW", "768")); FNB = int(E_("S_FNB", "4")); FNSTEP = int(E_("S_FNSTEP", "8"))
class FBlk(nn.Module):
    def __init__(s):
        super().__init__(); s.ln = nn.LayerNorm(FW, elementwise_affine=False); s.ada = nn.Linear(FW, 3 * FW); s.mlp = nn.Sequential(nn.Linear(FW, 4 * FW), nn.GELU(), nn.Linear(4 * FW, FW))
        nn.init.zeros_(s.ada.weight); nn.init.zeros_(s.ada.bias)
    def forward(s, h, c):
        sc, sh, g = s.ada(c).chunk(3, -1); return h + (1 + g) * s.mlp(s.ln(h) * (1 + sc) + sh)
class FMHead(nn.Module):
    def __init__(s):
        super().__init__(); s.K = FM_K; s.xin = nn.Linear(s.K, FW); s.cin = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, FW)); s.tin = nn.Sequential(nn.Linear(256, FW), nn.SiLU(), nn.Linear(FW, FW))
        s.blk = nn.ModuleList([FBlk() for _ in range(FNB)]); s.lno = nn.LayerNorm(FW, elementwise_affine=False); s.ado = nn.Linear(FW, 2 * FW); s.out = nn.Linear(FW, s.K)
        nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias); nn.init.zeros_(s.ado.weight); nn.init.zeros_(s.ado.bias)
        s.register_buffer("mu", torch.zeros(D_V)); s.register_buffer("U", torch.eye(D_V)); s.register_buffer("ws", torch.ones(D_V))
    def to_z(s, x): return ((x.float() - s.mu) @ s.U) / s.ws
    def from_z(s, z): return (z * s.ws) @ s.U.T + s.mu
    def temb(s, t):
        f = torch.exp(-math.log(10000) * torch.arange(128, device=t.device) / 128); a = t[:, None] * 1000 * f[None]; return torch.cat([a.sin(), a.cos()], -1)
    def forward(s, x, t, y):
        c = F.silu(s.cin(y) + s.tin(s.temb(t))); h = s.xin(x)
        for b in s.blk: h = b(h, c)
        sc, sh = s.ado(c).chunk(2, -1); return s.out(s.lno(h) * (1 + sc) + sh)
    @torch.no_grad()
    def sample(s, y, temp, zrest):
        x = torch.randn(y.shape[0], s.K, device=y.device) * temp
        for i in range(FNSTEP):
            t = torch.full((y.shape[0],), i / FNSTEP, device=y.device)
            with torch.autocast("cuda", dtype=torch.bfloat16): v = s(x, t, y).float()
            x = x + v / FNSTEP
        z = zrest.float().clone(); z[:, :s.K] = x; return s.from_z(z)

class S220(nn.Module):
    def __init__(s, emb):
        super().__init__()
        s.register_buffer("emb", emb.to(torch.bfloat16), persistent=False)
        s.tin = nn.Sequential(nn.Linear(D_TXT, d), nn.LayerNorm(d)); s.lang = nn.Embedding(3, d); s.reg = nn.Embedding(4, d)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, H, 4 * d, dropout=0.1, batch_first=True, norm_first=True), L_ENC)
        s.vin = nn.Sequential(nn.Linear(D_V, d), nn.LayerNorm(d)); s.bos = nn.Parameter(torch.randn(d) * 0.02)
        s.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, H, 4 * d, dropout=0.1, batch_first=True, norm_first=True), L_DEC)
        s.ln = nn.LayerNorm(d); s.out = nn.Linear(d, D_V); s.stop = nn.Linear(d, 1); s.adv = nn.Linear(d, NADV)
        s.register_buffer("mu", torch.zeros(D_V)); s.register_buffer("sd", torch.ones(D_V)); s.fm = FMHead()
        s.ev_up = nn.Sequential(nn.Linear(EVD_, 256), nn.GELU(), nn.Linear(256, d)); s.para_up = nn.Sequential(nn.Linear(PD, 256), nn.GELU(), nn.Linear(256, d)); s.lead_up = nn.Linear(EVD_, d)
        s.cin = nn.Sequential(nn.Linear(D_V, d), nn.LayerNorm(d)); s.ctx_kind = nn.Embedding(2, d)
        s.ctx_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, H, 4 * d, dropout=0.1, batch_first=True, norm_first=True), 2); s.ctx_out = nn.Linear(d, d)
        for z in (s.ev_up[2], s.para_up[2], s.lead_up, s.ctx_out): nn.init.zeros_(z.weight); nn.init.zeros_(z.bias)
        s.cond = None; s._off = 0; s._cpad = None
    def memory(s, ids, ipad, lang):
        x = s.tin(F.layer_norm(s.emb[ids].float(), (D_TXT,))) + s.lang(lang)[:, None]
        c = s.cond or {}
        if c.get("evv") is not None:
            G_ = float(E_("S_EVGAIN", "1")); ev_ = c["evv"].float(); e1 = s.ev_up(ev_)
            GS_ = float(E_("S_SPGAIN", str(G_)))
            if G_ != 1 or GS_ != 1:
                e0_ = ev_.clone(); e0_[..., :EV_NC + 2] = 0; e0 = s.ev_up(e0_); ep_ = ev_.clone(); ep_[..., EV_NC:EV_NC + 2] = 0; es_ = ev_.clone(); es_[..., :EV_NC] = 0
                e1 = e0 + G_ * (s.ev_up(ep_) - e0) + GS_ * (s.ev_up(es_) - e0)
            x = x + e1
        if c.get("lead") is not None and float(E_("S_EVGAIN", "1")) != 1: c["lead"] = c["lead"] * float(E_("S_EVGAIN", "1"))
        x = x + sinus(x.shape[1], d, x.device); tm = s.enc(x, src_key_padding_mask=ipad)
        s._off, s._cpad = 0, None
        if c.get("ctx") is not None:
            cx = torch.where(c["ckind"][:, None, None] == 0, s.cin(((c["ctxv"].float() - s.mu) / s.sd)), s.tin(F.layer_norm(s.emb[c["ctxt"]].float(), (D_TXT,))))
            cx = cx + s.ctx_kind(c["ckind"])[:, None] + sinus(cx.shape[1], d, cx.device)
            cm = s.ctx_out(s.ctx_enc(cx, src_key_padding_mask=c["cpad"])) * c["cuse"][:, None, None].to(cx.dtype)
            s._off, s._cpad = cm.shape[1], c["cpad"]; tm = torch.cat([cm, tm], 1)
        return tm

EV_NC = 9; EVD_ = EV_NC + 3; PD = 15
def make_cond(its, Li, dev, train=False):
    """Conditions: para [B,15] + pm; evv [B,Li,12]; lead [B,12]; context. None when empty."""
    B = len(its); c = {}
    pm = torch.tensor([float(x.get("para") is not None and (not train or True)) for x in its])
    if pm.any(): c["para"] = torch.stack([x["para"].float() if x.get("para") is not None else torch.zeros(PD) for x in its]).to(dev); c["pm"] = pm.to(dev)
    ue = [x.get("evv") is not None and (not train or True) for x in its]
    if any(ue):
        ev = torch.zeros(B, Li, EVD_)
        for j, x in enumerate(its):
            if ue[j]: ev[j, :len(x["evv"])] = x["evv"].float()
        c["evv"] = ev.to(dev)
        if any(ue[j] and x.get("lead") is not None for j, x in enumerate(its)):
            c["lead"] = torch.stack([x["lead"].float() if (ue[j] and x.get("lead") is not None) else torch.zeros(EVD_) for j, x in enumerate(its)]).to(dev)
    uc = [(x.get("ctxv") is not None or x.get("ctxt") is not None) and (not train or True) for x in its]
    if any(uc):
        Lc = max(max((len(x["ctxv"]) if x.get("ctxv") is not None else len(x["ctxt"])) for j, x in enumerate(its) if uc[j]), 1)
        cv = torch.zeros(B, Lc, D_V, dtype=torch.float16); ct = torch.zeros(B, Lc, dtype=torch.long); kd = torch.zeros(B, dtype=torch.long); cp = torch.ones(B, Lc, dtype=torch.bool)
        for j, x in enumerate(its):
            if not uc[j]: cp[j, 0] = False; continue
            if x.get("ctxv") is not None: n_ = len(x["ctxv"]); cv[j, :n_] = x["ctxv"]; cp[j, :n_] = False
            else: n_ = len(x["ctxt"]); ct[j, :n_] = x["ctxt"].long(); kd[j] = 1; cp[j, :n_] = False
        c.update(ctx=True, ctxv=cv.to(dev), ctxt=ct.to(dev), ckind=kd.to(dev), cpad=cp.to(dev), cuse=torch.tensor([float(u) for u in uc]).to(dev))
    return c or None


P_ID = 151650   # placeholder token that carries a timed sound event
def strip_sp(ch, mode=1):
    """Drop the spaces that tags leave between CJK characters (mode 1: around <strong> only; mode 2: all)."""
    out = []
    for i, c in enumerate(ch):
        if c["c"] == " " and not c["a"] and out and i + 1 < len(ch) and (ord(out[-1]["c"]) > 0x2E7F or ord(ch[i + 1]["c"]) > 0x2E7F):
            if mode == 2 or out[-1]["sp"][1] or ch[i + 1]["sp"][1]: continue
        out.append(c)
    return out

def load(model_dir, dev="cuda"):
    """model_dir/talker220: model.safetensors, embed.safetensors, tokenizer files. Returns (model, tokenizer)."""
    import transformers
    from safetensors.torch import load_file
    d_ = os.path.join(model_dir, "talker220")
    m = S220(load_file(os.path.join(d_, "embed.safetensors"))["emb"]).to(dev).eval(); m.load_state_dict(load_file(os.path.join(d_, "model.safetensors")), strict=False)
    return m, transformers.AutoTokenizer.from_pretrained(d_)
