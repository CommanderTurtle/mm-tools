# Copyright 2026 Adventists AI. Apache-2.0.
"""Sophon-Realtime v1 (Expressive) talker: inference-only model definition.

The talker reuses the thinker's own 32B layers (frozen) with a LoRA that is active only on the talker stream, and adds
small heads (decision, regression, flow) that write the next vector frame. Weights: <model>/base (shared) and
<model>/talker/model.safetensors (LoRA + heads)."""
import os, sys, math, json, re, copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import transformers
from safetensors import safe_open
from . import paths

E_ = os.environ.get
MD = paths.base_dir()


def _sd(prefix):
    m = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]; out = {}
    for f in sorted(set(v for k, v in m.items() if k.startswith(prefix))):
        with safe_open(f"{MD}/{f}", "pt") as fh:
            for k in fh.keys():
                if k.startswith(prefix): out[k[len(prefix):]] = fh.get_tensor(k)
    return out


def _chat(msgs, add_generation_prompt=True):
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
    if add_generation_prompt: s += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return s


FR = 0.16; KF, KA, KS, KSTY = 0, 1, 2, 3
PAD_ID = 151643
tc = json.load(open(f"{MD}/config.json"))["text_config"]
D, HQ, HKV, HD, NL = tc["hidden_size"], tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"], tc["num_hidden_layers"]
THETA = float(tc.get("rope_theta", 1e6))
TOKZ = transformers.AutoTokenizer.from_pretrained(MD)
W, NB, NSTEP = 1536, 8, 16
LORA_R, LORA_A = 256, 512.0
FM_K = int(E_("FM_K", "1024")); REGEN = True; STALL = int(E_("STALL", "10"))
EVFIX = True; EV_LGAIN = True; EV_NC = 9; EVD_ = EV_NC + 3; PD = 15
SPK_MODE = "ctx"; HID = True; HID_LAYER = -1
TEMPV = ""; CFG_W = 0.0; GC = False; FSDP = False
_PM = {}

def _is_p(i):
    i = int(i)
    if i not in _PM: d_ = TOKZ.decode([i]); _PM[i] = ("\ufffd" not in d_) and re.search(r"\w", d_) is None and not d_.startswith("<|")
    return _PM[i]
def ev_shift(v, ids):
    if not EVFIX or v is None: return v
    v = v.clone(); n = min(len(ids), v.shape[0]); C = EV_NC
    for j in range(n - 1):
        if v[j, :C].sum() > 0 and _is_p(ids[j + 1]):
            k = j + 1
            while k + 1 < n and _is_p(ids[k + 1]): k += 1
            if v[k, :C].sum() == 0: v[k, :C] = v[j, :C]; v[j, :C] = 0
    return v
PROMPT_IDS = TOKZ(_chat([{"role": "user", "content": "请随便说一句话。"}]), add_special_tokens=False).input_ids
P = len(PROMPT_IDS)
_tpl = _chat([{"role": "user", "content": "XQXQ"}])
GEN_TAIL = TOKZ(_tpl.split("XQXQ<|im_end|>\n", 1)[1], add_special_tokens=False).input_ids
TURN_HEAD = {r: TOKZ(f"<|im_start|>{r}\n", add_special_tokens=False).input_ids for r in ("user", "assistant")}; TURN_END = TOKZ("<|im_end|>\n", add_special_tokens=False).input_ids

def qwen3_model(dtype):
    cfg = transformers.Qwen3Config(**{k: v for k, v in tc.items() if k not in ("architectures", "model_type")})
    with torch.device("meta"): m = transformers.Qwen3Model(cfg)
    sd = {k: v.to(dtype) for k, v in _sd("language_model.model.").items()}; miss = m.load_state_dict(sd, strict=False, assign=True)
    assert not [k for k in miss.missing_keys if "rotary" not in k], miss.missing_keys[:5]
    m.rotary_emb = type(m.rotary_emb)(config=cfg)
    for n_, b_ in m.named_buffers():
        assert b_.device.type != "meta", n_
    return m.to(dtype)
def rope_cs(pos, dev):
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, device=dev, dtype=torch.float32) / HD))
    f = pos.float()[..., None] * inv; e = torch.cat([f, f], -1); return e.cos()[:, None], e.sin()[:, None]
def rope(x, c, s):
    x1, x2 = x[..., : HD // 2], x[..., HD // 2:]; return (x.float() * c + torch.cat([-x2, x1], -1).float() * s).to(torch.bfloat16)
def layer_fwd(layer, x, c, s, Kc, Vc, mask):
    B, L, _ = x.shape; a = layer.self_attn
    h = layer.input_layernorm(x)
    q = a.q_norm(a.q_proj(h).view(B, L, HQ, HD)).transpose(1, 2)
    k = a.k_norm(a.k_proj(h).view(B, L, HKV, HD)).transpose(1, 2)
    v = a.v_proj(h).view(B, L, HKV, HD).transpose(1, 2).to(torch.bfloat16)
    q, k = rope(q, c, s), rope(k, c, s)
    K = k if Kc is None else torch.cat([Kc, k], 2); Vv = v if Vc is None else torch.cat([Vc, v], 2)
    o = F.scaled_dot_product_attention(q, K, Vv, attn_mask=mask, enable_gqa=True)
    x = x + a.o_proj(o.transpose(1, 2).reshape(B, L, HQ * HD))
    x = x + layer.mlp(layer.post_attention_layernorm(x))
    return x, k, v

class LoRA(nn.Module):
    ON = False
    def __init__(s, base, r, alpha):
        super().__init__(); s.base = base; s.scale = alpha / r
        s.lora_A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features)); s.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
    def forward(s, x):
        y = s.base(x)
        if LoRA.ON: y = y + (F.linear(F.linear(x, s.lora_A.to(x.dtype)), s.lora_B.to(x.dtype)) * s.scale).to(y.dtype)
        return y
def add_lora(model, r, alpha):
    for layer in model.layers:
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")), (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for nm in names:
                b = getattr(parent, nm); lo = LoRA(b, r, alpha).to(b.weight.device); _dt = torch.bfloat16 if E_("LORA_BF16", "0" if "LOCAL_RANK" in os.environ else "1") == "1" else torch.float32; lo.lora_A.data = lo.lora_A.data.to(_dt); lo.lora_B.data = lo.lora_B.data.to(_dt); setattr(parent, nm, lo)
def layer_fwd_t(layer, x, c, s, Kc, Vc, mask):
    LoRA.ON = True
    try: return layer_fwd(layer, x, c, s, Kc, Vc, mask)
    finally: LoRA.ON = False

class Blk(nn.Module):
    def __init__(s):
        super().__init__(); s.ln = nn.LayerNorm(W, elementwise_affine=False); s.ada = nn.Linear(W, 3 * W); s.mlp = nn.Sequential(nn.Linear(W, 4 * W), nn.GELU(), nn.Linear(4 * W, W))
        nn.init.zeros_(s.ada.weight); nn.init.zeros_(s.ada.bias)
    def forward(s, h, c):
        sc, sh, g = s.ada(c).chunk(3, -1); return h + (1 + g) * s.mlp(s.ln(h) * (1 + sc) + sh)
class FMHead(nn.Module):
    def __init__(s):
        super().__init__(); s.K = FM_K or D; s.xin = nn.Linear(s.K, W); s.cin = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, W)); s.tin = nn.Sequential(nn.Linear(256, W), nn.SiLU(), nn.Linear(W, W))
        s.blk = nn.ModuleList([Blk() for _ in range(NB)]); s.lno = nn.LayerNorm(W, elementwise_affine=False); s.ado = nn.Linear(W, 2 * W); s.out = nn.Linear(W, s.K)
        nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias); nn.init.zeros_(s.ado.weight); nn.init.zeros_(s.ado.bias)
        s.register_buffer("mu", torch.zeros(D)); s.register_buffer("sd", torch.ones(D))
        s.register_buffer("U", torch.eye(D)); s.register_buffer("ws", torch.ones(D)); s.register_buffer("white", torch.zeros(()))
        s.null_y = nn.Parameter(torch.zeros(D))
    def to_z(s, x):
        if float(s.white) > 0: return ((x.float() - s.mu) @ s.U) / s.ws
        return (x.float() - s.mu) / s.sd
    def from_z(s, z):
        if float(s.white) > 0: return (z * s.ws) @ s.U.T + s.mu
        return z * s.sd + s.mu
    def temb(s, t):
        f = torch.exp(-math.log(10000) * torch.arange(128, device=t.device) / 128); a = t[:, None] * 1000 * f[None]; return torch.cat([a.sin(), a.cos()], -1)
    def forward(s, x, t, y):
        c = F.silu(s.cin(y) + s.tin(s.temb(t))); h = s.xin(x)
        for b in s.blk: h = b(h, c)
        sc, sh = s.ado(c).chunk(2, -1); return s.out(s.lno(h) * (1 + sc) + sh)
    @torch.no_grad()
    def sample(s, y, temp=1.0, n=NSTEP, zrest=None):
        x = torch.randn(y.shape[0], s.K, device=y.device)
        if TEMPV:
            tv = torch.empty(s.K, device=y.device); a = 0
            for seg in TEMPV.split(","):
                k, tt = seg.split(":"); tv[a:int(k)] = float(tt); a = int(k)
            tv[a:] = tv[a - 1]; x = x * tv
        else: x = x * temp
        yn = s.null_y[None].expand_as(y).float() if CFG_W > 0 else None
        for i in range(n):
            t = torch.full((y.shape[0],), i / n, device=y.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = s(x, t, y).float()
                if yn is not None: v = v + CFG_W * (v - s(x, t, yn).float())
            x = x + v / n
        if s.K < D: z = zrest.float().clone(); z[:, :s.K] = x; x = z
        return s.from_z(x)
class RefEnc(nn.Module):
    def __init__(s, d=512, dz_tok=8, dz_utt=32):
        super().__init__(); s.inp = nn.Sequential(nn.Linear(D, d), nn.LayerNorm(d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 4 * d, dropout=0.1, batch_first=True, norm_first=True), 2)
        s.tok = nn.Linear(d, dz_tok); s.mu = nn.Linear(d, dz_utt); s.lv = nn.Linear(d, dz_utt)
    def forward(s, vec, vm):
        f = s.enc(s.inp(vec / 30.0), src_key_padding_mask=~vm); p = (f * vm[..., None]).sum(1) / vm.sum(1, keepdim=True).clamp(min=1)
        return s.tok(f), s.mu(p), s.lv(p).clamp(-8, 4)
class SpkEnc(nn.Module):
    def __init__(s, H=1024):
        super().__init__(); s.inp = nn.Sequential(nn.Linear(D, H), nn.LayerNorm(H)); s.q = nn.Parameter(torch.randn(4, H) * 0.02)
        s.att = nn.MultiheadAttention(H, 8, batch_first=True); s.out = nn.Linear(H, D); nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)
    def forward(s, v, vm):
        h = s.inp(v.float() / 30.0); ok = vm.any(1); vm2 = vm.clone(); vm2[~ok, 0] = True
        o, _ = s.att(s.q[None].expand(v.shape[0], -1, -1), h, h, key_padding_mask=~vm2); return s.out(o.mean(1)) * ok[:, None].float()
class TalkerLT(nn.Module):
    def __init__(s, TH):
        super().__init__(); s.__dict__["_lmh"] = TH.LMH
        s.layers = TH.m.layers; s.norm = copy.deepcopy(TH.m.norm).float()
        for p in s.norm.parameters(): p.requires_grad_(True)
        s.reg = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D))
        s.fm = FMHead(); s.dec = nn.Sequential(nn.Linear(D, 1024), nn.GELU(), nn.Linear(1024, 3)); s.ref = RefEnc()
        s.bos = nn.Parameter(torch.zeros(D)); s.zutt_up = nn.Linear(32, D); s.ztok_up = nn.Linear(8, D); s.spk_up = nn.Linear(192, D)
        for l in (s.zutt_up, s.ztok_up, s.spk_up): nn.init.zeros_(l.weight); nn.init.zeros_(l.bias)
        if SPK_MODE == "enc": s.spk_enc = SpkEnc()
        s.para_up = nn.Sequential(nn.Linear(PD, 512), nn.GELU(), nn.Linear(512, D)); nn.init.zeros_(s.para_up[2].weight); nn.init.zeros_(s.para_up[2].bias)
        if E_("PARA_INIT", "zero") == "small": nn.init.normal_(s.para_up[2].weight, std=float(E_("PARA_STD", "1e-3")))
        if HID: s.hid_up = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1024), nn.GELU(), nn.Linear(1024, D)); nn.init.zeros_(s.hid_up[3].weight); nn.init.zeros_(s.hid_up[3].bias)
        s.ev_up = nn.Sequential(nn.Linear(EVD_, 512), nn.GELU(), nn.Linear(512, D)); nn.init.normal_(s.ev_up[2].weight, std=1e-3); nn.init.zeros_(s.ev_up[2].bias)
        s.ev_gp = nn.Parameter(torch.tensor(math.log(float(E_("EV_GP0", "8"))))); s.ev_gs = nn.Parameter(torch.tensor(math.log(float(E_("EV_GS0", "1")))))
    def ev_embed(s, v):
        if not EV_LGAIN: return s.ev_up(v)
        vb = v.clone(); vb[..., :EV_NC + 2] = 0; eb = s.ev_up(vb)
        vp = v.clone(); vp[..., EV_NC:EV_NC + 2] = 0; vs = v.clone(); vs[..., :EV_NC] = 0
        return eb + s.ev_gp.exp() * (s.ev_up(vp) - eb) + s.ev_gs.exp() * (s.ev_up(vs) - eb)
    def llm_params(s): return [p for n, p in s.layers.named_parameters() if "lora_" in n] + list(s.norm.parameters())
    def head_params(s):
        ids = {id(p) for p in s.llm_params()}; return [p for n, p in s.named_parameters() if id(p) not in ids and not n.startswith("layers.")]
    def small_state(s): return {k: v for k, v in s.state_dict().items() if not k.startswith("layers.") or ".lora_" in k}
    def trunk(s, x, pos, Kc, Vc, mask, cache=None):
        c, sn = rope_cs(pos, x.device); new = []
        for l, layer in enumerate(s.layers):
            kc = Kc[l] if cache is None else torch.cat([Kc[l], cache[l][0]], 2) if cache[l] is not None else Kc[l]
            vc = Vc[l] if cache is None else torch.cat([Vc[l], cache[l][1]], 2) if cache[l] is not None else Vc[l]
            if GC and s.training: x, k, v = torch.utils.checkpoint.checkpoint(layer_fwd_t, layer, x, c, sn, kc, vc, mask, use_reentrant=False)
            else: x, k, v = layer_fwd_t(layer, x, c, sn, kc, vc, mask)
            new.append((k, v))
        return s.norm(x), new
class Thinker:
    def __init__(s, dev):
        m = qwen3_model(torch.bfloat16)
        if FSDP:
            _L = m.layers; del m._modules["layers"]; m.to(dev); m._modules["layers"] = _L
        else: m = m.to(dev)
        m.eval()
        for p in m.parameters(): p.requires_grad_(False)
        add_lora(m, LORA_R, LORA_A)
        s.LMH = _sd("language_model.lm_head.")["weight"].to(dev, torch.bfloat16)
        s.m = m; s.E = m.embed_tokens.weight; s.dev = dev
    @torch.no_grad()
    def kv(s, ids, valid):
        B, L = ids.shape[:2]; x = s.E[ids] if ids.dtype == torch.long else ids; pos = torch.arange(L, device=s.dev)[None].expand(B, -1); c, sn = rope_cs(pos, s.dev)
        cm = torch.tril(torch.ones(L, L, dtype=torch.bool, device=s.dev))[None, None] & valid[:, None, None, :]
        cm = cm | torch.eye(L, dtype=torch.bool, device=s.dev)[None, None]
        Ks, Vs = [], []
        hl = None
        for li, layer in enumerate(s.m.layers):
            x, k, v = layer_fwd(layer, x, c, sn, None, None, cm); Ks.append(k); Vs.append(v)
            if li == HID_LAYER: hl = x
        return Ks, Vs, (hl if hl is not None else s.m.norm(x))

