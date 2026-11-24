# Copyright 2026 Adventists AI. Apache-2.0.
"""Sophon-Realtime v1 speaking engine: streaming talker (shared 32B trunk + LoRA) -> vector frames -> E2 -> HiFT, 24 kHz.
Usage: core = Core(); s = Session(core, lang, voice, emit); s.add_text(...); s.finish(); s.run()"""
import os, sys, re, time, math, threading, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sophon_expressive import config as _cfg; _cfg.apply_defaults()
from sophon_expressive import paths as _P
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchaudio, librosa
import transformers
from sophon_expressive import talker_model as L
T32_EV = os.environ.get("T32_EV", "0") == "1"
from sophon_expressive import ev_tags as MK
EV_GAIN = float(os.environ.get("EV_GAIN", "1"))
dev = "cuda"
TALKER_CK = _P.talker_weights()
E2CK = _P.e2_dir()
from safetensors.torch import load_file as _load_sf
VOICE = _load_sf(_P.voice_file(os.environ.get("SOPHON_VOICE", "chelsie")))
REF = {lg: lg for lg in ("zh", "en", "ja") if f"spk.{lg}" in VOICE}
K0, RHO = 2, 1.0
FP8 = os.environ.get('T32_FP8', '0') == '1'; E2B = os.environ.get('T32_E2B', '0') == '1' or os.environ.get('T32_SCHED', '0') == '1'; SCHED = os.environ.get('T32_SCHED', '0') == '1'; HABS = os.environ.get('E2V2_HIFT_ABS', '0') == '1'; FUSED = FP8 or os.environ.get('T32_FUSED', '0') == '1' or os.environ.get('T32_M2', '0') == '1'; M2 = os.environ.get('T32_M2', '0') == '1'
from contextlib import nullcontext as _NULL
import hift_abs as _HA
READ1 = os.environ.get('T32_READ1', '0') == '1'; MINNF = int(os.environ.get('T32_MINNF', '0')); TRACE = os.environ.get('T32_TRACE', '0') == '1'
JUNK = re.compile(r"[*#`>_~\[\]{}<>|\\/]|[\U0001F000-\U0001FFFF☀-➿]"); QUOTE = re.compile(r"[“”「」『』\"《》()（）【】]")
FIRST, NEXT, HOLD = int(os.environ.get("ST_FIRST", "4")), int(os.environ.get("ST_NEXT", "6")), int(os.environ.get("ST_HOLD", "2"))
PRE = int(os.environ.get("T32_PRE", "3600"))
WIN, XF = int(os.environ.get("E2_WIN", "60")), int(os.environ.get("E2_XF", "8"))
NT_FIRST = int(os.environ.get("E2_NT_FIRST", os.environ.get("E2_NT", "10")))
PAR_E2 = os.environ.get("T32_PAR_E2", "1") == "1"; FIRST_EXCL = os.environ.get("T32_FIRST_EXCL", "1") == "1"
DBG = os.environ.get("T32_DBG", "1") == "1"
PAR_AHEAD = float(os.environ.get("T32_PAR_AHEAD", "1.0"))
RAMP = [int(x) for x in os.environ.get("ST_RAMP", "2,3,4").split(",") if x]
TAIL_TOK = int(os.environ.get("ST_TAIL_TOK", "3")); NT = int(os.environ.get("E2_NT", "10")); DIT_STREAM = os.environ.get("ST_DIT_STREAM", "1") == "1"


class Core:
    def __init__(s):
        t0 = time.time()
        s.TH = L.Thinker(dev); m = L.TalkerLT(s.TH).to(dev).eval()
        if TALKER_CK.endswith(".safetensors"):
            from safetensors.torch import load_file; ck = dict(m=load_file(TALKER_CK))
        else: ck = torch.load(TALKER_CK, map_location="cpu", weights_only=False)
        miss = m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["m"].items()}, strict=False); s.m = m
        if os.environ.get("T32_LORA16", "1") == "1":
            n16 = 0
            for mod in m.modules():
                if isinstance(mod, L.LoRA):
                    mod.A16 = mod.lora_A.detach().to(torch.bfloat16).contiguous(); mod.B16 = mod.lora_B.detach().to(torch.bfloat16).contiguous()
                    mod.lora_A.data = torch.empty(0, device=dev); mod.lora_B.data = torch.empty(0, device=dev); n16 += 1
            def _fwd(mod, x):
                y = mod.base(x)
                if L.LoRA.ON: y = y + (F.linear(F.linear(x.to(torch.bfloat16), mod.A16), mod.B16) * mod.scale).to(y.dtype)
                return y
            L.LoRA.forward = _fwd; torch.cuda.empty_cache(); print("LoRA bf16", n16, flush=True)
        s.tok = L.TOKZ; s.IM = s.tok.convert_tokens_to_ids("<|im_end|>")
        pid = torch.tensor(L.PROMPT_IDS, device=dev)[None]
        with torch.no_grad(): s.Kp, s.Vp, _ = s.TH.kv(pid, torch.ones_like(pid, dtype=torch.bool))
        global PL; PL = len(L.PROMPT_IDS); s.Kp_l = {lg: (s.Kp, s.Vp) for lg in ("zh", "en", "ja")}
        if os.environ.get("SPK_CTX"):
            spk = os.environ.get("SOPHON_VOICE", "chelsie")
            E = s.TH.E
            with torch.no_grad():
                for lg in ("zh", "en", "ja"):
                    v = VOICE[f"ctx.{lg}"].to(dev, E.dtype)
                    ce = torch.cat([E[torch.tensor(L.TURN_HEAD["assistant"], device=dev)], v, E[torch.tensor(L.TURN_END, device=dev)], E[torch.tensor(L.PROMPT_IDS, device=dev)]])[None]
                    K_, V_, _ = s.TH.kv(ce, torch.ones(ce.shape[:2], dtype=torch.bool, device=dev)); s.Kp_l[lg] = (K_, V_)
            PL = ce.shape[1]; s.Kp, s.Vp = s.Kp_l["zh"]; print("SPK_CTX", spk, "prefix", PL, flush=True)
        s.zb = m.ztok_up(torch.zeros(8, device=dev)).detach()
        s.x0 = (m.bos + m.zutt_up(torch.zeros(1, 32, device=dev)) + m.spk_up(torch.zeros(1, 192, device=dev)))[:, None].detach()
        from sophon_expressive import e2_model as _E2
        s.m2, s.dec, s.aff, s.hift, _e2cfg = _E2.load_all(dev)
        s.voices = {}
        for lg in REF: s.voices[f"brand_{lg}"] = VOICE[f"spk.{lg}"].to(dev)
        s.prompts = {}
        if os.environ.get("E2_PROMPT", "0") == "1":
            for lg in REF: s.prompts[lg] = (VOICE[f"prompt_mu.{lg}"].to(dev), VOICE[f"prompt_mel.{lg}"].to(dev))
            if os.environ.get("E2V2", "0") == "1":
                from e2v2_stream import Stream
                t1 = time.time()
                with torch.no_grad():
                    for lg, (mur, mr) in s.prompts.items():
                        st_ = Stream(s.dec, mur, mr, s.aff(s.voices.get(f"brand_{lg}", s.voices["brand_zh"])), ch=16, left=int(os.environ.get("E2V2_LEFT", "12")), nt=int(os.environ.get("E2V2_NT", "10")))
                        for _ in range(st_.left + 2): st_.gen(torch.zeros(1, 80, 16, device=dev))
                print("E2V2 warm %.1fs" % (time.time() - t1), flush=True)
        if FUSED: (build_fused8 if FP8 else build_fused)(s)
        s.e2lock = threading.Lock(); s.e2stream = torch.cuda.Stream()
        s.lock = threading.RLock(); NG = int(os.environ.get("T32_POOL", "3")) if os.environ.get("T32_GRAPH", "1") == "1" else 0
        NG = 0 if M2 else NG; s.pool = [(GStepF if FUSED else GStep)(s) for _ in range(NG)]; s.gs = s.pool[0] if NG else None
        for g in s.pool:
            with torch.no_grad(): g.capture(); g.capture_ctx()
        if M2: setup_m2(s)
        if E2B: setup_e2b(s)
        if SCHED: s.engine = Engine(s)
        print("T32CORE ready %.1fs talker=%s e2=%s" % (time.time() - t0, TALKER_CK, E2CK), flush=True)

    @torch.no_grad()
    def spk_from_wav(s, w16):
        raise NotImplementedError("custom voices are not available in this release")

    @torch.no_grad()
    def ctx_extend(s, Kc, Vc, ids_new, L0):
        k = len(ids_new); x = s.TH.E[torch.tensor(ids_new, device=dev)][None]; pos = torch.arange(L0, L0 + k, device=dev)[None]; c, sn = L.rope_cs(pos, dev)
        m = torch.ones(1, 1, k, L0 + k, dtype=torch.bool, device=dev); m[0, 0, :, L0:] = torch.tril(torch.ones(k, k, dtype=torch.bool, device=dev))
        nK, nV = [], []
        for l, layer in enumerate(s.TH.m.layers):
            x, kk, vv = L.layer_fwd(layer, x, c, sn, Kc[l], Vc[l], m); nK.append(torch.cat([Kc[l], kk], 2)); nV.append(torch.cat([Vc[l], vv], 2))
        return nK, nV



class _DevGraph(torch.cuda.graph):
    _streams = {}
    def __init__(self, cuda_graph, pool=None, stream=None, **kw):
        if stream is None:
            d = torch.cuda.current_device(); stream = _DevGraph._streams.setdefault(d, torch.cuda.Stream(device=d))
        super().__init__(cuda_graph, pool=pool, stream=stream, **kw)
torch.cuda.graph = _DevGraph
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = os.environ.get("T32_BF16_RED", "0") == "1"
FL = []
def build_fused(core):
    bf = torch.bfloat16
    with torch.no_grad():
        for layer in core.m.layers:
            a, p = layer.self_attn, layer.mlp; mods = (a.q_proj, a.k_proj, a.v_proj, a.o_proj, p.gate_proj, p.up_proj, p.down_proj)
            for mod in mods: mod.Wc = mod.base.weight.data.cpu(); mod.base.weight = None; mod.Ac = mod.A16.cpu(); mod.Bc = (mod.B16.float().cpu() * mod.scale).to(bf); mod.A16 = mod.B16 = None
            torch.cuda.empty_cache(); r = a.q_proj.Ac.shape[0]
            oq, ok, ov = (m_.Wc.shape[0] for m_ in (a.q_proj, a.k_proj, a.v_proj))
            Bb = torch.zeros(oq + ok + ov, 3 * r, dtype=bf); Bb[:oq, :r] = a.q_proj.Bc; Bb[oq:oq + ok, r:2 * r] = a.k_proj.Bc; Bb[oq + ok:, 2 * r:] = a.v_proj.Bc
            og = p.gate_proj.Wc.shape[0]; Bg = torch.stack([p.gate_proj.Bc.t(), p.up_proj.Bc.t()]).contiguous()
            d = dict(Aqkv=torch.cat([a.q_proj.Ac, a.k_proj.Ac, a.v_proj.Ac], 0), Wqkv=torch.cat([torch.cat([a.q_proj.Wc, a.k_proj.Wc, a.v_proj.Wc], 0), Bb], 1),
                     Ao=a.o_proj.Ac, Wo=torch.cat([a.o_proj.Wc, a.o_proj.Bc], 1), Agu=torch.cat([p.gate_proj.Ac, p.up_proj.Ac], 0),
                     Wgu=torch.cat([p.gate_proj.Wc, p.up_proj.Wc], 0), Bgu=Bg, Ad=p.down_proj.Ac, Wd=torch.cat([p.down_proj.Wc, p.down_proj.Bc], 1))
            for mod in mods: mod.Wc = mod.Ac = mod.Bc = None
            EDV = [x for x in os.environ.get("T32_ENGINE_DEVS", "").split(",") if x] or [dev]; dv = EDV[len(FL) * len(EDV) // L.NL]
            layer.to(dv); d = {k: v.contiguous().to(dv) for k, v in d.items()}; d.update(sp=(oq, ok, ov), og=og, layer=layer, dev=dv); FL.append(d)
    torch.cuda.empty_cache(); print("FUSED layers %d, mem %.1f GB" % (len(FL), torch.cuda.memory_allocated() / 1e9), flush=True)
def flayer(d, x, c, sn, Kb, Vb, idx, mask, msk, Kfull=None, Vfull=None, rows=None):
    layer = d["layer"]; a = layer.self_attn; B, Lq, _ = x.shape; bf = torch.bfloat16
    h = layer.input_layernorm(x).to(bf); t = F.linear(h, d["Aqkv"]) * msk
    qkv = F.linear(torch.cat([h, t], -1), d["Wqkv"]); oq, ok, ov = d["sp"]
    q = a.q_norm(qkv[..., :oq].reshape(B, Lq, L.HQ, L.HD)).transpose(1, 2); k = a.k_norm(qkv[..., oq:oq + ok].reshape(B, Lq, L.HKV, L.HD)).transpose(1, 2)
    v = qkv[..., oq + ok:].reshape(B, Lq, L.HKV, L.HD).transpose(1, 2).to(bf)
    q, k = L.rope(q, c, sn), L.rope(k, c, sn)
    if rows is not None: Kb[rows[0], :, rows[1]] = k[:, :, 0].to(bf); Vb[rows[0], :, rows[1]] = v[:, :, 0]
    else: Kb.index_copy_(2, idx, k.to(bf)); Vb.index_copy_(2, idx, v)
    KK = Kb if Kfull is None else Kfull; VV = Vb if Vfull is None else Vfull
    if Lq == 1 and os.environ.get("T32_MANUAL_ATTN", "1") == "1":
        G = L.HQ // L.HKV; qg = q.reshape(B, L.HKV, G, L.HD)
        sc = torch.matmul(qg, KK.transpose(-1, -2)).float() * (L.HD ** -0.5)
        sc = sc + (mask.float() if mask.dtype != torch.bool else torch.where(mask, 0.0, float("-inf")).float()).reshape(B, 1, 1, -1)
        o = torch.matmul(torch.softmax(sc, -1).to(bf), VV).reshape(B, L.HQ, 1, L.HD)
    else:
        o = F.scaled_dot_product_attention(q, KK, VV, attn_mask=mask, enable_gqa=True)
    o = o.transpose(1, 2).reshape(B, Lq, L.HQ * L.HD).to(bf); t = F.linear(o, d["Ao"]) * msk
    x = x + F.linear(torch.cat([o, t], -1), d["Wo"]).to(x.dtype)
    h = layer.post_attention_layernorm(x).to(bf); t = F.linear(h, d["Agu"]) * msk
    gu = F.linear(h, d["Wgu"]); og = d["og"]; r2 = t.shape[-1] // 2
    gu = gu + torch.bmm(t.reshape(-1, 2, r2).transpose(0, 1), d["Bgu"]).transpose(0, 1).reshape(gu.shape); y = F.silu(gu[..., :og]) * gu[..., og:]
    t = F.linear(y, d["Ad"]) * msk
    return x + F.linear(torch.cat([y, t], -1), d["Wd"]).to(x.dtype)

PL = 0
CAP_C, CAP_T = int(os.environ.get("T32_CAPC", "512")), int(os.environ.get("T32_CAPT", "2048"))

def _pt(s, k, t):
    if os.environ.get("T32_PROF") == "1": torch.cuda.synchronize(); n = time.perf_counter(); s.stats[k] = s.stats.get(k, 0) + 1000 * (n - t); return n
    return t


def _tonefix(w, sr=24000):
    import numpy as _np
    from scipy.signal import butter as _bt, sosfiltfilt as _sf
    if len(w) < 1024: return w
    N, H = 512, 256; win = _np.hanning(N); fr = _np.fft.rfftfreq(N, 1 / sr); band = (fr >= 4000) & (fr <= 8000)
    nf = 1 + (len(w) - N) // H; bad = _np.zeros(nf, bool)
    for i in range(nf):
        S = _np.abs(_np.fft.rfft(w[i * H: i * H + N] * win)) ** 2; sb = S[band]; tot = S.sum() + 1e-12
        bad[i] = sb.max() > 31.6 * (sb.mean() + 1e-12) and sb.max() > 0.3 * tot
    run = _np.convolve(bad.astype(int), [1, 1], "same") >= 2
    if not run.any(): return w
    m = _np.zeros(len(w)); 
    for i in _np.nonzero(run)[0]: m[i * H: i * H + N] = 1.0
    k = int(0.005 * sr); m = _np.convolve(m, _np.ones(k) / k, "same").clip(0, 1)
    lp = _sf(_bt(6, 3500, "low", fs=sr, output="sos"), w)
    return w * (1 - m) + lp * m

def _pend0(s):
    p = getattr(s, 'v2_pend', None)
    return p is None or p.shape[2] == 0

class GStep:
    def __init__(s, core):
        s.c = core; m = core.m; s.NL = L.NL; S = CAP_C + CAP_T
        s.K = [torch.zeros(1, L.HKV, S, L.HD, dtype=torch.bfloat16, device=dev) for _ in range(L.NL)]; s.V = [torch.zeros_like(k) for k in s.K]
        P = PL
        for l in range(L.NL): s.K[l][:, :, :P] = core.Kp[l]; s.V[l][:, :, :P] = core.Vp[l]
        s.bias = torch.full((1, 1, 1, S), float("-inf"), dtype=torch.bfloat16, device=dev)
        s.pos = torch.zeros(1, 1, dtype=torch.long, device=dev); s.idx = torch.zeros(1, dtype=torch.long, device=dev)
        s.x_in = torch.zeros(1, 1, L.D, device=dev); s.y = None; s.graph = None; s.owner = None
        s.H = torch.zeros(CAP_C, L.D, dtype=torch.bfloat16, device=dev) if L.HID else None
    def _body(s, TB=None):
        TB = CAP_T if TB is None else TB; SL = CAP_C + TB
        m = s.c.m; L.LoRA.ON = True
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c, sn = L.rope_cs(s.pos, dev); x = s.x_in
                for l, layer in enumerate(m.layers):
                    a = layer.self_attn; h = layer.input_layernorm(x)
                    q = a.q_norm(a.q_proj(h).view(1, 1, L.HQ, L.HD)).transpose(1, 2)
                    k = a.k_norm(a.k_proj(h).view(1, 1, L.HKV, L.HD)).transpose(1, 2)
                    v = a.v_proj(h).view(1, 1, L.HKV, L.HD).transpose(1, 2).to(torch.bfloat16)
                    q, k = L.rope(q, c, sn), L.rope(k, c, sn)
                    s.K[l].index_copy_(2, s.idx, k.to(torch.bfloat16)); s.V[l].index_copy_(2, s.idx, v)
                    o = F.scaled_dot_product_attention(q, s.K[l][:, :, :SL], s.V[l][:, :, :SL], attn_mask=s.bias[..., :SL], enable_gqa=True)
                    x = x + a.o_proj(o.transpose(1, 2).reshape(1, 1, L.HQ * L.HD))
                    x = x + layer.mlp(layer.post_attention_layernorm(x))
                return m.norm(x)
        finally: L.LoRA.ON = False
    TBS = tuple(int(x) for x in os.environ.get("T32_TBUCKETS", "128,256,512,1024").split(",") if x)
    def capture(s):
        s.graphs = {}; s.ys = {}
        for TB in sorted(set([t for t in s.TBS if t < CAP_T] + [CAP_T])):
            s.pos.fill_(PL); s.idx.fill_(CAP_C + TB - 1)
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(2): s._body(TB)
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): y = s._body(TB)
            s.graphs[TB] = g; s.ys[TB] = y
        s.graph = s.graphs[CAP_T]; s.y = s.ys[CAP_T]
    def reset(s, sess):
        s.owner = sess; P = PL; s.bias.fill_(float("-inf"))
        Kp, Vp = s.c.Kp_l.get(sess.lang, (s.c.Kp, s.c.Vp))
        for l in range(L.NL): s.K[l][:, :, :P] = Kp[l]; s.V[l][:, :, :P] = Vp[l]
    BUCKETS = (1, 2, 4, 8, 16, 32)
    def _ctx_body(s, kb):
        b = s.cb[kb]; c, sn = L.rope_cs(b["pos"], dev); x = s.c.TH.E[b["ids"]][None]
        for l, layer in enumerate(s.c.TH.m.layers):
            a = layer.self_attn; h = layer.input_layernorm(x)
            q = a.q_norm(a.q_proj(h).view(1, kb, L.HQ, L.HD)).transpose(1, 2)
            k = a.k_norm(a.k_proj(h).view(1, kb, L.HKV, L.HD)).transpose(1, 2)
            v = a.v_proj(h).view(1, kb, L.HKV, L.HD).transpose(1, 2).to(torch.bfloat16)
            q, k = L.rope(q, c, sn), L.rope(k, c, sn)
            s.K[l].index_copy_(2, b["idx"], k.to(torch.bfloat16)); s.V[l].index_copy_(2, b["idx"], v)
            o = F.scaled_dot_product_attention(q, s.K[l][:, :, :CAP_C], s.V[l][:, :, :CAP_C], attn_mask=b["mask"], enable_gqa=True)
            x = x + a.o_proj(o.transpose(1, 2).reshape(1, kb, L.HQ * L.HD))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        if s.H is not None: s.H.index_copy_(0, b["idx"], s.c.TH.m.norm(x)[0].to(torch.bfloat16))
        return x
    def capture_ctx(s):
        s.cb = {}; s.cg = {}
        for kb in s.BUCKETS:
            P = PL
            s.cb[kb] = dict(ids=torch.full((kb,), 151643, dtype=torch.long, device=dev), pos=(P + torch.arange(kb, device=dev))[None],
                            idx=CAP_C - kb + torch.arange(kb, device=dev), mask=torch.zeros(1, 1, kb, CAP_C, dtype=torch.bfloat16, device=dev))
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(2): s._ctx_body(kb)
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): s._ctx_body(kb)
            s.cg[kb] = g
    def ctx_extend(s, ids_new, L0):
        if getattr(s, "cg", None) is None or os.environ.get("T32_CTXGRAPH", "1") != "1": return s.ctx_extend_eager(ids_new, L0)
        i = 0
        while i < len(ids_new):
            part = ids_new[i: i + 32]; k = len(part); kb = next(b for b in s.BUCKETS if b >= k)
            if L0 + kb > CAP_C: return s.ctx_extend_eager(ids_new[i:], L0)
            b = s.cb[kb]; b["ids"].fill_(151643); b["ids"][:k].copy_(torch.tensor(part, dtype=torch.long).pin_memory(), non_blocking=True)
            ar = torch.arange(kb, device=dev); b["pos"].copy_((L0 + ar)[None]); b["idx"].copy_(L0 + ar)
            j = torch.arange(CAP_C, device=dev)[None]; b["mask"].copy_(torch.where(j <= (L0 + ar)[:, None], 0.0, float("-inf")).to(torch.bfloat16)[None, None])
            s.cg[kb].replay(); L0 += k; i += k
    def ctx_extend_eager(s, ids_new, L0):
        k = len(ids_new); x = s.c.TH.E[torch.tensor(ids_new, device=dev)][None]; pos = torch.arange(L0, L0 + k, device=dev)[None]; c, sn = L.rope_cs(pos, dev)
        mk = torch.ones(1, 1, k, L0 + k, dtype=torch.bool, device=dev); mk[0, 0, :, L0:] = torch.tril(torch.ones(k, k, dtype=torch.bool, device=dev))
        for l, layer in enumerate(s.c.TH.m.layers):
            x, kk, vv = L.layer_fwd(layer, x, c, sn, s.K[l][:, :, :L0], s.V[l][:, :, :L0], mk)
            s.K[l][:, :, L0:L0 + k] = kk; s.V[l][:, :, L0:L0 + k] = vv
        if s.H is not None: s.H[L0:L0 + k] = s.c.TH.m.norm(x)[0].to(torch.bfloat16)
    def step(s, x, P, vis, p):
        if s.graph is None: s.capture()
        s.x_in.copy_(x.view(1, 1, -1).float()); s.pos.fill_(P + p); s.idx.fill_(CAP_C + p)
        s.bias[..., :CAP_C].fill_(float("-inf")); s.bias[..., : P + vis] = 0; s.bias[..., CAP_C + p] = 0
        TB = next(t for t in sorted(s.graphs) if t > p); s.graphs[TB].replay(); return s.ys[TB]


class GStepF(GStep):
    def __init__(s, core):
        super().__init__(core); s.msk1 = torch.ones(1, 1, 1, device=dev, dtype=torch.bfloat16); s.msk0 = torch.zeros(1, 1, 1, device=dev, dtype=torch.bfloat16)
    def _body(s, TB=None):
        TB = CAP_T if TB is None else TB; SL = CAP_C + TB
        c, sn = L.rope_cs(s.pos, dev); x = s.x_in; bias = s.bias[..., :SL]
        for l, d in enumerate(FL): x = flayer(d, x, c, sn, s.K[l], s.V[l], s.idx, bias, s.msk1, s.K[l][:, :, :SL], s.V[l][:, :, :SL])
        return s.c.m.norm(x)
    def capture(s):
        s.graphs = {}; s.ys = {}
        for TB in sorted(set([t for t in s.TBS if t < CAP_T] + [CAP_T])):
            j = CAP_C + TB - 1; s.pos.fill_(PL); s.idx.fill_(j); s.bias.fill_(float("-inf")); s.bias[..., j] = 0
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(2): s._body(TB)
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): y = s._body(TB)
            s.graphs[TB] = g; s.ys[TB] = y
            with torch.no_grad():
                for l in range(L.NL): s.K[l][:, :, j].zero_(); s.V[l][:, :, j].zero_()
        s.bias.fill_(float("-inf")); s.graph = s.graphs[CAP_T]; s.y = s.ys[CAP_T]
    def _ctx_body(s, kb):
        b = s.cb[kb]; c, sn = L.rope_cs(b["pos"], dev); x = s.c.TH.E[b["ids"]][None]
        for l, d in enumerate(FL): x = flayer(d, x, c, sn, s.K[l], s.V[l], b["idx"], b["mask"], s.msk0, s.K[l][:, :, :CAP_C], s.V[l][:, :, :CAP_C])
        if s.H is not None: s.H.index_copy_(0, b["idx"], s.c.TH.m.norm(x)[0].to(torch.bfloat16))
        return x
    def capture_ctx(s):
        super().capture_ctx()
        with torch.no_grad():
            for l in range(L.NL): s.K[l][:, :, CAP_C - 32: CAP_C].zero_(); s.V[l][:, :, CAP_C - 32: CAP_C].zero_()
    def ctx_extend_eager(s, ids_new, L0):
        k = len(ids_new); x = s.c.TH.E[torch.tensor(ids_new, device=dev)][None]; pos = torch.arange(L0, L0 + k, device=dev)[None]; c, sn = L.rope_cs(pos, dev)
        mk = torch.ones(1, 1, k, L0 + k, dtype=torch.bool, device=dev); mk[0, 0, :, L0:] = torch.tril(torch.ones(k, k, dtype=torch.bool, device=dev))
        ii = torch.arange(L0, L0 + k, device=dev)
        for l, d in enumerate(FL): x = flayer(d, x, c, sn, s.K[l], s.V[l], ii, mk, s.msk0, s.K[l][:, :, :L0 + k], s.V[l][:, :, :L0 + k])
        if s.H is not None: s.H[L0:L0 + k] = s.c.TH.m.norm(x)[0].to(torch.bfloat16)

class BatchSlots:
    def __init__(s, core, NS):
        s.c = core; s.NS = NS; S = CAP_C + CAP_T; s.S = S; bf = torch.bfloat16
        s.K = [torch.zeros(NS, L.HKV, S + 1, L.HD, dtype=bf, device=dev) for _ in FL]; s.V = [torch.zeros_like(k) for k in s.K]
        s.H = torch.zeros(NS, CAP_C, L.D, dtype=bf, device=dev) if L.HID else None
        s.x_in = torch.zeros(NS, 1, L.D, device=dev); s.bias = torch.full((NS, 1, 1, S), float("-inf"), dtype=bf, device=dev)
        s.pos = torch.zeros(NS, 1, dtype=torch.long, device=dev); s.idx = torch.full((NS,), S, dtype=torch.long, device=dev)
        s.rows = torch.arange(NS, device=dev); s.msk = torch.zeros(NS, 1, 1, device=dev, dtype=bf)
        s.TBS = sorted(set([t for t in GStep.TBS if t < CAP_T] + [CAP_T])); s.graphs = None
        s.q = []; s.cv = threading.Condition(); s.nstep = 0; s.nrows = 0; s.tstep = 0.0
        s.views = [SlotView(s, i) for i in range(NS)]
        threading.Thread(target=s._loop, daemon=True).start()
    def _body(s, TB):
        SL = CAP_C + TB; c, sn = L.rope_cs(s.pos, dev); x = s.x_in; bias = s.bias[..., :SL]
        for l, d in enumerate(FL): x = flayer(d, x, c, sn, s.K[l], s.V[l], None, bias, s.msk, s.K[l][:, :, :SL], s.V[l][:, :, :SL], rows=(s.rows, s.idx))
        return s.c.m.norm(x)
    def capture(s):
        s.graphs = {}
        s.idx.fill_(s.S); s.msk.zero_(); s.pos.zero_(); s.bias.fill_(float("-inf")); s.bias[..., 0] = 0
        for TB in s.TBS:
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(2): s._body(TB)
            torch.cuda.current_stream().wait_stream(st)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr): out = s._body(TB)
            s.graphs[TB] = (gr, out)
        with torch.no_grad():
            for l in range(L.NL): s.K[l][:, :, s.S].zero_(); s.V[l][:, :, s.S].zero_()
        print("M2 batch graphs", s.TBS, flush=True)
    def submit(s, i, x, P, vis, p):
        r = dict(i=i, x=x, P=P, vis=vis, p=p, ev=threading.Event(), y=None)
        with s.cv: s.q.append(r); s.cv.notify()
        r["ev"].wait(); return r["y"]
    def _loop(s):
        win = float(os.environ.get("T32_BATCH_WIN", "0.0015"))
        while True:
            with s.cv:
                while not s.q: s.cv.wait()
            t_end = time.perf_counter() + float(os.environ.get("T32_BATCH_WMAX", "0.008"))
            while True:
                nexp = sum(1 for v in s.views if v.owner is not None and getattr(v.owner, "_stepping", False) and not v.owner.finished and not v.owner.cancelled)
                with s.cv: nq = len(s.q)
                if nq >= nexp or time.perf_counter() >= t_end: break
                time.sleep(0.0002)
            with s.cv: reqs, s.q = s.q, []
            try:
                with torch.no_grad():
                    t0 = time.perf_counter()
                    if s.graphs is None: s.capture()
                    s.msk.zero_(); s.idx.fill_(s.S); s.bias.fill_(float("-inf")); s.bias[..., 0] = 0; s.pos.zero_()
                    mp = 0
                    for r in reqs:
                        i = r["i"]; P, vis, p = r["P"], r["vis"], r["p"]; mp = max(mp, p)
                        s.x_in[i].copy_(r["x"].view(1, -1).float()); s.pos[i, 0] = P + p; s.idx[i] = CAP_C + p; s.msk[i] = 1
                        s.bias[i, ..., 0] = float("-inf"); s.bias[i, ..., : P + vis] = 0; s.bias[i, ..., CAP_C: CAP_C + p + 1] = 0
                    TB = next(t for t in s.TBS if t > mp)
                    gr, out = s.graphs[TB]
                    if os.environ.get("T32_EAGER_BATCH", "0") == "1": out = s._body(TB)
                    else: gr.replay()
                    for r in reqs: r["y"] = out[r["i"]: r["i"] + 1].clone()
                    s.nstep += 1; s.nrows += len(reqs)
                    if os.environ.get("T32_BSTAT", "0") == "1": torch.cuda.synchronize(); s.tstep += time.perf_counter() - t0
            except Exception as e:
                import traceback; traceback.print_exc(); print("BATCH ERROR", repr(e)[:300], flush=True)
            for r in reqs: r["ev"].set()
class SlotView(GStepF):
    def __init__(s, bs, i):
        s.bs = bs; s.i = i; s.c = bs.c; s.owner = None; s.graph = 1; s.cg = None
        s.K = [k[i:i + 1] for k in bs.K]; s.V = [v[i:i + 1] for v in bs.V]; s.H = bs.H[i] if bs.H is not None else None
        s.msk0 = torch.zeros(1, 1, 1, device=dev, dtype=torch.bfloat16)
    def capture(s): pass
    def capture_ctx(s, pool=None):
        s.cb = {}; s.cg = {}
        for kb in s.BUCKETS:
            s.cb[kb] = dict(ids=torch.full((kb,), 151643, dtype=torch.long, device=dev), pos=(PL + torch.arange(kb, device=dev))[None],
                            idx=CAP_C - kb + torch.arange(kb, device=dev), mask=torch.zeros(1, 1, kb, CAP_C, dtype=torch.bfloat16, device=dev))
            st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(2): s._ctx_body(kb)
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool): s._ctx_body(kb)
            s.cg[kb] = g
        with torch.no_grad():
            for l in range(L.NL): s.K[l][:, :, CAP_C - 32: CAP_C].zero_(); s.V[l][:, :, CAP_C - 32: CAP_C].zero_()
    def reset(s, sess):
        s.owner = sess; P = PL; Kp, Vp = s.c.Kp_l.get(sess.lang, (s.c.Kp, s.c.Vp))
        for l in range(L.NL): s.K[l][:, :, :P] = Kp[l]; s.V[l][:, :, :P] = Vp[l]
    def step(s, x, P, vis, p): return s.bs.submit(s.i, x, P, vis, p)
    def ctx_extend(s, ids_new, L0):
        with s.c.ctxlock: return GStep.ctx_extend(s, ids_new, L0)
def setup_m2(core):
    if os.environ.get("T32_KEEP_LMH", "1") != "1" and getattr(core.TH, "LMH", None) is not None: core.TH.LMH = None
    import gc; gc.collect(); torch.cuda.empty_cache()
    NS = int(os.environ.get("T32_SLOTS", "8")); core.ctxlock = threading.Lock(); core.bs = BatchSlots(core, NS); core.pool = core.bs.views; core.gs = core.pool[0]
    with torch.no_grad():
        core.bs.capture()
        if os.environ.get("T32_CTXGRAPH", "1") == "1":
            pool = torch.cuda.graph_pool_handle()
            for v in core.bs.views: v.capture_ctx(pool)
    print("M2 slots %d, mem %.1f GB" % (NS, torch.cuda.memory_allocated() / 1e9), flush=True)

E4 = torch.float8_e4m3fn
def _q8(W):
    W = W.to(dev).float(); s = (W.abs().amax(1, keepdim=True) / 448.0).clamp(min=1e-12)
    return (W / s).to(E4).contiguous(), s.t().contiguous()
import triton, triton.language as tl
@triton.jit
def _rowq(X, Y, S, M, K, BLOCK: tl.constexpr):
    r = tl.program_id(0); offs = tl.arange(0, BLOCK); mx = tl.zeros([BLOCK], dtype=tl.float32); ok = r < M
    for k0 in range(0, K, BLOCK):
        x = tl.load(X + r * K + k0 + offs, mask=(k0 + offs < K) & ok, other=0.0).to(tl.float32); mx = tl.maximum(mx, tl.abs(x))
    sc = tl.maximum(tl.max(mx, 0) / 448.0, 1e-8); tl.store(S + r, sc)
    for k0 in range(0, K, BLOCK):
        x = tl.load(X + r * K + k0 + offs, mask=(k0 + offs < K) & ok, other=0.0).to(tl.float32)
        tl.store(Y + r * K + k0 + offs, (x / sc).to(tl.float8e4nv), mask=k0 + offs < K)
def f8lin(x, W8, ws):
    if os.environ.get("T32_F8TRI", "1") == "1":
        sh = x.shape; x2 = x.reshape(-1, sh[-1]).contiguous(); M, K = x2.shape; Mp = (M + 15) // 16 * 16
        y8 = torch.empty(Mp, K, device=x.device, dtype=E4); xs = torch.empty(Mp, 1, device=x.device, dtype=torch.float32)
        _rowq[(Mp,)](x2, y8, xs, M, K, BLOCK=1024)
        return torch._scaled_mm(y8, W8.t(), scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16)[:M].reshape(*sh[:-1], W8.shape[0])
    sh = x.shape; x2 = x.reshape(-1, sh[-1]); M = x2.shape[0]; Mp = (M + 15) // 16 * 16
    if Mp != M: x2 = F.pad(x2, (0, 0, 0, Mp - M))
    xs = (x2.abs().amax(1, keepdim=True).float() / 448.0).clamp(min=1e-8)
    y = torch._scaled_mm((x2.float() / xs).to(E4), W8.t(), scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16)
    return y[:M].reshape(*sh[:-1], W8.shape[0])
def build_fused8(core):
    bf = torch.bfloat16
    with torch.no_grad():
        for layer in core.m.layers:
            a, p = layer.self_attn, layer.mlp; mods = (a.q_proj, a.k_proj, a.v_proj, a.o_proj, p.gate_proj, p.up_proj, p.down_proj)
            r = a.q_proj.A16.shape[0]
            Bs = {id(m_): (m_.B16.float() * m_.scale).to(bf) for m_ in mods}
            oq, ok, ov = (m_.base.weight.shape[0] for m_ in (a.q_proj, a.k_proj, a.v_proj)); og = p.gate_proj.base.weight.shape[0]
            Bb = torch.zeros(oq + ok + ov, 3 * r, dtype=bf, device=dev); Bb[:oq, :r] = Bs[id(a.q_proj)]; Bb[oq:oq + ok, r:2 * r] = Bs[id(a.k_proj)]; Bb[oq + ok:, 2 * r:] = Bs[id(a.v_proj)]
            d = dict(Aqkv=torch.cat([a.q_proj.A16, a.k_proj.A16, a.v_proj.A16], 0).contiguous(), Bqkv=Bb, Ao=a.o_proj.A16.contiguous(), Bo=Bs[id(a.o_proj)].contiguous(),
                     Agu=torch.cat([p.gate_proj.A16, p.up_proj.A16], 0).contiguous(), Bgu=torch.stack([Bs[id(p.gate_proj)].t(), Bs[id(p.up_proj)].t()]).contiguous(),
                     Ad=p.down_proj.A16.contiguous(), Bd=Bs[id(p.down_proj)].contiguous())
            d["W8qkv"], d["sqkv"] = _q8(torch.cat([a.q_proj.base.weight, a.k_proj.base.weight, a.v_proj.base.weight], 0))
            d["W8o"], d["so"] = _q8(a.o_proj.base.weight)
            d["W8gu"], d["sgu"] = _q8(torch.cat([p.gate_proj.base.weight, p.up_proj.base.weight], 0))
            d["W8d"], d["sd"] = _q8(p.down_proj.base.weight)
            for m_ in mods: m_.base.weight = None; m_.A16 = m_.B16 = None
            d.update(sp=(oq, ok, ov), og=og, layer=layer, dev=dev); FL.append(d); torch.cuda.empty_cache()
    torch.cuda.empty_cache(); print("FUSED8 layers %d, mem %.1f GB" % (len(FL), torch.cuda.memory_allocated() / 1e9), flush=True)
def flayer8(d, x, c, sn, Kb, Vb, idx, mask, msk, Kfull=None, Vfull=None, rows=None):
    layer = d["layer"]; a = layer.self_attn; B, Lq, _ = x.shape; bf = torch.bfloat16
    h = layer.input_layernorm(x).to(bf); t = F.linear(h, d["Aqkv"]) * msk
    qkv = f8lin(h, d["W8qkv"], d["sqkv"]) + F.linear(t, d["Bqkv"]); oq, ok, ov = d["sp"]
    q = a.q_norm(qkv[..., :oq].reshape(B, Lq, L.HQ, L.HD)).transpose(1, 2); k = a.k_norm(qkv[..., oq:oq + ok].reshape(B, Lq, L.HKV, L.HD)).transpose(1, 2)
    v = qkv[..., oq + ok:].reshape(B, Lq, L.HKV, L.HD).transpose(1, 2).to(bf)
    q, k = L.rope(q, c, sn), L.rope(k, c, sn)
    if rows is not None: Kb[rows[0], :, rows[1]] = k[:, :, 0].to(bf); Vb[rows[0], :, rows[1]] = v[:, :, 0]
    else: Kb.index_copy_(2, idx, k.to(bf)); Vb.index_copy_(2, idx, v)
    KK = Kb if Kfull is None else Kfull; VV = Vb if Vfull is None else Vfull
    if Lq == 1 and os.environ.get("T32_MANUAL_ATTN", "1") == "1":
        G = L.HQ // L.HKV; qg = q.reshape(B, L.HKV, G, L.HD)
        sc = torch.matmul(qg, KK.transpose(-1, -2)).float() * (L.HD ** -0.5)
        sc = sc + (mask.float() if mask.dtype != torch.bool else torch.where(mask, 0.0, float("-inf")).float()).reshape(B, 1, 1, -1)
        o = torch.matmul(torch.softmax(sc, -1).to(bf), VV).reshape(B, L.HQ, 1, L.HD)
    else:
        o = F.scaled_dot_product_attention(q, KK, VV, attn_mask=mask, enable_gqa=True)
    o = o.transpose(1, 2).reshape(B, Lq, L.HQ * L.HD).to(bf); t = F.linear(o, d["Ao"]) * msk
    x = x + (f8lin(o, d["W8o"], d["so"]) + F.linear(t, d["Bo"])).to(x.dtype)
    h = layer.post_attention_layernorm(x).to(bf); t = F.linear(h, d["Agu"]) * msk
    gu = f8lin(h, d["W8gu"], d["sgu"]); og = d["og"]; r2 = t.shape[-1] // 2
    gu = gu + torch.bmm(t.reshape(-1, 2, r2).transpose(0, 1), d["Bgu"]).transpose(0, 1).reshape(gu.shape)
    y = F.silu(gu[..., :og]) * gu[..., og:]
    t = F.linear(y, d["Ad"]) * msk
    return x + (f8lin(y, d["W8d"], d["sd"]) + F.linear(t, d["Bd"])).to(x.dtype)
_flayer16 = flayer
def flayer(d, *a, **k): return flayer8(d, *a, **k) if "W8qkv" in d else _flayer16(d, *a, **k)

def _est_new_g(e, xt, mt, ct, sp, t, L, KVb, frc, frn, kb):
    from x_transformers.x_transformers import apply_rotary_pos_emb as _rope
    R = xt.shape[0]; tt = e.time_embed(t.repeat(R))
    h = e.input_embed(xt.transpose(1, 2), ct.transpose(1, 2), mt.transpose(1, 2), sp)[:, -L:]; res = h; ks, vs = [], []
    for l, b in enumerate(e.transformer_blocks):
        norm, g_msa, sh, sc, g_mlp = b.attn_norm(h, emb=tt); a = b.attn
        q = a.to_q(norm); k = a.to_k(norm); v = a.to_v(norm); ks.append(k); vs.append(v)
        qr = _rope(q, frn); kr = _rope(k, frn)
        K = torch.cat([_rope(KVb[l, 0], frc), kr], 1); V = torch.cat([KVb[l, 1], v], 1)
        H = a.heads; hd = K.shape[-1] // H
        o = F.scaled_dot_product_attention(qr.view(R, L, H, hd).transpose(1, 2), K.view(R, -1, H, hd).transpose(1, 2), V.view(R, -1, H, hd).transpose(1, 2), attn_mask=kb)
        o = a.to_out[1](a.to_out[0](o.transpose(1, 2).reshape(R, L, H * hd)))
        h = h + g_msa.unsqueeze(1) * o
        h = h + g_mlp.unsqueeze(1) * b.ff(b.ff_norm(h) * (1 + sc[:, None]) + sh[:, None])
    if e.long_skip_connection is not None: h = e.long_skip_connection(torch.cat((h, res), dim=-1))
    h = e.norm_out(h, tt); return e.proj_out(h).transpose(1, 2), torch.stack([torch.stack(ks), torch.stack(vs)], 1)
def _kvc_graph(eb, st, Bp, NB):
    key = ("kvc", Bp, NB, id(st.est))
    if key in eb.G: return eb.G[key]
    e = st.est; dv = st.noise.device; R = Bp * NB; NL = len(e.transformer_blocks); inner = e.transformer_blocks[0].attn.to_k.out_features
    Tm = int(os.environ.get("E2V2_PMAX", "260")) + st.ch * st.left; LT = st.ch + 64
    fr_all, _ = e.rotary_embed.forward_from_seq_len(Tm + st.ch + 8); dr = fr_all.shape[-1]
    g = dict(xr=torch.zeros(R, 80, LT, device=dv), mr=torch.zeros(R, 80, LT, device=dv), cr=torch.zeros(R, 80, LT, device=dv), sp=torch.zeros(R, 80, device=dv), t=torch.zeros(1, device=dv),
             KV=torch.zeros(NL, 2, R, Tm, inner, device=dv), frc=torch.zeros(R, Tm, dr, device=dv), frn=torch.zeros(R, st.ch, dr, device=dv),
             kb=torch.zeros(R, 1, 1, Tm + st.ch, device=dv), Tm=Tm, fr_all=fr_all[0], L=st.ch)
    g["kb"][:, :, :, 1:Tm] = float("-inf")
    def body(): return _est_new_g(e, g["xr"], g["mr"], g["cr"], g["sp"], g["t"], st.ch, g["KV"], g["frc"], g["frn"], g["kb"])
    with torch.cuda.stream(eb.st):
        for _ in range(2): body()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, pool=eb.pool, stream=eb.st): g["out"] = body()
    torch.cuda.current_stream().wait_stream(eb.st)
    g["graph"] = gr; eb.G[key] = g; print("E2B kvc graph B=%d NB=%d Tm=%d" % (Bp, NB, Tm), flush=True); return g
def _run_kvcg(eb, ps):
    import e2v2_kvc as KV
    st0 = ps[0]["s"]; NB = 1 if st0.cfg == 0 else 2; B = len(ps); Bp = next(b for b in eb.BK if b >= B)
    g = _kvc_graph(eb, st0, Bp, NB); L = g["L"]; Tm = g["Tm"]; fr = g["fr_all"]; LT = g["xr"].shape[2]
    cs = [KV.start(p["s"], p["mu"]) for p in ps]
    for j, (p, c) in enumerate(zip(ps, cs)):
        Tc = c["mu_ctx"].shape[2] - L; r0, r1 = j * NB, (j + 1) * NB; c["Tc"] = Tc
        g["frc"][r0:r1, :Tc] = fr[:Tc]; g["frn"][r0:r1] = fr[Tc:Tc + L]; g["kb"][r0:r1, :, :, :Tc] = 0; g["kb"][r0:r1, :, :, Tc:Tm] = float("-inf")
        g["mr"][r0:r1, :, :] = KV.rows(p["s"], c["mu_ctx"][:, :, -LT:], c["mu_ctx"][:, :, -LT:], c["cond"][:, :, -LT:])[1]
        g["cr"][r0:r1, :, :] = KV.rows(p["s"], c["mu_ctx"][:, :, -LT:], c["mu_ctx"][:, :, -LT:], c["cond"][:, :, -LT:])[2]
        g["sp"][r0:r1] = KV.rows(p["s"], c["mu_ctx"][:, :, -LT:], c["mu_ctx"][:, :, -LT:], c["cond"][:, :, -LT:])[3]
    for k in range(st0.nt):
        g["t"].copy_(st0.ts[k:k + 1]); dt = st0.ts[k + 1] - st0.ts[k]
        for j, (p, c) in enumerate(zip(ps, cs)):
            s_ = p["s"]; r0, r1 = j * NB, (j + 1) * NB; o = 0
            xs = torch.cat([s_.pstates[k]] + [stt[k] for stt, _ in c["ctx"]] + [c["x"]], 2)[:, :, -LT:]
            g["xr"][r0:r1] = xs
            for piece in [s_.pkv[k]] + [stt.kv[k] for stt, _ in c["ctx"]]:
                n = piece.shape[3]; g["KV"][:, :, r0:r1, o:o + n] = piece; o += n
        g["graph"].replay(); v, kvn = g["out"]
        for j, (p, c) in enumerate(zip(ps, cs)):
            r0, r1 = j * NB, (j + 1) * NB
            c["states"].kv.append(kvn[:, :, r0:r1].clone()); c["x"] = c["x"] + dt * KV.comb(p["s"], v[r0:r1]); c["states"].append(c["x"])
    for p, c in zip(ps, cs): p["r"]["y"] = KV.finish(p["s"], c)

class E2Batch:
    BK = (1, 2, 4, 8)
    def __init__(s, core):
        s.c = core; s.q = []; s.cv = threading.Condition(); s.G = {}; s.pool = torch.cuda.graph_pool_handle(); s.st = torch.cuda.Stream()
        s.nreq = 0; s.nbat = 0; s.hist = {}
        threading.Thread(target=s._loop, daemon=True).start()
    def gen(s, stm, mu):
        ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream())
        r = dict(s=stm, mu=mu, ev_in=ev, done=threading.Event(), y=None, err=None)
        with s.cv: s.q.append(r); s.cv.notify()
        r["done"].wait()
        if r["err"] is not None: raise r["err"]
        torch.cuda.current_stream().wait_event(r["ev_out"]); r["y"].record_stream(torch.cuda.current_stream()); return r["y"]
    def _loop(s):
        win = float(os.environ.get("T32_E2B_WIN", "0.003"))
        while True:
            with s.cv:
                while not s.q: s.cv.wait()
            nact = sum(1 for g in s.c.pool if g.owner is not None and not g.owner.finished and not g.owner.cancelled)
            if nact > 1: time.sleep(win)
            with s.cv: reqs, s.q = s.q, []
            with torch.cuda.stream(s.st):
                for r in reqs: s.st.wait_event(r["ev_in"]); r["mu"].record_stream(s.st)
                plans = [s._plan(r) for r in reqs]; groups = {}
                for p in plans: groups.setdefault(p["key"], []).append(p)
                for key, ps in groups.items():
                    try:
                        with torch.no_grad():
                            for i in range(0, len(ps), s.BK[-1]): s._run(ps[i:i + s.BK[-1]])
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        for p in ps: p["r"]["err"] = e
                ev = torch.cuda.Event(); ev.record(s.st)
            s.nreq += len(reqs); s.nbat += len(groups); s.hist[len(reqs)] = s.hist.get(len(reqs), 0) + 1
            for r in reqs: r["ev_out"] = ev; r["done"].set()
    def _plan(s, r):
        st, mu_new = r["s"], r["mu"]; L = mu_new.shape[2]; ctx = st.chunks[-st.left:]
        nt1 = getattr(st, "nt_once", None); sd = (st.nt // nt1) if (nt1 and nt1 < st.nt and st.nt % nt1 == 0) else 1
        T = st.P + sum(m.shape[2] for _, m in ctx) + L
        ok = L == st.ch and st.P > 0
        return dict(r=r, s=st, mu=mu_new, L=L, sd=sd, ctx=ctx, key=(T, st.P, L, sd, st.nt, float(st.cfg), st.ch, st.left, len(ctx)) if ok else ("solo", id(r)))
    def _graph(s, st, B, T, P):
        NB = 1 if st.cfg == 0 else 2; key = (B, T, P, st.left, st.ch, NB)
        if key in s.G: return s.G[key]
        dv = st.noise.device; B2 = NB * B
        g = dict(x=torch.zeros(B2, 80, T, device=dv), mu=torch.zeros(B2, 80, T, device=dv), c=torch.zeros(B2, 80, T, device=dv), sp=torch.zeros(B2, 80, device=dv), t=torch.zeros(1, device=dv))
        i = torch.arange(T, device=dv); ci = torch.where(i < P, torch.full_like(i, -1), (i - P) // st.ch)
        vis = (ci[None, :] <= ci[:, None]) & (ci[None, :] >= ci[:, None] - st.left); vis[:, :P] = True; g["am"] = vis[None, None].expand(B2, 1, T, T).contiguous()
        m = torch.ones(B2, 1, T, device=dv)
        def body(): st._t = g["t"]; return st._est(g["x"], m, g["mu"], None, g["sp"], g["c"], g["am"])
        for _ in range(2): body()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, pool=s.pool, stream=s.st): g["out"] = body()
        g["graph"] = gr; s.G[key] = g; print("E2B graph B=%d T=%d P=%d" % (B, T, P), flush=True); return g
    def _run_kvc(s, ps):
        import e2v2_kvc as KV
        cs = [KV.start(p["s"], p["mu"]) for p in ps]; st0 = ps[0]["s"]; NB = 1 if st0.cfg == 0 else 2
        for k in range(st0.nt):
            ins = [KV.inputs(p["s"], c, k) for p, c in zip(ps, cs)]; T = ins[0][5]; dt = st0.ts[k + 1] - st0.ts[k]
            v, kv = KV.est_new(st0.est, torch.cat([a[0] for a in ins]), torch.cat([a[1] for a in ins]), torch.cat([a[2] for a in ins]), torch.cat([a[3] for a in ins]),
                               st0.ts[k:k + 1], cs[0]["L"], torch.cat([a[4] for a in ins], 2), T)
            for j, (p, c) in enumerate(zip(ps, cs)):
                c["states"].kv.append(kv[:, :, j * NB:(j + 1) * NB]); c["x"] = c["x"] + dt * KV.comb(p["s"], v[j * NB:(j + 1) * NB]); c["states"].append(c["x"])
        for p, c in zip(ps, cs): p["r"]["y"] = KV.finish(p["s"], c)
    def _run(s, ps):
        if os.environ.get("E2V2_KVC", "0") == "1" and ps[0]["key"][0] != "solo":
            import e2v2_kvc as KV
            if all(KV.ok(p["s"], p["L"], p["ctx"]) for p in ps): return _run_kvcg(s, ps) if os.environ.get("E2V2_KVCG", "1") == "1" else s._run_kvc(ps)
        if ps[0]["key"][0] == "solo":
            for p in ps: p["r"]["y"] = p["s"].gen0(p["mu"])
            return
        B = len(ps); Bp = next(b for b in s.BK if b >= B); st0 = ps[0]["s"]; T, P, L, sd = ps[0]["key"][:4]
        xs0 = []
        for p in ps:
            st = p["s"]; NN = st.noise.shape[2]; off = st.P + (st.n * st.ch + getattr(st, 'noff', 0)) % max(st.ch, NN - st.P - st.ch); st.n += 1; st.nt_once = None
            p["x"] = st.noise[:, :, off: off + L].float(); p["states"] = [p["x"]]
            p["mu_ctx"] = torch.cat([st.mur] + [m for _, m in p["ctx"]] + [p["mu"].float()], 2)
            cond = torch.zeros_like(p["mu_ctx"]); cond[:, :, : st.P] = st.mr; p["cond"] = cond
        g = s._graph(st0, Bp, T, P); pad = lambda t: torch.cat([t] + [t[:1]] * (Bp - B), 0)
        MU = pad(torch.cat([p["mu_ctx"] for p in ps], 0)); C = pad(torch.cat([p["cond"] for p in ps], 0)); SP = pad(torch.cat([p["s"].spk for p in ps], 0))
        g["mu"][:Bp].copy_(MU); g["mu"][Bp:].zero_(); g["c"][:Bp].copy_(C); g["c"][Bp:].zero_(); g["sp"][:Bp].copy_(SP); g["sp"][Bp:].zero_()
        nst = st0.nt // sd; cfg = float(st0.cfg); nocfg = cfg == 0
        for kk in range(1, nst + 1):
            k0, k = (kk - 1) * sd, kk * sd; dt = st0.ts[k] - st0.ts[k0]
            X = pad(torch.cat([torch.cat([p["s"].pstates[k0]] + [stt[k0] for stt, _ in p["ctx"]] + [p["x"]], 2) for p in ps], 0))
            g["x"][:Bp].copy_(X)
            if not nocfg: g["x"][Bp:].copy_(X)
            g["t"].copy_(st0.ts[k0:k0 + 1])
            g["graph"].replay(); v = g["out"]; v = (v[:Bp] if nocfg else ((1 + cfg) * v[:Bp] - cfg * v[Bp:]))[:, :, -L:]
            for j, p in enumerate(ps):
                xn = p["x"] + dt * v[j:j + 1]
                for jj in range(1, sd): p["states"].append(p["x"] + (xn - p["x"]) * jj / sd)
                p["x"] = xn; p["states"].append(xn)
        for p in ps:
            st = p["s"]; st.chunks.append((p["states"], p["mu"].float()))
            if len(st.chunks) == 2 and st.chunks[0][1].shape[2] + st.chunks[1][1].shape[2] == st.ch and st.chunks[0][1].shape[2] < st.ch:
                (s0, m0), (s1, m1) = st.chunks; st.chunks = [([torch.cat([a_, b_], 2) for a_, b_ in zip(s0, s1)], torch.cat([m0, m1], 2))]
            if len(st.chunks) > st.left + 2: st.chunks = st.chunks[-(st.left + 2):]
            p["r"]["y"] = p["x"]
def setup_e2b(core):
    import e2v2_stream as _ES
    if not hasattr(_ES.Stream, "gen0"): _ES.Stream.gen0 = _ES.Stream.gen
    core.e2b = E2Batch(core); eb = core.e2b
    def _g(st, mu_new): return eb.gen(st, mu_new)
    _ES.Stream.gen = torch.no_grad()(_g); print("E2B on", flush=True)
    if os.environ.get("T32_E2B_WARM", "1") == "1":
        t0 = time.time()
        with torch.no_grad(), torch.cuda.stream(eb.st):
            for lg, (mur, mr) in core.prompts.items():
                st = _ES.Stream(core.dec, mur, mr, core.aff(core.voices.get(f"brand_{lg}", core.voices["brand_zh"])), ch=16, left=int(os.environ.get("E2V2_LEFT", "12")), nt=int(os.environ.get("E2V2_NT", "10")))
                for nc in range(st.left + 1):
                    for B in eb.BK: eb._graph(st, B, st.P + 16 * nc + 16, st.P)
        torch.cuda.synchronize(); print("E2B warm %.1fs graphs %d mem %.1f GB" % (time.time() - t0, len(eb.G), torch.cuda.memory_allocated() / 1e9), flush=True)

def _gen_safe_g(st, mu, tries=2, hf_th=-1.5, max_th=2.3):
    mel = yield ("e2", st, mu); st.n_retry = getattr(st, "n_retry", 0); st.n_bad = getattr(st, "n_bad", 0)
    for _ in range(tries):
        if float(mel[:, 72:].max()) <= hf_th and float(mel.max()) <= max_th: break
        st.n -= 1; st.chunks.pop(); st.noff = getattr(st, "noff", 0) + 7919; st.n_retry += 1; mel = yield ("e2", st, mu)
    else:
        if float(mel[:, 72:].max()) > hf_th or float(mel.max()) > max_th:
            st.n_bad += 1; mel = mel.clone(); mel[:, 72:] = mel[:, 72:].clamp(max=-6.0); mel = mel.clamp(max=1.8)
            stt, m_ = st.chunks[-1]; stt[-1] = mel.clone()
    if os.environ.get("E2V2_HFFLOOR_OUT", "") != "": mel = mel.clone(); mel[:, 72:] = float(os.environ["E2V2_HFFLOOR_OUT"])
    return mel
def _bs_run_sync(bs, reqs):
    bs.msk.zero_(); bs.idx.fill_(bs.S); bs.bias.fill_(float("-inf")); bs.bias[..., 0] = 0; bs.pos.zero_(); mp = 0
    for i, x, P, vis, p in reqs:
        mp = max(mp, p); bs.x_in[i].copy_(x.view(1, -1).float()); bs.pos[i, 0] = P + p; bs.idx[i] = CAP_C + p; bs.msk[i] = 1
        bs.bias[i, ..., 0] = float("-inf"); bs.bias[i, ..., : P + vis] = 0; bs.bias[i, ..., CAP_C: CAP_C + p + 1] = 0
    TB = next(t for t in bs.TBS if t > mp); gr, out = bs.graphs[TB]; gr.replay(); bs.nstep += 1; bs.nrows += len(reqs)
    return [out[i: i + 1].clone() for i, *_ in reqs]
def _e2b_run_sync(eb, reqs):
    rs = [dict(s=st, mu=mu, y=None, err=None) for st, mu in reqs]; groups = {}
    for p in [eb._plan(r) for r in rs]: groups.setdefault(p["key"], []).append(p)
    for key, ps in groups.items():
        for i in range(0, len(ps), eb.BK[-1]): eb._run(ps[i:i + eb.BK[-1]])
    eb.nreq += len(rs); eb.nbat += len(groups); eb.hist[len(rs)] = eb.hist.get(len(rs), 0) + 1
    return [r["y"] for r in rs]
def _hift_batch(hift, reqs):
    sg = _HA.patch(hift); out = [None] * len(reqs); groups = {}
    for j, (se, ws_, fin) in enumerate(reqs): groups.setdefault((se.mel_cum.shape[2] - ws_, bool(fin)), []).append(j)
    for (Tw, fin), js in groups.items():
        plans = [_HA.plan(reqs[j][0], reqs[j][1]) for j in js]; sg._abs = plans
        try:
            w, _ = hift.inference(speech_feat=torch.cat([reqs[j][0].mel_cum[:, :, reqs[j][1]:] for j in js], 0), finalize=fin)
            for b, j in enumerate(js): _HA.commit(reqs[j][0], plans[b], sg._cum_out[b]); out[j] = w[b:b + 1]
        finally: sg._abs = None
    return out
_K65 = {}
def _keep65(j0, nT, dv):
    if dv not in _K65: _K65[dv] = torch.tensor([(j % 26) != 25 for j in range(26 * 2000)], device=dv)
    return _K65[dv][j0: j0 + nT].repeat_interleave(2)
def _ad_batch(m2, frs):
    out = [None] * len(frs); groups = {}
    for j, fr in enumerate(frs): groups.setdefault(tuple(fr.shape), []).append(j)
    for shp, js in groups.items():
        _, mu, _ = m2(torch.cat([frs[j] for j in js], 0))
        for b, j in enumerate(js): out[j] = mu[b:b + 1]
    return out
class Engine:
    def __init__(s, core):
        s.c = core; s.sess = []; s.cv = threading.Condition(); s.ntick = 0
        threading.Thread(target=s._loop, daemon=True).start(); print("SCHED engine on", flush=True)
    def add(s, se):
        se.par_on = False; se.rev = threading.Event()
        with s.cv: s.sess.append(se); s.cv.notify()
    def _finish(s, se):
        se.finished = True
        for k in ("ms_ctx", "ms_talker", "ms_e2"): se.stats[k] = round(se.stats[k])
        se.stats["sec"] = round(se.speech_off / 24000, 2); se.stats.pop("t0", None)
        with s.cv:
            if se in s.sess: s.sess.remove(se)
        se._done_ev.set()
    def _loop(s):
        while True:
            with s.cv:
                while not s.sess: s.cv.wait()
                act = list(s.sess)
            try:
                if os.environ.get("T32_CPROF") and not hasattr(s, "_pr"):
                    import cProfile; s._pr = cProfile.Profile(); s._prn = 0
                if getattr(s, "_pr", None) is not None and len(act) >= 6:
                    s._pr.enable()
                    with torch.no_grad(): busy = s.tick(act)
                    s._pr.disable(); s._prn += 1
                    if s._prn == int(os.environ["T32_CPROF"]):
                        import pstats, io; o_ = io.StringIO(); pstats.Stats(s._pr, stream=o_).sort_stats("tottime").print_stats(45); print("CPROF\n" + o_.getvalue(), flush=True)
                else:
                    with torch.no_grad(): busy = s.tick(act)
            except Exception as e:
                import traceback; traceback.print_exc(); print("SCHED ERROR", repr(e)[:300], flush=True)
                for se in act: se.cancelled = True; s._finish(se)
                busy = True
            s.ntick += 1
            if not busy: time.sleep(0.002)
    def tick(s, act):
        busy = False; alive = []; T0 = time.perf_counter(); s.tm = getattr(s, "tm", {})
        def _acc(k, t): s.tm[k] = s.tm.get(k, 0.0) + time.perf_counter() - t; return time.perf_counter()
        for se in act:
            if se.cancelled: s._finish(se); continue
            se.ev.clear()
            if se._commit(): busy = True
            alive.append(se)
        T0 = _acc("commit", T0)
        SB = int(os.environ.get("T32_SERIAL_BUDGET", "2")); RB = int(os.environ.get("T32_SCHED_BUDGET", "2"))
        gens = {}
        for se in alive:
            bud = SB if not se.started else RB
            if se.started and len(se.frames) - getattr(se, "v2_units", 0) < s.c.m2.ad.la + 4: bud = max(bud, int(os.environ.get("T32_CATCHUP_BUDGET", "8")))
            t0 = time.perf_counter(); g = se._steps_g(bud)
            try: gens[se] = (g, next(g), t0)
            except StopIteration: se.stats["ms_talker"] += 1000 * (time.perf_counter() - t0)
        OVL = os.environ.get("T32_OVERLAP", "1") == "1"; E2S = s.c.e2stream; rg = {}
        if OVL:
            with torch.cuda.stream(E2S):
                E2S.wait_stream(torch.cuda.default_stream()); rg, b_ = s._render_start(alive); busy = busy or b_
        while gens or rg:
            if gens:
                ev = torch.cuda.Event(); ev.record(torch.cuda.default_stream())
                items = list(gens.items()); ys = _bs_run_sync(s.c.bs, [(se.g.i, *rq) for se, (g, rq, _) in items]); busy = True; new = {}
            if rg:
                with torch.cuda.stream(E2S):
                    if gens: E2S.wait_event(ev)
                    rg = s._render_step(rg, _acc); busy = True
            if gens:
                for (se, (g, rq, t0)), y in zip(items, ys):
                    try: new[se] = (g, g.send(y), t0)
                    except StopIteration: pass
                gens = new
        T0 = _acc("talker", T0)
        for _pass in range(int(os.environ.get("T32_RENDER_PASSES", "2"))):
            if OVL and _pass == 0: continue
            with torch.cuda.stream(E2S if OVL else torch.cuda.current_stream()):
                if OVL: E2S.wait_stream(torch.cuda.default_stream())
                busy = s._render_pass(alive if _pass == 0 else [se for se in alive if s._ahead(se) < float(os.environ.get("T32_LOW_AHEAD", "0.6"))], _acc) or busy
        T0 = _acc("render", T0)
        for se in alive:
            if se.done and se.mel_out >= len(se.frames) * 8 and _pend0(se): s._finish(se)
        return busy
    def _ahead(s, se):
        t1 = se.stats.get("t_first_audio")
        if t1 is None: return 0.0
        return se.speech_off / 24000 - (time.perf_counter() - se.stats["t0"] - t1)
    def _render_start(s, alive):
        busy = False; rg = {}
        for se in alive:
            g = se._render_v2_g()
            try: rg[se] = (g, next(g))
            except StopIteration as e: busy = busy or bool(e.value)
        return rg, busy
    def _render_pass(s, alive, _acc):
        rg, busy = s._render_start(alive)
        while rg: rg = s._render_step(rg, _acc); busy = True
        return busy
    def _render_step(s, rg, _acc):
        if True:
            items = list(rg.items()); res = {}; new = {}
            e2i = [(se, rq) for se, (g, rq) in items if rq[0] == "e2"]; hfi = [(se, rq) for se, (g, rq) in items if rq[0] == "hift"]; adi = [(se, rq) for se, (g, rq) in items if rq[0] == "ad"]
            if adi:
                T1 = time.perf_counter(); mus = _ad_batch(s.c.m2, [rq[1] for se, rq in adi]); _acc("ad", T1)
                for (se, _), m_ in zip(adi, mus): res[se] = m_
            if e2i:
                T1 = time.perf_counter(); mels = _e2b_run_sync(s.c.e2b, [(rq[1], rq[2]) for se, rq in e2i]); _acc("dit", T1)
                for (se, _), m_ in zip(e2i, mels): res[se] = m_
            if hfi:
                T1 = time.perf_counter(); ws = _hift_batch(s.c.hift, [(se, rq[2], rq[3]) for se, rq in hfi]); _acc("hift", T1)
                for (se, _), w_ in zip(hfi, ws): res[se] = w_
            for se, (g, rq) in items:
                try: new[se] = (g, g.send(res[se]))
                except StopIteration: pass
            return new

class Session:
    def __init__(s, core, lang, voice_key, emit):
        s.c = core; s.lang = lang; s.emit = emit; s.voice = core.voices[voice_key] if voice_key in core.voices else core.voices.get(f"brand_{lang}", core.voices["brand_zh"])
        s.text = ""; s.final = False; s.cancelled = False; s.finished = False; s.ev = threading.Event()
        s._evv = None; s._evl = None; s._evch = []; s._evlead = 0; s._ev0 = False
        s.ids = []; s.Kc, s.Vc = (list(x) for x in core.Kp_l.get(lang, (core.Kp, core.Vp))); s.Lc = PL
        s.cache = [None] * L.NL; s.x = core.x0.clone(); s.r = 0; s.nf = 0; s.p = 0; s.run_ = 0; s.done = False
        s.win_start = 0; s.tail = None
        s.frames = []; s.mel_out = 0; s.mel_cum = None; s.speech_off = 0; s.started = False; s.pending = np.zeros(0, np.float32)
        s.g = None; s.t_start = time.time()
        if core.gs is not None:
            for _ in range(400):
                with core.lock:
                    free = [g for g in core.pool if g.owner is None or g.owner.finished]
                    if free: s.g = free[0]; s.g.reset(s); break
                    live = [g for g in core.pool if not g.owner.cancelled]
                    if live: min(live, key=lambda g: g.owner.t_start).owner.cancel()
                time.sleep(0.005)
            if s.g is None: raise RuntimeError("no free talker slot")
        s.stats = dict(t0=time.perf_counter(), t_first_audio=None, n_tok=0, n_frames=0, n_step=0, ms_ctx=0.0, ms_talker=0.0, ms_e2=0.0)
    def add_text(s, t):
        if "t_text0" not in s.stats: s.stats["t_text0"] = round(time.perf_counter() - s.stats["t0"], 3)
        s.text += t; s.ev.set()
    def finish(s): s.final = True; s.ev.set()
    def cancel(s): s.cancelled = True; s.ev.set()

    def _ev_clean(s):
        raw = s.text
        if not s.final: raw = re.sub(r"(\[[a-z_]*|</?[a-z_]*)$", "", raw)
        ch, lead = MK.walk(raw); keep = []
        for c_ in ch:
            if JUNK.fullmatch(c_["c"]) or QUOTE.fullmatch(c_["c"]):
                if c_["a"] and keep and not keep[-1]["a"]: keep[-1]["a"] = c_["a"]
                continue
            keep.append(c_)
        while keep and keep[0]["c"].isspace(): keep.pop(0)
        if s.final:
            while keep and keep[-1]["c"].isspace(): keep.pop()
        s._evch = keep; s._evlead = lead
        return "".join(c_["c"] for c_ in keep)
    def _ev_build(s, clean):
        enc = s.c.tok(clean, add_special_tokens=False, return_offsets_mapping=True); n = len(enc.input_ids); ch = s._evch
        v = torch.zeros(n + 1, L.EVD_); v[:, L.EV_NC + 2] = 1.0
        for k, (b0, b1) in enumerate(enc.offset_mapping):
            for c in range(b0, b1):
                if ch[c]["a"] and ch[c]["a"] <= L.EV_NC: v[k, ch[c]["a"] - 1] = 1.0
                v[k, L.EV_NC] = max(v[k, L.EV_NC], ch[c]["sp"][0]); v[k, L.EV_NC + 1] = max(v[k, L.EV_NC + 1], ch[c]["sp"][1])
        s._evv = L.ev_shift(v, list(enc.input_ids) + [s.c.IM])
        lv = None
        if s._evlead and s._evlead <= L.EV_NC: lv = torch.zeros(L.EVD_); lv[s._evlead - 1] = 1.0; lv[L.EV_NC + 2] = 1.0
        s._evl = lv
    @torch.no_grad()
    def _ev_add(s):
        m = s.c.m
        if s.r >= 1:
            if s._evv is not None and s.r - 1 < len(s._evv):
                e_ = m.ev_embed(s._evv[s.r - 1].to(dev).float())
                if EV_GAIN != 1.0 and not L.EV_LGAIN:
                    v0_ = s._evv[s.r - 1].to(dev).float().clone(); v0_[:L.EV_NC] = 0; fb_ = m.ev_up(v0_); e_ = fb_ + EV_GAIN * (e_ - fb_)
                s.x = s.x + e_[None, None]
        elif s._evl is not None: s.x = s.x + m.ev_embed(s._evl.to(dev).float())[None, None]

    @torch.no_grad()
    def _commit(s):
        if T32_EV: clean = s._ev_clean()
        else: clean = QUOTE.sub("", JUNK.sub("", s.text)); clean = clean.strip() if s.final else clean.lstrip()
        if not s.final and len(s.ids) >= s.r + K0 + int(math.floor(RHO * s.nf)) + 1 + int(os.environ.get("T32_AHEAD", "6")): return False
        ids_all = s.c.tok(clean, add_special_tokens=False).input_ids
        target = ids_all + [s.c.IM] if s.final else ids_all[: max(0, len(ids_all) - TAIL_TOK)]
        if T32_EV: s._ev_build(clean)
        if len(target) <= len(s.ids): return False
        if target[: len(s.ids)] != s.ids: target = s.ids + target[len(s.ids):]
        new = target[len(s.ids):]; t0 = time.perf_counter()
        with (_NULL() if (M2 and s.g is not None) else s.c.lock):
            if s.g is not None:
                if s.g.owner is not s or s.cancelled: s.cancelled = True; return False
                if s.Lc + len(new) > CAP_C: new = new[: CAP_C - s.Lc]; target = s.ids + new
                if not new: return False
                s.g.ctx_extend(new, s.Lc)
            else: s.Kc, s.Vc = s.c.ctx_extend(s.Kc, s.Vc, new, s.Lc)
        s.Lc += len(new); s.ids = list(target); s.stats["n_tok"] = len(s.ids); s.stats["ms_ctx"] += 1000 * (time.perf_counter() - t0)
        return True

    @torch.no_grad()
    def _steps(s, budget=6):
        c = s.c; m = c.m; n = len(s.ids); fin = s.final and n and s.ids[-1] == c.IM; did = 0; t0 = time.perf_counter()
        mf = int(n * 4 + 20)
        while did < budget and not s.done and not s.cancelled:
            if FIRST_EXCL and s.mel_out == 0 and len(s.frames) >= FIRST + HOLD: break
            need = s.r + K0 + int(math.floor(RHO * s.nf))
            if not fin and n < need + 1: break
            if T32_EV and s.p == 0 and not s._ev0: s._ev0 = True; s._ev_add()
            vis = min(n, need); P = PL
            cm = torch.zeros(1, 1, 1, s.Lc + s.p + 1, dtype=torch.bool, device=dev); cm[..., : P + vis] = True; cm[..., s.Lc:] = True
            if s.g is not None:
                if s.g.owner is not s or s.p >= CAP_T - 1: s.cancelled = True; break
                with (_NULL() if M2 else c.lock):
                    if s.g.owner is not s or s.cancelled: s.cancelled = True; break
                    y = s.g.step(s.x, P, vis, s.p)[:, -1].float()
            else:
                with c.lock, torch.autocast("cuda", dtype=torch.bfloat16):
                    y, new = m.trunk(s.x.float(), torch.tensor([[P + s.p]], device=dev), s.Kc, s.Vc, cm, cache=s.cache)
                s.cache = [(nk, nv) if cc is None else (torch.cat([cc[0], nk], 2), torch.cat([cc[1], nv], 2)) for cc, (nk, nv) in zip(s.cache, new)]
                y = y[:, -1].float()
            lg = m.dec(y)[0].float(); s.p += 1; did += 1
            if s.r >= n: lg[L.KA] = -1e9
            if s.r < n or s.nf == 0 or not fin: lg[L.KS] = -1e9
            if s.nf < MINNF: lg[L.KS] = -1e9
            if READ1 and s.r == 0: lg[L.KF] = -1e9
            cap = s.nf >= mf or (L.STALL > 0 and s.run_ >= L.STALL and s.r < n)
            if cap: lg[L.KF] = -1e9
            if cap and s.r >= n and fin: lg[L.KS] = 1e9
            a = int(lg.argmax())
            if TRACE: s.__dict__.setdefault('trace', []).append('SAF'[[L.KS, L.KA, L.KF].index(a)] if a in (L.KS, L.KA, L.KF) else '?')
            if a == L.KS: s.done = True; break
            if a == L.KF:
                f = m.fm.from_z(m.reg(y).float()); s.frames.append(f[0]); s.x = f[:, None]; s.nf += 1; s.run_ += 1
                if s.nf in (1, 4) and f"t_f{s.nf}" not in s.stats: s.stats[f"t_f{s.nf}"] = round(time.perf_counter() - s.stats["t0"], 3); s.stats[f"tok_at_f{s.nf}"] = len(s.ids)
            else:
                s.x = (c.TH.E[s.ids[s.r]].float() + c.zb)[None, None]
                if L.HID and s.g is not None and s.g.H is not None: s.x = s.x + m.hid_up(s.g.H[PL + s.r].float())[None, None]
                s.r += 1; s.run_ = 0
            if T32_EV: s._ev_add()
            if s.p > mf + n + 5 and fin: s.done = True
        s.stats["ms_talker"] += 1000 * (time.perf_counter() - t0); s.stats["n_step"] = s.p; s.stats["n_frames"] = s.nf
        return did > 0

    @torch.no_grad()
    def _render_v2(s):
        c = s.c; K = len(s.frames); la = c.m2.ad.la; last = s.done; u0 = getattr(s, "v2_units", 0)
        if u0 < 2 * int(os.environ.get("E2V2_LA_FIRST_N", "1")) and os.environ.get("E2V2_LA_FIRST", "") != "": la = min(la, int(os.environ["E2V2_LA_FIRST"]))
        SP65 = os.environ.get("SPEED65", "0") == "1"; pend = getattr(s, "v2_pend", None); npend = 0 if pend is None else pend.shape[2]
        if u0 >= K and last and npend == 0: return False
        u1 = min(u0 + 2, K); end = min(K, u0 + 2 + la)
        if not last and K < u0 + 2 + la: return False
        if u1 <= u0 and npend == 0: return False
        t0 = time.perf_counter()
        import contextlib
        with (_NULL() if E2B else c.e2lock), (torch.cuda.stream(c.e2stream) if s.par_on else contextlib.nullcontext()):
            if s.par_on: c.e2stream.wait_stream(torch.cuda.default_stream())
            if getattr(s, "v2", None) is None:
                from e2v2_stream import Stream
                PR = c.prompts.get(s.lang) or c.prompts.get("zh"); mur, mr = PR
                s.v2 = Stream(c.dec, mur, mr, c.aff(s.voice), ch=16, left=int(os.environ.get("E2V2_LEFT", "12")), nt=int(os.environ.get("E2V2_NT", "10")))
            if u1 > u0:
                st = max(0, end - int(os.environ.get("E2V2_ADWIN", "30")))
                _tp = _pt(s, "_", time.perf_counter()); fr = torch.stack(s.frames[st:end])[None].float(); _, mu, _ = c.m2(fr); mu = mu.float()[:, :, (u0 - st) * 8:(u1 - st) * 8]; _pt(s, "ms_ad", _tp)
            else: mu = None
            if SP65:
                if mu is not None:
                    j0 = 4 * u0; nT = mu.shape[2] // 2
                    keep = _keep65(j0, nT, mu.device)
                    mu = mu[:, :, : 2 * nT][:, :, keep]; s.v2_pend = mu if pend is None else torch.cat([pend, mu], 2)
                fin_units = last and u1 >= K; s.v2_units = u1; s.mel_out = u1 * 8
                _nb = getattr(s, 'v2_nblk', 0); _bs = 8 if (os.environ.get('E2V2_FIRST8', '0') == '1' and _nb < 2) else 16
                if s.v2_pend.shape[2] < _bs and not fin_units: return True
                mu = s.v2_pend[:, :, :_bs]; s.v2_pend = s.v2_pend[:, :, _bs:]; s.v2_nblk = _nb + 1
                if s.v2_pend.shape[2] > 0: fin_units = False
            if int(os.environ.get('E2V2_NT_FIRST', '0')) and getattr(s, 'v2_nfirst', 0) < int(os.environ.get('E2V2_NT_FIRST_N', '1')): s.v2.nt_once = int(os.environ['E2V2_NT_FIRST']); s.v2_nfirst = getattr(s, 'v2_nfirst', 0) + 1
            _tp = _pt(s, "_", time.perf_counter()); mel = s.v2.gen_safe(mu) if os.environ.get("E2V2_SAFE", "1") == "1" else s.v2.gen(mu); _tp = _pt(s, "ms_dit", _tp); s.v2_units = max(u1, getattr(s, "v2_units", 0)); s.stats["v2_retry"] = getattr(s.v2, "n_retry", 0); s.stats["v2_bad"] = getattr(s.v2, "n_bad", 0); s.mel_out = u1 * 8
            s.mel_cum = mel if s.mel_cum is None else torch.cat([s.mel_cum, mel], 2)
            done_all = last and u1 >= K and (not SP65 or s.v2_pend.shape[2] == 0)
            HW = int(os.environ.get("E2V2_HIFT_CTX", "100")); nm = mel.shape[2]
            T_ = s.mel_cum.shape[2]; ws_ = max(0, T_ - nm - HW)
            if os.environ.get("E2V2_HIFT_ANCHOR", "0") == "1":
                an = getattr(s, "hift_ws", 0); MX = int(os.environ.get("E2V2_HIFT_MAX", "600"))
                if T_ - an > MX and float(s.mel_cum[0, :72, max(0, T_ - nm - 2):T_ - nm].mean()) < float(os.environ.get("E2V2_HIFT_QUIET", "-7.0")):
                    an = max(0, T_ - nm - HW); s.stats["hift_reanchor"] = s.stats.get("hift_reanchor", 0) + 1
                s.hift_ws = an; ws_ = an
            w = _HA.infer_one(c.hift, s, s.mel_cum[:, :, ws_:], ws_, done_all) if HABS else c.hift.inference(speech_feat=s.mel_cum[:, :, ws_:], finalize=done_all)[0]; _pt(s, "ms_hift", _tp)
            if os.environ.get("E2V2_BURSTFIX", "0") == "1":
                def _bad(w_):
                    t_ = w_[0, max(0, s.speech_off - ws_ * 480):].float()
                    if t_.numel() < 512: return False
                    S_ = torch.fft.rfft(t_ * torch.hann_window(t_.numel(), device=t_.device)).abs() ** 2; f_ = torch.fft.rfftfreq(t_.numel(), 1 / 24000).to(t_.device)
                    hf = float(S_[(f_ >= 3000) & (f_ <= 8000)].sum() / (S_.sum() + 1e-12)); pk = float(t_.abs().max())
                    return pk > float(os.environ.get("E2V2_BURST_PK", "0.6")) and hf > float(os.environ.get("E2V2_BURST_HF", "0.5"))
                for _try in range(2):
                    if not _bad(w) or mu is None: break
                    s.stats["burst_retry"] = s.stats.get("burst_retry", 0) + 1
                    s.mel_cum = s.mel_cum[:, :, : s.mel_cum.shape[2] - nm]; s.v2.n -= 1; s.v2.chunks.pop(); s.v2.noff = getattr(s.v2, "noff", 0) + 7919
                    mel = s.v2.gen(mu); s.mel_cum = torch.cat([s.mel_cum, mel], 2)
                    w, _ = c.hift.inference(speech_feat=s.mel_cum[:, :, ws_:], finalize=done_all)
                if _bad(w): s.stats["burst_left"] = s.stats.get("burst_left", 0) + 1; w = w.clone(); w[0, max(0, s.speech_off - ws_ * 480):] *= 0.3
        w = w.float()[0].cpu().numpy(); w_new = w[max(0, s.speech_off - ws_ * 480):]; s.speech_off += len(w_new)
        _ax = np.abs(w_new); _o = _ax > 0.6
        if _o.any(): w_new = w_new.copy(); w_new[_o] = np.sign(w_new[_o]) * (0.6 + 0.4 * np.tanh((_ax[_o] - 0.6) / 0.4))
        if os.environ.get('E2V2_TONEFIX', '0') == '1': w_new = _tonefix(w_new)
        _og = float(os.environ.get('E2V2_OUT_GAIN', '1'))
        if _og != 1.0: w_new = w_new * _og
        s.stats["ms_e2"] += 1000 * (time.perf_counter() - t0); s.stats["n_render"] = s.stats.get("n_render", 0) + 1
        if "ms_r0" not in s.stats: s.stats["ms_r0"] = round(1000 * (time.perf_counter() - t0)); s.stats["r0_frames"] = len(s.frames); s.stats["t_r0"] = round(t0 - s.stats["t0"], 3)
        s.stats.setdefault("wlens", []); (len(s.stats["wlens"]) < 4) and s.stats["wlens"].append((round(time.perf_counter() - s.stats["t0"], 3), len(w_new), u1, int(len(np.nonzero(np.abs(w_new) > 0.02)[0]))))
        HS = float(os.environ.get("E2V2_HOLD_SEC", "0.3"))
        if not s.started and not done_all:
            hb = w_new if getattr(s, "v2_hold", None) is None else np.concatenate([s.v2_hold, w_new]); nz = np.nonzero(np.abs(hb) > 0.02)[0]
            if len(nz) == 0 or len(hb) - nz[0] < HS * 24000: s.v2_hold = hb; return True
            s.v2_hold = None; w_new = hb
        if getattr(s, "v2_hold", None) is not None: w_new = np.concatenate([s.v2_hold, w_new]); s.v2_hold = None
        if os.environ.get("E2V2_SAVE"):
            s.v2_wav = getattr(s, "v2_wav", []); s.v2_wav.append(w_new)
            if done_all:
                d = os.environ["E2V2_SAVE"]; os.makedirs(d, exist_ok=True); tag = time.strftime("%H%M%S") + "_%d" % (id(s) % 1000)
                torch.save(dict(mel=s.mel_cum.cpu(), frames=torch.stack(s.frames).cpu(), wav=np.concatenate(s.v2_wav), lang=s.lang, text=s.text), f"{d}/{tag}.pt")
        s._out(w_new, done_all); return True

    def _steps_g(s, budget=6):
        c = s.c; m = c.m; n = len(s.ids); fin = s.final and n and s.ids[-1] == c.IM; did = 0; t0 = time.perf_counter()
        mf = int(n * 4 + 20)
        while did < budget and not s.done and not s.cancelled:
            if FIRST_EXCL and s.mel_out == 0 and len(s.frames) >= FIRST + HOLD: break
            need = s.r + K0 + int(math.floor(RHO * s.nf))
            if not fin and n < need + 1: break
            if T32_EV and s.p == 0 and not s._ev0: s._ev0 = True; s._ev_add()
            vis = min(n, need); P = PL
            cm = torch.zeros(1, 1, 1, s.Lc + s.p + 1, dtype=torch.bool, device=dev); cm[..., : P + vis] = True; cm[..., s.Lc:] = True
            if s.g is not None:
                if s.g.owner is not s or s.p >= CAP_T - 1: s.cancelled = True; break
                with (_NULL() if M2 else c.lock):
                    if s.g.owner is not s or s.cancelled: s.cancelled = True; break
                    y = (yield (s.x, P, vis, s.p))[:, -1].float()
            else:
                with c.lock, torch.autocast("cuda", dtype=torch.bfloat16):
                    y, new = m.trunk(s.x.float(), torch.tensor([[P + s.p]], device=dev), s.Kc, s.Vc, cm, cache=s.cache)
                s.cache = [(nk, nv) if cc is None else (torch.cat([cc[0], nk], 2), torch.cat([cc[1], nv], 2)) for cc, (nk, nv) in zip(s.cache, new)]
                y = y[:, -1].float()
            lg = m.dec(y)[0].float(); s.p += 1; did += 1
            if s.r >= n: lg[L.KA] = -1e9
            if s.r < n or s.nf == 0 or not fin: lg[L.KS] = -1e9
            if s.nf < MINNF: lg[L.KS] = -1e9
            if READ1 and s.r == 0: lg[L.KF] = -1e9
            cap = s.nf >= mf or (L.STALL > 0 and s.run_ >= L.STALL and s.r < n)
            if cap: lg[L.KF] = -1e9
            if cap and s.r >= n and fin: lg[L.KS] = 1e9
            a = int(lg.argmax())
            if TRACE: s.__dict__.setdefault('trace', []).append('SAF'[[L.KS, L.KA, L.KF].index(a)] if a in (L.KS, L.KA, L.KF) else '?')
            if a == L.KS: s.done = True; break
            if a == L.KF:
                f = m.fm.from_z(m.reg(y).float()); s.frames.append(f[0]); s.x = f[:, None]; s.nf += 1; s.run_ += 1
                if s.nf in (1, 4) and f"t_f{s.nf}" not in s.stats: s.stats[f"t_f{s.nf}"] = round(time.perf_counter() - s.stats["t0"], 3); s.stats[f"tok_at_f{s.nf}"] = len(s.ids)
            else:
                s.x = (c.TH.E[s.ids[s.r]].float() + c.zb)[None, None]
                if L.HID and s.g is not None and s.g.H is not None: s.x = s.x + m.hid_up(s.g.H[PL + s.r].float())[None, None]
                s.r += 1; s.run_ = 0
            if T32_EV: s._ev_add()
            if s.p > mf + n + 5 and fin: s.done = True
        s.stats["ms_talker"] += 1000 * (time.perf_counter() - t0); s.stats["n_step"] = s.p; s.stats["n_frames"] = s.nf
        return did > 0


    def _render_v2_g(s):
        c = s.c; K = len(s.frames); la = c.m2.ad.la; last = s.done; u0 = getattr(s, "v2_units", 0)
        if u0 < 2 * int(os.environ.get("E2V2_LA_FIRST_N", "1")) and os.environ.get("E2V2_LA_FIRST", "") != "": la = min(la, int(os.environ["E2V2_LA_FIRST"]))
        SP65 = os.environ.get("SPEED65", "0") == "1"; pend = getattr(s, "v2_pend", None); npend = 0 if pend is None else pend.shape[2]
        if u0 >= K and last and npend == 0: return False
        u1 = min(u0 + 2, K); end = min(K, u0 + 2 + la)
        if not last and K < u0 + 2 + la: return False
        if u1 <= u0 and npend == 0: return False
        t0 = time.perf_counter()
        import contextlib
        with (_NULL() if E2B else c.e2lock), (torch.cuda.stream(c.e2stream) if s.par_on else contextlib.nullcontext()):
            if s.par_on: c.e2stream.wait_stream(torch.cuda.default_stream())
            if getattr(s, "v2", None) is None:
                from e2v2_stream import Stream
                PR = c.prompts.get(s.lang) or c.prompts.get("zh"); mur, mr = PR
                s.v2 = Stream(c.dec, mur, mr, c.aff(s.voice), ch=16, left=int(os.environ.get("E2V2_LEFT", "12")), nt=int(os.environ.get("E2V2_NT", "10")))
            if u1 > u0:
                st = max(0, end - int(os.environ.get("E2V2_ADWIN", "30")))
                _tp = _pt(s, "_", time.perf_counter()); fr = torch.stack(s.frames[st:end])[None].float(); mu = (yield ("ad", fr)); mu = mu.float()[:, :, (u0 - st) * 8:(u1 - st) * 8]; _pt(s, "ms_ad", _tp)
            else: mu = None
            if SP65:
                if mu is not None:
                    j0 = 4 * u0; nT = mu.shape[2] // 2
                    keep = _keep65(j0, nT, mu.device)
                    mu = mu[:, :, : 2 * nT][:, :, keep]; s.v2_pend = mu if pend is None else torch.cat([pend, mu], 2)
                fin_units = last and u1 >= K; s.v2_units = u1; s.mel_out = u1 * 8
                _nb = getattr(s, 'v2_nblk', 0); _bs = 8 if (os.environ.get('E2V2_FIRST8', '0') == '1' and _nb < 2) else 16
                if s.v2_pend.shape[2] < _bs and not fin_units: return True
                mu = s.v2_pend[:, :, :_bs]; s.v2_pend = s.v2_pend[:, :, _bs:]; s.v2_nblk = _nb + 1
                if s.v2_pend.shape[2] > 0: fin_units = False
            if int(os.environ.get('E2V2_NT_FIRST', '0')) and getattr(s, 'v2_nfirst', 0) < int(os.environ.get('E2V2_NT_FIRST_N', '1')): s.v2.nt_once = int(os.environ['E2V2_NT_FIRST']); s.v2_nfirst = getattr(s, 'v2_nfirst', 0) + 1
            _tp = _pt(s, "_", time.perf_counter()); mel = yield from _gen_safe_g(s.v2, mu); _tp = _pt(s, "ms_dit", _tp); s.v2_units = max(u1, getattr(s, "v2_units", 0)); s.stats["v2_retry"] = getattr(s.v2, "n_retry", 0); s.stats["v2_bad"] = getattr(s.v2, "n_bad", 0); s.mel_out = u1 * 8
            s.mel_cum = mel if s.mel_cum is None else torch.cat([s.mel_cum, mel], 2)
            done_all = last and u1 >= K and (not SP65 or s.v2_pend.shape[2] == 0)
            HW = int(os.environ.get("E2V2_HIFT_CTX", "100")); nm = mel.shape[2]
            T_ = s.mel_cum.shape[2]; ws_ = max(0, T_ - nm - HW)
            if os.environ.get("E2V2_HIFT_ANCHOR", "0") == "1":
                an = getattr(s, "hift_ws", 0); MX = int(os.environ.get("E2V2_HIFT_MAX", "600"))
                if T_ - an > MX and float(s.mel_cum[0, :72, max(0, T_ - nm - 2):T_ - nm].mean()) < float(os.environ.get("E2V2_HIFT_QUIET", "-7.0")):
                    an = max(0, T_ - nm - HW); s.stats["hift_reanchor"] = s.stats.get("hift_reanchor", 0) + 1
                s.hift_ws = an; ws_ = an
            w = (yield ("hift", s, ws_, done_all)) if HABS else c.hift.inference(speech_feat=s.mel_cum[:, :, ws_:], finalize=done_all)[0]; _pt(s, "ms_hift", _tp)
            if os.environ.get("E2V2_BURSTFIX", "0") == "1":
                def _bad(w_):
                    t_ = w_[0, max(0, s.speech_off - ws_ * 480):].float()
                    if t_.numel() < 512: return False
                    S_ = torch.fft.rfft(t_ * torch.hann_window(t_.numel(), device=t_.device)).abs() ** 2; f_ = torch.fft.rfftfreq(t_.numel(), 1 / 24000).to(t_.device)
                    hf = float(S_[(f_ >= 3000) & (f_ <= 8000)].sum() / (S_.sum() + 1e-12)); pk = float(t_.abs().max())
                    return pk > float(os.environ.get("E2V2_BURST_PK", "0.6")) and hf > float(os.environ.get("E2V2_BURST_HF", "0.5"))
                for _try in range(2):
                    if not _bad(w) or mu is None: break
                    s.stats["burst_retry"] = s.stats.get("burst_retry", 0) + 1
                    s.mel_cum = s.mel_cum[:, :, : s.mel_cum.shape[2] - nm]; s.v2.n -= 1; s.v2.chunks.pop(); s.v2.noff = getattr(s.v2, "noff", 0) + 7919
                    mel = s.v2.gen(mu); s.mel_cum = torch.cat([s.mel_cum, mel], 2)
                    w, _ = c.hift.inference(speech_feat=s.mel_cum[:, :, ws_:], finalize=done_all)
                if _bad(w): s.stats["burst_left"] = s.stats.get("burst_left", 0) + 1; w = w.clone(); w[0, max(0, s.speech_off - ws_ * 480):] *= 0.3
        w = w.float()[0].cpu().numpy(); w_new = w[max(0, s.speech_off - ws_ * 480):]; s.speech_off += len(w_new)
        _ax = np.abs(w_new); _o = _ax > 0.6
        if _o.any(): w_new = w_new.copy(); w_new[_o] = np.sign(w_new[_o]) * (0.6 + 0.4 * np.tanh((_ax[_o] - 0.6) / 0.4))
        if os.environ.get('E2V2_TONEFIX', '0') == '1': w_new = _tonefix(w_new)
        _og = float(os.environ.get('E2V2_OUT_GAIN', '1'))
        if _og != 1.0: w_new = w_new * _og
        s.stats["ms_e2"] += 1000 * (time.perf_counter() - t0); s.stats["n_render"] = s.stats.get("n_render", 0) + 1
        if "ms_r0" not in s.stats: s.stats["ms_r0"] = round(1000 * (time.perf_counter() - t0)); s.stats["r0_frames"] = len(s.frames); s.stats["t_r0"] = round(t0 - s.stats["t0"], 3)
        s.stats.setdefault("wlens", []); (len(s.stats["wlens"]) < 4) and s.stats["wlens"].append((round(time.perf_counter() - s.stats["t0"], 3), len(w_new), u1, int(len(np.nonzero(np.abs(w_new) > 0.02)[0]))))
        HS = float(os.environ.get("E2V2_HOLD_SEC", "0.3"))
        if not s.started and not done_all:
            hb = w_new if getattr(s, "v2_hold", None) is None else np.concatenate([s.v2_hold, w_new]); nz = np.nonzero(np.abs(hb) > 0.02)[0]
            if len(nz) == 0 or len(hb) - nz[0] < HS * 24000: s.v2_hold = hb; return True
            s.v2_hold = None; w_new = hb
        if getattr(s, "v2_hold", None) is not None: w_new = np.concatenate([s.v2_hold, w_new]); s.v2_hold = None
        if os.environ.get("E2V2_SAVE"):
            s.v2_wav = getattr(s, "v2_wav", []); s.v2_wav.append(w_new)
            if done_all:
                d = os.environ["E2V2_SAVE"]; os.makedirs(d, exist_ok=True); tag = time.strftime("%H%M%S") + "_%d" % (id(s) % 1000)
                torch.save(dict(mel=s.mel_cum.cpu(), frames=torch.stack(s.frames).cpu(), wav=np.concatenate(s.v2_wav), lang=s.lang, text=s.text), f"{d}/{tag}.pt")
        s._out(w_new, done_all); return True


    @torch.no_grad()
    def _render(s):
        if os.environ.get("E2V2", "0") == "1": return s._render_v2()
        c = s.c; K = len(s.frames); last = s.done; ws = s.win_start; we = min(K, ws + WIN)
        full = (we == ws + WIN) and not (last and we == K)
        fin = we if (last and we == K) else (we if full else we - HOLD)
        final = last and we == K
        emit_to = we if final else min(fin, ws + WIN - XF)
        nr = s.stats.get("n_render", 0); need = FIRST if s.mel_out == 0 else (RAMP[nr - 1] if 0 < nr <= len(RAMP) else NEXT)
        if not full and not final and (emit_to * 8 - s.mel_out < need * 8): return False
        t0 = time.perf_counter()
        if "t_r0" not in s.stats: s.stats["t_r0"] = round(t0 - s.stats["t0"], 3)
        import contextlib
        with (_NULL() if E2B else c.e2lock), (torch.cuda.stream(c.e2stream) if s.par_on else contextlib.nullcontext()):
            if s.par_on: c.e2stream.wait_stream(torch.cuda.default_stream())
            fr = torch.stack(s.frames[ws:we])[None].float()
            _tA = time.perf_counter(); _, mux, _ = c.m2(fr); mux = mux.float()[:, :, : (fin - ws) * 8]
            if DBG and s.mel_out == 0: torch.cuda.synchronize(); s.stats["r0_m2"] = round(1000 * (time.perf_counter() - _tA))
            torch.manual_seed(0); cond = torch.zeros_like(mux); off = 0
            PR = c.prompts.get(s.lang) or c.prompts.get("zh")
            if PR is not None: mur, mr = PR; off = mur.shape[2]; mux = torch.cat([mur, mux], 2); cond = torch.zeros_like(mux); cond[:, :, :off] = mr
            mel, _ = c.dec(mu=mux, mask=torch.ones(1, 1, mux.shape[2], device=dev), n_timesteps=(NT_FIRST if not s.par_on else NT), spks=c.aff(s.voice), cond=cond, streaming=DIT_STREAM)
            mel = mel[:, :, off:]
            if DBG and s.mel_out == 0: torch.cuda.synchronize(); s.stats["r0_dec"] = round(1000 * (time.perf_counter() - _tA))
            if s.tail is not None:
                o = min(s.tail.shape[2], mel.shape[2]); r_ = torch.linspace(0, 1, o, device=dev)
                mel = torch.cat([s.tail[:, :, :o] * (1 - r_) + mel[:, :, :o] * r_, mel[:, :, o:]], 2)
            a, b = s.mel_out - ws * 8, emit_to * 8 - ws * 8
            new = mel[:, :, a:b] if b > a else mel[:, :, :0]
            if new.shape[2]: s.mel_out = emit_to * 8; s.mel_cum = new if s.mel_cum is None else torch.cat([s.mel_cum, new], 2)
            if full:
                s.tail = mel[:, :, (WIN - XF) * 8: WIN * 8].clone(); s.win_start = ws + WIN - XF
            done_all = final
            w, _ = c.hift.inference(speech_feat=s.mel_cum, finalize=done_all) if s.mel_cum is not None else (torch.zeros(1, 0, device=dev), None)
        w = w.float()[0].cpu().numpy(); w_new = w[s.speech_off:]; s.speech_off = len(w)
        if DBG and "r0_all" not in s.stats: s.stats["r0_all"] = round(1000 * (time.perf_counter() - _tA))
        s.stats["ms_e2"] += 1000 * (time.perf_counter() - t0); s.stats["n_render"] = s.stats.get("n_render", 0) + 1
        if "ms_r0" not in s.stats: s.stats["ms_r0"] = round(1000 * (time.perf_counter() - t0)); s.stats["r0_frames"] = len(s.frames)
        s._out(w_new, done_all); return True

    def _out(s, w, last):
        if not s.started:
            w = np.concatenate([s.pending, w]); idx = np.nonzero(np.abs(w) > 0.02)[0]
            if len(idx) == 0:
                s.pending = w[-PRE:]
                if not last: return
                w = w[:0]
            else:
                w = w[max(0, idx[0] - PRE):].copy(); s.started = True
                if os.environ.get('E2V2_FADEIN', '0') == '1': _k = min(len(w), 360); w[:_k] *= np.linspace(0, 1, _k)
        if len(w):
            if s.stats["t_first_audio"] is None: s.stats["t_first_audio"] = round(time.perf_counter() - s.stats["t0"], 3)
            for i in range(0, len(w), 4800): s.emit((np.clip(w[i:i + 4800], -1, 1) * 32767).astype(np.int16).tobytes())

    def _render_loop(s):
        try:
            while not s.cancelled:
                if not s._render(): s.rev.wait(0.005); s.rev.clear()
                if s.done and s.mel_out >= len(s.frames) * 8 and _pend0(s): break
        except Exception as e: print("RENDER ERROR", repr(e)[:300], flush=True); s.cancelled = True

    def run(s):
        if SCHED:
            s._done_ev = threading.Event(); s.c.engine.add(s); s._done_ev.wait(); return
        busy = False; s.rev = threading.Event(); s.par_on = False; rt = None
        try:
            while not s.cancelled:
                if not busy: s.ev.wait(0.01)
                s.ev.clear()
                if s.cancelled: break
                s._commit(); busy = bool(s._steps(6 if rt else (int(os.environ.get("T32_SERIAL_BUDGET", "6")) if not s.started else 2)))
                if rt: s.rev.set()
                else:
                    busy = busy | bool(s._render())
                    ahead = s.speech_off / 24000 - (time.perf_counter() - s.stats["t0"] - (s.stats.get("t_first_audio") or 1e9))
                    if PAR_E2 and s.mel_out > 0 and not s.done and (os.environ.get("T32_PAR_EARLY", "0") == "1" or (s.stats.get("t_first_audio") is not None and ahead >= PAR_AHEAD)):
                        s.par_on = True; rt = threading.Thread(target=s._render_loop, daemon=True); rt.start()
                if s.done and (rt or (s.mel_out >= len(s.frames) * 8 and _pend0(s))): break
            if rt: rt.join()
        finally:
            s.finished = True
            for k in ("ms_ctx", "ms_talker", "ms_e2"): s.stats[k] = round(s.stats[k])
            s.stats["sec"] = round(s.speech_off / 24000, 2); s.stats.pop("t0", None)

    def say_all(s, text):
        s.add_text(text); s.finish(); out = []; s.emit = out.append; s.run(); return np.frombuffer(b"".join(out), np.int16).astype(np.float32) / 32767


_steps_orig = Session._steps
def _steps_flag(s, budget=6):
    s._stepping = True
    try: return _steps_orig(s, budget)
    finally: s._stepping = False
Session._steps = _steps_flag
