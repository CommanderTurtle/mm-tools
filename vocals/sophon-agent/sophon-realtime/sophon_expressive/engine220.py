# Copyright 2026 Adventists AI. Apache-2.0.
"""Sophon-Realtime E2E speaking engine (single GPU): streaming reply text -> talker (CUDA Graph, all sessions batched in one step)
-> vector frames -> adapter -> 2-step DiT (batched) -> HiFT, 24 kHz PCM16.
API: core = Core(); s = Session(core, lang, voice, emit); s.add_text(...); s.finish(); s.run()"""
import os, sys, time, threading, re, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.dirname(HERE), HERE):
    if p not in sys.path: sys.path.insert(0, p)
os.environ.setdefault("E2V2_CFG", "0"); os.environ.setdefault("E2V2_KVC", "1"); os.environ.setdefault("S_EVGAIN", "1"); os.environ.setdefault("S220_PREP_FP32", "0"); os.environ.setdefault("S220_PREP_PAD", "16")
_a = sys.argv; sys.argv = [sys.argv[0]]
import talker220_model as S, talker220_engine as FG
_pc = (lambda: (torch.cuda.synchronize(), time.perf_counter())[1]) if os.environ.get("S220_PROF_SYNC") == "1" else time.perf_counter
PROF_PRINT = float(os.environ.get("S220_PROF_PRINT", "0"))
FRAME_CAP = int(os.environ.get("S220_FRAME_CAP", "1800"))
RENDER_EVERY = int(os.environ.get("S220_RENDER_EVERY", "2"))
AHEAD = float(os.environ.get("S220_AHEAD", "1.5"))
EVTOK = os.environ.get("S220_EVTOK", "1") == "1"; os.environ.setdefault("S_SPGAIN", "3")
EVCAP_END = float(os.environ.get("S220_EVCAP_END", "10"))
EVCAP_AT, EVCAP_NEXT, EVCAP_LEAD = [float(os.environ.get(k, d)) for k, d in (("S220_EVCAP_AT", "20"), ("S220_EVCAP_NEXT", "20"), ("S220_EVCAP_LEAD", "20"))]
sys.argv = _a
import engine as EN
import hift_abs as HA, ev_tags as MK
MK.PT.update({"[cry]": 9, "[clear]": 6, "[gasp]": 1})   # extra sound tags the talker knows
from sophon_expressive import e2_model as E2M
from e2v2_stream import Stream
import e2v2_stream as _ES
from safetensors.torch import load_file
sys.setswitchinterval(0.0002)
dev = "cuda"
NS = int(os.environ.get("S220_SLOTS", "16")); START_TOK = int(os.environ.get("S220_START_TOK", "4"))
JUNK = re.compile(r"[*#`>_~{}|\\/]|[\U0001F000-\U0001FFFF☀-➿]")
_K65 = {}
def keep65(j0, nT):
    if dev not in _K65: _K65[dev] = torch.tensor([(j % 26) != 25 for j in range(26 * 4000)], device=dev)
    return _K65[dev][j0: j0 + nT].repeat_interleave(2)

def ad_batch(core, frs):
    """Adapter batches (grouped by shape), each group through a captured graph."""
    if getattr(core, "adg", None) is None: return EN._ad_batch(core.m2, frs)
    out = [None] * len(frs); groups = {}
    for j, fr in enumerate(frs): groups.setdefault(tuple(fr.shape), []).append(j)
    for shp, js in groups.items():
        mu = core.adg(torch.cat([frs[j] for j in js], 0))
        for b, j in enumerate(js): out[j] = mu[b:b + 1]
    return out

def e2b_all(eb, reqs):
    """Batch the DiT across sessions regardless of context length / prompt (the KV-cache graph handles the context per row)."""
    if os.environ.get("S220_E2B_ALL", "1") != "1": return EN._e2b_run_sync(eb, reqs)
    import e2v2_kvc as KV
    rs = [dict(s=st, mu=mu, y=None, err=None) for st, mu in reqs]; ps = [eb._plan(r) for r in rs]; fast = {}; rest = {}
    for p_ in ps:
        if p_["key"][0] != "solo" and KV.ok(p_["s"], p_["L"], p_["ctx"]): fast.setdefault((p_["L"], p_["s"].nt, float(p_["s"].cfg), id(p_["s"].est)), []).append(p_)
        else: rest.setdefault(p_["key"], []).append(p_)
    M = eb.BK[-1]
    for g_ in fast.values():
        for i in range(0, len(g_), M): EN._run_kvcg(eb, g_[i:i + M])
    for g_ in rest.values():
        for i in range(0, len(g_), M): eb._run(g_[i:i + M])
    eb.nreq += len(rs); eb.nbat += len(fast) + len(rest)
    return [r["y"] for r in rs]

class Render:
    def __init__(s, C, lang):
        mur, mr = C.prompts.get(lang, C.prompts["zh"]); s.C = C; s.v2 = Stream(C.dec, mur, mr, C.aff(C.spk.get(lang, C.spk["zh"])), ch=16, left=12, nt=int(os.environ.get("S220_NT", "2")))
        s.u = 0; s.pend = None; s.mel_cum = None; s.off = 0
    def ad_need(s, K, last):
        la = s.C.m2.ad.la
        if s.u < 6: la = min(la, 1)
        if s.u >= K: return None
        u0 = s.u; u1 = min(u0 + 2, K); end = min(K, u0 + 2 + la)
        if (not last and K < u0 + 2 + la) or u1 <= u0: return None
        return (max(0, end - 30), end, u0, u1)
    def ad_apply(s, mu_full, pl):
        st, end, u0, u1 = pl; mu = mu_full.float()[:, :, (u0 - st) * 8:(u1 - st) * 8]; nT = mu.shape[2] // 2
        mu = mu[:, :, :2 * nT][:, :, keep65(4 * u0, nT)]; s.pend = mu if s.pend is None else torch.cat([s.pend, mu], 2); s.u = u1
    def take(s, last, K):
        if s.pend is None: return None
        fin = last and s.u >= K
        if s.pend.shape[2] < 16 and not fin: return None
        mu = s.pend[:, :, :16]; s.pend = s.pend[:, :, 16:]
        if s.pend.shape[2] > 0: fin = False
        return (mu, fin) if mu.shape[2] else None
    def after_mel(s, mel, fin):
        s.mel_cum = mel if s.mel_cum is None else torch.cat([s.mel_cum, mel], 2); T_ = s.mel_cum.shape[2]
        return max(0, T_ - mel.shape[2] - 40), fin and s.pend.shape[2] == 0
    def after_wav(s, w, ws):
        w = w.float()[0].cpu().numpy(); wn = w[max(0, s.off - ws * 480):]; s.off += len(wn); wn = wn * float(os.environ.get("E2V2_OUT_GAIN", "0.5"))
        a = np.abs(wn); o = a > 0.6
        if o.any(): wn = wn.copy(); wn[o] = np.sign(wn[o]) * (0.6 + 0.4 * np.tanh((a[o] - 0.6) / 0.4))
        return wn

class Core:
    def __init__(s):
        import transformers
        t0 = time.time()
        s.m, s.tok = S.load(os.environ["SOPHON_MODEL_DIR"], dev); s.IM = s.tok.convert_tokens_to_ids("<|im_end|>")
        s.eng = FG.Engine(s.m, NS); s.eng.done.fill_(True)
        s.m2, s.dec, s.aff, s.hift, _ = E2M.load_all(dev); s.dec.inference_cfg_rate = 0.0
        if os.environ.get("S220_HIFT_FOLD", "1") == "1":
            import torch.nn.utils.parametrize as _pz
            for _m in s.hift.modules():
                if _pz.is_parametrized(_m, "weight"): _pz.remove_parametrizations(_m, "weight")
        V = load_file(os.path.join(os.environ["SOPHON_MODEL_DIR"], "voices", "chelsie.safetensors"))
        s.prompts = {lg: (V[f"prompt_mu.{lg}"].to(dev), V[f"prompt_mel.{lg}"].to(dev)) for lg in ("zh", "en", "ja")}
        s.spk = {lg: V[f"spk.{lg}"].to(dev).float() for lg in ("zh", "en", "ja")}
        s.voices = {f"brand_{lg}": s.spk[lg] for lg in s.spk}; s.lock = threading.Lock()
        if not hasattr(_ES.Stream, "gen0"): _ES.Stream.gen0 = _ES.Stream.gen
        s.pool = []; EN.E2Batch.BK = (1, 2, 4, 8, 16); s.eb = EN.E2Batch(s)
        s.slots = [None] * NS; s.waiting = []; s.cv = threading.Condition(); s.stop = False
        s.th = threading.Thread(target=s._loop, daemon=True); s.th.start()
        if os.environ.get("S220_KVC_WARM", "1") == "1":
            _r = Render(s, "zh")
            for _B in EN.E2Batch.BK: EN._kvc_graph(s.eb, _r.v2, _B, 1 if _r.v2.cfg == 0 else 2)
        if os.environ.get("S220_AD_GRAPH", "1") == "1":
            import graphs220 as hiftg; s.adg = hiftg.GraphFn(lambda x_: s.m2(x_)[1]); _D = s.eng.fout.shape[-1]; _t = time.time()
            with torch.no_grad():
                for _T in (3, 5, 7, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30):
                    for _B in ((1, 2, 4, 8, 16) if _T == 30 else (1, 2, 4)):
                        for _ in range(4): s.adg(torch.randn(_B, _T, _D, device=dev))
            print("AD graph", len(s.adg.g), "shapes %.1fs" % (time.time() - _t), flush=True)
        else: s.adg = None
        _pn = int(os.environ.get("S220_PREP_PAD", "0")); _nw = int(os.environ.get("S220_PREP_WARM", "1024" if _pn > 0 else "320"))
        if _nw > 0:
            _t = time.time(); _k = 0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _n in (range(_pn, _nw + 1, _pn) if _pn > 0 else range(1, _nw + 1)):
                    for _rep in range(2):
                        _ids = torch.full((1, _n), 1000, device=dev); _ip = torch.zeros_like(_ids, dtype=torch.bool)
                        if _pn > 0 and _rep: _ip[:, -3:] = True
                        s.m.cond = None; s.m.memory(_ids, _ip, torch.tensor([0], device=dev)); _k += 1
            print("PREP warm %d shapes %.1fs" % (_k, time.time() - _t), flush=True)
        if os.environ.get("S220_HIFT_GRAPH", "1") == "1":
            import graphs220 as hiftg; s.hg = hiftg.HiftGraph(s.hift, max_b=16); _t = time.time()
            def _mk(B, Tw):
                rs = []
                for _ in range(B):
                    r = Render(s, "zh"); r.mel_cum = torch.randn(1, 80, Tw, device=dev) - 4; r.pend = torch.zeros(1, 80, 0, device=dev); rs.append((r, 0, False))
                return rs
            with torch.inference_mode():
                s.hg.warm(_mk, Tws=(56,), Bs=(1, 2, 4, 8, 16)); s.hg.warm(_mk, Tws=(16, 32, 48), Bs=(1, 2, 4, 8))
            print("HIFT graph", len(s.hg.g), "shapes %.1fs" % (time.time() - _t), flush=True)
        print("E2E core ready %.1fs slots %d" % (time.time() - t0, NS), flush=True)
    def submit(s, sess):
        with s.cv: s.waiting.append(sess); s.cv.notify()
    # ---- one scheduler thread drives every session
    @torch.no_grad()
    def _loop(s):
        E = s.eng; import collections as _c; s.prof = _c.Counter()
        while not s.stop:
            with s.cv:
                if not s.waiting and all(x is None for x in s.slots): s.cv.wait(0.05); continue
                for sess in list(s.waiting):
                    if sess.cancelled: s.waiting.remove(sess); sess._finish(); continue
                    if not sess.ready(): continue
                    free = [j for j, x in enumerate(s.slots) if x is None]
                    if not free: break
                    j = free[0]; sess.stats.setdefault("t_slot", round(time.perf_counter() - sess.stats["t0"], 3)); s.slots[j] = sess; s.waiting.remove(sess); sess.slot = j; E.reset_slot(j); sess.R = Render(s, sess.lang); sess.loaded = -1
            act = [(j, x) for j, x in enumerate(s.slots) if x is not None]
            _t0 = _pc(); curl = E.cur.tolist() if act else []
            for j, x in act:
                if x.cancelled: E.done[j] = True; s.slots[j] = None; x._finish(); continue
                try: x.refresh(s, j, curl[j])
                except Exception as _e:
                    x.stats["error"] = repr(_e)[:200]; print("SESSION_ERR", x.stats["error"], flush=True); E.done[j] = True; s.slots[j] = None; x._finish()
            act = [(j, x) for j, x in enumerate(s.slots) if x is not None]
            if not act: continue
            if AHEAD > 0 and not s.waiting:
                _now = time.perf_counter(); _ld = min((x.stats["sec"] - (_now - x.t_a0)) if (x.t_a0 is not None and not x.talk_done) else -1.0 for j, x in act)
                if _ld > AHEAD:
                    s.prof["sleep"] = s.prof.get("sleep", 0) + 1
                    with s.cv: s.cv.wait(min(0.05, _ld - AHEAD))
                    continue
            s._tick = getattr(s, "_tick", 0) + 1; _rt = (s._tick % RENDER_EVERY == 0)
            _t1 = _pc(); E.step(); f = E.fout.clone(); al = (~E.done).tolist(); adp = []; reqs = []; _t2 = _pc(); _t3 = _t4 = _t5 = _t2
            for j, x in act:
                if x.talk_done: continue
                if al[j]:
                    x.frames.append(f[j])
                    if len(x.frames) >= min(E.Tmax - 4, FRAME_CAP): x.talk_done = True; E.done[j] = True; x.stats["truncated"] = "frame_limit"
                    if len(x.frames) == 1: x.stats["t_f1"] = round(time.perf_counter() - x.stats["t0"], 3)
                else: x.talk_done = True
            for j, x in act:
                K = len(x.frames)
                if x.talk_done:
                    while True:
                        pl = x.R.ad_need(K, True)
                        if pl is None: break
                        x.R.ad_apply(ad_batch(s, [torch.stack(x.frames[pl[0]:pl[1]])[None].float()])[0], pl)
                    while True:
                        b = x.R.take(True, K)
                        if b is None: break
                        reqs.append((j, x, b[0], b[1]))
                elif _rt or "t_mel1" not in x.stats:
                    pl = x.R.ad_need(K, False)
                    if pl is not None: adp.append((j, x, pl))
            if adp:
                outs = ad_batch(s, [torch.stack(x.frames[pl[0]:pl[1]])[None].float() for j, x, pl in adp])
                for (j, x, pl), mu in zip(adp, outs):
                    x.R.ad_apply(mu, pl); b = x.R.take(False, len(x.frames))
                    if b is not None: reqs.append((j, x, b[0], b[1]))
            _t3 = _pc()
            if reqs:
                mels = e2b_all(s.eb, [(x.R.v2, mu) for j, x, mu, fin in reqs]); hr = []
                for (j, x, mu, fin), mel in zip(reqs, mels):
                    ws, dn = x.R.after_mel(mel, fin); hr.append((x.R, ws, dn)); x.stats.setdefault("t_mel1", round(time.perf_counter() - x.stats["t0"], 3)); x.stats.setdefault("nf_mel1", len(x.frames))
                _t4 = _pc(); wavs = EN._hift_batch(s.hift, hr); _t5 = _pc()
                for (j, x, mu, fin), w, (_, ws, dn) in zip(reqs, wavs, hr): x._out(x.R.after_wav(w, ws))
            if os.environ.get("S220_PROFLOG") == "1":
                if not hasattr(s, "proflog"): s.proflog = []
                s.proflog.append((_t0, len(act), len(reqs), _t1 - _t0, _t2 - _t1, _t3 - _t2, _t4 - _t3, _t5 - _t4, time.perf_counter() - _t0))
            P_ = s.prof; P_["ticks"] += 1; P_["act"] += len(act); P_["refresh"] += _t1 - _t0; P_["step"] += _t2 - _t1; P_["adapter"] += _t3 - _t2; P_["dit"] += (_t4 - _t3) if reqs else 0; P_["hift"] += _t5 - _t4; P_["render_ticks"] += bool(reqs); P_["reqs"] += len(reqs); P_["total"] += time.perf_counter() - _t0
            if PROF_PRINT > 0 and time.perf_counter() - getattr(s, "_plast", 0) > PROF_PRINT:
                n_ = max(1, P_["ticks"]); w_ = time.perf_counter() - getattr(s, "_plast", time.perf_counter())
                if P_["ticks"] > 3: print("PROF5 win %.1fs ticks %d act %.1f reqs/render %.1f | refresh %.1f step %.1f adapter %.1f dit %.1f hift %.1f total %.1f ms/tick | busy %d%% sleeps %d" % (w_, P_["ticks"], P_["act"] / n_, P_["reqs"] / max(1, P_["render_ticks"]), *[1000 * P_[k] / n_ for k in ("refresh", "step", "adapter", "dit", "hift", "total")], 100 * P_["total"] / max(w_, 1e-3), P_.get("sleep", 0)), flush=True)
                for k in list(P_): P_[k] = 0
                s._plast = time.perf_counter()
            for j, x in act:
                if x.talk_done and x.R.u >= len(x.frames) and (x.R.pend is None or x.R.pend.shape[2] == 0):
                    s.slots[j] = None; E.done[j] = True; x._finish()

class Session:
    def __init__(s, core, lang, voice_key, emit):
        s.c = core; s.lang = lang if lang in ("zh", "en", "ja") else "zh"; s.emit = emit; s.text = ""; s.final = False; s.cancelled = False; s.finished = False
        s.t_a0 = None; s.frames = []; s.talk_done = False; s.slot = None; s.loaded = -1; s.ver = 0; s.ntok = 0; s.loaded_final = False; s.loaded_chars = 0; s.done_ev = threading.Event()
        s.stats = dict(t0=time.perf_counter(), t_first_audio=None, sec=0.0, ms_talker=0, ms_e2=0)
    def add_text(s, t):
        if "t_text0" not in s.stats: s.stats["t_text0"] = round(time.perf_counter() - s.stats["t0"], 3)
        if s.final: return
        if len(s.text) + len(t) > 300 and len(s.c.tok(s.text + t, add_special_tokens=False).input_ids) > s.c.eng.Smax - 64:
            s.stats["truncated"] = "text_limit"; s.final = True; s.ver += 1; return
        s.text += t; s.ver += 1
    def finish(s): s.final = True; s.ver += 1
    def cancel(s): s.cancelled = True
    def ready(s): return s.final or len(s.c.tok(s.text, add_special_tokens=False).input_ids) >= START_TOK
    def _clean(s):
        raw = s.text
        if not s.final: raw = re.sub(r"(\[[a-z_]*|</?[a-z_]*)$", "", raw)
        ch, lead = MK.walk(raw); keep = [c for c in ch if not JUNK.fullmatch(c["c"])]
        _ns = int(os.environ.get("S220_NOSP", "1"))
        if _ns and s.lang in ("zh", "ja"): keep = S.strip_sp(keep, _ns)
        txt = "".join(c["c"] for c in keep).strip(); return txt, keep, lead
    def refresh(s, core, j, curj=0):
        if s.loaded == s.ver or s.talk_done: return
        if s.loaded >= 0 and not (s.final and not s.loaded_final) and s.ntok - curj > int(os.environ.get("S220_REFRESH_AHEAD", "12")): return
        if s.loaded >= 0 and not (s.final and not s.loaded_final) and len(s.text) - s.loaded_chars < int(os.environ.get("S220_REFRESH_MIN", "8")): return
        s.loaded = s.ver; s.loaded_final = s.final; s.loaded_chars = len(s.text); txt, ch, lead = s._clean()
        if not txt: txt = "。"
        enc = core.tok(txt, add_special_tokens=False, return_offsets_mapping=True); n = len(enc.input_ids); s.ntok = n
        v = torch.zeros(n + 1, S.EVD_); v[:, S.EV_NC + 2] = 1.0; off0 = len(("".join(c["c"] for c in ch)) ) - len(("".join(c["c"] for c in ch)).lstrip())
        cs = "".join(c["c"] for c in ch)[off0:]; cl = ch[off0:]
        for k, (b0, b1) in enumerate(enc.offset_mapping):
            for c in range(b0, min(b1, len(cl))):
                if cl[c]["a"] and cl[c]["a"] <= S.EV_NC: v[k, cl[c]["a"] - 1] = 1.0
                v[k, S.EV_NC] = max(v[k, S.EV_NC], cl[c]["sp"][0]); v[k, S.EV_NC + 1] = max(v[k, S.EV_NC + 1], cl[c]["sp"][1])
        lv = None
        if lead and lead <= S.EV_NC: lv = torch.zeros(S.EVD_); lv[lead - 1] = 1.0; lv[S.EV_NC + 2] = 1.0
        has_ev = bool(v[:, :S.EV_NC + 2].sum() > 0) or lv is not None
        if os.environ.get("S220_FORCE_EV") == "1": has_ev = True
        if os.environ.get("S220_DBG_REF") == "1": print("REF", repr(txt), enc.input_ids, v[:, S.EV_NC:].tolist(), flush=True)
        cond = dict(evv=v, lead=lv) if has_ev else None
        if EVTOK and has_ev:
            nid, nev = [], []
            def _row(c_): r_ = torch.zeros(S.EVD_); r_[c_] = 1.0; r_[S.EV_NC + 2] = 1.0; return r_
            if lv is not None: nid.append(S.P_ID); nev.append(_row(int(lv[:S.EV_NC].argmax())))
            for k in range(n):
                r_ = v[k].clone(); pc = int(r_[:S.EV_NC].argmax()) if float(r_[:S.EV_NC].sum()) > 0 else -1; r_[:S.EV_NC] = 0; nid.append(enc.input_ids[k]); nev.append(r_)
                if pc >= 0: nid.append(S.P_ID); nev.append(_row(pc))
            nid.append(core.tok.convert_tokens_to_ids("<|im_end|>")); nev.append(v[n].clone())
            v = torch.stack(nev); lv = None; s.ntok = len(nid) - 1; cond = dict(evv=v, lead=None, ids=nid)
        mem, nn, off, g = FG.prep(core.m, core.tok, [txt], [s.lang], [cond])[0]
        core.eng.load(j, mem, nn, off, g)
        lim = torch.full((core.eng.Smax,), float(core.eng.stall_n))
        for k in (v[:, :S.EV_NC].sum(1) > 0).nonzero().flatten().tolist(): lim[k] = EVCAP_AT; lim[min(k + 1, core.eng.Smax - 1)] = EVCAP_NEXT
        if lv is not None: lim[0] = EVCAP_LEAD
        if EVCAP_END < 20 and cond is not None and cond.get("ids") is not None:
            _ids = cond["ids"]
            for k in (v[:, :S.EV_NC].sum(1) > 0).nonzero().flatten().tolist():
                _rest = core.tok.decode([t for t in _ids[k + 1:-1] if t != S.P_ID])
                if not re.search(r"[0-9A-Za-z\u3040-\u30ff\u4e00-\u9fff]", _rest):
                    lim[k] = min(float(lim[k]), EVCAP_END)
                    for j_ in range(k + 1, len(_ids) - 1): lim[j_] = min(float(lim[j_]), 4.0)
        core.eng.slim[j] = lim.to(core.eng.slim.device)
        if not s.final: core.eng.n[j] = nn + 10 ** 6
    def _out(s, w):
        if not len(w): return
        if s.stats["t_first_audio"] is None: s.stats["t_first_audio"] = round(time.perf_counter() - s.stats["t0"], 3); s.t_a0 = time.perf_counter()
        s.stats["sec"] = round(s.stats["sec"] + len(w) / 24000, 2)
        if not s.cancelled: s.emit((np.clip(w, -1, 1) * 32767).astype(np.int16).tobytes())
    def _finish(s): s.finished = True; s.done_ev.set()
    def run(s):
        s.c.submit(s); s.done_ev.wait()
