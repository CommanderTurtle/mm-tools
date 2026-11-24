"""Sophon-Realtime-32B-Flash-4bit on Apple silicon (MLX): the full-duplex loop of the server plugin (sophon_realtime/duplex.py + proc.py + talker.py), ported.
Hearing, turn-taking and replying are all done by our own model: the user's audio goes straight into the thinker through its own audio tower; whether the
user has finished is the thinker's own answer to a turn question asked inside the reply's context (same check schedule and thresholds as the server),
so the audio is computed once and the reply continues from the same cache."""
import collections, hashlib, json, os, random, re, threading, time, uuid, queue
import numpy as np, mlx.core as mx
from mlx_lm import load as lm_load, stream_generate
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_audio.tts.utils import load_model as tts_load
from mlx_ear import load_ear, embed_audio

SR = 16000; AUD = 151669; LET = "ABCDEFGHIJ"
MINI_AB = {"zh": "用户说完了吗？A. 说完了 B. 没说完 只答字母。", "en": "Has the user finished? A. finished B. not finished Letter only."}
STREAM_S = float(__import__("os").environ.get("SOPHON_STREAM_S", "0.08"))   # talker chunk length
FIRST_MS = int(__import__("os").environ.get("SOPHON_FIRST_MS", "20"))       # first turn question after this much silence
CHECKS = [0.2, 0.35, 0.5, 0.8, 1.3, 2.0]; THRS = [0.95, 0.85, 0.7, 0.5, 0.35, 0.25]; FALLBACK_S = 2.0; MERGE_WINDOW = 2.0
SHARED = True        # turn question inside the reply context (one shared prefix); False = the server plugin's separate prompt
HIST_TURNS = 8; HIST_SEC = 120.0; LEAD = int(__import__("os").environ.get("SOPHON_LEAD", "3")); MAX_REPLY_TOK = 400
SPK_CHECK_MS = 900          # speech this long is compared with the owner's voice (a shorter stretch gives too noisy an embedding)
OWNER_N = 5                 # the owner's voice is the mean of up to this many of their utterances
BYSTANDER_QUIET_MS = 600    # after speech that is not the owner's, input is ignored until the room has been this quiet
NOISE_CAP = 0.02   # the ambient-floor estimate never rises above this RMS, so speech is still detectable in a loud room
ECHO_K_START, ECHO_K, ECHO_ADAPT_S = 0.5, 0.12, 3.0   # speech threshold while playing = k x playback loudness (see Session.feed)
BC_RESUME, BC_THR, BC_WINDOW = True, 0.5, 4.0   # user only says "mm-hmm" over a reply that got cut: continue that reply (as the server)
RECHECK_MS = 900   # a turn question takes ~0.85 s on an M4 Max: don't start one that would still be running at the fallback
TURN_Q = {"zh": ("用户现在处于什么话轮状态？", ["话说完了，可以接话", "句子没说完，还在继续", "简短附和，不是要接话", "还在组织语言，或要求先等一下"]),
          "en": ("Has the user finished speaking?", ["finished, the assistant can reply", "not finished, still talking",
                                                      "just a backchannel, not taking the turn", "hesitating or asking to wait"])}
TPL = {"en": ("Question: {q}", "Options:", "Answer with only the letter of the correct option.", "The user said: <|audio|>", 'The assistant had just said: "{c}"'),
       "zh": ("问题：{q}", "选项：", "请只回答正确选项的字母。", "用户说：<|audio|>", "助手刚才说的是：“{c}”")}


def perm_of(qid, opts, seed=0):
    key = "\x1f".join([qid, "\x1e".join(opts), str(seed)]).encode()
    rng = random.Random(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))
    p = list(range(len(opts))); rng.shuffle(p); return p


def turn_question(lang, said):
    q, opts = TURN_Q[lang]; tq, to, ti, head, sd = TPL[lang]; p = perm_of("turn", opts)
    s = " ".join((said or "").split())[-300:]
    text = (sd.format(c=s) + "\n\n" if s else "") + head + "\n\n" + "\n".join([tq.format(q=q), "", to] + [f"{LET[i]}. {opts[j]}" for i, j in enumerate(p)] + ["", ti])
    return text, p


POINT = {"[laughter]", "[breath]", "[sigh]", "[cough]", "[quick_breath]"}; SPAN = {"laughter", "strong"}
CLOSE = {"[": "]", "<": ">", "(": ")", "（": "）", "【": "】"}


def tag_clean(s, final):
    """= proc.tag_clean: keep whitelisted sound tags, drop stage directions and markdown."""
    out = []; open_ = []; buf = s

    def emit_tag(t):
        m = re.fullmatch(r"<(/?)([a-z_\-]+)>", t)
        if t in POINT:
            out.append(t); return
        if m and m.group(2) in SPAN:
            name = m.group(2)
            if not m.group(1):
                if name not in open_: open_.append(name); out.append(t)
            elif name in open_:
                open_.remove(name); out.append(t)

    def text(v):
        v = v.replace("→", ", so ").replace("⇒", ", so ").replace("≥", " at least ").replace("≤", " at most ")
        if "strong" in open_:
            m = re.search(r"[，。！？、,.!?;；:：]", v)
            if m:
                if m.start(): out.append(v[:m.start()])
                out.append("</strong>"); open_.remove("strong"); v = v[m.start():]
        if v: out.append(v)

    while buf:
        idx = [k for k in (buf.find(c) for c in list(CLOSE) + ["*", "#", "`"]) if k >= 0]
        i = min(idx) if idx else -1
        if i < 0:
            text(buf); buf = ""; break
        if i > 0:
            text(buf[:i]); buf = buf[i:]
        c = buf[0]
        if c in "*#`":
            buf = buf[1:]; continue
        j = buf.find(CLOSE[c], 1)
        if j < 0:
            if len(buf) > 40:
                buf = buf[1:]; continue
            break
        tok, buf = buf[:j + 1], buf[j + 1:]
        if c in "[<": emit_tag(tok)
    if final:
        if buf:
            t = re.sub(r"[\[<(（【].*$", "", buf)
            if t: text(t)
        for name in reversed(open_): out.append(f"</{name}>")
    return "".join(out)


def resample(a, si, so):
    n = int(round(len(a) * so / si))
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32) if n > 1 else a


class Brain:
    """The three parts of the model on MLX. Used from ONE thread (the GPU worker)."""

    def __init__(self, root, lang="zh", voice="chelsie", sys_prompt=None):
        import os
        ex = lambda *p: next((os.path.join(root, x) for x in p if os.path.exists(os.path.join(root, x))), os.path.join(root, p[0]))
        self.lang = lang
        self.ear, self.fe = load_ear(ex("ear", "thinker"), mx.bfloat16)
        self.lm, self.tok = lm_load(ex("thinker_mlx", "mlx_thinker"))
        self.tts = tts_load(f"{root}/talker"); cfg = self.tts.config; tc = cfg.talker_config
        self.SYS = json.load(open(sys_prompt or f"{root}/system_prompt.json"))[lang]
        self.LID = [self.tok.encode(c, add_special_tokens=False)[0] for c in LET[:4]]
        self.te = lambda ids: self.tts.talker.text_projection(self.tts.talker.get_text_embeddings()(mx.array([ids])))
        ce = lambda ids: self.tts.talker.get_input_embeddings()(mx.array([ids]))
        self.bos, self.eos, self.pad = [self.te([i]) for i in (cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id)]
        xv = mx.array(np.load(f"{root}/voices/{voice}_{lang}.npy")).reshape(1, 1, -1).astype(self.bos.dtype)
        lid = tc.codec_language_id["chinese" if lang == "zh" else "english"]
        self.codec = mx.concatenate([ce([tc.codec_think_id, tc.codec_think_bos_id, lid, tc.codec_think_eos_id]), xv,
                                     ce([tc.codec_pad_id, tc.codec_bos_id])], 1)
        self.role = self.te(self.tts.tokenizer.encode("<|im_start|>assistant\n"))
        self.cache = make_prompt_cache(self.lm); self.sig = []; self._emb = {}
        # Turn question style, set per model in the weights' config.json: "short" (default; the two-option question inside the reply
        # context, tuned on 32B) or "separate" (the server plugin's four-option question in its own prompt; the 4B model needs it:
        # Smart Turn sample, 4B 8-bit: short 72 / 74 % vs separate 90 / 90 % en / zh).
        mcfg = os.path.join(root, "config.json")
        self.turn_style = (json.load(open(mcfg)).get("turn_question") if os.path.exists(mcfg) else None) or "short"

    def embed(self, key, a):
        if key not in self._emb:
            e, _ = embed_audio(self.ear, self.fe, a); self._emb[key] = e
        return self._emb[key]

    def _layout(self, msgs, audios):
        """Chat template -> token ids, input embeddings and a signature (audio tokens identified by clip key) for KV-cache reuse."""
        s = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        parts = s.split("<|audio|>"); ids, sig, embs = [], [], []
        for i, p in enumerate(parts):
            t = self.tok.encode(p, add_special_tokens=False)
            if t: ids += t; sig += t; embs.append(self.lm.model.embed_tokens(mx.array(t)))
            if i < len(audios):
                k, a = audios[i]; e = self.embed(k, a).astype(mx.bfloat16)
                ids += [AUD] * e.shape[0]; sig += [(k, j) for j in range(e.shape[0])]; embs.append(e)
        return ids, mx.concatenate(embs), sig

    def _prefill(self, ids, E, sig, upto, cancel=None, chunk=64):
        """Run tokens [.., upto) through the thinker reusing the KV cache for the longest prefix already there (same tokens / same audio clips).
        Returns the logits of position upto-1 when it had to be computed (else None)."""
        n = 0
        while n < min(len(sig), len(self.sig), upto) and sig[n] == self.sig[n]: n += 1
        n = min(n, upto - 1)
        have = self.cache[0].offset
        if have > n: trim_prompt_cache(self.cache, have - n)
        self.last_prefill = upto - n; s = n
        while s < upto:                                   # in chunks, so a speculative prefill can be dropped as soon as the user speaks again
            if cancel is not None and s > n and cancel():
                self.sig = sig[:s]; return None
            e = min(upto, s + chunk)
            out = self.lm(mx.array([ids[s:e]]), cache=self.cache, input_embeddings=E[s:e][None]); mx.eval(out, [c.state for c in self.cache])
            self.sig = sig[:e]; s = e
        return out[0, -1]

    def _conv(self, history):
        msgs = [{"role": "system", "content": self.SYS}]; audios = []
        for h in history:
            msgs.append({"role": "user", "content": "<|audio|>"}); audios.append((h["ukey"], h["user"]))
            if h.get("a_audio") is not None:
                msgs.append({"role": "assistant", "content": "<|audio|>"}); audios.append((h["akey"], h["a_audio"]))
            elif h.get("a_text"):
                msgs.append({"role": "assistant", "content": h["a_text"]})
        return msgs, audios

    def turn_check_shared(self, history, user, cancel=None, full=False):
        """Turn question asked inside the reply's own context (system prompt + history + this user audio, then the question), so the
        expensive part -- everything up to the end of the user audio -- is computed once and reused by the reply.
        Default: the short two-option question (~21 tokens instead of ~64; same accuracy on Smart Turn, see results doc). full=True: the
        server's four-option question, which also gives P(backchannel) -- used right after a barge-in, when "mm-hmm" must resume the reply."""
        if full:
            text, perm = turn_question(self.lang, "")
            block = text.split("\n\n", 1)[1]                          # drop the "用户说：<|audio|>" header
        else:
            block = MINI_AB[self.lang]; perm = None
        msgs, audios = self._conv(history)
        msgs.append({"role": "user", "content": "<|audio|>\n\n" + block}); audios.append(user)
        ids, E, sig = self._layout(msgs, audios)
        lg = self._prefill(ids, E, sig, len(ids), cancel)
        if lg is None: return None, None
        if perm is None:
            z = np.array(mx.softmax(lg.astype(mx.float32)[mx.array(self.LID[:2])])); return float(z[0]), 0.0
        z = np.array(mx.softmax(lg.astype(mx.float32)[mx.array(self.LID)]))
        return float(z[perm.index(0)]), float(z[perm.index(2)])

    def prep(self, history, user):
        """Speculative prefill as soon as the user pauses: system prompt + history + this user audio go into the KV cache, so the turn
        question (and then the reply) only add their own few tokens."""
        msgs, audios = self._conv(history)
        msgs.append({"role": "user", "content": "<|audio|>"}); audios.append(user)
        ids, E, sig = self._layout(msgs, audios)
        end = max(i for i, x in enumerate(sig) if isinstance(x, tuple) and x[0] == user[0]) + 1
        self._prefill(ids, E, sig, end)
        return end

    def warmup(self):
        a = np.zeros(SR, np.float32); self.turn_check("warm", a, ""); self.reply([], ("warm2", a), lambda x: None, lambda: False, max_tok=8)
        self.sig = []; self.cache = make_prompt_cache(self.lm)
        # keep the system prompt (and later the history) in the KV cache: each reply then only prefills the new speech
        ids, E, sig = self._layout([{"role": "system", "content": self.SYS}, {"role": "user", "content": "<|audio|>"}], [("warm3", a)])
        n = sig.index(("warm3", 0)) if ("warm3", 0) in sig else 0
        self.lm(mx.array([ids[:n]]), cache=self.cache, input_embeddings=E[:n][None]); mx.eval([c.state for c in self.cache]); self.sig = sig[:n]
        self._emb.clear()

    def turn_p(self, history, user, said, cancel=None, full=False):
        """P(finished), P(backchannel) for the user audio, in the model's turn question style."""
        if self.turn_style == "separate": return self.turn_check(user[0], user[1], said)
        return self.turn_check_shared(history, user, cancel=cancel, full=full)

    def turn_check(self, k, audio, said):
        text, perm = turn_question(self.lang, said)
        ids, E, _ = self._layout([{"role": "user", "content": text}], [(k, audio)])
        z = np.array(mx.softmax(self.lm(mx.array([ids]), input_embeddings=E[None])[0, -1].astype(mx.float32)[mx.array(self.LID)]))
        self._emb.pop(k, None)
        return float(z[perm.index(0)]), float(z[perm.index(2)])

    def reply(self, history, user, emit, cancelled, max_tok=MAX_REPLY_TOK, said=""):
        """Thinker writes, talker speaks while it writes (proc.py rules: cleaned text, stable tokens only, LEAD tokens before the first chunk)."""
        msgs, audios = self._conv(history)
        msgs.append({"role": "user", "content": "<|audio|>"}); audios.append(user)
        ids, E, sig = self._layout(msgs, audios)
        if said:                                   # resume: what was already spoken starts the assistant turn, the model simply continues it
            t = self.tok.encode(said, add_special_tokens=False); ids = ids + t; sig = sig + t
            E = mx.concatenate([E, self.lm.model.embed_tokens(mx.array(t))])
        self._prefill(ids, E, sig, len(ids) - 1)
        gen = stream_generate(self.lm, self.tok, mx.array(ids[-1:]), input_embeddings=E[-1:], prompt_cache=self.cache, max_tokens=max_tok,
                              sampler=make_sampler(temp=0.7, top_p=0.9))
        st = {"raw": "", "clean": "", "sent": 0, "buf": [], "done": False, "ntok": 0, "calls": 0}
        t0 = time.time(); self.timing = {}
        brain = self

        def pull():
            fin = False
            try:
                r = next(gen); st["raw"] += r.text; st["ntok"] += 1
                if st["ntok"] == 1: brain.timing["tok1"] = round(time.time() - t0, 3)
            except StopIteration:
                fin = True
            txt = tag_clean(st["raw"], fin).lstrip(); tid = brain.tok.encode(txt, add_special_tokens=False) if txt else []
            stable = tid if fin else tid[:-1]
            if len(stable) > st["sent"]:
                st["buf"] += [brain.te([t]) for t in stable[st["sent"]:]]; st["sent"] = len(stable)
            if fin: st["buf"].append(brain.eos); st["done"] = True
            st["clean"] = txt

        while not st["done"] and len(st["buf"]) < LEAD:
            if cancelled(): break
            pull()
        out = []; self.timing["lead"] = round(time.time() - t0, 3)
        if st["buf"] and not cancelled():
            first = st["buf"].pop(0)

            class Trail:
                @property
                def shape(s):
                    st["calls"] += 1
                    if cancelled(): raise KeyboardInterrupt
                    if not st["done"] and st["calls"] % 2 == 0: pull()
                    while not st["done"] and len(st["buf"]) <= st.get("used", 0):   # never let the talker run out of text mid-reply
                        if cancelled(): raise KeyboardInterrupt
                        pull()
                    return (1, len(st["buf"]), brain.pad.shape[-1])

                def __getitem__(s, k):
                    st["used"] = k[1].start + 1; return st["buf"][k[1].start]

            pre = mx.concatenate([self.role, mx.concatenate([mx.broadcast_to(self.pad, (1, 5, self.pad.shape[-1])), self.bos], 1) + self.codec[:, :-1],
                                  first + self.codec[:, -1:]], 1)
            self.tts._prepare_generation_inputs = lambda *a, **k: (pre, Trail(), self.pad)
            try:
                for r in self.tts.generate(text="x", lang_code="chinese", stream=True, streaming_interval=STREAM_S, temperature=0.7, max_tokens=4000):
                    if cancelled(): break
                    a = np.array(r.audio, dtype=np.float32); out.append(a); emit(a)
            except KeyboardInterrupt:
                pass
        while not st["done"] and not cancelled(): pull()       # finish the text for the history even if the talker stopped early
        self.sig += [("gen", i) for i in range(st["ntok"])]
        return st["clean"] or st["raw"], (np.concatenate(out) if out else None)


class Session:
    """One conversation = the plugin's SophonBuffer logic: energy VAD on 20 ms frames, turn questions at 0.2/0.35/0.5/0.8/1.3/2.0 s of silence
    (thresholds 0.95..0.25), fallback at 2 s, merge-and-abort when the user resumes within 2 s of a reply, barge-in cuts the reply and the history."""

    def __init__(self, brain, play, log=print, speaker=None, speaker_thr=0.5, dump_dir=None):
        self.dump_dir = dump_dir
        self.spk = speaker; self.spk_thr = speaker_thr; self.owner = None; self.owner_n = 0; self.uid = 0; self.spk_uid = -1
        self.bystander = False; self.quiet_ms = 0
        self.b = brain; self.play = play; self.log = log; self.jobs = queue.Queue(); self.lock = threading.Lock()
        self.history = []; self.last_said = ""; self.noise = 0.003; self.floor = collections.deque(maxlen=200); self.floor_n = 0; self.gen = 0; self.sgen = 0; self._reset()
        self.far_hist = collections.deque([0.0] * 10, maxlen=10); self.played_s = 0.0; self.cut = None
        self.pre = bytearray(); self.prev_user = None; self.t_reply = 0.0; self.reply = None; self.events = []
        threading.Thread(target=self._worker, daemon=True).start()
        if speaker is not None:
            self.spk_q = queue.Queue(); threading.Thread(target=self._speaker_worker, daemon=True).start()

    def _speaker_worker(self):
        """Compares the voice of the utterance in progress with the owner's, off the model worker's thread (which may be busy for
        seconds with a reply). The first voice heard after connecting is the owner's; it is the one that said the wake word."""
        while True:
            uid, audio = self.spk_q.get()
            try:
                e = self.spk.embed(audio)
                if e is None: continue
                with self.lock:
                    if uid != self.uid: continue
                    if self.owner is None:
                        self.owner = e.copy(); self.owner_n = 1; self.ev("owner_enrolled", speech_s=round(len(audio) / SR, 2)); continue
                    o = self.owner / np.linalg.norm(self.owner); score = float(e @ o); mine = score >= self.spk_thr
                    self.ev("speaker", score=round(score, 3), owner=mine)
                    if self.dump_dir:
                        import wave; os.makedirs(self.dump_dir, exist_ok=True)
                        with wave.open(os.path.join(self.dump_dir, f"{time.strftime('%H%M%S')}-{score:.2f}-{'owner' if mine else 'other'}.wav"), "wb") as w:
                            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
                    if mine:
                        if self.owner_n < OWNER_N: self.owner = self.owner + e; self.owner_n += 1     # the voice model improves with more of the owner's speech
                    else: self._bystander()
            except Exception as ex:
                self.ev("speaker_error", error=repr(ex))

    def _bystander(self):
        """The speech in progress is not the owner's: drop it, and put back the reply it cut off. Input stays ignored until the room has
        been quiet for BYSTANDER_QUIET_MS, so continuous speech (a television) does not start it again a moment later."""
        self.bystander = True; self.quiet_ms = 0; self.ev("bystander")
        if self.cut is not None: self._resume()
        else: self._reset()

    def _reset(self):
        self.utt = bytearray(); self.voiced = 0; self.speech_ms = 0; self.silence_ms = 0; self.next_check = 0; self.gen += 1
        self.inflight = False; self.pending = False; self.t_sil = 0.0; self.p_last = -1.0; self.pb_last = 0.0; self.max_pf = 0.0

    def ev(self, name, **kw):
        kw = {"ev": name, "t": round(time.time(), 3), **kw}; self.events.append(kw); self.log(json.dumps(kw, ensure_ascii=False))

    def _worker(self):
        while True:
            kind, arg = self.jobs.get()
            try:
                if kind == "check" and arg[7] != self.sgen:
                    continue                                       # the user spoke again or a reply already started: this check is stale, skip it
                if kind == "check":
                    k, audio, said, gen, thr, idx, t_sil, sg = arg
                    t = time.time()
                    if SHARED: pf, pb = self.b.turn_p(list(self.history), (k, audio), said, cancel=lambda: sg != self.sgen,
                                                      full=self.cut is not None and time.time() - self.cut["t"] < BC_WINDOW + FALLBACK_S)
                    else: pf, pb = self.b.turn_check(k, audio, said)
                    if pf is None:
                        with self.lock: self.inflight = False if self.inflight == k else self.inflight
                        self.ev("turn_check_dropped", n=idx, tokens=self.b.last_prefill, sec=round(time.time() - t, 3)); continue
                    with self.lock:
                        stale = gen != self.gen; self.inflight = False if self.inflight == k else self.inflight
                        r = self.reply
                        if stale and sg == self.sgen and r is not None and not r.get("started") and r.get("sgen") == sg:
                            r["ukey"], r["user"] = k, audio      # a reply was queued meanwhile (fallback): let it reuse this question's prefill
                        if not stale:
                            sil = time.time() - t_sil
                            # the answer arrives later than on the server: judge it with the threshold for the silence we are at NOW,
                            # and keep it, so that later check points with a lower threshold can still accept it without recomputing
                            thr = min(thr, THRS[max([0] + [i for i, c in enumerate(CHECKS) if c <= sil])]); self.p_last = pf
                            self.ckey = k; self.caudio = audio; self.cgen = gen; self.pb_last = pb; self.max_pf = max(self.max_pf, pf)
                            if self._is_bc(pf, pb): self.pending = "resume"
                            elif pf >= thr: self.pending = True
                    self.ev("turn_check", n=idx, p_finished=round(pf, 3), p_backchannel=round(pb, 3), thr=thr, sec=round(time.time() - t, 3),
                            after_silence_s=round(time.time() - t_sil, 3), stale=stale, decision="reply" if (pf >= thr and not stale) else "wait")
                elif kind == "prep":
                    k, audio, gen = arg
                    if gen == self.gen:
                        t = time.time(); n = self.b.prep(list(self.history), (k, audio)); self.ev("prep", tokens=self.b.last_prefill, sec=round(time.time() - t, 3))
                elif kind == "reply":
                    self._run_reply(arg)
                elif kind == "resume":
                    self._run_resume(arg)
            except Exception as exc:  # noqa: BLE001
                import traceback; self.ev("error", where=kind, err=repr(exc)[:300], tb=traceback.format_exc()[-800:])

    def _run_reply(self, r):
        t0 = time.time()
        with self.lock: r["started"] = True

        def emit(a):
            if r["cancel"]: return
            if r["t_out"] is None:
                r["t_out"] = time.time(); self.ev("first_audio", after_silence_s=round(r["t_out"] - r["t_sil"], 3), prefill_tok=self.b.last_prefill, since_start=round(r["t_out"] - t0, 3), **self.b.timing)
            r["gen_s"] += len(a) / 24000; self.play(a)

        text, a24 = self.b.reply(self.history, (r["ukey"], r["user"]), emit, lambda: r["cancel"])
        r["text"] = text; r["a16"] = resample(a24, 24000, SR) if a24 is not None else None; r["done"] = True
        with self.lock:
            if not r["cancel"]:
                self._commit(r, r["a16"], text); self.last_said = text
            elif r.get("cut") is not None and r.get("entry") is None:      # barged in while still generating: keep what was actually heard
                played, frac = r["cut"]; said = text[: int(len(text) * frac)]
                a16 = r["a16"][: int(played * SR)] if r["a16"] is not None and played > 0.5 else None
                if said or a16 is not None: self._commit(r, a16, said + "……（被用户打断）")
                self.last_said = said
        self.ev("reply_done", text=text, audio_s=round(r["gen_s"], 2), cancelled=r["cancel"], sec=round(time.time() - t0, 2), hist=len(self.history))

    def _is_bc(self, pf, pb, extra=0.0):
        """The user only back-channelled ("mm-hmm", "yeah") right after cutting a reply: continue that reply instead of answering."""
        c = self.cut
        return (BC_RESUME and c is not None and c["r"].get("entry") is not None and time.time() - c["t"] < BC_WINDOW + extra
                and self.speech_ms < 1500 and pb >= BC_THR and pb > pf)

    def _run_resume(self, r):
        t0 = time.time(); r["started"] = True; r0 = r["resume"]; e = r0["entry"]
        said = e["a_text"].replace("……（被用户打断）", "")
        def emit(a):
            if r["cancel"]: return
            if r["t_out"] is None:
                r["t_out"] = time.time(); self.ev("first_audio", after_silence_s=round(r["t_out"] - r["t_sil"], 3), prefill_tok=self.b.last_prefill, resume=True)
            r["gen_s"] += len(a) / 24000; self.play(a)
        hist = [h for h in self.history if h is not e]
        text, a24 = self.b.reply(hist, (e["ukey"], e["user"]), emit, lambda: r["cancel"], said=said)
        r["text"] = text; r["done"] = True
        with self.lock:
            if any(h is e for h in self.history):    # identity: entries hold numpy arrays, == would compare them
                full = said + text
                if r["cancel"] and r.get("cut") is not None: full = said + text[: int(len(text) * r["cut"][1])] + "……（被用户打断）"
                e["a_text"] = full; e["a_audio"] = None; r["entry"] = e
        self.ev("reply_done", text=text, resumed_from=said[-30:], cancelled=r["cancel"], sec=round(time.time() - t0, 2), hist=len(self.history))

    def _commit(self, r, a16, text):
        e = {"user": r["user"], "ukey": r["ukey"], "a_audio": a16, "akey": uuid.uuid4().hex, "a_text": text}
        r["entry"] = e; self.history.append(e)
        while len(self.history) > HIST_TURNS or sum(len(h["user"]) + (len(h["a_audio"]) if h["a_audio"] is not None else 0) for h in self.history) / SR > HIST_SEC:
            self.history.pop(0)

    def feed(self, f, far=None):
        """Call every 20 ms with 320 float32 samples (16 kHz mono), after echo cancellation; far = what the speaker played in those 20 ms.
        While the speaker plays, what is left of its echo must not count as the user speaking: the speech threshold follows the
        loudness of the last 200 ms of playback (higher during the first seconds, while the echo canceller is still adapting)."""
        f = np.asarray(f, np.float32); r = float(np.sqrt(np.mean(f * f))); fb = f.tobytes()
        if far is not None:
            fr = float(np.sqrt(np.mean(np.square(far, dtype=np.float32)))); self.far_hist.append(fr)
            if fr > 1e-3: self.played_s += 0.02
        with self.lock:
            k = ECHO_K_START if self.played_s < ECHO_ADAPT_S else ECHO_K
            # ambient floor: the quietest tenth of the last 4 s, speech or not. The estimate below only moves while nobody
            # is speaking, so a room louder than the fixed threshold would otherwise read as one endless utterance.
            self.floor.append(r); self.floor_n += 1
            if self.floor_n % 25 == 0 and len(self.floor) == self.floor.maxlen:
                self.noise = max(self.noise, min(sorted(self.floor)[len(self.floor) // 10], NOISE_CAP))
            thr = max(0.012, self.noise * 3.5, k * max(self.far_hist))
            if self.bystander:
                self.quiet_ms = 0 if r > thr else self.quiet_ms + 20
                if self.quiet_ms >= BYSTANDER_QUIET_MS: self.bystander = False; self.voiced = 0
                return
            if self.speech_ms == 0:
                self.pre.extend(fb)
                if len(self.pre) > int(0.6 * SR) * 4: del self.pre[: len(self.pre) - int(0.6 * SR) * 4]
                if r > thr:
                    self.voiced += 20
                    if self.voiced >= 100: self._onset()
                else:
                    self.voiced = 0; self.noise = 0.98 * self.noise + 0.02 * min(r, 0.05)
            else:
                self.utt.extend(fb)
                if r > thr:
                    if self.silence_ms > 0: self.gen += 1; self.sgen += 1; self.inflight = False; self.pending = False; self.p_last = -1.0
                    self.speech_ms += 20; self.silence_ms = 0; self.next_check = 0
                else:
                    if self.silence_ms == 0: self.t_sil = time.time()
                    self.silence_ms += 20
            if self.spk is not None and self.spk_uid != self.uid and self.speech_ms >= SPK_CHECK_MS:
                self.spk_uid = self.uid; self.spk_q.put((self.uid, np.frombuffer(bytes(self.utt), np.float32).copy()))
            if self.speech_ms < 120: return
            if SHARED:
                # an earlier answer for this pause meets the (lower) threshold of the check point we have reached: reply, no new compute
                if not self.pending and self.p_last >= 0 and self.cgen == self.gen:
                    i = max([0] + [j for j, c in enumerate(CHECKS) if self.silence_ms >= c * 1000])
                    if self.p_last >= THRS[i]: self.pending = True; self.ev("accept_later", p=round(self.p_last, 3), thr=THRS[i])
                start = FIRST_MS if self.next_check == 0 else CHECKS[max(self.next_check, 3)] * 1000 if self.next_check < len(CHECKS) else 1e9
                if not self.inflight and not self.pending and self.speech_ms >= 300 and self.silence_ms >= start and (self.next_check == 0 or (self.p_last < THRS[4] and self.silence_ms + RECHECK_MS <= FALLBACK_S * 1000)):
                    # first question as soon as the pause starts (prefill + question in one pass); one re-check with the longer audio from 0.8 s
                    self.next_check = 1 if self.next_check == 0 else max(self.next_check + 1, 4)
                    while self.next_check < len(CHECKS) and self.silence_ms >= CHECKS[self.next_check] * 1000: self.next_check += 1
                    key, audio = uuid.uuid4().hex, np.frombuffer(bytes(self.utt), np.float32).copy(); self.inflight = key   # id of the question in flight
                    self.jobs.put(("check", (key, audio, self.last_said, self.gen, THRS[max(0, self.next_check - 1)], self.next_check, self.t_sil, self.sgen)))
            elif not self.inflight and not self.pending and self.next_check < len(CHECKS) and self.silence_ms >= CHECKS[self.next_check] * 1000:
                thr_ = THRS[self.next_check]; self.next_check += 1
                key, audio = uuid.uuid4().hex, np.frombuffer(bytes(self.utt), np.float32).copy(); self.inflight = key
                self.ckey = key; self.caudio = audio; self.cgen = self.gen
                self.jobs.put(("check", (key, audio, self.last_said, self.gen, thr_, self.next_check, self.t_sil, self.sgen)))
            if self.silence_ms >= FALLBACK_S * 1000 and not self.pending:
                self.pending = "resume" if self._is_bc(self.max_pf, self.pb_last, extra=FALLBACK_S) else True; self.ev("fallback")
            if self.pending == "resume": self._resume(); return
            if self.pending:
                if SHARED and getattr(self, "cgen", None) == self.gen: ukey, user = self.ckey, self.caudio     # reply from the audio the last check heard (cached)
                else: ukey, user = uuid.uuid4().hex, np.frombuffer(bytes(self.utt), np.float32).copy()
                r = {"user": user, "ukey": ukey, "cancel": False, "done": False, "gen_s": 0.0, "t_sil": self.t_sil, "t_out": None, "sgen": self.sgen}
                self.ev("reply_start", user_s=round(len(user) / SR, 2), after_silence_s=round(time.time() - self.t_sil, 3))
                self.reply = r; self.prev_user = user; self.t_reply = time.time(); self._reset(); self.jobs.put(("reply", r))

    def _resume(self):
        """Continue the reply that speech cut off (a backchannel, or a bystander)."""
        cut = self.cut; self.cut = None; r0 = cut["r"]
        r = {"resume": r0, "cancel": False, "done": False, "gen_s": 0.0, "t_sil": self.t_sil, "t_out": None, "sgen": self.sgen,
             "user": None, "ukey": None}
        self.ev("resume_start", after_silence_s=round(time.time() - self.t_sil, 3), p_backchannel=round(self.pb_last, 3))
        self.reply = r; self.prev_user = None; self.t_reply = time.time(); self._reset(); self.jobs.put(("resume", r))

    def _onset(self):
        r = self.reply; now = time.time(); self.sgen += 1; self.uid += 1
        if r is not None and not r["cancel"] and not (r["done"] and r["t_out"] is not None and now - r["t_out"] - 0.25 >= r["gen_s"] - 0.3):
            if self.prev_user is not None and now - self.t_reply < MERGE_WINDOW:
                # user resumed right after we started replying: we cut in too early -> abort that reply and merge the speech into this turn
                r["cancel"] = True; self.play(None); self.utt.extend(self.prev_user.tobytes())
                if r.get("entry") is not None: self.history = [h for h in self.history if h is not r["entry"]]   # by identity (entries hold arrays)
                self.ev("merge_abort", since_reply_s=round(now - self.t_reply, 2))
            elif r["t_out"] is not None and r.get("resume") is not None:
                # cut again while continuing a reply: keep the original history entry; _run_resume writes what was actually said into it
                played = max(0.0, now - r["t_out"] - 0.25); r["cancel"] = True; self.play(None)
                r["cut"] = (played, max(0.0, min(1.0, played / max(r["gen_s"], 1e-3)))); self.cut = {"t": now, "r": r["resume"]}
                self.ev("barge_in", played_s=round(played, 2), said="", resumed=True)
            elif r["t_out"] is not None:
                played = now - r["t_out"] - 0.25
                r["cancel"] = True; self.play(None)
                txt = r.get("text") or ""; said = txt[: int(len(txt) * max(0.0, min(1.0, played / max(r["gen_s"], 1e-3))))]
                if r.get("entry") is not None: self.history = [h for h in self.history if h is not r["entry"]]   # by identity (entries hold arrays)
                r["cut"] = (max(played, 0.0), max(0.0, min(1.0, played / max(r["gen_s"], 1e-3))))
                if r["done"]:
                    a16 = r.get("a16"); a16 = a16[: int(max(played, 0) * SR)] if a16 is not None and played > 0.5 else None
                    self._commit(r, a16, said + "……（被用户打断）")
                self.last_said = said; self.ev("barge_in", played_s=round(played, 2), said=said)
                self.cut = {"t": now, "r": r}                       # what was said is known once the reply has finished generating
        self.prev_user = None
        self.utt.extend(self.pre); self.pre.clear(); self.speech_ms = self.voiced; self.silence_ms = 0; self.next_check = 0
