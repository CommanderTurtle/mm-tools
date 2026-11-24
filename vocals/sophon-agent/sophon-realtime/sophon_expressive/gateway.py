# Copyright 2026 Adventists AI. Apache-2.0.
"""Sophon-Realtime v1 gateway: browser / client WebSocket (/ws) -> turn-taking by the thinker -> reply (thinker + reply LoRA) -> speaking engine -> audio.
Serves the web demo at /."""
import argparse, asyncio, base64, hashlib, io, json, os, random, re, time, collections
import numpy as np, soundfile as sf, aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--dj", default="http://127.0.0.1:8300")
ap.add_argument("--tts", default="http://127.0.0.1:8097")
ap.add_argument("--style-dj", default="", help="instance for turn-taking / style questions (empty = same as --dj)")
ap.add_argument("--tts-model", default="")
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--root-path", default="")
ap.add_argument("--rate", type=int, default=30, help="max turns per IP per minute")
ap.add_argument("--max-sec", type=float, default=30.0)
ap.add_argument("--reply-max-tokens", type=int, default=1000, help="max tokens per reply")
ap.add_argument("--reply-temp", type=float, default=0.5, help="reply sampling temperature")
ap.add_argument("--reply-rep", type=float, default=1.0)
ap.add_argument("--sys-append", default="")
ap.add_argument("--reply-model", default="sophon", help="served LoRA name used for replies")
ap.add_argument("--sys-tag", action="store_true", default=True)
ap.add_argument("--keep-tags", action="store_true", default=True)
ap.add_argument("--style-dims", default="emo", choices=["all", "emo"])
ap.add_argument("--allow-angry", action="store_true")
ap.add_argument("--tts-temp", type=float, default=0.6)
ap.add_argument("--tts-topk", type=int, default=50)
ap.add_argument("--turn-checks", default="0.3,0.5,0.8,1.3,2.0,3.0", help="pause lengths (s) at which turn-taking is checked")
ap.add_argument("--turn-thresholds", default="0.8,0.6,0.5,0.35,0.25,0.15",)
ap.add_argument("--spec-at", type=float, default=0.08)
ap.add_argument("--merge-window", type=float, default=2.0)
ap.add_argument("--turn-mode", default="vad", choices=["vad", "poll"])
ap.add_argument("--poll-ms", type=int, default=100)
ap.add_argument("--turn-dither", type=float, default=0.0)
ap.add_argument("--poll-th", type=float, default=0.9)
ap.add_argument("--poll-spec-th", type=float, default=-1)
ap.add_argument("--poll-n", type=int, default=2)
ap.add_argument("--poll-gate", type=float, default=0.0)
ap.add_argument("--poll-win", type=float, default=8.0)
ap.add_argument("--vad-fallback", type=float, default=4.0, help="reply anyway after this much silence (s)")
ap.add_argument("--lookahead", type=float, default=0.7)
ap.add_argument("--talker", default="http://127.0.0.1:8099")
ap.add_argument("--talker12", default="http://127.0.0.1:8100")
ap.add_argument("--talker32", default="http://127.0.0.1:8120", help="speaking engine address")
ap.add_argument("--realtime", default="http://127.0.0.1:8110")
ap.add_argument("--dj-small", default="http://127.0.0.1:8302")
ap.add_argument("--brand-ref", default="")
ap.add_argument("--tts-backend", default="talker32", choices=["talker32"])
ap.add_argument("--cosy", default="http://127.0.0.1:8098")
ap.add_argument("--hist-turns", type=int, default=8, help="previous turns kept as context")
ap.add_argument("--hist-sec", type=float, default=120.0, help="max user audio (s) kept as context")
ap.add_argument("--log", default="")
A = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
CHECKS = [float(x) for x in A.turn_checks.split(",") if x.strip()]
THS = [float(x) for x in A.turn_thresholds.split(",") if x.strip()]; THS += [THS[-1]] * (len(CHECKS) - len(THS))

EMO = ["平静", "开心", "兴奋", "温柔安抚", "严肃", "难过", "生气", "惊讶", "俏皮", "关切", "好奇疑惑", "无奈", "紧张急切"]
EMO_EN = ["calm", "happy", "excited", "gentle and soothing", "serious", "sad", "angry", "surprised", "playful", "caring", "curious", "resigned", "tense and urgent"]
ENG = ["轻柔低声", "正常", "高昂响亮"]; SPD = ["慢", "中", "快"]
LET = "ABCDEFGHIJKLM"
VOICES = ["vivian", "serena", "uncle_fu", "dylan", "eric", "ryan", "aiden", "ono_anna", "sohee"]
SYS = {"zh": "你是智子（Sophon），AI降临派（Adventists AI）训练的 Sophon 系列模型，能听、能看，也能说。你会直接听到用户的语音，你的回复会被直接合成为语音说出来。请像真人对话一样，用自然的口语一两句话回应用户，注意用户的情绪。只写要说出口的话，不要写括号说明、表情或格式符号；数字、日期、金额、符号和英文缩写都直接写成口语念法。",
       "en": "You are Sophon, a model of the Sophon series trained by Adventists AI (AI降临派); you can hear, see and speak. You hear the user's speech directly, and your reply will be spoken aloud. Respond the way a person would in conversation, in one or two natural spoken sentences, paying attention to the user's emotion. Write only the words to be spoken: no parenthetical notes, emoji or formatting; write numbers and symbols the way they are read aloud."}
CLAUSE = re.compile(r"[，。！？,.!?；;：:…]")
_SYS_TAG = {
 "zh": "你是智子（Sophon），AI降临派（Adventists AI）训练的 Sophon 系列模型，能听、能看，也能说。你会直接听到用户的语音，你的回复会被直接合成为语音说出来。请像真人对话一样，用自然的口语一两句话回应用户，注意用户的情绪。只写要说出口的话，不要写括号说明、表情或格式符号；数字、日期、金额、符号和英文缩写都直接写成口语念法。需要时可以用声音标签表现笑声、咳嗽和重读。",
 "en": "You are Sophon, a model of the Sophon series trained by Adventists AI (AI降临派); you can hear, see and speak. You hear the user's speech directly, and your reply will be spoken aloud. Respond the way a person would in conversation, in one or two natural spoken sentences, paying attention to the user's emotion. Write only the words to be spoken: no parenthetical notes, emoji or formatting; write numbers and symbols the way they are read aloud. When it fits, you may use sound tags for laughter, coughs and emphasis.",
}
CLEAN = re.compile(r"[（(\[【][^）)\]】]*[）)\]】]|[*#_`~>]")
CJK = re.compile(r"[一-鿿]")
TURN_Q = {"zh": ("用户现在处于什么话轮状态？", ["话说完了，可以接话", "句子没说完，还在继续", "简短附和，不是要接话", "还在组织语言，或要求先等一下"]),
          "en": ("Has the user finished speaking?", ["finished, the assistant can reply", "not finished, still talking",
                                                      "just a backchannel, not taking the turn", "hesitating or asking to wait"])}
TPL = {"en": ("Question: {q}", "Options:", "Answer with only the letter of the correct option.", "The user said: <|audio|>", 'The assistant had just said: "{c}"'),
       "zh": ("问题：{q}", "选项：", "请只回答正确选项的字母。", "用户说：<|audio|>", "助手刚才说的是：“{c}”")}

def perm_of(qid, opts, seed=0):
    key = "\x1f".join([qid, "\x1e".join(opts), str(seed)]).encode()
    rng = random.Random(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))
    p = list(range(len(opts))); rng.shuffle(p); return p

def q_text(dim):
    opts = {"emo": EMO, "eng": ENG, "spd": SPD}[dim]; name = {"emo": "情绪/态度", "eng": "声音能量", "spd": "语速"}[dim]
    return f"你接下来要用语音回应上面这段话。你回应时的{name}应该是哪一个？" + " ".join(f"{LET[i]}. {o}" for i, o in enumerate(opts)) + "\n只回答一个字母。"

def instruction(lang, e, g, s):
    if lang == "zh":
        return "，".join([f"用{EMO[e]}的语气"] + ({0: ["声音轻柔低沉"], 2: ["声音高昂响亮"]}.get(g, [])) + ({0: ["语速放慢"], 2: ["语速加快"]}.get(s, [])))
    return ", ".join([f"Speak in a {EMO_EN[e]} tone"] + ({0: ["softly"], 2: ["loudly and energetically"]}.get(g, [])) + ({0: ["slowly"], 2: ["quickly"]}.get(s, [])))

def cosy_instruction(lang, e, g, s):
    if lang == "zh":
        extra = ({0: "，声音轻柔低沉", 2: "，声音高昂响亮"}.get(g, "")) + ({0: "，语速放慢", 2: "，语速加快"}.get(s, ""))
        return f"请用{EMO[e]}的语气说这句话{extra}。"
    extra = ({0: ", softly and quietly", 2: ", loudly and energetically"}.get(g, "")) + ({0: ", slowly", 2: ", quickly"}.get(s, ""))
    return f"Please say this in a {EMO_EN[e]} tone{extra}."

def wav_b64(a16):
    b = io.BytesIO(); sf.write(b, a16, 16000, format="WAV", subtype="PCM_16"); return base64.b64encode(b.getvalue()).decode()

def to_wav16_b64(b64):
    a, sr = sf.read(io.BytesIO(base64.b64decode(b64)), dtype="float32")
    a = a.mean(1) if a.ndim > 1 else a
    if sr != 16000:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), 16000); a = resample_poly(a, 16000 // g, int(sr) // g).astype(np.float32)
    return wav_b64(a), len(a) / 16000

def seglen(s):
    return len(CJK.findall(s)) + len(re.findall(r"[A-Za-z0-9']+", s))

def max_codec(text):
    return int(12.5 * (0.45 * len(CJK.findall(text)) + 0.6 * len(re.findall(r"[A-Za-z0-9']+", text)) + 1.5))

def trim_silence(b, lead, keep_ms, thr=350):
    x = np.frombuffer(b, dtype=np.int16); w = 240
    if len(x) < w: return b
    n = len(x) // w; amp = np.abs(x[: n * w].reshape(n, w)).max(1)
    voiced = np.nonzero(amp >= thr)[0]; keep = keep_ms // 10
    if len(voiced) == 0: return b"" if lead else x[: keep * w].tobytes()
    if lead: return x[max(0, voiced[0] - keep) * w:].tobytes()
    end = min(n, voiced[-1] + 1 + keep); return x[: end * w].tobytes() if end < n else b

app = FastAPI(root_path=A.root_path)
SESSION: aiohttp.ClientSession = None
HITS = collections.defaultdict(collections.deque)

@app.on_event("startup")
async def _start():
    global SESSION
    SESSION = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120), connector=aiohttp.TCPConnector(limit=256, keepalive_timeout=300))
    asyncio.create_task(_brand())

async def _brand():
    if not A.brand_ref: return
    for lang, n in (("zh", "052"), ("en", "047")):
        try:
            a, sr = sf.read(A.brand_ref.format(lang=lang, n=n), dtype="float32", always_2d=True); a = a.mean(1)[: sr * 6]
            b = io.BytesIO(); sf.write(b, a, sr, format="WAV")
            async with SESSION.post(A.cosy + "/voice", json={"wav_b64": base64.b64encode(b.getvalue()).decode()}) as r: j = await r.json()
            BRAND_COSY[lang] = j["voice_id"]; print("brand voice", lang, j["voice_id"], flush=True)
        except Exception as e: print("brand voice failed", lang, e, flush=True)

def small_dj(): return A.style_dj or A.dj

def user_audio(b64):
    return {"role": "user", "content": [{"type": "text", "text": "<|audio|>"}, {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}

def parse_history(xs):
    out = []
    for x in xs or []:
        r = x.get("role") or x.get("from")
        if r == "user":
            if x.get("audio_b64"): b, sec = to_wav16_b64(x["audio_b64"]); out.append({"b64": b, "sec": sec, "text": ""})
            else: out.append({"b64": "", "user_text": str(x.get("text") or x.get("value") or ""), "sec": 0.0, "text": ""})
        elif r in ("assistant", "gpt") and out and not out[-1]["text"]: out[-1]["text"] = str(x.get("text") or x.get("value") or "")
    return out[-A.hist_turns:]

def msgs(lang, b64, extra=None, hist=None):
    m = [{"role": "system", "content": (_SYS_TAG.get(lang, _SYS_TAG["en"]) if A.sys_tag else SYS[lang]) + ((" " + A.sys_append) if A.sys_append else "")}]
    for h in hist or []:
        m.append(user_audio(h["b64"]) if h.get("b64") else {"role": "user", "content": h.get("user_text", "")})
        if h.get("text"): m.append({"role": "assistant", "content": h["text"]})
    m.append(user_audio(b64))
    if extra: m.append({"role": "user", "content": extra})
    return m

def pick(top, n):
    best = None
    for t in top:
        k = t["token"].strip()
        if len(k) == 1 and k in LET[:n] and (best is None or t["logprob"] > best[1]): best = (k, t["logprob"])
    return best

async def mcq(lang, b64, dim, hist=None):
    n = {"emo": 13, "eng": 3, "spd": 3}[dim]
    body = {"model": "dj", "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20, "chat_template_kwargs": {"enable_thinking": False}, "messages": msgs(lang, b64, q_text(dim), hist)}
    async with SESSION.post(small_dj() + "/v1/chat/completions", json=body) as r:
        j = await r.json()
    best = pick(j["choices"][0]["logprobs"]["content"][0]["top_logprobs"], n)
    i = LET.index(best[0]) if best else {"emo": 0, "eng": 1, "spd": 1}[dim]
    if dim == "emo" and i == EMO.index("生气") and not A.allow_angry: i = EMO.index("严肃")
    return i

def tdz(a):
    return np.clip(a + np.random.standard_normal(len(a)).astype(np.float32) * A.turn_dither, -1, 1) if A.turn_dither > 0 else a
async def turn_prob(lang, b64, said):
    q, opts = TURN_Q[lang]; tq, to, ti, head, sd = TPL[lang]; p = perm_of("turn", opts)
    s = " ".join((said or "").split())[-300:]
    text = (sd.format(c=s) + "\n\n" if s else "") + head + "\n\n" + "\n".join([tq.format(q=q), "", to] + [f"{LET[i]}. {opts[j]}" for i, j in enumerate(p)] + ["", ti])
    body = {"model": "dj", "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [{"type": "text", "text": text}, {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}]}
    async with SESSION.post(small_dj() + "/v1/chat/completions", json=body) as r:
        j = await r.json()
    top = {}
    for tk in j["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
        k = tk["token"].strip()
        if k not in top or tk["logprob"] > top[k]: top[k] = tk["logprob"]
    z = np.array([top.get(LET[i], -1e9) for i in range(len(opts))]); z = np.exp(z - z.max()); z /= z.sum()
    return float(z[p.index(0)])

BACKENDS = ("cosy", "qwen", "cosyb", "talker", "talker12", "talker32", "realtime")
BRAND_COSY = {}
def writer_of(backend): return A.dj_small if backend in ("cosyb", "talker", "talker12") else A.dj

async def reply_stream(lang, b64, q, hist=None, url=None):
    body = {"model": A.reply_model, "stream": True, "max_tokens": A.reply_max_tokens, "temperature": A.reply_temp, "top_p": 0.9, "repetition_penalty": A.reply_rep, "chat_template_kwargs": {"enable_thinking": False}, "messages": msgs(lang, b64, None, hist)}
    async with SESSION.post((url or A.dj) + "/v1/chat/completions", json=body) as r:
        async for line in r.content:
            s = line.decode().strip()
            if not s.startswith("data:") or s[5:].strip() == "[DONE]": continue
            ch = json.loads(s[5:])["choices"]
            if ch and ch[0].get("delta", {}).get("content"): await q.put(ch[0]["delta"]["content"])
    await q.put(None)

async def tts_stream(text, ins, voice, seed, out_q):
    body = {"model": A.tts_model, "input": text, "voice": voice, "instructions": ins, "language": "Chinese" if CJK.search(text) else "English",
            "task_type": "CustomVoice", "extra_params": {"temperature": A.tts_temp, "top_k": A.tts_topk}, "seed": seed,
            "max_new_tokens": max_codec(text), "stream": True, "stream_format": "sse", "response_format": "pcm"}
    try:
        async with SESSION.post(A.tts + "/v1/audio/speech", json=body) as r:
            async for line in r.content:
                s = line.decode().strip()
                if not s.startswith("data:"): continue
                try: j = json.loads(s[5:])
                except Exception: continue
                if isinstance(j, dict) and j.get("audio"): await out_q.put(base64.b64decode(j["audio"]))
    finally:
        await out_q.put(None)

_HEARD_TOK = re.compile(r"\[[a-z_]+\]|</?[a-z_]+>|[A-Za-z0-9']+|[\u3040-\u30ff\u4e00-\u9fff]|[，。！？、；：,.!?;:…]|\s+|.")
def heard_text(text, sec):
    """Part of a reply the user actually heard before interrupting, estimated from the playback time."""
    t = 0.0; out = []
    for m in _HEARD_TOK.finditer(text):
        s = m.group(0)
        if s.startswith("[") and s.endswith("]"): d = 0.5
        elif s.startswith("<"): d = 0.0
        elif re.match(r"[A-Za-z0-9']+$", s): d = 0.37
        elif re.match(r"[\u3040-\u30ff\u4e00-\u9fff]$", s): d = 0.22
        elif re.match(r"[，。！？、；：,.!?;:…]$", s): d = 0.15
        else: d = 0.0
        if t + d > sec: break
        t += d; out.append(s)
    return "".join(out)

class Out:
    def __init__(self, ws, gate=None):
        self.ws, self.gate = ws, gate; self.q = asyncio.Queue(); self.on_first_bytes = None
        self.task = asyncio.create_task(self._pump())
    async def _pump(self):
        if self.gate is not None: await self.gate.wait()
        while True:
            kind, x = await self.q.get()
            if kind == "end": return
            if kind == "json": await self.ws.send_json(x)
            else:
                await self.ws.send_bytes(x)
                if self.on_first_bytes: cb, self.on_first_bytes = self.on_first_bytes, None; cb()
    async def json(self, d): self.q.put_nowait(("json", d))
    async def bytes(self, b): self.q.put_nowait(("bytes", b))
    async def close(self):
        self.q.put_nowait(("end", None)); await self.task
    def cancel(self):
        if not self.task.done(): self.task.cancel()

async def run_turn(out, b64, sec, lang, voice, ip, sess=None, extra_metrics=None, backend=None, cosy_voice=None, utt_ref=None, hist_override=None):
    T0 = time.perf_counter(); M = dict(extra_metrics or {}); M["audio_sec"] = round(sec, 2)
    hist = list(sess.history) if (sess is not None and sess.use_hist) else None
    if hist_override is not None: hist = hist_override
    M["hist_turns"] = len(hist or []); M["hist_sec"] = round(sum(h["sec"] for h in hist or []), 1)
    t = lambda: round(time.perf_counter() - T0, 3)
    seed = random.randint(1, 2 ** 31 - 1)
    if (backend or A.tts_backend) in ("talker", "talker12", "talker32", "realtime"):
        async def _emo(): return (0, 1, 1)
        style_task = asyncio.ensure_future(_emo())
    elif A.style_dims == "emo":
        async def _emo(): return (await mcq(lang, b64, "emo", hist), 1, 1)
        style_task = asyncio.ensure_future(_emo())
    else:
        style_task = asyncio.ensure_future(asyncio.gather(mcq(lang, b64, "emo", hist), mcq(lang, b64, "eng", hist), mcq(lang, b64, "spd", hist)))
    style_task.add_done_callback(lambda _f: M.setdefault("style_ready", t()))
    BK = backend or A.tts_backend
    q = asyncio.Queue(); rtask = asyncio.create_task(asyncio.sleep(0) if BK == "realtime" else reply_stream(lang, b64, q, hist, writer_of(BK)))
    seg_ready = asyncio.Queue(); style = None; tts_tasks = []
    play = {"end": None}

    def remaining():
        return 9.9 if play["end"] is None else play["end"] - time.perf_counter()

    async def segmenter():
        nonlocal style
        buf, n, first, full, done = "", 0, True, "", False
        async def launch(text):
            nonlocal style
            text = CLEAN.sub("", text)
            if not text.strip(): return
            if style is None:
                e, g, s = await style_task; style = (e, g, s); M["style"] = t()
                await out.json({"type": "style", "emo": EMO[e], "energy": ENG[g], "speed": SPD[s], "instruction": instruction(lang, e, g, s), "t": M["style"]})
            aq = asyncio.Queue(); task = asyncio.create_task(tts_stream(text, instruction(lang, *style), voice, seed, aq)); tts_tasks.append(task)
            M.setdefault("seg1_sent", t()); M["n_segs"] = M.get("n_segs", 0) + 1
            await out.json({"type": "seg", "text": text})
            await seg_ready.put((aq, task))
        while not done or buf.strip():
            if not done:
                try: d = await asyncio.wait_for(q.get(), timeout=0.04)
                except asyncio.TimeoutError: d = ""
                if d is None: done = True; M["reply_done"] = t()
                elif d:
                    if "first_token" not in M: M["first_token"] = t()
                    buf += d; full += d; n += 1; M["_partial"] = full
                    await out.json({"type": "text", "delta": d})
            L = seglen(buf)
            if first:
                if (CLAUSE.search(buf) and L >= 4) or (n >= A.k and L >= 3) or (done and buf.strip()):
                    await launch(buf); buf = ""; first = False
            elif done:
                if buf.strip(): await launch(buf); buf = ""
            elif remaining() < A.lookahead:
                cut = max((m.end() for m in CLAUSE.finditer(buf)), default=0)
                if cut and seglen(buf[:cut]) >= 4: await launch(buf[:cut]); buf = buf[cut:]
        M["reply_chars"] = len(full); M["reply_text"] = full
        await seg_ready.put(None)

    async def emitter():
        idx = 0
        def _first_sent():
            M["first_audio"] = t()
            if M.get("eos_wall"): M["eos_to_audio"] = round(time.time() - M["eos_wall"], 3)
        async def send(b):
            if "audio_bytes" not in M: out.on_first_bytes = _first_sent; M["first_ready"] = t()
            await out.bytes(b)
            M["audio_bytes"] = M.get("audio_bytes", 0) + len(b)
            now = time.perf_counter(); base = now if (out.gate is None or out.gate.is_set()) else now + 0.3
            play["end"] = max(play["end"] or base, base) + len(b) / 48000
        while True:
            it = await seg_ready.get()
            if it is None: break
            aq, task = it; pend = None; head = True
            while True:
                b = await aq.get()
                if b is None: break
                if head and idx > 0:
                    b = trim_silence(b, lead=True, keep_ms=60)
                    if not b: continue
                if head and idx == 0: await send(b); head = False; continue
                head = False
                if pend is not None: await send(pend)
                pend = b
            if pend is not None:
                pend = trim_silence(pend, lead=False, keep_ms=150)
                if pend: await send(pend)
            idx += 1
        M["audio_done"] = t()

    async def cosy_path():
        nonlocal style
        tk = BK in ("talker", "talker12", "talker32")
        e, g, s_ = await style_task; style = (e, g, s_); M["style"] = t()
        if tk: await out.json({"type": "style", "emo": "", "t": M["style"]})
        else: await out.json({"type": "style", "emo": EMO[e], "energy": ENG[g], "speed": SPD[s_], "instruction": cosy_instruction(lang, e, g, s_), "t": M["style"]})
        full = ""; first = True
        async with SESSION.ws_connect(((A.talker32 if BK == "talker32" else A.talker12 if BK == "talker12" else A.talker) if tk else A.cosy).replace("http", "ws") + "/tts", max_msg_size=0, compress=0) as cws:
            if tk: vid = cosy_voice or f"brand_{lang}"
            elif BK == "cosyb": vid = cosy_voice or BRAND_COSY.get(lang) or ("vivian_zh" if lang == "zh" else "ryan_en")
            else: vid = cosy_voice or ("vivian_zh" if lang == "zh" else "ryan_en")
            await cws.send_json({"type": "start", "voice": vid, "lang": lang, "instruct": cosy_instruction(lang, e, g, s_)}); M["seg1_sent"] = t(); M["n_segs"] = 1
            async def feed():
                nonlocal full
                while True:
                    d = await q.get()
                    if d is None: M["reply_done"] = t(); await cws.send_json({"type": "end"}); return
                    if "first_token" not in M: M["first_token"] = t()
                    full += d; M["_partial"] = full; await out.json({"type": "text", "delta": d})
                    c = d if (A.keep_tags and BK == "talker32") else CLEAN.sub("", d)
                    if c: await cws.send_json({"type": "text", "t": c})
            ft = asyncio.create_task(feed())
            try:
                async for m in cws:
                    if m.type == aiohttp.WSMsgType.BINARY:
                        if "audio_bytes" not in M: out.on_first_bytes = _first_sent_cosy; M["first_ready"] = t()
                        await out.bytes(m.data); M["audio_bytes"] = M.get("audio_bytes", 0) + len(m.data)
                        now = time.perf_counter(); base = now if (out.gate is None or out.gate.is_set()) else now + 0.3
                        play["end"] = max(play["end"] or base, base) + len(m.data) / 48000
                    elif m.type == aiohttp.WSMsgType.TEXT:
                        j = json.loads(m.data)
                        if j.get("type") == "done":
                            if j.get("segs"): M["n_segs"] = len(j["segs"]); M["seg1_ms"] = j["segs"][0].get("ms_talker", 0) + j["segs"][0].get("ms_e2", 0) + j["segs"][0].get("ms_hid", 0)
                            break
                        if j.get("type") == "error": await out.json({"type": "error", "message": "合成失败：" + j.get("message", "")})
                    else: break
            finally:
                ft.cancel()
        M["reply_chars"] = len(full); M["reply_text"] = full; M["audio_done"] = t()
    async def realtime_path():
        await out.json({"type": "style", "emo": "sophon-realtime 单模型", "t": t()}); M["style"] = t()
        full = ""
        async with SESSION.ws_connect(A.realtime.replace("http", "ws") + "/chat", max_msg_size=0, compress=0) as rws:
            await rws.send_json({"type": "start", "lang": lang, "voice": cosy_voice or f"brand_{lang}", "audio_b64": b64, "history": [{"b64": h.get("b64", ""), "user_text": h.get("user_text", ""), "text": h.get("text", "")} for h in (hist or [])]})
            M["seg1_sent"] = t(); M["n_segs"] = 1
            try:
                async for m in rws:
                    if m.type == aiohttp.WSMsgType.BINARY:
                        if "audio_bytes" not in M: out.on_first_bytes = _first_sent_cosy; M["first_ready"] = t()
                        await out.bytes(m.data); M["audio_bytes"] = M.get("audio_bytes", 0) + len(m.data)
                        now = time.perf_counter(); base = now if (out.gate is None or out.gate.is_set()) else now + 0.3
                        play["end"] = max(play["end"] or base, base) + len(m.data) / 48000
                    elif m.type == aiohttp.WSMsgType.TEXT:
                        j = json.loads(m.data)
                        if j.get("type") == "text":
                            if "first_token" not in M: M["first_token"] = t()
                            full += j["delta"]; M["_partial"] = full; await out.json({"type": "text", "delta": j["delta"]})
                        elif j.get("type") == "done": M["reply_done"] = t(); M["rt_stats"] = j.get("stats"); break
                        elif j.get("type") == "error": await out.json({"type": "error", "message": "sophon-realtime：" + j.get("message", "")})
                    else: break
            except asyncio.CancelledError:
                try: await rws.send_json({"type": "cancel"})
                except Exception: pass
                raise
        M["reply_chars"] = len(full); M["reply_text"] = full; M["audio_done"] = t()
    def _first_sent_cosy():
        M["first_audio"] = t()
        if M.get("eos_wall"): M["eos_to_audio"] = round(time.time() - M["eos_wall"], 3)

    try:
        if BK == "realtime": await realtime_path()
        elif BK in ("cosy", "cosyb", "talker", "talker12", "talker32"): await cosy_path()
        else: await asyncio.gather(segmenter(), emitter())
    except asyncio.CancelledError:
        out.cancel()
        if sess is not None and M.get("first_audio") is not None and M.get("_partial", "").strip():
            heard = heard_text(M["_partial"], max(0.0, min(t() - M["first_audio"] - 0.3, M.get("audio_bytes", 0) / 48000)))
            if heard.strip(): sess.assistant_said = heard; sess.add_history(b64, sec, heard + "……（被用户打断）", utt_ref)
        raise
    finally:
        rtask.cancel()
        if not style_task.done(): style_task.cancel()
        for tk in tts_tasks:
            if not tk.done(): tk.cancel()
    reply_text = M.pop("reply_text", ""); M.pop("_partial", None)
    if sess is not None:
        sess.assistant_said = reply_text; sess.play_end = play["end"]
        if reply_text.strip(): sess.add_history(b64, sec, reply_text, utt_ref)
    M["out_sec"] = round(M.get("audio_bytes", 0) / 48000, 2)
    M.update(lang=lang, voice=(cosy_voice or "default") if BK != "qwen" else voice, tts=BK, engine=BK, emo=EMO[style[0]] if (style and BK not in ("talker", "talker12", "talker32", "realtime")) else None); M.pop("eos_wall", None)
    if out.gate is not None: await out.gate.wait()
    while not out.q.empty(): await asyncio.sleep(0.005)
    await asyncio.sleep(0)
    await out.json({"type": "metrics", **{k: v for k, v in M.items() if k not in ("voice", "tts")}})
    await out.json({"type": "done"}); await out.close()
    if A.log:
      with open(A.log, "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ip": ".".join(ip.split(".")[:2]) if ip else "", **M}, ensure_ascii=False) + "\n")

class Session:
    FR = 320
    def __init__(self, ws, ip):
        self.ws, self.ip = ws, ip; self.lang, self.voice = "zh", None; self.backend = A.tts_backend; self.cosy_voice = None
        self.noise = 0.004; self.voiced_ms = 0; self.speaking = False; self.pre = collections.deque(maxlen=15)
        self.utt = []; self.sil = 0.0; self.next_check = 0; self.last_voiced_wall = None
        self.turn = None; self.gate = None; self.checking = None; self.assistant_said = ""; self.play_end = None
        self.rem = np.zeros(0, np.int16); self.last_commit = None
        self.ring = collections.deque(maxlen=int(A.poll_win * 50)); self.poll_seq = 0; self.poll_last = 0; self.poll_run = 0; self.since_poll = 0.0; self.inflight = 0
        self.history = []; self.use_hist = True; self.drop_ref = None

    def add_history(self, b64, sec, text, ref):
        if ref is not None and ref is self.drop_ref: return
        self.history.append({"b64": b64, "sec": sec, "text": text, "ref": ref})
        while len(self.history) > A.hist_turns or (len(self.history) > 1 and sum(h["sec"] for h in self.history) > A.hist_sec):
            self.history.pop(0)

    def assistant_active(self):
        busy = self.turn is not None and not self.turn.done() and self.gate is not None and self.gate.is_set()
        playing = self.play_end is not None and time.perf_counter() < self.play_end
        return busy or playing

    async def send(self, d):
        try: await self.ws.send_json(d)
        except Exception: pass

    def cancel_turn(self):
        if self.turn is not None and not self.turn.done(): self.turn.cancel()
        if getattr(self, "out", None) is not None: self.out.cancel()
        self.turn = None; self.gate = None; self.out = None

    def start_turn(self, gated):
        a = np.concatenate(self.utt).astype(np.float32) / 32768
        sec = len(a) / 16000
        if sec > A.max_sec: a = a[-int(A.max_sec * 16000):]; sec = A.max_sec
        self.gate = asyncio.Event() if gated else None
        out = Out(self.ws, self.gate); self.out = out
        extra = {"mode": "stream", "eos_wall": self.last_voiced_wall}; ref = self.utt
        async def go():
            try: await run_turn(out, wav_b64(a), sec, self.lang, self.voice, self.ip, sess=self, extra_metrics=extra, backend=self.backend, cosy_voice=self.cosy_voice, utt_ref=ref)
            except asyncio.CancelledError: pass
            except Exception as e: await self.send({"type": "error", "message": str(e)[:300]})
        self.turn = asyncio.create_task(go()); self.turn_t0 = time.perf_counter()
        return a

    async def check(self, at_sil, th):
        if self.turn is None or self.turn.done(): self.start_turn(gated=True)
        a = np.concatenate(self.utt).astype(np.float32) / 32768
        t0 = time.perf_counter()
        if os.environ.get("SOPHON_DEBUG"): sf.write(f"/tmp/sophon_turn_{int(time.time()*1000)}_{at_sil}.wav", a, 16000)
        try: p = await turn_prob(self.lang, wav_b64(tdz(a[-int(A.max_sec * 16000):])), self.assistant_said)
        except Exception as e: p = 0.0; await self.send({"type": "error", "message": "话轮判断失败：" + str(e)[:200]})
        ms = round((time.perf_counter() - t0) * 1000)
        if not self.speaking: return
        if self.sil < at_sil - 0.05: return
        await self.send({"type": "turn_check", "pause": at_sil, "p_finished": round(p, 3), "threshold": th, "ms": ms})
        if p >= th: await self.commit(f"duplexjev p={p:.2f}", p)

    async def poll(self, seq):
        a = np.concatenate(list(self.ring)).astype(np.float32) / 32768; self.inflight += 1; t0 = time.perf_counter()
        try: p = await turn_prob(self.lang, wav_b64(tdz(a)), self.assistant_said)
        except Exception as e: p = 0.0
        finally: self.inflight -= 1
        if not self.speaking or seq <= self.poll_last: return
        if A.poll_gate > 0 and self.sil < A.poll_gate - 1e-6: self.poll_run = 0; return
        self.poll_last = seq; sth = A.poll_spec_th if A.poll_spec_th >= 0 else A.poll_th
        await self.send({"type": "turn_check", "mode": "poll", "pause": round(self.sil, 2), "p_finished": round(p, 3), "threshold": A.poll_th, "ms": round((time.perf_counter() - t0) * 1000)})
        if p >= sth and (self.turn is None or self.turn.done()): self.start_turn(gated=True)
        self.poll_run = self.poll_run + 1 if p >= A.poll_th else 0
        if p < sth and self.turn is not None and self.gate is not None and not self.gate.is_set(): self.cancel_turn()
        if self.poll_run >= A.poll_n: await self.commit(f"duplexjev poll p={p:.2f} x{A.poll_n}", p)

    async def commit(self, why, p=None):
        if not self.speaking: return
        self.speaking = False
        if not allow(self.ip):
            self.cancel_turn(); self.utt = []; await self.send({"type": "error", "message": "请求太频繁，请一分钟后再试"}); return
        if self.turn is None or self.turn.done(): self.start_turn(gated=False)
        await self.send({"type": "turn_commit", "why": why, "p_finished": p, "pause": round(self.sil, 2),
                         "spec_ms": round((time.perf_counter() - self.turn_t0) * 1000)})
        if self.gate is not None: self.gate.set()
        self.last_commit = (time.perf_counter(), self.utt); self.utt = []; self.ring.clear(); self.poll_run = 0

    async def feed(self, pcm):
        x = np.concatenate([self.rem, np.frombuffer(pcm, dtype=np.int16)])
        n = len(x) // self.FR; self.rem = x[n * self.FR:]
        for i in range(n):
            f = x[i * self.FR:(i + 1) * self.FR]; self.ring.append(f); r = float(np.sqrt(np.mean((f.astype(np.float32) / 32768) ** 2)))
            thr = max(0.012, self.noise * 3.5) * (2.0 if self.assistant_active() else 1.0)
            if not self.speaking:
                self.pre.append(f)
                if r > thr:
                    self.voiced_ms += 20
                    if self.voiced_ms >= (300 if self.assistant_active() else 100):
                        merge = []
                        if self.assistant_active():
                            if self.last_commit and time.perf_counter() - self.last_commit[0] < A.merge_window:
                                merge = self.last_commit[1]
                                self.drop_ref = merge
                                if self.history and self.history[-1]["ref"] is merge: self.history.pop()
                            self.cancel_turn(); self.play_end = None
                            await self.send({"type": "barge_in", "merged": bool(merge)})
                        self.speaking = True; self.utt = merge + list(self.pre); self.sil = 0.0; self.next_check = 0; self.poll_run = 0; self.since_poll = 0.0
                        self.last_voiced_wall = time.time(); await self.send({"type": "vad", "speech": True})
                else:
                    self.voiced_ms = 0; self.noise = 0.98 * self.noise + 0.02 * min(r, 0.05)
            else:
                self.utt.append(f)
                if r > thr:
                    if self.sil >= 0.06 and self.turn is not None and self.gate is not None and not self.gate.is_set():
                        self.cancel_turn()
                    self.sil = 0.0; self.next_check = 0; self.last_voiced_wall = time.time()
                else:
                    self.sil += 0.02
                    if A.turn_mode == "vad" and self.sil >= A.spec_at and (self.turn is None or self.turn.done()): self.start_turn(gated=True)
                    if A.turn_mode == "vad" and self.next_check < len(CHECKS) and self.sil >= CHECKS[self.next_check]:
                        at, th = CHECKS[self.next_check], THS[self.next_check]; self.next_check += 1
                        asyncio.create_task(self.check(at, th))
                    if self.sil >= A.vad_fallback:
                        await self.commit("vad 兜底")
                if A.turn_mode == "poll" and self.speaking:
                    self.since_poll += 0.02
                    if A.poll_gate > 0 and self.sil < A.poll_gate - 1e-6: self.since_poll = A.poll_ms / 1000; self.poll_run = 0
                    elif self.since_poll >= A.poll_ms / 1000 - 1e-6 and self.inflight < 3:
                        self.since_poll = 0.0; self.poll_seq += 1; asyncio.create_task(self.poll(self.poll_seq))
                if sum(len(u) for u in self.utt) > A.max_sec * 16000: await self.commit("太长")

@app.get("/v1/health")
async def health():
    out = {}
    for name, url in (("duplexjev", A.dj + "/health"), ("speech_b", A.tts + "/health"), ("duplexjev_small", small_dj() + "/health"), ("speech", A.cosy + "/health"),
                      ("duplexjev_writer4b", A.dj_small + "/health"), ("talker", A.talker + "/health"), ("talker12", A.talker12 + "/health"), ("talker32", A.talker32 + "/health"), ("realtime", A.realtime + "/health")):
        try:
            async with SESSION.get(url, timeout=aiohttp.ClientTimeout(total=3)) as r: out[name] = r.status == 200
        except Exception: out[name] = False
    return JSONResponse({"ok": all(out.values()), **out, "k": A.k, "turn_checks": CHECKS, "turn_thresholds": THS, "spec_at": A.spec_at, "vad_fallback": A.vad_fallback, "turn_mode": A.turn_mode, "poll": [A.poll_ms, A.poll_th, A.poll_spec_th, A.poll_n, A.poll_gate]})

@app.get("/")
async def index():
    return HTMLResponse(open(os.path.join(HERE, "static", "index.html"), encoding="utf-8").read())

def allow(ip):
    if ip in ("127.0.0.1", "::1") or ip.startswith("172.16."): return True
    now = time.time(); d = HITS[ip]
    while d and now - d[0] > 60: d.popleft()
    if len(d) >= A.rate: return False
    d.append(now); return True

@app.websocket("/ws")
async def ws_ep(ws: WebSocket):
    await ws.accept()
    ip = ws.headers.get("x-real-ip") or ws.headers.get("x-forwarded-for", "").split(",")[0].strip() or (ws.client.host if ws.client else "")
    sess = Session(ws, ip); cur = None
    try:
        while True:
            m = await ws.receive()
            if m.get("type") == "websocket.disconnect": break
            if m.get("bytes") is not None:
                await sess.feed(m["bytes"]); continue
            msg = json.loads(m.get("text") or "{}")
            if msg.get("type") == "config":
                sess.lang = msg.get("lang") if msg.get("lang") in ("zh", "en") else "zh"
                sess.voice = msg.get("voice") if msg.get("voice") in VOICES else ("vivian" if sess.lang == "zh" else "ryan")
                if msg.get("tts") in BACKENDS: sess.backend = msg["tts"]
                cv = msg.get("my_voice") or msg.get("cosy_voice")
                sess.cosy_voice = cv if cv else None
                if "history" in msg: sess.use_hist = bool(msg["history"])
            elif msg.get("type") == "history":
                try:
                    for h in parse_history(msg.get("messages")): sess.history.append({**h, "ref": None})
                    sess.history = sess.history[-A.hist_turns:]; await ws.send_json({"type": "history_ok", "turns": len(sess.history)})
                except Exception as e: await ws.send_json({"type": "error", "message": "history 格式不对：" + str(e)[:200]})
            elif msg.get("type") == "reset":
                sess.history = []; sess.assistant_said = ""
                await ws.send_json({"type": "reset_ok"})
            elif msg.get("type") == "clone" and False:
                try:
                    async with SESSION.post(A.cosy + "/voice", json={"wav_b64": msg["wav_b64"]}) as r: j = await r.json()
                    try:
                        if "voice_id" in j:
                            for _tu in (A.talker, A.talker12, A.talker32, A.realtime):
                                try:
                                    async with SESSION.post(_tu + "/voice", json={"wav_b64": msg["wav_b64"], "voice_id": j["voice_id"]}) as r2: await r2.json()
                                except Exception: pass
                    except Exception: pass
                    if "voice_id" in j: sess.cosy_voice = j["voice_id"]; await ws.send_json({"type": "clone_ok", "voice_id": j["voice_id"]})
                    else: await ws.send_json({"type": "error", "message": j.get("error", "克隆失败")})
                except Exception as e: await ws.send_json({"type": "error", "message": "克隆失败：" + str(e)[:200]})
            elif msg.get("type") == "ping":
                await ws.send_json({"type": "pong", "t": msg.get("t")})
            elif msg.get("type") == "cancel":
                sess.cancel_turn(); sess.play_end = None
                if cur and not cur.done(): cur.cancel()
            elif msg.get("type") == "turn":
                if cur and not cur.done(): cur.cancel()
                if not allow(ip): await ws.send_json({"type": "error", "message": "请求太频繁，请一分钟后再试"}); continue
                lang = msg.get("lang") if msg.get("lang") in ("zh", "en") else "zh"
                voice = msg.get("voice") if msg.get("voice") in VOICES else ("vivian" if lang == "zh" else "ryan")
                async def go(m=msg, lang=lang, voice=voice):
                    try:
                        b64, sec = to_wav16_b64(m["audio_b64"])
                        if sec > A.max_sec: raise ValueError(f"语音太长（{sec:.1f} 秒）")
                        await run_turn(Out(ws), b64, sec, lang, voice, ip, extra_metrics={"mode": "utterance"}, backend=m.get("tts"), cosy_voice=m.get("cosy_voice"), hist_override=parse_history(m["history"]) if m.get("history") else None)
                    except asyncio.CancelledError: pass
                    except Exception as e:
                        try: await ws.send_json({"type": "error", "message": str(e)[:300]})
                        except Exception: pass
                cur = asyncio.create_task(go())
    except WebSocketDisconnect:
        pass
    finally:
        sess.cancel_turn()
        if cur and not cur.done(): cur.cancel()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=A.port, log_level="warning", ws_max_size=16 * 1024 * 1024, ws_per_message_deflate=False)
