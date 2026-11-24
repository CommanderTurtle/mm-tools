# Copyright 2026 Adventists AI. Apache-2.0.
"""Speaking engine service.
WebSocket /tts: {"type":"start","lang"} -> {"type":"text","t"} (streamed tokens) -> {"type":"end"}; {"type":"cancel"} at any time.
Server -> binary PCM16 24 kHz chunks ... -> {"type":"done","sec","segs"}.  GET /health."""
import os, sys, io, json, time, base64, argparse, asyncio, threading, queue, hashlib
import numpy as np, soundfile as sf, librosa
import faulthandler, signal; faulthandler.register(signal.SIGUSR1, all_threads=True)
from aiohttp import web, WSMsgType
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8120); A = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import engine220 as T
core = T.Core()

def wav16(b64):
    w, s = sf.read(io.BytesIO(base64.b64decode(b64)), dtype="float32", always_2d=True); w = w.mean(1)
    return librosa.resample(w, orig_sr=s, target_sr=16000) if s != 16000 else w

import re
SENT_END = re.compile(r"[。！？!?]|[.](?=\s)|\n")
MINLEN = {"zh": 10, "ja": 10, "en": 25}
MAXC = {"zh": 55, "ja": 55, "en": 170}
CLAUSE = re.compile(r"[,;:](?=\s)|[，、；：]")
class MultiSession:
    def __init__(s, lang, voice, emit):
        s.lang, s.voice, s.emit = lang, voice, emit; s.lock = threading.Lock(); s.cur = None; s.pend = ""; s.final = False; s.cancelled = False
        s.ev = threading.Event(); s.stats = dict(t_first_audio=None, n_sent=0, ms_talker=0, ms_e2=0, sec=0.0); s.t0 = time.perf_counter(); s.cur_text = ""
    def _route(s):
        if s.cur is None or s.cur.final: return
        m = None
        for mm in SENT_END.finditer(s.pend):
            if len((s.cur_text + s.pend[: mm.end()]).strip()) >= MINLEN.get(s.lang, 10): m = mm; break
        if m is None and len(s.cur_text) + len(s.pend) > MAXC.get(s.lang, 60):
            for mm in CLAUSE.finditer(s.pend):
                if len((s.cur_text + s.pend[: mm.end()]).strip()) >= MINLEN.get(s.lang, 10) * 2: m = mm; break
        if m:
            s.cur.add_text(s.pend[: m.end()]); s.cur_text += s.pend[: m.end()]; s.pend = s.pend[m.end():]; s.cur.finish()
        else:
            keep = 1 if (not s.final and s.pend[-1:] in (".", ",", ";", ":")) else 0
            give = s.pend[: len(s.pend) - keep]
            s.cur.add_text(give); s.cur_text += give; s.pend = s.pend[len(give):]
            if s.final: s.cur.finish()
    def add_text(s, t):
        with s.lock: s.pend += t; s._route()
        s.ev.set()
    def finish(s):
        with s.lock: s.final = True; s._route()
        s.ev.set()
    def cancel(s):
        s.cancelled = True
        with s.lock:
            if s.cur is not None: s.cur.cancel()
        s.ev.set()
    def run(s):
        first = True
        while not s.cancelled:
            with s.lock:
                if not s.pend.strip() and s.final: break
                if not s.pend.strip():
                    pass
                else:
                    s.cur = T.Session(core, s.lang, s.voice, s.emit if first else s._emit_after_gap); s.cur_text = ""; s._gap_done = first; s._route()
                    if s.cancelled: s.cur.cancel()
            if s.cur is None or (s.cur is not None and s.cur.finished):
                s.ev.wait(0.02); s.ev.clear(); continue
            s.cur.run(); st = s.cur.stats
            if first and st.get("t_first_audio") is not None: s.stats["t_first_audio"] = round(time.perf_counter() - s.t0 - 0, 3) if False else st["t_first_audio"]
            for k in ("ms_talker", "ms_e2"): s.stats[k] += st.get(k, 0)
            s.stats["sec"] = round(s.stats["sec"] + st.get("sec", 0), 2); s.stats["n_sent"] += 1; first = False
            with s.lock: s.cur = None
    def _emit_after_gap(s, b):
        if not s._gap_done: s.emit(np.zeros(int(0.12 * 24000), np.int16).tobytes()); s._gap_done = True
        s.emit(b)
SPLIT = os.environ.get("T32_SPLIT", "1") == "1"

async def ws_tts(request):
    ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024, compress=False); await ws.prepare(request)
    loop = asyncio.get_running_loop(); out_q = queue.Queue(); sess = None; ptask = None; nbytes = 0; RX = []
    async def pump():
        nonlocal nbytes
        while True:
            x = await loop.run_in_executor(None, out_q.get)
            if x is None: break
            nbytes += len(x); await ws.send_bytes(x)
        if not ws.closed: await ws.send_json({"type": "done", "sec": round(nbytes / 48000, 2), "segs": [sess.stats] if sess else []})
    def runner(s):
        try: s.run()
        except Exception as e: print("T32 ERROR", repr(e)[:300], flush=True)
        finally: out_q.put(None)
    try:
        async for m in ws:
            if m.type != WSMsgType.TEXT: continue
            j = json.loads(m.data)
            if j["type"] == "start" and sess is None:
                sess = (MultiSession if SPLIT else lambda l, v, e: T.Session(core, l, v, e))(j.get("lang") or "zh", j.get("voice") or "", out_q.put)
                threading.Thread(target=runner, args=(sess,), daemon=True).start(); ptask = asyncio.create_task(pump())
            elif j["type"] == "text" and sess: sess.add_text(j.get("t", "")); RX.append(j.get("t", ""))
            elif j["type"] == "end" and sess: sess.finish()
            elif j["type"] == "cancel" and sess: sess.cancel()
    finally:
        if sess and not getattr(sess, "finished", False): sess.cancel()
        if os.environ.get("T32_LOG"):
            with open(os.environ["T32_LOG"], "a") as f: f.write(json.dumps(dict(t=time.time(), rx=RX, cancelled=bool(sess and sess.cancelled), stats=sess.stats if sess else None), ensure_ascii=False) + "\n")
        if ptask:
            try: await asyncio.wait_for(ptask, 30)
            except Exception: pass
    return ws

async def post_voice(request):
    return web.json_response({"error": "voice cloning is not available in this build"}, status=501)
async def _post_voice_v1(request):
    j = await request.json(); w = wav16(j["wav_b64"])
    if len(w) / 16000 < 2: return web.json_response({"error": "reference audio too short (min 2 s)"}, status=400)
    vid = j.get("voice_id") or ("u_" + hashlib.sha1(base64.b64decode(j["wav_b64"])[:200000]).hexdigest()[:10])
    with core.lock: core.voices[vid] = core.spk_from_wav(w)
    return web.json_response({"voice_id": vid})

async def health(request): return web.json_response({"ok": True, "model": "sophon-realtime-e2e"})
sys.setswitchinterval(0.0002)
s = T.Session(core, "zh", "brand_zh", lambda b: None); s.add_text("你好，很高兴认识你。"); s.finish(); s.run()
s = T.Session(core, "en", "brand_en", lambda b: None); s.add_text("Hello, nice to meet you."); s.finish(); s.run()
print("T32_SERVER up", A.port, s.stats, flush=True)
app = web.Application(client_max_size=64 * 1024 * 1024)
app.add_routes([web.get("/tts", ws_tts), web.post("/voice", post_voice), web.get("/health", health)])
web.run_app(app, host="127.0.0.1", port=A.port, print=None)
