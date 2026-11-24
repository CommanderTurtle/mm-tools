"""OpenAI Realtime compatible face (wss://<host>/v1/realtime) on the same Sophon engine as the GPT-Live face.

Implemented (GA event names): session.created / session.update -> session.updated; input_audio_buffer.append / commit / clear;
input_audio_buffer.speech_started / speech_stopped / committed; conversation.item.input_audio_transcription.completed;
response.created / output_item.added / content_part.added / output_audio.delta / output_audio_transcript.delta /
output_audio.done / output_audio_transcript.done / content_part.done / output_item.done / response.done;
function calling: response.output_item.added(function_call) / function_call_arguments.delta / .done, then the client's
conversation.item.create(function_call_output) + response.create makes the model answer from the result;
conversation.item.truncate -> conversation.item.truncated; response.cancel. Audio: audio/pcm 24 kHz, audio/pcmu, audio/pcma
(8 kHz G.711), plus the beta names pcm16 / g711_ulaw / g711_alaw.
Turn-taking is the engine's own (DuplexJev); speech_started / stopped are reported from an input energy detector so clients can
truncate playback on barge-in."""
from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
import os
import time
import uuid

import numpy as np
import websockets

from .adapters import VllmOmniAdapter
from . import grounding
from . import backend

log = logging.getLogger("sophon_live.realtime")

# speech_started / stopped drive client-side truncation (OpenAI Realtime clients cut our playback on speech_started),
# so false starts from background noise mute the agent. Use Silero (robust to steady noise and bursts) when available.
SILERO_PATH = os.environ.get("SOPHON_LIVE_SILERO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "silero_vad.onnx"))
GROUND = os.environ.get("SOPHON_LIVE_GROUND", "1") != "0"          # tool-argument grounding guard (see grounding.py)
VAD_KIND = os.environ.get("SOPHON_LIVE_VAD", "energy")              # energy | silero (silero misses band-limited telephony speech)
VAD_RMS = float(os.environ.get("SOPHON_LIVE_VAD_RMS", "0.015"))       # energy VAD: minimum frame RMS threshold
VAD_K = float(os.environ.get("SOPHON_LIVE_VAD_K", "3.0"))             # threshold = max(VAD_RMS, K x noise floor); floor = 20th pct of the last 5 s
VAD_E_START_MS = int(os.environ.get("SOPHON_LIVE_VAD_E_START_MS", "300"))  # voiced this long before speech_started (horns, sneezes, "mm-hm" don't cut the agent)
VAD_ON_P = float(os.environ.get("SOPHON_LIVE_VAD_ON", "0.6"))      # speech prob to count a frame as voiced
VAD_OFF_P = float(os.environ.get("SOPHON_LIVE_VAD_OFF", "0.35"))
VAD_START_MS = int(os.environ.get("SOPHON_LIVE_VAD_START_MS", "250"))  # voiced this long -> speech_started (filters coughs, horns)
VAD_STOP_MS = int(os.environ.get("SOPHON_LIVE_VAD_STOP_MS", "500"))
_silero_sess = None


def _silero():
    global _silero_sess
    if _silero_sess is None:
        try:
            import onnxruntime as ort
            o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
            _silero_sess = ort.InferenceSession(SILERO_PATH, sess_options=o, providers=["CPUExecutionProvider"])
        except Exception:  # noqa: BLE001
            log.warning("silero VAD unavailable, falling back to energy VAD", exc_info=True); _silero_sess = False
    return _silero_sess or None


class SileroVAD:
    """Streaming Silero v5 at 16 kHz, 512-sample (32 ms) frames; returns per-frame speech probabilities."""
    def __init__(self, sess):
        self.sess = sess; self.state = np.zeros((2, 1, 128), np.float32); self.ctx = np.zeros(64, np.float32); self.rem = np.zeros(0, np.float32)
        self.sr = np.array(16000, np.int64)

    def feed(self, a: np.ndarray) -> list:
        x = np.concatenate([self.rem, a]) if self.rem.size else a
        n = x.size // 512; self.rem = x[n * 512:].copy(); out = []
        for i in range(n):
            fr = x[i * 512:(i + 1) * 512]
            inp = np.concatenate([self.ctx, fr])[None, :].astype(np.float32); self.ctx = fr[-64:]
            p, self.state = self.sess.run(None, {"input": inp, "state": self.state, "sr": self.sr})
            out.append(float(p[0][0]))
        return out
DEFAULT_VOICE = os.environ.get("SOPHON_LIVE_VOICE", "chelsie")
CJK_MIN = 4

# ---------------------------------------------------------------- G.711
_SEG_END = np.array([0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF])


def ulaw_decode(b: bytes) -> np.ndarray:
    u = ~np.frombuffer(b, np.uint8).astype(np.int32) & 0xFF
    sign, exp, man = u & 0x80, (u >> 4) & 0x07, u & 0x0F
    x = ((man << 3) + 0x84) << exp
    x = np.where(sign != 0, 0x84 - x, x - 0x84)
    return (x / 32768.0).astype(np.float32)


def ulaw_encode(x: np.ndarray) -> bytes:
    s = np.clip(x * 32768.0, -32635, 32635).astype(np.int32)
    sign = np.where(s < 0, 0x80, 0); s = np.abs(s) + 0x84
    exp = np.searchsorted(_SEG_END, s, side="left"); exp = np.minimum(exp, 7)
    man = (s >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | man) & 0xFF).astype(np.uint8).tobytes()


def alaw_decode(b: bytes) -> np.ndarray:
    a = np.frombuffer(b, np.uint8).astype(np.int32) ^ 0x55
    sign, exp, man = a & 0x80, (a >> 4) & 0x07, a & 0x0F
    x = np.where(exp == 0, (man << 4) + 8, ((man << 4) + 0x108) << np.maximum(exp - 1, 0))
    x = np.where(sign != 0, x, -x)
    return (x / 32768.0).astype(np.float32)


def alaw_encode(x: np.ndarray) -> bytes:
    s = np.clip(x * 32768.0, -32767, 32767).astype(np.int32)
    sign = np.where(s >= 0, 0x80, 0); s = np.abs(s)
    exp = np.zeros_like(s); man = (s >> 4) & 0x0F
    big = s >= 256
    e = np.floor(np.log2(np.maximum(s, 1))).astype(np.int32) - 7
    e = np.clip(e, 1, 7)
    exp = np.where(big, e, 0)
    man = np.where(big, (s >> (exp + 3)) & 0x0F, (s >> 4) & 0x0F)
    return ((sign | (exp << 4) | man) ^ 0x55).astype(np.uint8).tobytes()


FORMATS = {"audio/pcm": ("pcm", 24000), "pcm16": ("pcm", 24000), "audio/pcmu": ("ulaw", 8000), "g711_ulaw": ("ulaw", 8000),
           "audio/pcma": ("alaw", 8000), "g711_alaw": ("alaw", 8000)}


def _fmt(spec) -> tuple[str, int]:
    if isinstance(spec, dict):
        kind, rate = FORMATS.get(spec.get("type", "audio/pcm"), ("pcm", 24000))
        return kind, int(spec.get("rate") or rate)
    if isinstance(spec, str):
        return FORMATS.get(spec, ("pcm", 24000))
    return ("pcm", 24000)


def decode_audio(b64: str, fmt: tuple[str, int]) -> tuple[np.ndarray, int]:
    raw = base64.b64decode(b64 or "")
    kind, rate = fmt
    if kind == "ulaw":
        return ulaw_decode(raw), rate
    if kind == "alaw":
        return alaw_decode(raw), rate
    return np.frombuffer(raw[: len(raw) // 2 * 2], "<i2").astype(np.float32) / 32768.0, rate


def encode_audio(x: np.ndarray, fmt: tuple[str, int]) -> str:
    kind, _ = fmt
    if kind == "ulaw":
        return base64.b64encode(ulaw_encode(x)).decode()
    if kind == "alaw":
        return base64.b64encode(alaw_encode(x)).decode()
    return base64.b64encode(np.clip(x * 32768, -32768, 32767).astype("<i2").tobytes()).decode()


def _id(p: str) -> str:
    return f"{p}_{uuid.uuid4().hex[:20]}"


# 有工具时默认附加的调用守则（接口层产品行为，可用 SOPHON_LIVE_TOOL_RULES=0 关闭）。
# 针对的问题：用户还没说就拿编造的值调工具；姓名/邮箱听写错了不让用户拼。
TOOL_RULES_EN = (
    "\n\nTool-use rules:"
    "\n- Only call a tool with argument values the user actually said or that came from an earlier tool result. "
    "Never guess, invent or use placeholder values (no example names, emails, zip codes or IDs). If something is missing, ask for it first and wait for the answer."
    "\n- Names, emails and IDs are easy to mishear. When a lookup fails, ask the user to spell it letter by letter, read it back, then try again with the spelled version."
    "\n- Before any action that changes something (cancel, exchange, return, modify, book, pay), summarize the exact details and get an explicit yes first."
    "\n- Say one thing at a time and keep each reply short; do not start a new reply before the previous one is finished."
)
TOOL_RULES_ZH = (
    "\n\n调用工具的守则："
    "\n- 工具参数只能用用户亲口说过的内容，或者之前工具返回的结果；绝不猜测、编造或用示例值（姓名、邮箱、邮编、编号都一样）。缺什么先问，等用户回答。"
    "\n- 姓名、邮箱、编号容易听错。查不到时请用户逐字（逐个字母）拼一遍，复述确认后再用拼写的版本重查。"
    "\n- 取消、换货、退货、修改、预订、付款这类会改动东西的操作，先把具体细节复述一遍，得到用户明确同意再执行。"
    "\n- 一次只说一件事，回复简短；上一句没说完不要开始新的回复。"
)


class RealtimeSession:
    def __init__(self, ws, args, asr, asr_lock, resample, model: str):
        self.ws, self.args, self.asr, self.asr_lock, self.resample = ws, args, asr, asr_lock, resample
        self.id = _id("sess"); self.model = model or "sophon-realtime"
        self.cfg = {"type": "realtime", "object": "realtime.session", "id": self.id, "model": self.model, "output_modalities": ["audio"],
                    "instructions": "", "tools": [], "tool_choice": "auto",
                    "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}, "transcription": None,
                                        "turn_detection": {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 500}},
                              "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "alloy", "speed": 1.0}}}
        self.in_fmt = ("pcm", 24000); self.out_fmt = ("pcm", 24000)
        self.engine: VllmOmniAdapter | None = None
        self.closed = False
        self.tasks: list = []
        self.lang = "en"
        self.resp: dict | None = None          # current response {id, item_id, text, done, calls:[...], audio_s}
        self.pending_outputs: list = []        # (call_id, output) from conversation.item.create(function_call_output)
        self.heard: list = []                  # ASR text of each caller turn (always transcribed, for the grounding guard)
        self.seen: list = []                   # tool outputs (+ instructions / tool schema) the agent has seen
        self.asr_tasks: list = []
        self.guard_counts: dict = {}
        self.calls: dict = {}                  # call_id -> response id
        self.in_ms = 0.0                       # input audio received (ms)
        self.vad_on = False; self.vad_voiced = 0; self.vad_silence = 0.0; self.speech_t0 = 0.0
        self.user_audio: list = []; self.user_item: str | None = None
        self.last_client_audio = 0.0
        self.bo = backend.BackOffice(self.id) if backend.MODEL else None   # front + back-office mode (backend.py)
        self.bk_wait: dict = {}                # call_id -> future for back-office tool calls
        self.hold: dict | None = None           # non-interruptible say in progress
        self.held: list = []                   # caller audio held back meanwhile

    async def send(self, ev: dict) -> None:
        ev.setdefault("event_id", _id("event"))
        try:
            await self.ws.send(json.dumps(ev, ensure_ascii=False))
        except websockets.ConnectionClosed:
            self.closed = True

    async def error(self, message: str, code: str = "invalid_request_error", event_id=None, param=None) -> None:
        await self.send({"type": "error", "error": {"type": "invalid_request_error", "code": code, "message": message, "param": param,
                                                      "event_id": event_id}})

    async def open(self) -> None:
        await self.send({"type": "session.created", "session": self.cfg})

    # ------------------------------------------------------------ session
    async def on_update(self, ev: dict) -> None:
        s = ev.get("session") or {}
        before = (copy.deepcopy(self.cfg), self.lang, self.in_fmt, self.out_fmt)
        for k in ("instructions", "tool_choice", "output_modalities"):
            if k in s and s[k] is not None:
                self.cfg[k] = s[k]
        if "modalities" in s:                                  # beta name
            self.cfg["output_modalities"] = [m for m in s["modalities"] if m in ("audio", "text")] or ["audio"]
        if isinstance(s.get("tools"), list):
            self.cfg["tools"] = s["tools"]
        a = s.get("audio") or {}
        ain, aout = a.get("input") or {}, a.get("output") or {}
        if "format" in ain: self.cfg["audio"]["input"]["format"] = ain["format"]
        if "input_audio_format" in s: self.cfg["audio"]["input"]["format"] = s["input_audio_format"]
        if "format" in aout: self.cfg["audio"]["output"]["format"] = aout["format"]
        if "output_audio_format" in s: self.cfg["audio"]["output"]["format"] = s["output_audio_format"]
        for k in ("transcription", "turn_detection", "noise_reduction"):
            if k in ain: self.cfg["audio"]["input"][k] = ain[k]
        if "input_audio_transcription" in s: self.cfg["audio"]["input"]["transcription"] = s["input_audio_transcription"]
        if "turn_detection" in s: self.cfg["audio"]["input"]["turn_detection"] = s["turn_detection"]
        if aout.get("voice") or s.get("voice"): self.cfg["audio"]["output"]["voice"] = aout.get("voice") or s.get("voice")
        self.in_fmt = _fmt(self.cfg["audio"]["input"]["format"]); self.out_fmt = _fmt(self.cfg["audio"]["output"]["format"])
        ins = self.cfg.get("instructions") or ""
        cjk = sum(1 for ch in ins if "一" <= ch <= "鿿")
        self.lang = os.environ.get("SOPHON_LIVE_LANG") or ("zh" if cjk >= max(CJK_MIN, 0.05 * max(1, len(ins))) else "en")
        tools = [t for t in self.cfg.get("tools") or [] if isinstance(t, dict) and t.get("type", "function") == "function"]
        if self.cfg.get("tool_choice") == "none":
            tools = []
        if self.bo is not None and (tools or backend.ALL):
            tools = [backend.DELEGATE_TOOL]                     # the front only delegates; the back office holds the real tools
        if self.engine is None:
            self.engine = VllmOmniAdapter(self.args.engine, self.args.bridge)
            try:
                await self.engine.start(instructions=self._sys(), voice=DEFAULT_VOICE, lang=self.lang, tools=tools, history=[])
            except Exception as e:  # noqa: BLE001
                log.exception("engine start failed"); await self.error(f"engine unavailable: {e}", "server_error"); return
            self.tasks = [asyncio.create_task(self._engine_events()), asyncio.create_task(self._idle_feed())]
            log.info("realtime %s started lang=%s tools=%d in=%s out=%s", self.id, self.lang, len(tools), self.in_fmt, self.out_fmt)
        else:
            old_tools = self.engine.cfg["extra_body"].get("tools")
            self.engine.cfg["extra_body"]["tools"] = tools
            ok, err = await self.engine.set_instructions(self._sys())
            if not ok:                                         # the engine kept the previous session: say so instead of session.updated
                self.engine.cfg["extra_body"]["tools"] = old_tools
                self.cfg, self.lang, self.in_fmt, self.out_fmt = before
                err = err or {}
                log.warning("realtime %s: engine refused session.update: %s", self.id, err)
                await self.error(f"session.update was not applied: {err.get('message') or 'refused by the engine'}",
                                 err.get("code") or "session_update_failed", ev.get("event_id"))
                return
        await self.send({"type": "session.updated", "session": self.cfg})

    def _sys(self) -> str:
        rule = ("\n\n你的回复会被直接合成为语音说出来：用自然的口语；只写要说出口的话，不要写括号说明、表情或格式符号。" if self.lang == "zh" else
                "\n\nYour reply is spoken aloud: use natural spoken language; write only the words to be spoken, no parenthetical notes, emoji or markdown.")
        tools = [t for t in self.cfg.get("tools") or [] if isinstance(t, dict)]
        if tools and self.cfg.get("tool_choice") != "none" and self.bo is not None:
            rule += backend.FRONT_RULES_EN
        elif not tools and self.bo is not None and backend.ALL:
            rule += backend.FRONT_RULES_ALL_EN
        elif tools and self.cfg.get("tool_choice") != "none" and os.environ.get("SOPHON_LIVE_TOOL_RULES", "1") != "0":
            rule += TOOL_RULES_ZH if self.lang == "zh" else TOOL_RULES_EN
        return (self.cfg.get("instructions") or "") + rule

    # ------------------------------------------------------------ client events
    async def on_event(self, ev: dict) -> None:
        t = ev.get("type", ""); eid = ev.get("event_id")
        if t == "session.update":
            return await self.on_update(ev)
        if self.engine is None and t not in ("session.update",):
            await self.on_update({"session": {}})             # clients may start streaming without an update
        if t == "input_audio_buffer.append":
            x, rate = decode_audio(ev.get("audio") or "", self.in_fmt)
            a = self.resample(x, rate, 16000)
            self.last_client_audio = time.monotonic()
            if self.hold is not None:                          # a non-interruptible say is playing: keep the caller's audio for later
                h = self.hold
                if h["done"] and time.monotonic() >= h.get("t_first", 0) + h.get("audio_s", 0) + 0.3:   # played to the end
                    await self._release_hold()
                else:
                    self.held.append(a); await self.engine.send_audio(np.zeros_like(a)); return
            await self._vad(a)
            await self.engine.send_audio(a)
        elif t == "input_audio_buffer.commit":
            await self.send({"type": "input_audio_buffer.committed", "previous_item_id": None, "item_id": self.user_item or _id("item")})
        elif t == "input_audio_buffer.clear":
            self.user_audio = []
            await self.send({"type": "input_audio_buffer.cleared"})
        elif t == "conversation.item.create":
            item = ev.get("item") or {}
            if item.get("type") == "function_call_output":
                fut = self.bk_wait.pop(item.get("call_id"), None)
                if fut is not None:
                    if not fut.done(): fut.set_result(str(item.get("output", "")))
                else:
                    self.pending_outputs.append((item.get("call_id"), str(item.get("output", ""))))
                item = dict(item, id=item.get("id") or _id("item"), object="realtime.item", status="completed")
                await self.send({"type": "conversation.item.added", "previous_item_id": ev.get("previous_item_id"), "item": item})
                await self.send({"type": "conversation.item.done", "previous_item_id": ev.get("previous_item_id"), "item": item})
                await self.send({"type": "conversation.item.created", "previous_item_id": ev.get("previous_item_id"), "item": item})
            else:
                item = dict(item, id=item.get("id") or _id("item"), object="realtime.item", status="completed")
                await self.send({"type": "conversation.item.created", "previous_item_id": ev.get("previous_item_id"), "item": item})
                log.info("realtime %s: text/other item ignored (speech input only): %s", self.id, str(item)[:120])
        elif t == "response.create" and isinstance((ev.get("response") or {}).get("say"), str):
            await self._say(ev)
        elif t == "response.create":
            meta = (ev.get("response") or {}).get("metadata")
            if self.resp is not None and not self.resp["done"] and self.resp["audio"]:
                self.resp["closing"] = True                   # the lead-in still playing: its remaining audio stays with it
            r = await self._begin(meta)
            r["forced"] = True; self._track(r)
            r["empty_timer"] = asyncio.get_running_loop().call_later(12.0, lambda: asyncio.ensure_future(self._finish_if_empty(r)))
            if self.pending_outputs:
                outs, self.pending_outputs = self.pending_outputs, []
                for i, (cid, out) in enumerate(outs):
                    self.seen.append(out)
                    if GROUND: out = out + grounding.failure_hint(out, self.heard)
                    await self.engine.tool_result(cid, out, speak=(i == len(outs) - 1))
                log.info("realtime %s: %d tool result(s) -> follow-up", self.id, len(outs))
            # otherwise the engine answers on its own turn detection; nothing to force
        elif t == "response.cancel":
            pass
        elif t == "conversation.item.truncate":
            await self.send({"type": "conversation.item.truncated", "item_id": ev.get("item_id"), "content_index": ev.get("content_index", 0),
                             "audio_end_ms": ev.get("audio_end_ms", 0)})
        elif t in ("conversation.item.delete", "conversation.item.retrieve", "output_audio_buffer.clear"):
            pass
        else:
            await self.error(f"Unsupported event type: {t}", "unsupported_event", eid)

    # ------------------------------------------------------------ say: speak given text verbatim (extension)
    async def _say(self, ev: dict) -> None:
        """response.create {"response": {"say": "...", "interruptible": true}}: the session voice speaks exactly this text, and it
        becomes part of the conversation history. interruptible=false: until it has been spoken, the caller's audio is held back
        (not heard by the model, so no barge-in) and delivered right after."""
        rq = ev.get("response") or {}
        text = rq["say"].strip()
        if not text:
            return await self.error("response.say is empty", "invalid_value", ev.get("event_id"), "response.say")
        if self.resp is not None and not self.resp["done"] and self.resp["audio"]:
            self.resp["closing"] = True
        r = await self._begin(rq.get("metadata"))
        r["forced"] = True; r["say"] = text; self._track(r)
        if rq.get("interruptible") is False:
            self.hold = r
        if not await self.engine.say(text):
            self.hold = None
            await self._finish(r, "failed")
            return await self.error("this engine does not support response.say (needs the Flash / Tag / E2E plugin 1.6.0+)", "unsupported", ev.get("event_id"))
        r["empty_timer"] = asyncio.get_running_loop().call_later(60.0, lambda: asyncio.ensure_future(self._finish_if_empty(r)))
        log.info("realtime %s: say %r interruptible=%s", self.id, text[:80], rq.get("interruptible", True))

    async def _release_hold(self) -> None:
        held, self.held, self.hold = self.held, [], None
        for a in held:
            await self._vad(a); await self.engine.send_audio(a)

    # ------------------------------------------------------------ input VAD + transcription
    async def _vad(self, a: np.ndarray) -> None:
        if VAD_KIND == "silero":
            sess = _silero()
            if sess is not None:
                return await self._vad_silero(a, sess)
        return await self._vad_energy(a)

    async def _vad_silero(self, a: np.ndarray, sess) -> None:
        if getattr(self, "svad", None) is None:
            self.svad = SileroVAD(sess); self.sv_buf = []
        self.sv_buf.append(a)
        if self.vad_on:
            self.user_audio.append(a)
        for p in self.svad.feed(a):
            self.in_ms += 32
            if p >= VAD_ON_P:
                self.vad_voiced += 32; self.vad_silence = 0.0
                if not self.vad_on and self.vad_voiced >= VAD_START_MS:
                    self.vad_on = True; self.user_item = _id("item"); self.speech_t0 = self.in_ms - self.vad_voiced
                    self.user_audio = [np.concatenate(self.sv_buf[-20:])]
                    await self.send({"type": "input_audio_buffer.speech_started", "audio_start_ms": int(self.speech_t0), "item_id": self.user_item})
            elif p < VAD_OFF_P or not self.vad_on:
                self.vad_voiced = 0
                if self.vad_on:
                    self.vad_silence += 32
                    if self.vad_silence >= VAD_STOP_MS:
                        await self._vad_end()
        self.sv_buf = self.sv_buf[-20:]

    async def _vad_end(self) -> None:
        self.vad_on = False
        item = self.user_item
        await self.send({"type": "input_audio_buffer.speech_stopped", "audio_end_ms": int(self.in_ms - self.vad_silence), "item_id": item})
        await self.send({"type": "input_audio_buffer.committed", "previous_item_id": None, "item_id": item})
        await self.send({"type": "conversation.item.created", "previous_item_id": None, "item": {
            "id": item, "object": "realtime.item", "type": "message", "role": "user", "status": "completed",
            "content": [{"type": "input_audio", "transcript": None}]}})
        if self.asr is not None:
            self.asr_tasks.append(asyncio.create_task(self._asr(np.concatenate(self.user_audio) if self.user_audio else np.zeros(0, np.float32), item)))
        self.user_audio = []

    async def _vad_energy(self, a: np.ndarray) -> None:
        n = 320
        if a.size:
            st = getattr(self, "_lvl", None)
            if st is None: st = self._lvl = []
            st.append(float(np.sqrt((a ** 2).mean())))
            if len(st) >= 500:
                q = np.percentile(st, [10, 50, 90]); self._lvl = []
                log.info("realtime %s: input rms p10/p50/p90 = %.4f/%.4f/%.4f (chunk %d samples)", self.id, q[0], q[1], q[2], a.size)
        for i in range(0, a.size - n + 1, n):
            fr = a[i:i + n]; rms = float(np.sqrt((fr ** 2).mean()))
            self.in_ms += 20
            h = getattr(self, "_rms5", None)
            if h is None:
                from collections import deque
                h = self._rms5 = deque(maxlen=250)
            h.append(rms)
            if len(h) % 10 == 0 or not hasattr(self, "_thr"):
                self._thr = max(VAD_RMS, float(np.percentile(h, 20)) * VAD_K)
            if self.vad_on:
                self.user_audio.append(fr)
            else:
                pa = getattr(self, "pre_audio", None)
                if pa is None: pa = self.pre_audio = []
                pa.append(fr); del pa[:-25]
            if rms > self._thr:
                self.vad_voiced += 1; self.vad_silence = 0.0
                if not self.vad_on and self.vad_voiced * 20 >= VAD_E_START_MS:
                    self.vad_on = True; self.user_item = _id("item"); self.speech_t0 = self.in_ms - self.vad_voiced * 20
                    self.user_audio = list(getattr(self, "pre_audio", [])) + [fr]
                    await self.send({"type": "input_audio_buffer.speech_started", "audio_start_ms": int(self.speech_t0), "item_id": self.user_item})
                    log.info("realtime %s: vad start at %.1fs", self.id, self.in_ms / 1000)
            else:
                self.vad_voiced = max(0, self.vad_voiced - 2)          # tolerate short dips inside a word
                if self.vad_on:
                    self.vad_silence += 20
                    if self.vad_silence >= 500:
                        self.vad_on = False
                        item = self.user_item
                        log.info("realtime %s: vad end at %.1fs (%d frames buffered)", self.id, self.in_ms / 1000, len(self.user_audio))
                        await self.send({"type": "input_audio_buffer.speech_stopped", "audio_end_ms": int(self.in_ms - self.vad_silence), "item_id": item})
                        await self.send({"type": "input_audio_buffer.committed", "previous_item_id": None, "item_id": item})
                        await self.send({"type": "conversation.item.created", "previous_item_id": None, "item": {
                            "id": item, "object": "realtime.item", "type": "message", "role": "user", "status": "completed",
                            "content": [{"type": "input_audio", "transcript": None}]}})
                        if self.asr is not None:
                            self.asr_tasks.append(asyncio.create_task(self._asr(np.concatenate(self.user_audio) if self.user_audio else np.zeros(0, np.float32), item)))
                        self.user_audio = []

    async def _asr(self, a: np.ndarray, item: str) -> None:
        if a.size < 3200:
            log.info("realtime %s: asr skipped (%.2fs)", self.id, a.size / 16000)
            return
        try:
            async with self.asr_lock:
                text = await asyncio.to_thread(self.asr, a)
        except Exception:  # noqa: BLE001
            log.exception("asr"); return
        log.info("realtime %s: asr %.1fs -> %r", self.id, a.size / 16000, (text or "")[:200])
        if text and text.strip():
            self.heard.append(text.strip())
            if self.bo is not None: self.bo.add("caller", text)
        if self.cfg["audio"]["input"].get("transcription"):
            await self.send({"type": "conversation.item.input_audio_transcription.completed", "item_id": item, "content_index": 0, "transcript": text})

    async def _idle_feed(self) -> None:
        z = np.zeros(1600, np.float32)
        while not self.closed:
            await asyncio.sleep(0.1)
            if time.monotonic() - self.last_client_audio > 0.3:
                await self.engine.send_audio(z)

    # ------------------------------------------------------------ responses
    async def _begin(self, metadata=None, erid=None) -> dict:
        r = {"id": _id("resp"), "item_id": _id("item"), "text": "", "calls": [], "audio": False, "done": False, "t0": time.monotonic(),
             "metadata": metadata, "erid": erid}
        self.resp = r
        await self.send({"type": "response.created", "response": {"id": r["id"], "object": "realtime.response", "status": "in_progress",
                                                                  "status_details": None, "output": [], "conversation_id": "conv_" + self.id[5:],
                                                                  "output_modalities": self.cfg.get("output_modalities", ["audio"]),
                                                                  "metadata": metadata}})
        return r

    async def _begin_message(self, r: dict) -> None:
        if r["audio"]:
            return
        r["audio"] = True
        item = {"id": r["item_id"], "object": "realtime.item", "type": "message", "role": "assistant", "status": "in_progress", "content": []}
        await self.send({"type": "response.output_item.added", "response_id": r["id"], "output_index": 0, "item": item})
        await self.send({"type": "conversation.item.added", "previous_item_id": None, "item": item})
        await self.send({"type": "conversation.item.created", "previous_item_id": None, "item": item})
        await self.send({"type": "response.content_part.added", "response_id": r["id"], "item_id": r["item_id"], "output_index": 0,
                         "content_index": 0, "part": {"type": "audio", "transcript": ""}})

    async def _finish(self, r: dict, status: str = "completed") -> None:
        if r["done"]:
            return
        r["done"] = True
        if self.bo is not None and r.get("text"): self.bo.add("front", r["text"])
        output = []
        if r["audio"]:
            base = {"response_id": r["id"], "item_id": r["item_id"], "output_index": 0, "content_index": 0}
            await self.send({"type": "response.output_audio.done", **base})
            await self.send({"type": "response.output_audio_transcript.done", **base, "transcript": r["text"]})
            await self.send({"type": "response.content_part.done", **base, "part": {"type": "audio", "transcript": r["text"]}})
            item = {"id": r["item_id"], "object": "realtime.item", "type": "message", "role": "assistant", "status": status if status != "cancelled" else "incomplete",
                    "content": [{"type": "output_audio", "transcript": r["text"]}]}
            await self.send({"type": "response.output_item.done", "response_id": r["id"], "output_index": 0, "item": item})
            await self.send({"type": "conversation.item.done", "previous_item_id": None, "item": item})
            output.append(item)
        output += r["calls"]
        await self.send({"type": "response.done", "response": {
            "id": r["id"], "object": "realtime.response", "status": status, "status_details": {"type": status} if status != "completed" else None,
            "output": output, "conversation_id": "conv_" + self.id[5:], "output_modalities": self.cfg.get("output_modalities", ["audio"]),
            "metadata": r.get("metadata"), "usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0, "input_token_details": {"text_tokens": 0, "audio_tokens": 0, "cached_tokens": 0},
                      "output_token_details": {"text_tokens": 0, "audio_tokens": 0}}}})
        if self.resp is r:
            self.resp = None

    async def _finish_if_empty(self, r: dict) -> None:
        if not r["done"] and not r["audio"] and not r["calls"]:
            await self._finish(r)

    async def _guard(self, e: dict) -> bool:
        """True = call rejected and sent back to the model (not shown to the client)."""
        if os.environ.get("SOPHON_LIVE_UNKNOWN_TOOL", "1") != "0":
            names = {(t.get("name") or (t.get("function") or {}).get("name")) for t in (self.cfg.get("tools") or []) if isinstance(t, dict)}
            nm = e.get("name", "")
            if names and nm not in names:
                k = "?" + nm
                self.guard_counts[k] = self.guard_counts.get(k, 0) + 1
                if self.guard_counts[k] <= 3:
                    log.info("realtime %s: unknown tool %s rejected (%d)", self.id, nm, self.guard_counts[k])
                    msg = (f"The tool '{nm}' does not exist; you can only call these tools: {', '.join(sorted(n for n in names if n))}. "
                           "If this is something the customer has to check or change on their own phone or device, ask them to do it "
                           "step by step and tell you what they see. Do not call a tool for it.")
                    await self.engine.tool_result(e.get("call_id"), msg, speak=True)
                    return True
        if not GROUND or self.asr is None:
            return False
        pend = [t for t in self.asr_tasks if not t.done()]
        if pend:
            await asyncio.wait(pend, timeout=2.5)
        self.asr_tasks = [t for t in self.asr_tasks if not t.done()]
        args = e.get("arguments") or {}
        if isinstance(args, str):
            try: args = json.loads(args)
            except Exception: args = {}  # noqa: E722
        ctx = [self.cfg.get("instructions") or "", json.dumps(self.cfg.get("tools") or [])] + self.seen
        bad = grounding.check(e.get("name", ""), args, self.heard, ctx)
        if not bad:
            return False
        key = e.get("name", "") + json.dumps(args, sort_keys=True)
        self.guard_counts[key] = self.guard_counts.get(key, 0) + 1
        if self.guard_counts[key] > 2:                          # the model insists: let it through rather than loop forever
            log.info("realtime %s: grounding guard let through after 2 rejections: %s", self.id, key[:160]); return False
        msg = grounding.guard_message(bad, self.heard)
        log.info("realtime %s: grounding guard rejected %s(%s): %s", self.id, e.get("name"), json.dumps(args)[:160], [b[:2] for b in bad])
        log.info("realtime %s: heard so far: %s", self.id, " || ".join(self.heard[-4:])[-600:])
        await self.engine.tool_result(e.get("call_id"), msg, speak=True)
        return True

    async def _call(self, e: dict) -> None:
        r = self.resp if (self.resp is not None and not self.resp["done"]) else await self._begin()
        idx = (1 if r["audio"] else 0) + len(r["calls"])
        args = json.dumps(e.get("arguments") or {}, ensure_ascii=False)
        item = {"id": _id("item"), "object": "realtime.item", "type": "function_call", "status": "in_progress", "name": e["name"],
                "call_id": e["call_id"], "arguments": ""}
        await self.send({"type": "response.output_item.added", "response_id": r["id"], "output_index": idx, "item": item})
        await self.send({"type": "conversation.item.added", "previous_item_id": None, "item": item})
        await self.send({"type": "response.function_call_arguments.delta", "response_id": r["id"], "item_id": item["id"], "output_index": idx,
                         "call_id": e["call_id"], "delta": args})
        await self.send({"type": "response.function_call_arguments.done", "response_id": r["id"], "item_id": item["id"], "output_index": idx,
                         "call_id": e["call_id"], "name": e["name"], "arguments": args})
        item = dict(item, status="completed", arguments=args)
        await self.send({"type": "response.output_item.done", "response_id": r["id"], "output_index": idx, "item": item})
        await self.send({"type": "conversation.item.done", "previous_item_id": None, "item": item})
        r["calls"].append(item)
        log.info("realtime %s: function call %s(%s)", self.id, e["name"], args[:200])
        if not r["audio"]:                                     # a call with no spoken lead-in: the response ends here
            r["no_audio_wait"] = asyncio.get_running_loop().call_later(1.5, lambda: asyncio.ensure_future(self._finish(r)))

    async def _delegate(self, e: dict) -> None:
        args = e.get("arguments") or {}
        if isinstance(args, str):
            try: args = json.loads(args)
            except Exception: args = {"task": args}  # noqa: E722
        task = str(args.get("task") or "")
        log.info("realtime %s: delegate %r", self.id, task[:200])
        pend = [t for t in self.asr_tasks if not t.done()]      # the caller's last words must be in the transcript
        if pend:
            await asyncio.wait(pend, timeout=3.0)
        loop = asyncio.get_running_loop()

        async def call_tool(name: str, a: dict) -> str:
            cid = _id("call"); fut = loop.create_future(); self.bk_wait[cid] = fut
            await self._call({"name": name, "call_id": cid, "arguments": a})
            try:
                out = await asyncio.wait_for(fut, 90)
            except asyncio.TimeoutError:
                self.bk_wait.pop(cid, None); out = "Error: no result from the system"
            self.seen.append(out)
            return out
        try:
            tools = [t for t in self.cfg.get("tools") or [] if isinstance(t, dict)]
            ans = await self.bo.run(self.cfg.get("instructions") or "", tools, task, call_tool)
        except Exception as ex:  # noqa: BLE001
            log.exception("delegate"); ans = "Sorry, something went wrong on my side. Could you say that again?"
        log.info("realtime %s: delegate result %r", self.id, ans[:200])
        if not self.closed:
            await self.engine.tool_result(e.get("call_id"), ans, speak=True)

    def _track(self, r: dict) -> None:
        self._open = [x for x in getattr(self, "_open", []) if not x["done"]] + [r]

    def _route(self, erid):
        opn = [x for x in getattr(self, "_open", []) if not x["done"]]
        if erid is not None:
            for x in opn:
                if x.get("erid") == erid:
                    return x
        # an engine reply we haven't seen: give it the newest open response that has no engine reply yet (e.g. one created by
        # response.create), otherwise, without reply ids, keep appending to the newest open response
        r = self.resp
        if r is not None and not r["done"] and r.get("erid") is None:
            r["erid"] = erid if erid is not None else "_"
            self._track(r)
            return r
        if erid is None and r is not None and not r["done"]:
            return r
        return None

    async def _engine_events(self) -> None:
        q = self.engine.q
        while not self.closed:
            e = await q.get(); t = e["t"]
            if t in ("audio", "text"):
                erid = e.get("rid")
                r = self._route(erid)
                if r is None:
                    r = await self._begin(erid=erid); self._track(r)
                if r.get("no_audio_wait") is not None:
                    r["no_audio_wait"].cancel(); r["no_audio_wait"] = None
                    if not r["audio"] and r["calls"]:          # speech after a bare call belongs to a new response
                        await self._finish(r); r = await self._begin()
                await self._begin_message(r)
                base = {"response_id": r["id"], "item_id": r["item_id"], "output_index": 0, "content_index": 0}
                if t == "audio":
                    if r.get("say") is not None:                # playback clock of a say (non-interruptible hold ends when it has been played)
                        r.setdefault("t_first", time.monotonic()); r["audio_s"] = r.get("audio_s", 0.0) + e["pcm"].size / e["sr"]
                    x = self.resample(e["pcm"], e["sr"], self.out_fmt[1])
                    step = self.out_fmt[1] // 10                # 100 ms deltas
                    for i in range(0, x.size, step):
                        await self.send({"type": "response.output_audio.delta", **base, "delta": encode_audio(x[i:i + step], self.out_fmt)})
                else:
                    d = e["delta"]; r["text"] += d
                    await self.send({"type": "response.output_audio_transcript.delta", **base, "delta": d})
            elif t == "tool_call":
                if self.bo is not None and e.get("name") == "delegate":
                    self.tasks.append(asyncio.create_task(self._delegate(e)))
                    continue
                if await self._guard(e):
                    continue
                await self._call(e)
            elif t == "done":
                erid = e.get("rid")
                r = next((x for x in self._open if x.get("erid") == erid and not x["done"]), None) if erid else None
                if r is None:
                    r = next((x for x in self._open if not x["done"] and x["audio"]), None)
                if r is not None:
                    await self._finish(r)
            elif t == "cancel":
                if self.resp is not None and not self.resp["done"]:
                    await self._finish(self.resp, "cancelled")
            elif t == "error":
                log.warning("realtime %s engine: %s", self.id, e.get("message"))
                if "closed" in (e.get("message") or ""):
                    await self.error("engine connection lost", "server_error"); await self.close(); return

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for t in self.tasks:
            t.cancel()
        if self.engine is not None:
            await self.engine.close()
        log.info("realtime %s closed", self.id)
