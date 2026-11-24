"""GPT-Live compatible server (wss://<host>/v1/live/sessions) in front of a Sophon-Realtime engine.

Client protocol (same as OpenAI GPT-Live with client delegation):
  in : session.start, session.input_audio.append, session.commentary.append, session.thinking.append, session.instructions.append, session.close
  out: session.started, session.output_audio.delta (continuous ~100 ms frames, silence when idle), session.output_transcript.delta,
       session.input_transcript.delta, session.delegation.created, session.commentary.appended, session.closed, error
Delegation: the model gets one tool, `delegate`. When it calls it, the gateway emits session.delegation.created; commentary (to be
said, in the model's own words) and thinking (background only) appended for that delegation go back to the model as the tool
result and it answers aloud."""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from collections import deque

import numpy as np
import websockets
from websockets.http11 import Response
from websockets.datastructures import Headers

from .adapters import VllmOmniAdapter
from .realtime import RealtimeSession

log = logging.getLogger("sophon_live")

FRAME_MS = 100
TRANSCRIPT_LEAD_MS = 200            # GPT-Live's output transcript runs ~200 ms ahead of the audio
DELEG_QUIET_S = float(os.environ.get("SOPHON_LIVE_DELEG_QUIET", "0.6"))   # commentary chunks arriving this close together are merged
DEFAULT_VOICE = os.environ.get("SOPHON_LIVE_VOICE", "chelsie")
API_KEYS = {k.strip() for k in os.environ.get("SOPHON_LIVE_API_KEYS", "").split(",") if k.strip()}
CJK = re.compile(r"[㐀-鿿]")
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍～~*#`_>]+")


def clean_text(t: str, lang: str) -> str:
    """Transcript as spoken: no emoji / markdown symbols, no line breaks."""
    t = EMOJI.sub("", t).replace("\r", "")
    t = re.sub(r"\s*\n\s*", "" if lang == "zh" else " ", t)
    return re.sub(r" {2,}", " ", t)

DELEGATE = {
    "zh": {"type": "function", "name": "delegate",
           "description": "把需要后端完成的事交给后端助手：查实时或外部信息（时间、日期、天气、新闻、搜索）、查用户的记忆和资料、控制设备或调用任何功能、"
                          "看图。调用前先用一句很短的口语告诉用户你去处理一下。闲聊、常识和你自己就知道的知识直接回答，不要调用。",
           "parameters": {"type": "object", "properties": {"task": {"type": "string", "description": "一句话写清楚用户要什么"}}, "required": ["task"]}},
    "en": {"type": "function", "name": "delegate",
           "description": "Hand work that needs the backend to the backend assistant: live or external information (time, date, weather, news, search), "
                          "the user's memories and profile, controlling devices or calling any function, looking at images. Before calling, say one very short spoken phrase that you're on it. Answer chit-chat, general "
                          "knowledge and anything you already know yourself without calling it.",
           "parameters": {"type": "object", "properties": {"task": {"type": "string", "description": "What the user wants, in one sentence"}}, "required": ["task"]}},
}
SPOKEN_RULE = {"zh": "\n\n你的回复会被直接合成为语音说出来：用自然的口语，一般一两句；只写要说出口的话，不要写括号说明、表情或格式符号。",
               "en": "\n\nYour reply is spoken aloud: use natural spoken language, usually one or two sentences; write only the words to be "
                     "spoken, no parenthetical notes, emoji or markdown."}
RESULT_FMT = {"zh": ("后端结果（只能根据这段内容回答，用你自己的口语自然地告诉用户，不要提到后端或工具；内容里没有的信息绝对不要自己编。"
                     "如果它说没查到、想不起来或请用户再说一遍，就照这个意思告诉用户）：\n{c}",
                     "背景信息（不要照读，只在需要时用自己的话简短回应；不要编造其中没有的事实）：\n{c}"),
              "en": ("Backend result (answer only from this text, in your own natural spoken words; don't mention the backend or tools; never "
                     "invent anything that is not in it. If it says nothing was found or asks the user to repeat, tell the user exactly that):\n{c}",
                     "Background information (don't read it out; use it only if needed, in your own words; don't invent facts not in it):\n{c}")}
PROGRESS_FMT = {"zh": "后端还在处理（这是进度，不是结果）。用一句很短的口语让用户稍等，不要重复你刚说过的话，不要编结果：\n{c}",
                "en": "The backend is still working (progress, not the result). Say one very short phrase asking the user to hold on; don't repeat "
                      "what you just said and don't invent a result:\n{c}"}
DELEG_MODE = os.environ.get("SOPHON_LIVE_DELEG_MODE", "model")   # model: every backend update goes to the model, which decides to speak or <wait>; rules: gateway heuristics
IDLE_UPDATE_S = float(os.environ.get("SOPHON_LIVE_IDLE_UPDATE", "7.0"))   # model mode: no backend news by then -> tell the model once, it may say a short "hold on"
UPDATE_FMT = {
    "zh": ("【后端更新｜委派开始后 {t:.0f} 秒｜距你上次开口 {gap} 】\n{body}\n你最近说过：「{last}」\n\n"
           "怎么回应：\n- 如果上面有用户要的新结果（答案、查到的内容、已经办好、或者没查到）：用你自己的口语自然地告诉用户，只依据上面的内容，"
           "绝对不要编造，不要提后端或工具。\n- 如果只是进度、后台背景，或者和你已经说过的意思重复：你上次开口不到 6 秒的话，只输出 <wait>，"
           "不要说任何话；已经等了比较久的话，可以说一句很短、自然、和之前不重复的话让用户稍等。"),
    "en": ("[Backend update | {t:.0f} s since the delegation | {gap} since you last spoke]\n{body}\nYou last said: \"{last}\"\n\n"
           "How to respond:\n- If there is a new result the user wants (an answer, what was found, done, or nothing found): tell the user in your "
           "own natural spoken words, based only on the text above; never invent anything; don't mention the backend or tools.\n- If it is only "
           "progress, background, or repeats what you already said: if you spoke less than 6 s ago, output only <wait> and say nothing; if the "
           "user has been waiting a while, you may say one very short, natural line (not a repeat) asking them to hold on."),
}
UPDATE_PARTS = {"zh": ("要转达给用户的内容：", "后端的进度 / 思考（不用照读）：", "（还没有新内容）"),
                "en": ("Content to pass on to the user: ", "Backend progress / thinking (don't read it out): ", "(nothing new yet)")}
RESULT_MIN_CHARS = int(os.environ.get("SOPHON_LIVE_RESULT_MIN_CHARS", "15"))    # shorter commentary = a progress line
PROGRESS_AFTER_S = float(os.environ.get("SOPHON_LIVE_PROGRESS_AFTER", "5.0"))  # speak held progress once if no result by then
RESULT_MERGE_S = float(os.environ.get("SOPHON_LIVE_RESULT_MERGE", "1.6"))     # multi-stage backends: merge result pieces arriving this close together
SHORT_WAIT_S = float(os.environ.get("SOPHON_LIVE_SHORT_WAIT", "1.8"))          # how long a short commentary waits to see if the backend keeps working
INPUT_CONTENT_TYPES = {"developer": {"input_text"}, "user": {"input_text"}, "assistant": {"output_text", "text"}}


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32)
    try:
        from scipy.signal import resample_poly
        g = np.gcd(sr_in, sr_out)
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)
    except Exception:  # noqa: BLE001
        n = int(round(x.size * sr_out / sr_in))
        return np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x).astype(np.float32)


def split_pieces(text: str) -> list[str]:
    """1-2 character pieces, punctuation on its own (GPT-Live transcript granularity)."""
    out, i = [], 0
    while i < len(text):
        if text[i] in "。，,、！!？?;；.":
            out.append(text[i]); i += 1; continue
        if CJK.match(text[i]):
            out.append(text[i:i + 2]); i += 2; continue
        j = i
        while j < len(text) and not CJK.match(text[j]) and text[j] not in "。，,、！!？?;；":
            j += 1
            if text[j - 1] == " " and j - i > 1:
                break
        out.append(text[i:j]); i = j
    return [p for p in out if p]


class ASR:
    """Input transcript (GPT-Live sends session.input_transcript.delta; duplex clients build the delegated task from it)."""

    def __init__(self, path: str, device: str = "cuda"):
        import torch
        import transformers
        self.torch = torch
        self.proc = transformers.AutoProcessor.from_pretrained(path)
        self.model = transformers.AutoModelForMultimodalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
        self.device = device

    def __call__(self, a16k: np.ndarray) -> str:
        with self.torch.inference_mode():
            inp = self.proc.apply_transcription_request(audio=a16k).to(self.device, self.torch.bfloat16)
            o = self.model.generate(**inp, max_new_tokens=200)
            return self.proc.decode(o[:, inp["input_ids"].shape[1]:], return_format="transcription_only")[0].strip()


BARGE_MS = int(os.environ.get("SOPHON_LIVE_BARGE_MS", "200"))          # user voiced this long while Sophon is talking -> stop playback
BARGE_MIN_RMS = float(os.environ.get("SOPHON_LIVE_BARGE_RMS", "0.015"))

class LiveSession:
    def __init__(self, ws, args, asr, asr_lock):
        self.ws, self.args, self.asr, self.asr_lock = ws, args, asr, asr_lock
        self.id = "sess_" + uuid.uuid4().hex[:24]
        self.rate = 16000
        self.lang = "zh"
        self.instructions = ""
        self.started = False
        self.closed = False
        self.engine: VllmOmniAdapter | None = None
        self.timeline_ms = 0
        self.play: deque = deque()                     # (np.int16 samples at client rate, rid)
        self.play_buf = np.zeros(0, np.int16)
        self.resp: dict = {}                           # rid -> {"text": str, "released": int, "audio_s": float, "played_s": float, "start_ms": int|None, "done": bool, "asr_wait": bool}
        self.cur_rid: str | None = None
        self.user_audio: list = []                     # f32 16 kHz chunks since the last cut
        self.user_t0: int | None = None
        self.asr_task: asyncio.Task | None = None
        self.deleg: dict = {}                          # delegation id -> {"call_id", "chunks": [(kind, text)], "timer"}
        self.last_client_audio = 0.0
        self.tasks: list = []
        self.bi_rem = np.zeros(0, np.float32)          # barge-in VAD: 20 ms frame remainder
        self.bi_noise = 0.003; self.bi_voiced = 0; self.bi_gap = 0
        self.model = "gpt-live-1"
        self.last_text = ""
        self.last_voice_t = 0.0                        # when the assistant's voice was last heard (model-mode updates)
        self.last_said = ""                            # the assistant's latest utterance

    # ------------------------------------------------------------- out
    async def send(self, ev: dict) -> None:
        ev.setdefault("event_id", "ev_" + uuid.uuid4().hex[:12])
        try:
            await self.ws.send(json.dumps(ev, ensure_ascii=False))
        except websockets.ConnectionClosed:
            self.closed = True

    async def error(self, code: str, message: str, **kw) -> None:
        await self.send({"type": "error", "error": {"code": code, "message": message, **kw}})

    # ------------------------------------------------------------- session start
    async def on_start(self, ev: dict) -> bool:
        sess = ev.get("session") or {}
        fmt = (sess.get("audio") or {}).get("format") or {}
        rate = int(fmt.get("rate") or 16000)
        if fmt.get("type", "audio/pcm") != "audio/pcm" or rate not in (16000, 24000):
            await self.error("invalid_value", f"Invalid value: audio.format {fmt}. Supported: audio/pcm at 16000 or 24000 Hz."); return False
        items = sess.get("input") or []
        hist = []
        for it in items:
            role = (it or {}).get("role", "")
            allowed = INPUT_CONTENT_TYPES.get(role)
            if allowed is None:
                await self.error("invalid_value", f"Invalid value: {role!r}. Unsupported role."); return False
            texts = []
            for part in (it or {}).get("content") or []:
                if (part or {}).get("type") not in allowed:
                    await self.error("invalid_value", f"Invalid value: {(part or {}).get('type')!r}."); return False
                texts.append(str(part.get("text", "")))
            if role in ("user", "assistant"):
                hist.append({"role": role, "text": "".join(texts)})
            else:                                            # developer messages join the instructions
                sess["instructions"] = (sess.get("instructions") or "") + "\n" + "".join(texts)
        self.rate = rate
        self.model = sess.get("model") or "gpt-live-1"
        self.instructions = sess.get("instructions") or ""
        probe = self.instructions + "".join(h["text"] for h in hist)
        self.lang = "zh" if (not probe or len(CJK.findall(probe)) >= max(4, 0.05 * len(probe))) else "en"
        if os.environ.get("SOPHON_LIVE_LANG"):
            self.lang = os.environ["SOPHON_LIVE_LANG"]
        voice = (sess.get("audio") or {}).get("output", {}).get("voice") or "marin"
        delegation = sess.get("delegation") or {}
        tools = [DELEGATE[self.lang]] if delegation.get("type") == "client" else []
        self.engine = VllmOmniAdapter(self.args.engine, self.args.bridge)
        try:
            await self.engine.start(instructions=self._sys(), voice=os.environ.get(f"SOPHON_LIVE_VOICE_{voice.upper()}", DEFAULT_VOICE),
                                    lang=self.lang, tools=tools, history=hist)
        except Exception as e:  # noqa: BLE001
            log.exception("engine start failed")
            await self.error("server_error", f"engine unavailable: {e}"); return False
        self.started = True
        await self.send({"type": "session.started", "session": {
            "id": self.id, "object": "live.session", "model": self.model, "status": "active", "expires_at": int(time.time()) + 7200,
            "audio": sess.get("audio") or {"format": {"type": "audio/pcm", "rate": rate}}, "delegation": delegation, "store": bool(sess.get("store", False))}})
        log.info("session %s started lang=%s rate=%d instructions=%d chars history=%d delegation=%s", self.id, self.lang, rate,
                 len(self.instructions), len(hist), delegation.get("type"))
        self.tasks = [asyncio.create_task(self._pump()), asyncio.create_task(self._engine_events()), asyncio.create_task(self._idle_feed())]
        return True

    def _sys(self) -> str:
        return (self.instructions or "") + SPOKEN_RULE[self.lang]

    # ------------------------------------------------------------- client events
    async def on_event(self, ev: dict) -> None:
        t = ev.get("type", "")
        if not self.started:
            if t == "session.start":
                if not await self.on_start(ev):
                    await self.close("start_failed")
            else:
                await self.error("invalid_request_error", "first event must be session.start")
            return
        if t == "session.input_audio.append":
            pcm = np.frombuffer(base64.b64decode(ev.get("audio") or ""), "<i2").astype(np.float32) / 32768.0
            a = resample(pcm, self.rate, 16000)
            self.last_client_audio = time.monotonic()
            self._user(a)
            self._barge_check(a)
            await self.engine.send_audio(a)
        elif t in ("session.commentary.append", "session.thinking.append"):
            did = ev.get("delegation_id"); content = str(ev.get("content") or "")
            kind = "commentary" if t == "session.commentary.append" else "thinking"
            if kind == "commentary":
                await self.send({"type": "session.commentary.appended", "start_ms": self.timeline_ms, "end_ms": self.timeline_ms, "client_event_id": ev.get("event_id")})
            d = self.deleg.get(did) if did else None
            if d is None:
                log.warning("%s with unknown delegation_id=%s is accepted but has no effect", t, did); return
            d["chunks"].append((kind, content))
            if d.get("timer"):
                d["timer"].cancel()
            d["timer"] = asyncio.get_running_loop().call_later(DELEG_QUIET_S, lambda: asyncio.ensure_future(self._flush_deleg(did)))
        elif t == "session.instructions.append":
            await self.send({"type": "session.commentary.appended", "start_ms": self.timeline_ms, "end_ms": self.timeline_ms, "client_event_id": ev.get("event_id")})
            self.instructions = (self.instructions + "\n" + str(ev.get("content") or "")).strip()
            await self.engine.set_instructions(self._sys())
        elif t == "session.update":
            await self.error("unknown_parameter", "Unknown parameter: session.update is not supported after session.start.")
        elif t == "session.close":
            await self.close("close_requested")
        else:
            await self.error("invalid_request_error", f"Unknown event type: {t!r}")

    async def _flush_deleg(self, did: str) -> None:
        if DELEG_MODE == "model":
            return await self._flush_model(did)
        return await self._flush_rules(did)

    async def _flush_model(self, did: str) -> None:
        """Model mode: every backend update (merged over DELEG_QUIET_S) goes to the model with timing and what it said last;
        the model answers, says a short natural "hold on", or outputs <wait> (silent)."""
        d = self.deleg.get(did)
        if d is None or not d["chunks"]:
            return
        chunks, d["chunks"] = d["chunks"], []
        sent = d.setdefault("sent", set())
        chunks = [(k, c) for k, c in chunks if c.strip() and c.strip() not in sent]
        sent.update(c.strip() for _, c in chunks)
        if not chunks:
            return
        await self._post_update(did, chunks)

    async def _post_update(self, did: str, chunks: list) -> None:
        d = self.deleg[did]
        if d.get("idle_timer") is not None:
            d["idle_timer"].cancel(); d["idle_timer"] = None
        pc, pt, pn = UPDATE_PARTS[self.lang]
        com = "".join(c for k, c in chunks if k == "commentary").strip()
        thk = "\n".join(c for k, c in chunks if k == "thinking").strip()
        body = "\n".join(x for x in ((pc + com) if com else "", (pt + thk) if thk else "") if x) or pn
        gap = time.monotonic() - self.last_voice_t if self.last_voice_t else None
        gap_s = ("还没开过口" if self.lang == "zh" else "never") if gap is None else (f"{gap:.0f} 秒" if self.lang == "zh" else f"{gap:.0f} s")
        out = UPDATE_FMT[self.lang].format(t=time.monotonic() - d["t0"], gap=gap_s, body=body, last=(self.last_said or "")[-80:])
        d["posted_any"] = True
        ok = await self.engine.tool_result(d["call_id"], out, speak=True)
        log.info("delegation %s -> update posted ok=%s (%d commentary chars, %d thinking chars)", did, ok, len(com), len(thk))

    async def _idle_update(self, did: str) -> None:
        d = self.deleg.get(did)
        if d is None or d.get("posted_any"):
            return
        await self._post_update(did, [])

    async def _flush_rules(self, did: str) -> None:
        """Waiting-phase logic (rules mode). Backends stream a delegation in stages, e.g. a short progress line ("我查一下"), then background
        thinking ("准备查一下…") while a tool runs, then the real result. Only the real result triggers a spoken follow-up:
          - thinking never triggers speech by itself; it marks the backend as "still working" and joins the result as context;
          - a short commentary (< RESULT_MIN_CHARS) is held for SHORT_WAIT_S: if thinking follows, it was a progress line
            (dropped, our own lead-in already covered it); if nothing follows, it was a short result and is spoken;
          - commentary that is long, or that comes after thinking, is the result;
          - if nothing real has arrived PROGRESS_AFTER_S after the delegation, one short "still working" line is spoken."""
        d = self.deleg.get(did)
        if d is None or not d["chunks"]:
            return
        chunks, d["chunks"] = d["chunks"], []
        sent = d.setdefault("sent", set())                       # backends often repeat the same line: use it once
        chunks = [(k, c) for k, c in chunks if c.strip() and c.strip() not in sent]
        sent.update(c.strip() for _, c in chunks)
        if not chunks:
            return
        held = d.setdefault("held", [])
        was_working = d.get("working", False)
        if any(k == "thinking" for k, _ in chunks):
            d["working"] = True
            if d.get("progress_timer") is None and not d.get("progress_spoken"):
                wait = max(0.0, d["t0"] + PROGRESS_AFTER_S - time.monotonic())
                d["progress_timer"] = asyncio.get_running_loop().call_later(wait, lambda: asyncio.ensure_future(self._progress(did)))
        com = "".join(c for k, c in chunks if k == "commentary").strip()
        if com and (len(com) >= RESULT_MIN_CHARS or was_working):
            if not d.get("working"):
                await self._result(did, chunks)
            else:                                                # multi-stage backend: it may still add to / correct this
                d.setdefault("res", []).extend(chunks)           # ("让我重新查一下…" then the real answer) -> merge into one reply
                if d.get("merge_timer") is not None:
                    d["merge_timer"].cancel()
                d["merge_timer"] = asyncio.get_running_loop().call_later(RESULT_MERGE_S, lambda: asyncio.ensure_future(self._merged(did)))
        elif com:                                                # short and the backend hasn't shown it is still working
            held.extend(chunks)
            if d.get("short_timer") is not None:
                d["short_timer"].cancel()
            d["short_timer"] = asyncio.get_running_loop().call_later(SHORT_WAIT_S, lambda: asyncio.ensure_future(self._short_done(did)))
            log.info("delegation %s: short commentary held (%d chars)", did, len(com))
        else:
            held.extend(chunks)
            log.info("delegation %s: thinking held, backend still working (%d chars)", did, sum(len(c) for _, c in chunks))

    async def _merged(self, did: str) -> None:
        d = self.deleg.get(did)
        if d is None:
            return
        d["merge_timer"] = None
        res, d["res"] = d.get("res", []), []
        if res:
            await self._result(did, res)

    async def _short_done(self, did: str) -> None:
        d = self.deleg.get(did)
        if d is None:
            return
        d["short_timer"] = None
        if not d.get("working") and any(k == "commentary" for k, _ in d.get("held", [])):
            await self._result(did, [])                         # nothing followed: it was the (short) result

    async def _result(self, did: str, chunks: list) -> None:
        d = self.deleg[did]
        for key in ("progress_timer", "short_timer"):
            if d.get(key) is not None:
                d[key].cancel(); d[key] = None
        held = d.get("held", [])
        # held thinking joins as context; a held short commentary is a progress line if the backend kept working, else part of the result
        keep = [x for x in held if x[0] == "thinking" or not d.get("working")]
        d["held"] = []
        await self._post(did, keep + chunks, progress=False)

    async def _progress(self, did: str) -> None:
        d = self.deleg.get(did)
        if d is None or d.get("progress_spoken") or not d.get("held") or d.get("res"):
            return                                               # a result is already on its way: no progress line
        d["progress_timer"] = None; d["progress_spoken"] = True
        await self._post(did, d["held"], progress=True)
        d["held"] = [x for x in d["held"] if x[0] == "thinking"]

    async def _post(self, did: str, chunks: list, progress: bool) -> None:
        d = self.deleg[did]
        com = "".join(c for k, c in chunks if k == "commentary").strip()
        thk = "\n".join(c for k, c in chunks if k == "thinking").strip()
        fmt_c, fmt_t = RESULT_FMT[self.lang]
        if progress:
            out = PROGRESS_FMT[self.lang].format(c="\n".join(x for x in (com, thk) if x))
        else:
            out = "\n\n".join(x for x in (fmt_c.format(c=com) if com else "", fmt_t.format(c=thk) if thk else "") if x)
        ok = await self.engine.tool_result(d["call_id"], out, speak=True)
        log.info("delegation %s -> %s posted ok=%s (%d commentary chars, %d thinking chars)", did, "progress" if progress else "result", ok, len(com), len(thk))

    # ------------------------------------------------------------- input transcript
    def _user(self, a: np.ndarray) -> None:
        if self.user_t0 is None:
            self.user_t0 = self.timeline_ms
        self.user_audio.append(a)
        tot = sum(x.size for x in self.user_audio)
        while tot > 16000 * 30 and len(self.user_audio) > 1:      # keep at most the last 30 s
            tot -= self.user_audio.pop(0).size

    def _playing(self) -> bool:
        return bool(self.play) or self.play_buf.size > 0

    def _barge_check(self, a) -> None:
        """User talks over Sophon -> stop playback now. The engine only cuts its history on barge-in
        (the testbench client stops its own player); GPT-Live clients expect the server to go silent."""
        x = np.concatenate([self.bi_rem, a]) if self.bi_rem.size else a
        n = x.size // 320; self.bi_rem = x[n * 320:].copy()
        for i in range(n):
            f = x[i * 320:(i + 1) * 320]; r = float(np.sqrt(np.mean(f * f)))
            thr = max(BARGE_MIN_RMS, self.bi_noise * 3.5)
            if r > thr:
                self.bi_voiced += 20; self.bi_gap = 0
            else:
                self.bi_noise = 0.95 * self.bi_noise + 0.05 * r if r < self.bi_noise * 3 else self.bi_noise
                self.bi_gap += 20
                if self.bi_gap >= 120: self.bi_voiced = 0
            if self.bi_voiced >= BARGE_MS and self._playing():
                self.bi_voiced = 0
                self._drop_playback("barge-in (gateway VAD)")

    def _drop_playback(self, why: str) -> None:
        self.play.clear(); self.play_buf = np.zeros(0, np.int16)
        for r in self.resp.values():
            r["text"] = r["text"][: r["released"]]; r["done"] = True; r["dropped"] = True
        log.info("session %s %s: playback dropped", self.id, why)

    def _cut_user(self) -> asyncio.Task | None:
        """Close the user's segment (a reply or a delegation just started) and transcribe it."""
        if self.asr_task is not None and not self.asr_task.done():
            return self.asr_task
        if not self.user_audio:
            return None
        a = np.concatenate(self.user_audio); t0 = self.user_t0 or self.timeline_ms
        self.user_audio, self.user_t0 = [], None
        fr = a[: a.size // 320 * 320].reshape(-1, 320)
        voiced = np.flatnonzero(np.sqrt((fr ** 2).mean(1)) > 0.01) if fr.size else np.zeros(0, int)
        if voiced.size < 12 or self.asr is None:                  # < 0.24 s of speech: nothing to transcribe
            return None
        a = a[max(0, voiced[0] * 320 - 3200): min(a.size, (voiced[-1] + 1) * 320 + 3200)]
        start = t0 + int(voiced[0] * 20); end = t0 + int((voiced[-1] + 1) * 20)
        self.asr_task = asyncio.create_task(self._asr(a, start, end))
        return self.asr_task

    async def _asr(self, a, start, end) -> None:
        try:
            async with self.asr_lock:
                text = await asyncio.to_thread(self.asr, a)
        except Exception:  # noqa: BLE001
            log.exception("asr failed"); return
        pieces = split_pieces(text)
        if not pieces:
            return
        step = max(1, (end - start) // len(pieces))
        for i, p in enumerate(pieces):
            await self.send({"type": "session.input_transcript.delta", "delta": p, "start_ms": start + i * step, "end_ms": start + (i + 1) * step})
        log.info("session %s user: %s", self.id, text)

    # ------------------------------------------------------------- engine events
    async def _engine_events(self) -> None:
        q = self.engine.q
        while not self.closed:
            e = await q.get(); t = e["t"]
            if t in ("audio", "text"):
                rid = e.get("rid") or self.cur_rid or ("r_" + uuid.uuid4().hex[:10])
                if rid != self.cur_rid or rid not in self.resp:
                    self.cur_rid = rid
                    task = self._cut_user()
                    prev = self.last_text.rstrip()
                    lead = ("。" if self.lang == "zh" else ". ") if prev and prev[-1] not in "。！？.!?，,～…" else ""
                    self.resp[rid] = {"text": lead, "released": 0, "audio_s": 0.0, "played_s": 0.0, "start_ms": None, "done": False, "asr": task}
                r = self.resp[rid]
                if r.get("dropped"):
                    continue
                if t == "audio":
                    x = resample(e["pcm"], e["sr"], self.rate)
                    r["audio_s"] += x.size / self.rate
                    self.play.append((np.clip(x * 32768, -32768, 32767).astype(np.int16), rid))
                else:
                    d = clean_text(e["delta"], self.lang); r["text"] += d
                    if d.strip(): self.last_text = d
            elif t == "done":
                rid = e.get("rid") or self.cur_rid
                if rid in self.resp:
                    self.resp[rid]["done"] = True
                if rid == self.cur_rid:
                    self.cur_rid = None
            elif t == "cancel":
                self._drop_playback("barge-in (engine)")
            elif t == "tool_call":
                if e.get("name") != "delegate":
                    log.warning("ignoring call to unknown tool %s", e.get("name")); continue
                asyncio.create_task(self._delegate(e))           # never block the audio path while the transcript finishes
            elif t == "error":
                log.warning("session %s engine: %s", self.id, e.get("message"))
                if "closed" in (e.get("message") or ""):
                    await self.error("server_error", "engine connection lost"); await self.close("server_error"); return

    async def _delegate(self, e: dict) -> None:
        task = self._cut_user()
        if task is not None:                                     # the client builds the task from the user's transcript: send it first
            try:
                await asyncio.wait_for(asyncio.shield(task), 3)
            except Exception:  # noqa: BLE001
                pass
        did = "item_" + uuid.uuid4().hex[:16]
        what = (e.get("arguments") or {}).get("task")
        self.deleg[did] = {"call_id": e["call_id"], "chunks": [], "timer": None, "task": what, "t0": time.monotonic()}
        if DELEG_MODE == "model":
            self.deleg[did]["idle_timer"] = asyncio.get_running_loop().call_later(IDLE_UPDATE_S, lambda: asyncio.ensure_future(self._idle_update(did)))
        await self.send({"type": "session.delegation.created", "offset_ms": self.timeline_ms,
                         "delegation": {"id": did, "type": "delegation", "target": "client", "task": what}})
        log.info("session %s delegation %s task=%r", self.id, did, what)

    # ------------------------------------------------------------- output pump (continuous, real time)
    async def _pump(self) -> None:
        n = self.rate * FRAME_MS // 1000
        t_next = time.monotonic()
        idle = True
        while not self.closed:
            t_next += FRAME_MS / 1000
            while time.monotonic() < t_next:                   # when the first audio of a reply arrives while idle, send it at once
                if idle and self.play:                          # instead of waiting for the next 100 ms tick
                    t_next = time.monotonic(); break
                await asyncio.sleep(min(0.01, max(0.0, t_next - time.monotonic())))
            while self.play_buf.size < n and self.play:
                x, rid = self.play.popleft()
                r = self.resp.get(rid)
                if r is not None and r["start_ms"] is None:
                    r["start_ms"] = self.timeline_ms
                self.play_buf = np.concatenate([self.play_buf, x]) if self.play_buf.size else x
                self._play_rid = rid
            frame = self.play_buf[:n]; self.play_buf = self.play_buf[n:]
            voiced = frame.size
            idle = voiced == 0 and not self.play
            if frame.size < n:
                frame = np.concatenate([frame, np.zeros(n - frame.size, np.int16)])
            await self.send({"type": "session.output_audio.delta", "delta": base64.b64encode(frame.tobytes()).decode()})
            if voiced:
                self.last_voice_t = time.monotonic()
                r = self.resp.get(getattr(self, "_play_rid", None))
                if r is not None:
                    r["played_s"] += voiced / self.rate
                    self.last_said = r["text"][: r["released"]] or self.last_said
            await self._release_text()
            self.timeline_ms += FRAME_MS

    async def _release_text(self) -> None:
        for rid, r in list(self.resp.items()):
            if r["start_ms"] is None or r["released"] >= len(r["text"]):
                if r["done"] and r["start_ms"] is not None and r["released"] >= len(r["text"]) and not self.play and self.play_buf.size == 0:
                    self.resp.pop(rid, None)
                continue
            if r.get("asr") is not None and not r["asr"].done():
                continue                                          # the user's words go out before the reply's
            total = len(r["text"])
            finished_audio = r["done"] and not any(p[1] == rid for p in self.play) and self.play_buf.size == 0
            if finished_audio:
                upto = total
            else:
                rate = (total / r["audio_s"]) if (r["done"] and r["audio_s"] > 0) else (4.5 if CJK.search(r["text"]) else 14.0)
                upto = min(total, int((r["played_s"] + TRANSCRIPT_LEAD_MS / 1000) * rate))
            if upto > r["released"]:
                chunk = r["text"][r["released"]:upto]; r["released"] = upto
                for p in split_pieces(chunk):
                    await self.send({"type": "session.output_transcript.delta", "delta": p, "start_ms": self.timeline_ms, "end_ms": self.timeline_ms + FRAME_MS})

    async def _idle_feed(self) -> None:
        """The engine needs a steady input stream to notice silence (turn end, tool follow-ups): pad with silence when the client goes quiet."""
        z = np.zeros(1600, np.float32)
        while not self.closed:
            await asyncio.sleep(0.1)
            if time.monotonic() - self.last_client_audio > 0.3:
                self._user(z)
                await self.engine.send_audio(z)

    async def close(self, reason: str) -> None:
        if self.closed:
            return
        self.closed = True
        for t in self.tasks:
            t.cancel()
        try:
            await self.ws.send(json.dumps({"type": "session.closed", "reason": reason, "event_id": "ev_" + uuid.uuid4().hex[:12],
                                           "session": {"id": self.id, "status": "closed", "model": self.model, "expires_at": 0}}))
            await self.ws.close()
        except Exception:  # noqa: BLE001
            pass
        if self.engine is not None:
            await self.engine.close()
        log.info("session %s closed (%s)", self.id, reason)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--engine", default="ws://127.0.0.1:8000/v1/realtime")
    ap.add_argument("--bridge", default="http://127.0.0.1:8700")
    ap.add_argument("--asr", default=os.environ.get("SOPHON_LIVE_ASR", "Qwen/Qwen3-ASR-0.6B-hf"))
    ap.add_argument("--path", default="/v1/live/sessions")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asr = ASR(args.asr) if args.asr else None
    if asr is not None:
        asr(np.zeros(16000, np.float32))                         # warm up
    asr_lock = asyncio.Lock()

    page_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
    kf = os.environ.get("SOPHON_LIVE_REALTIME_KEYS_FILE", "")
    realtime_keys = {l.split("#")[0].strip() for l in open(kf)} - {""} if kf and os.path.exists(kf) else set()
    allow_origins = {o.strip().rstrip("/") for o in os.environ.get("SOPHON_LIVE_ALLOW_ORIGINS", "").split(",") if o.strip()}

    def process_request(conn, req):
        path = req.path.split("?")[0]
        if path in ("/", "/index.html") and req.headers.get("Upgrade", "").lower() != "websocket":
            try:                                                 # the test page (served from the same origin as the WebSocket)
                body = open(page_path, "rb").read()
            except OSError:
                return conn.respond(404, "Not Found\n")
            return Response(200, "OK", Headers([("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body))),
                                                ("Cache-Control", "no-cache")]), body)
        if path == "/health":
            return conn.respond(200, "ok\n")
        if path not in (args.path, "/v1/realtime"):
            return conn.respond(404, "Not Found\n")
        if path == "/v1/realtime" and realtime_keys:              # keyed access for external evaluators (e.g. Artificial Analysis)
            auth = req.headers.get("Authorization", "")
            proto = req.headers.get("Sec-WebSocket-Protocol", "")
            key = auth[7:].strip() if auth.startswith("Bearer ") else next(
                (p.strip()[len("openai-insecure-api-key."):] for p in proto.split(",") if p.strip().startswith("openai-insecure-api-key.")), "")
            if key not in realtime_keys:
                return conn.respond(401, "Unauthorized: invalid API key\n")
        origin = req.headers.get("Origin")
        if origin and origin.rstrip("/") not in allow_origins:   # like GPT-Live: no browser connections, except our own test page
            return conn.respond(403, "Forbidden: Origin header not allowed\n")
        if API_KEYS:
            auth = req.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:].strip() not in API_KEYS:
                return conn.respond(401, "Unauthorized\n")
        return None

    async def handler(ws):
        from urllib.parse import parse_qs, urlparse
        u = urlparse(ws.request.path)
        if u.path == "/v1/realtime":                             # OpenAI Realtime face
            s = RealtimeSession(ws, args, asr, asr_lock, resample, (parse_qs(u.query).get("model") or ["sophon-realtime"])[0])
            await s.open()
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    try:
                        ev = json.loads(raw)
                    except json.JSONDecodeError:
                        await s.error("Event is not valid JSON.", "invalid_json"); continue
                    await s.on_event(ev)
                    if s.closed:
                        break
            except websockets.ConnectionClosed:
                pass
            finally:
                await s.close()
            return
        s = LiveSession(ws, args, asr, asr_lock)
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    await s.error("invalid_json", "Event is not valid JSON."); continue
                await s.on_event(ev)
                if s.closed:
                    break
        except websockets.ConnectionClosed:
            pass
        finally:
            await s.close("client_disconnected")

    async def run():
        async with websockets.serve(handler, args.host, args.port, max_size=None, compression=None, process_request=process_request):
            log.info("Sophon Live gateway on ws://%s:%d%s -> %s", args.host, args.port, args.path, args.engine)
            await asyncio.Future()
    asyncio.run(run())


if __name__ == "__main__":
    main()
