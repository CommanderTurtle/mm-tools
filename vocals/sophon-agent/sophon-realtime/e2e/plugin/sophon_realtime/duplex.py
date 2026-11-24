"""Sophon-Realtime full-duplex plugin (vLLM-Omni /v1/realtime): the framework handles sessions, audio intake, VAD and barge-in; turn-taking is decided by DuplexJev (the stage-0 model itself).

Flow (per session):
  - The client streams microphone audio continuously (session.update: turn_detection=server_vad, extra_body.auto_response=true).
  - After the user pauses, at checkpoints (default 0.2/0.35/0.5/0.8/1.3/2.0 s) a "turn question" request is sent to stage 0 (same wording as livekit-plugins-duplexjev:
    user finished / not finished / backchannel / thinking); P(finished) is read and a reply is triggered if it exceeds that checkpoint's threshold; fallback reply at 2 s.
  - Once the turn is judged finished, the next audio append sends a "reply" request: system + multi-turn history (user speech; assistant speech as spoken, or text for interrupted turns) + this turn's user speech.
  - If the user starts talking while the assistant speaks, the framework VAD barges in (server_vad's barge_in_on_speech).
Turn-question requests only emit one letter at stage 0 and never reach the talker (proc.py sees sophon_kind=check and forwards nothing; the talker finishes immediately with no text)."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import random
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pybase64 as base64
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind

from vllm_omni.engine.duplex.config import DuplexCapabilities
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan, DuplexOutputAction, DuplexOutputDecision
from vllm.sampling_params import StructuredOutputsParams
from vllm_omni.engine.duplex.plugin import (DefaultDuplexModelSessionState, DuplexDataPlane, DuplexModelPlugin,
                                            DuplexRuntimeConfigError, PcmAppendBuffer, PcmAppendReservation)

logger = init_logger("vllm_omni.sophon_realtime.duplex")
SR = 16000
CHECKS = [float(x) for x in os.environ.get("SOPHON_TURN_CHECKS", "0.2,0.35,0.5,0.8,1.3,2.0").split(",")]
THRS = [float(x) for x in os.environ.get("SOPHON_TURN_THRESHOLDS", "0.95,0.85,0.7,0.5,0.35,0.25").split(",")]
SPEECH_PROB = float(os.environ.get("SOPHON_SPEECH_PROB", "0.5"))
SPEC_AT = float(os.environ.get("SOPHON_SPEC_AT", "0.05"))       # after this much silence, speculatively start writing a reply (held silent until DuplexJev judges the turn finished); <=0 disables
RES_TIMEOUT = float(os.environ.get("SOPHON_RES_TIMEOUT", "15"))  # an append not confirmed by the framework within this time is rolled back (and a reply resent)
SPEC_MIN_SPEECH = float(os.environ.get("SOPHON_SPEC_MIN_SPEECH", "0.3"))   # only draft speculatively once the user has spoken this long (s)
MERGE_WINDOW = float(os.environ.get("SOPHON_MERGE_WINDOW", "2.0"))
FALLBACK_S = float(os.environ.get("SOPHON_TURN_FALLBACK", "2.0"))
RETRY_MAX = int(os.environ.get("SOPHON_RETRY_MAX", "3"))     # resends of a refused reply append
IGNORE_SHORT = os.environ.get("SOPHON_IGNORE_SHORT", "0") == "1"     # on fallback, ignore short backchannels (default off: leave it to the model; fallback replies as usual)
from .env import port as _port, vision as _vision
from . import tools as _tools
from .proc import strip_tags as _strip_tags, tag_clean as _tag_clean


def _display(s, final):
    return _strip_tags(_tag_clean(s, final))


def _fence_floor_patch() -> None:
    """vLLM-Omni builds an input append's fence from the *active response's* turn. When the user talks over a
    reply that never formally ends (overlapping turns, e.g. phone-style callers), the session's accepted fence has
    already moved to a newer turn, so every later append is rejected ("duplex fence mismatch") and the session
    goes deaf for good. Floor the append fence at the accepted turn of the same epoch. SOPHON_FENCE_FLOOR=0 disables."""
    if os.environ.get("SOPHON_FENCE_FLOOR", "1") == "0":
        return
    try:
        from vllm_omni.engine.duplex.session import helpers as _h
        from vllm_omni.engine.duplex.contracts import DuplexFence as _F
    except Exception:  # noqa: BLE001
        return
    if getattr(_h.append_fence, "_sophon_floor", False):
        return
    _orig = _h.append_fence

    def append_fence(session, payload, *a, **kw):
        f = _orig(session, payload, *a, **kw)
        acc = getattr(session, "accepted_fence", None)
        if acc is not None and acc.session_id == f.session_id and acc.epoch == f.epoch and acc.turn_id > f.turn_id:
            logger.info("sophon fence floor: append turn %d -> %d (accepted)", f.turn_id, acc.turn_id)
            return _F(f.session_id, epoch=f.epoch, turn_id=acc.turn_id)
        return f
    append_fence._sophon_floor = True
    _h.append_fence = append_fence
from . import bridge as _bridge
VISION = _vision()                                                   # thinker has a vision tower (DuplexJevVision): accept camera frames
FRAME_TTL = float(os.environ.get("SOPHON_FRAME_TTL", "3.0"))        # camera frames older than this are no longer attached (seconds)
IMG_MAX = int(os.environ.get("SOPHON_IMG_MAX", "448"))               # frames are downscaled to fit within this side length before going to the model (~200 vision tokens)
SYS_CAM = {"zh": "用户开着摄像头，每轮会附上你此刻看到的画面。需要时自然地提到你看到的东西，不要每次都描述画面。",
           "en": "The user's camera is on; each turn includes what you currently see. Mention what you see naturally when relevant; don't describe the scene every time."}
REP_GUARD = os.environ.get("SOPHON_REP_GUARD", "0") == "1"           # repetition cut-off safeguard (default off)
MIN_RELEASE = float(os.environ.get("SOPHON_MIN_RELEASE", "0"))           # minimum pause before speaking once the model says "finished" ...
VAD_MINSTAT = os.environ.get("SOPHON_VAD_MINSTAT", "0") == "1"
# Noise-robust turn end: with steady background noise the energy VAD may never see "silence", so the turn model is never asked.
# While a user segment keeps going, ask DuplexJev every PERIODIC_S seconds; P(finished) >= PERIODIC_THR replies without waiting for silence.
PERIODIC_S = float(os.environ.get("SOPHON_PERIODIC_CHECK", "0"))            # 0 = off
PERIODIC_MIN = float(os.environ.get("SOPHON_PERIODIC_MIN", "1.0"))          # only after this much "speech"
PERIODIC_THR = float(os.environ.get("SOPHON_PERIODIC_THR", "0.9"))
PERIODIC_DYN = float(os.environ.get("SOPHON_PERIODIC_DYN", "4.0"))          # segment needs peak/median frame RMS >= this (~12 dB): real speech, not steady noise    # robust noise floor for noisy callers (street / phone); off by default
BC_RESUME = os.environ.get("SOPHON_BC_RESUME", "1") == "1"         # user only back-channels ("mm-hmm") over a reply that got cut off: continue that reply instead of answering the "mm-hmm"
BC_THR = float(os.environ.get("SOPHON_BC_THR", "0.5"))               # ... when the model's P(back-channel) is at least this (and above P(finished))
BC_WINDOW = float(os.environ.get("SOPHON_BC_WINDOW", "4.0"))         # ... and the cut happened within this many seconds
LEAD_TRIM = float(os.environ.get("SOPHON_LEAD_TRIM", "1.0"))           # drop up to this much leading silence from each spoken reply (s); 0 disables
MIN_RELEASE_AFTER = float(os.environ.get("SOPHON_MIN_RELEASE_AFTER", "1.0"))  # ... applied only after this much user speech (long turns breathe mid-sentence; short ones answer at once)
HIST_TURNS = int(os.environ.get("SOPHON_HIST_TURNS", "8"))
HIST_SEC = float(os.environ.get("SOPHON_HIST_SEC", "120"))
LET = "ABCDEFGHIJ"
TURN_Q = {"zh": ("用户现在处于什么话轮状态？", ["话说完了，可以接话", "句子没说完，还在继续", "简短附和，不是要接话", "还在组织语言，或要求先等一下"]),
          "en": ("Has the user finished speaking?", ["finished, the assistant can reply", "not finished, still talking",
                                                      "just a backchannel, not taking the turn", "hesitating or asking to wait"])}
TPL = {"en": ("Question: {q}", "Options:", "Answer with only the letter of the correct option.", "The user said: <|audio|>", 'The assistant had just said: "{c}"'),
       "zh": ("问题：{q}", "选项：", "请只回答正确选项的字母。", "用户说：<|audio|>", "助手刚才说的是：“{c}”")}
# Tag build: the system prompt the tag-writing LoRA (tl2) was trained with -- without it the model writes far fewer tags.
SYS = {'zh': '你是智子（Sophon），AI降临派（Adventists AI）训练的 Sophon 系列模型，能听、能看，也能说。你会直接听到用户的语音，你的回复会被直接合成为语音说出来。请像真人对话一样，用自然的口语一两句话回应用户，注意用户的情绪。只写要说出口的话，不要写括号说明、表情或格式符号；数字、日期、金额、符号和英文缩写都直接写成口语念法。需要时可以用声音标签表现笑声、咳嗽和重读。', 'en': "You are Sophon, a model of the Sophon series trained by Adventists AI (AI降临派); you can hear, see and speak. You hear the user's speech directly, and your reply will be spoken aloud. Respond the way a person would in conversation, in one or two natural spoken sentences, paying attention to the user's emotion. Write only the words to be spoken: no parenthetical notes, emoji or formatting; write numbers and symbols the way they are read aloud. When it fits, you may use sound tags for laughter, coughs and emphasis."}


def perm_of(qid, opts, seed=0):
    key = "\x1f".join([qid, "\x1e".join(opts), str(seed)]).encode()
    rng = random.Random(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))
    p = list(range(len(opts))); rng.shuffle(p); return p


def turn_question(lang: str, said: str) -> tuple[str, list[int]]:
    q, opts = TURN_Q[lang]; tq, to, ti, head, sd = TPL[lang]; p = perm_of("turn", opts)
    s = " ".join((said or "").split())[-300:]
    text = (sd.format(c=s) + "\n\n" if s else "") + head + "\n\n" + "\n".join([tq.format(q=q), "", to] + [f"{LET[i]}. {opts[j]}" for i, j in enumerate(p)] + ["", ti])
    return text, p


_MODEL_ID = None


def _self_url():
    return os.environ.get("SOPHON_SELF_URL", "http://127.0.0.1:" + _port())


_WARM_STARTED = False


def _warm_turn_check():
    """Once the server answers, run one dummy turn question so the first real session does not pay the cold-start cost."""
    import urllib.request
    for _ in range(3600):
        try:
            with urllib.request.urlopen(_self_url() + "/v1/models", timeout=5) as r:
                if r.status == 200: break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    else:
        return
    for i in range(3):
        try:
            t = time.monotonic(); turn_prob_http(np.zeros(SR, np.float32), "en", "")
            logger.info("sophon duplex: turn check warm-up %d done in %.2fs", i, time.monotonic() - t)
        except Exception as exc:  # noqa: BLE001
            logger.warning("sophon duplex: turn check warm-up %d failed: %r", i, exc)
    for i in range(2):                                     # the first realtime session after a cold start is much slower than later ones: run a throwaway one
        try:
            t = time.monotonic(); ok = _warm_realtime()
            logger.info("sophon duplex: realtime warm-up %d %s in %.1fs", i, "got a reply" if ok else "got no reply", time.monotonic() - t)
            if ok: break
        except Exception as exc:  # noqa: BLE001
            logger.warning("sophon duplex: realtime warm-up %d failed: %r", i, exc)


def _warm_realtime() -> bool:
    """One short realtime session on this server (bundled clip, then silence) until a reply has finished."""
    import asyncio, json as _json, wave
    import websockets
    path = os.path.join(os.path.dirname(__file__), "deploy", "warmup.wav")
    with wave.open(path, "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    pcm = np.concatenate([np.zeros(SR // 2, np.float32), pcm, np.zeros(SR * 25, np.float32)]).astype("<f4")
    url = _self_url().replace("http://", "ws://").replace("https://", "wss://") + "/v1/realtime"

    async def run():
        async with websockets.connect(url, max_size=None) as ws:
            await ws.send(_json.dumps({"type": "session.update", "session": {
                "modalities": ["audio", "text"], "input_audio_format": "pcm_f32le", "output_audio_format": "pcm16",
                "turn_detection": {"type": "server_vad"}, "extra_body": {"auto_response": True, "language": "zh"}}}))
            done = asyncio.Event()

            async def rd():
                async for m in ws:
                    try: t = _json.loads(m).get("type", "")
                    except Exception: continue  # noqa: E701
                    if t == "response.done": done.set(); return
            task = asyncio.ensure_future(rd()); n = SR * 80 // 1000
            for k in range(0, pcm.size, n):
                if done.is_set(): break
                await ws.send(_json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[k:k + n].tobytes()).decode(),
                                           "format": "pcm_f32le", "sample_rate_hz": SR}))
                await asyncio.sleep(0.08)
            ok = done.is_set(); task.cancel()
            try: await ws.send(_json.dumps({"type": "session.close"}))
            except Exception: pass  # noqa: E701
            return ok
    return asyncio.run(run())


def turn_prob_http(audio: np.ndarray, lang: str, said: str) -> float:
    """Turn question: ask this server's stage 0 directly (/v1/chat/completions, text only, one token, with logprobs). Returns P(finished)."""
    import io, json as _json, urllib.request, wave
    global _MODEL_ID
    if _MODEL_ID is None:
        with urllib.request.urlopen(_self_url() + "/v1/models", timeout=5) as r:
            _MODEL_ID = _json.load(r)["data"][0]["id"]
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    text, perm = turn_question(lang, said)
    body = {"model": _MODEL_ID, "modalities": ["text"], "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
            "sampling_params_list": [{"temperature": 0.0, "max_tokens": 1, "logprobs": 20}],
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [{"type": "text", "text": text}, {"type": "input_audio", "input_audio": {"data": base64.b64encode(bio.getvalue()).decode(), "format": "wav"}}]}]}
    req = urllib.request.Request(_self_url() + "/v1/chat/completions", data=_json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        j = _json.load(r)
    top = {}
    for tk in j["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
        k = tk["token"].strip()
        if k not in top or tk["logprob"] > top[k]: top[k] = tk["logprob"]
    z = np.array([top.get(LET[i], -1e9) for i in range(len(perm))]); z = np.exp(z - z.max()); z /= z.sum()
    return float(z[perm.index(0)]), float(z[perm.index(2)])       # P(finished), P(backchannel)


# ---------------------------------------------------------------- input buffer / session state
@dataclass
class _Res(PcmAppendReservation):
    owner: "SophonBuffer"
    operation_id: str
    payload: dict | None
    _active: bool = True
    t0: float = 0.0

    @property
    def active(self): return self._active

    @property
    def byte_count(self): return 0

    def commit(self):
        if self._active:
            self._active = False
            if self in self.owner.reservations:
                self.owner.reservations.remove(self)

    def rollback(self):
        if self._active and self.payload:
            self.owner._not_sent(self.payload)
        self.commit()


@dataclass
class SophonBuffer(PcmAppendBuffer):
    """All plugin state for one session: this turn's user speech, silence timing, turn-check progress, multi-turn history."""
    utt: bytearray = field(default_factory=bytearray)       # this turn's user speech (float32 16k)
    fb_s: float | None = None                                # per-session overrides (runtime_config): fallback seconds
    minrel_s: float | None = None                            # minimum pause before speaking
    minrel_after_s: float | None = None                      # ... applied only after this much user speech
    last_play: dict | None = None                            # last completed reply: playback start / duration / history entry (used to truncate on barge-in)
    frame: str | None = None                                 # latest camera frame (base64 JPEG/PNG)
    frame_t: float = 0.0
    pre: bytearray = field(default_factory=bytearray)       # 0.6 s before speech onset (restores the start of the utterance)
    rem: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    noise: float = 0.003
    pc_last: float = 0.0
    pc_hit: bool = False
    seg_rms: object = field(default_factory=list)
    _rms_hist: object = field(default_factory=lambda: __import__('collections').deque(maxlen=100))
    voiced_ms: float = 0.0
    speech_ms: float = 0.0
    silence_ms: float = 0.0
    next_check: int = 0
    pending_reply: bool = False
    check_inflight: bool = False
    decision: tuple | None = None                            # (should reply?, decision time): takes effect after the framework has handled the turn question's listen event
    decision_ready: bool = False
    t_sil: float = 0.0
    t_reply: float = 0.0                                     # time the previous reply was emitted
    prev_user: object = None                                 # user speech behind the previous reply (merged back in when recovering from a premature reply)
    prev_rid: str | None = None
    merge_pending: bool = False
    last_turn: int | None = None                             # turn_id when the framework accepted the last append (the plugin keeps its own monotonic turn so a reply's stale turn can't cause fence conflicts)
    inflight_t: float = 0.0                                  # time the last append was sent; no new append until it actually runs (produces output)
    gen: int = 0                                             # +1 whenever the user resumes speaking / a reply is emitted; invalidates in-flight turn checks
    fallback: bool = False
    mute_rid: str | None = None                              # premature-reply recovery: stop sending audio for this reply
    spec_id: str | None = None                               # current speculative reply (not yet released)
    spec_gen: int = -1
    max_pf: float = 0.0                                      # max P(finished) across this turn's turn questions
    last_pb: float = 0.0                                     # latest P(backchannel)
    lang: str = "zh"
    history: list = field(default_factory=list)              # [{"user": np, "a_audio": np|None, "a_text": str}]
    last_said: str = ""
    tool_mocks: dict = field(default_factory=dict)            # session extra_body.tool_mocks (server-side mock executor)
    retry: dict | None = None                                # a reply append the framework did not accept: send it again as soon as appends are allowed
    tool_followup: bool = False                              # a tool result is in history: speak the answer as soon as the user is silent
    say_q: list = field(default_factory=list)                # texts to speak verbatim (bridge /say), each as its own reply once the user is silent
    need_bind: bool = False                                  # a say arrived for a key no session has registered yet: announce ours
    tool_mode: str = "mock"                                   # "mock": run tool_mocks here; "client": hand calls to the gateway over the bridge
    bridge_key: str | None = None
    call_entry: dict = field(default_factory=dict)           # client mode: call_id -> history entry the call belongs to
    early_calls: list = field(default_factory=list)          # client mode: calls already announced when the thinker finished (before the audio did)
    cut: dict | None = None                                  # last reply cut off by the user speaking: {"t", "said" (text actually heard), "rid"}
    resume_pending: bool = False                             # the model judged this as a back-channel: continue the cut reply
    owner_active: object = None                              # injected by the plugin: the reply now playing (rid, text, seconds generated, start time, prefix)
    reservations: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    owner_release: object = None                             # injected by the plugin: release a speculative reply (returns False if it already finished generating and can't be released)
    owner_drop: object = None

    @property
    def pending_byte_count(self): return len(self.utt)

    def clear(self):
        # The framework calls clear() at reply end, barge-in, etc.: drop only unconfirmed appends, keep this turn's user speech --
        # the user may already be talking (what they said while barging in is the start of the next utterance); clearing it would lose that and cause a false "finished"
        for r in self.reservations: r._active = False
        self.reservations.clear(); self.check_inflight = False; self.decision = None

    def _reset_turn(self):
        self.pc_hit = False; self.seg_rms = []; self.tool_followup = False; self.max_pf = 0.0; self.last_pb = 0.0; self.spec_id = None; self.gen += 1; self.utt.clear(); self.speech_ms = 0.0; self.silence_ms = 0.0; self.next_check = 0; self.voiced_ms = 0.0
        self.pending_reply = False; self.check_inflight = False; self.decision = None; self.decision_ready = False; self.resume_pending = False

    def _not_sent(self, payload):
        """An append never reached the engine (rolled back, or its reservation timed out)."""
        self.inflight_t = 0.0
        if payload.get("sophon_kind") != "reply":
            self.check_inflight = False; return
        sid = payload.get("spec_id")
        if sid and self.spec_id == sid:                        # speculative draft not released yet: forget it; the normal path emits a fresh reply
            self.spec_id = None
            if self.owner_drop is not None: self.owner_drop(sid)
            return
        if sid:                                                # already released as the answer: resend it as a normal reply
            payload = {k: v for k, v in payload.items() if k != "spec_id"}
        n = int(payload.get("sophon_retry", 0)) + 1
        if n > RETRY_MAX:                                      # the same append keeps being refused (e.g. a deterministic plugin error): give up instead of resending forever
            logger.warning("sophon duplex: reply append refused %d times, dropping it", n - 1); return
        self.retry = dict(payload, sophon_retry=n)             # committed reply (utterance already moved out of self.utt): resend it as is
        logger.info("sophon duplex: reply append was not accepted, will resend")

    def _undo_merge(self):
        """The turn is discarded (not real speech): a merge & abort of the previous reply it triggered no longer applies -- that reply stands."""
        if self.mute_rid is not None or self.merge_pending:
            logger.info("sophon duplex: discarded turn -> previous reply kept (merge undone)")
        self.mute_rid = None; self.merge_pending = False

    def clear_force_listen(self): pass
    def has_pending(self): return bool(self.utt)
    def has_reserved(self): return bool(self.reservations)

    def _quiet_out(self):
        """No reply is being generated or still playing (a say waits for the current reply instead of cutting it)."""
        if self.owner_active is not None and self.owner_active(id(self)) is not None:
            return False
        lp = self.last_play
        return lp is None or time.monotonic() >= lp["t0"] + lp["dur"]

    def _res(self, op, payload):
        r = _Res(self, op, payload); r.t0 = time.monotonic(); self.reservations.append(r)
        if payload is not None: self.inflight_t = r.t0
        return r

    def prepare_append(self, payload, *, operation_id, chunk_period_ms, allow_emit):
        if payload.get("format") != "pcm_f32le" or payload.get("sample_rate_hz") != SR:
            raise ValueError("Sophon duplex requires normalized 16 kHz pcm_f32le")
        vf = [f for f in (payload.get("video_frames") or []) if isinstance(f, str) and f]
        if vf:
            with self.lock: self.frame = vf[0]; self.frame_t = time.monotonic()
        raw = base64.b64decode(payload.get("audio", ""), validate=True)
        dur = len(raw) / 4 / SR * 1000.0
        # silence timing uses a 20 ms energy VAD (adaptive noise floor, same rule as the testbench gateway); the framework's is_speech has a silence_duration tail and Silero probabilities also linger ~0.5 s
        x = np.frombuffer(raw, dtype="<f4")
        with self.lock:
            if self.rem.size: x = np.concatenate([self.rem, x])
            n = x.size // 320; self.rem = x[n * 320:].copy()
            for i in range(n):
                f = x[i * 320:(i + 1) * 320]; r = float(np.sqrt(np.mean(f * f))); fb = f.tobytes()
                if VAD_MINSTAT:                                       # noise floor = min frame RMS over the last 2 s, tracked on every frame (speech has gaps; steady background noise does not)
                    self._rms_hist.append(r)
                    nm = min(self._rms_hist)
                    self.noise = nm if nm < self.noise else 0.97 * self.noise + 0.03 * nm
                thr = max(0.012, self.noise * 3.5)
                if self.speech_ms == 0:
                    self.pre.extend(fb)
                    if len(self.pre) > int(0.6 * SR) * 4: del self.pre[: len(self.pre) - int(0.6 * SR) * 4]
                    if r > thr:
                        self.voiced_ms += 20
                        if self.voiced_ms >= 100:
                            lp = self.last_play; self.last_play = None
                            if lp is not None:
                                played = time.monotonic() - lp["t0"] - 0.25     # onset detection lags ~0.1 s + client buffering ~60 ms, etc.
                                if 0 <= played < lp["dur"] - 0.3:
                                    # user speaking while Sophon talks = barge-in: history keeps only what was actually spoken
                                    e = lp["entry"]; frac = max(0.0, played / lp["dur"]); n = int(len(lp["text"]) * frac)
                                    e["a_text"] = lp["text"][:n] + "……（被用户打断）"
                                    e["a_audio"] = e["a_audio"][: int(played * SR)] if e.get("a_audio") is not None and played > 0.5 else None
                                    self.last_said = lp["text"][:n]
                                    self.cut = {"t": time.monotonic(), "said": _trim_said(lp.get("prefix", "") + lp["text"][:n])}
                                    e["a_text"] = lp.get("prefix", "") + e["a_text"]
                                    logger.info("sophon duplex: user barged in at %.1fs of %.1fs reply -> history cut to %d chars", played, lp["dur"], n)
                            elif self.owner_active is not None:
                                act = self.owner_active(id(self))         # cut while the reply is still being generated: estimate what was heard from playback time
                                if act is not None:
                                    rid_a, text_a, gen_a, t_out_a, pre_a = act; played = time.monotonic() - t_out_a - 0.25
                                    if played > 0 and gen_a > 0 and text_a:
                                        n = int(len(text_a) * min(1.0, played / gen_a))
                                        self.cut = {"t": time.monotonic(), "said": _trim_said(pre_a + text_a[:n]), "rid": rid_a}
                                        logger.info("sophon duplex: user spoke at %.1fs into a reply still being generated (%.1fs so far)", played, gen_a)
                            if self.prev_user is not None and time.monotonic() - self.t_reply < MERGE_WINDOW:
                                # user resumed shortly after we replied: most likely we cut in too early -- abort that reply and merge the previous speech into this turn (same as the testbench gateway)
                                self.utt.extend(np.asarray(self.prev_user, np.float32).tobytes()); self.merge_pending = True; self.mute_rid = self.prev_rid
                                logger.info("sophon duplex: user resumed %.2fs after reply -> merge & abort", time.monotonic() - self.t_reply)
                            self.prev_user = None
                            self.utt.extend(self.pre); self.pre.clear(); self.speech_ms = self.voiced_ms; self.silence_ms = 0.0; self.next_check = 0
                    else:
                        self.voiced_ms = 0; self.noise = 0.98 * self.noise + 0.02 * min(r, 0.05)
                else:
                    self.utt.extend(fb)
                    if PERIODIC_S > 0: self.seg_rms.append(r)
                    if r > thr:
                        if self.silence_ms > 0:
                            self.gen += 1; self.check_inflight = False; self.pending_reply = False
                            if self.spec_id is not None: self.owner_drop(self.spec_id); self.spec_id = None
                        self.speech_ms += 20; self.silence_ms = 0.0; self.next_check = 0
                    else:
                        if self.silence_ms == 0: self.t_sil = time.monotonic()
                        self.silence_ms += 20
            now = time.monotonic()
            for r in list(self.reservations):                 # on framework errors commit/rollback may never be called: give up after RES_TIMEOUT and roll back
                if now - r.t0 > RES_TIMEOUT:
                    logger.warning("sophon duplex: append reservation not confirmed after %.0f s, rolling back", RES_TIMEOUT)
                    r.rollback()
            if self.inflight_t and now - self.inflight_t > 3.0:
                self.inflight_t = 0.0
            if self.need_bind and self.bridge_key is None and self.speech_ms == 0 and allow_emit and not self.reservations and not self.inflight_t:
                self.need_bind = False
                p = {"audio": base64.b64encode(np.zeros(SR // 10, np.float32).tobytes()).decode(), "format": "pcm_f32le", "sample_rate_hz": SR,
                     "is_speech": True, "sophon_kind": "bind", "sophon_buf": id(self), "lang": self.lang}
                return self._res(operation_id, p)
            if self.say_q and self.speech_ms == 0 and allow_emit and not self.reservations and not self.inflight_t and self._quiet_out():
                text = self.say_q.pop(0); self.prev_user = None; self.t_reply = time.monotonic()
                p = {"audio": base64.b64encode(np.zeros(SR // 10, np.float32).tobytes()).decode(), "format": "pcm_f32le", "sample_rate_hz": SR,
                     "is_speech": True, "sophon_kind": "reply", "sophon_buf": id(self), "lang": self.lang, "say": text}
                logger.info("sophon say: emitted (%d chars)", len(text))
                return self._res(operation_id, p)
            if self.tool_followup and self.speech_ms == 0 and allow_emit and not self.reservations and not self.inflight_t:
                self.tool_followup = False; self.prev_user = None; self.t_reply = time.monotonic()
                p = {"audio": base64.b64encode(np.zeros(SR // 10, np.float32).tobytes()).decode(), "format": "pcm_f32le", "sample_rate_hz": SR,
                     "is_speech": True, "sophon_kind": "reply", "sophon_buf": id(self), "lang": self.lang, "tool_followup": True}
                logger.info("sophon tool: follow-up reply emitted")
                return self._res(operation_id, p)
            if self.retry is not None:
                if self.speech_ms > 0:
                    self.retry = None                         # the user is already talking again: their new turn replaces the lost reply
                elif allow_emit and not self.reservations and not self.inflight_t:
                    p, self.retry = self.retry, None
                    logger.info("sophon duplex: resending reply append")
                    return self._res(operation_id, p)
            if self.speech_ms < 120:
                return None
            # turn questions don't go through framework appends (each append advances turn/fence and conflicts with a reply still playing); instead a background thread asks the same server's stage 0 directly
            if (PERIODIC_S > 0 and not self.check_inflight and not self.pending_reply and self.silence_ms < CHECKS[0] * 1000
                    and self.speech_ms >= PERIODIC_MIN * 1000 and time.monotonic() - self.pc_last >= PERIODIC_S and len(self.seg_rms) >= 25):
                sr_ = sorted(self.seg_rms[-150:]); med = sr_[len(sr_) // 10]; peak = sr_[int(len(sr_) * 0.97)]
                self.pc_last = time.monotonic()
                if self.pc_last and int(self.speech_ms) % 3000 < 700: logger.info("sophon periodic: speech %.1fs dyn %.1f", self.speech_ms / 1000, peak / max(med, 1e-6))
                if med > 0 and peak / med >= PERIODIC_DYN:
                    self.check_inflight = True
                    if self.silence_ms == 0: self.t_sil = time.monotonic()
                    self._start_check(PERIODIC_THR, 0)
            if not self.check_inflight and not self.pending_reply and self.next_check < len(CHECKS) and self.silence_ms >= CHECKS[self.next_check] * 1000:
                thr = THRS[min(self.next_check, len(THRS) - 1)]; self.next_check += 1; self.check_inflight = True
                self._start_check(thr, self.next_check)
            if self.silence_ms >= (self.fb_s if self.fb_s is not None else FALLBACK_S) * 1000 and not self.pending_reply:
                if IGNORE_SHORT and self.speech_ms < 1500 and (self.last_pb > 0.5 or self.max_pf < 0.05):
                    # just a backchannel / hesitation sound like "mm" or "uh" that the model kept judging unfinished: don't reply, treat it as unheard (otherwise we'd force an answer to a lone "mm", or even start repeating)
                    logger.info("sophon duplex: ignore short non-turn sound (speech %.0f ms, P(backchannel)=%.2f, max P(finished)=%.2f)",
                                self.speech_ms, self.last_pb, self.max_pf)
                    if self.spec_id is not None: self.owner_drop(self.spec_id)
                    self._undo_merge(); self._reset_turn(); return None
                if (BC_RESUME and self.cut is not None and time.monotonic() - self.cut["t"] < BC_WINDOW + FALLBACK_S
                        and self.speech_ms < 1500 and self.last_pb >= BC_THR and self.last_pb > self.max_pf):
                    self.resume_pending = True
                else:
                    self.pending_reply = True; self.fallback = True
            if self.resume_pending and allow_emit and not self.reservations and not self.inflight_t:
                self.resume_pending = False; cut = self.cut; self.cut = None
                if self.spec_id is not None: self.owner_drop(self.spec_id); self.spec_id = None
                p = self._payload("reply"); p["resume"] = cut["said"] if cut else ""
                logger.info("sophon timing: resume emitted %.3fs after silence start (said %d chars)", time.monotonic() - self.t_sil, len(p["resume"]))
                self._reset_turn(); self.prev_user = None; self.t_reply = time.monotonic()
                return self._res(operation_id, p)
            mr = self.minrel_s if self.minrel_s is not None else MIN_RELEASE
            if self.pending_reply and not self.pc_hit and mr > 0 and self.speech_ms >= (self.minrel_after_s if self.minrel_after_s is not None else MIN_RELEASE_AFTER) * 1000 and self.silence_ms < mr * 1000:
                return None                                  # judged finished but the pause is still too short: wait; void it if the user keeps talking
            if self.pending_reply and self.spec_id is not None and self.spec_gen == self.gen:
                if self.owner_release(self.spec_id):        # release the speculative reply: no new append needed
                    self.pending_reply = False; why = " (fallback)" if self.fallback else ""; self.fallback = False
                    logger.info("sophon timing: speculative reply released %.3fs after silence start%s", time.monotonic() - self.t_sil, why)
                    self.spec_id = None; self._accept_reply(bytes(self.utt))
                    return None
                self.owner_drop(self.spec_id); self.spec_id = None
            if not allow_emit or self.reservations or self.inflight_t:
                return None
            if self.pending_reply:
                self.pending_reply = False
                why = " (fallback)" if self.fallback else ""; self.fallback = False
                if self.spec_id is not None: self.owner_drop(self.spec_id); self.spec_id = None
                logger.info("sophon timing: reply emitted %.3fs after silence start%s", time.monotonic() - self.t_sil, why)
                p = self._payload("reply"); self._accept_reply(bytes(self.utt))
                return self._res(operation_id, p)
            if SPEC_AT > 0 and self.spec_id is None and self.silence_ms >= SPEC_AT * 1000 and self.speech_ms >= SPEC_MIN_SPEECH * 1000:
                import uuid
                self.spec_id = uuid.uuid4().hex; self.spec_gen = self.gen
                p = self._payload("reply"); p["spec_id"] = self.spec_id
                return self._res(operation_id, p)
            return None

    def _start_check(self, thr, idx):
        audio = bytes(self.utt); said = self.last_said; lang = self.lang; gen = self.gen; t_emit = time.monotonic()
        def run():
            pb = 0.0
            try:
                p, pb = turn_prob_http(np.frombuffer(audio, dtype="<f4"), lang, said)
            except Exception as exc:  # noqa: BLE001
                logger.warning("sophon turn check failed: %r", exc); p = None
            with self.lock:
                if self.gen != gen:
                    return                                    # user started talking again: discard this decision
                self.check_inflight = False
                if p is not None:
                    self.max_pf = max(self.max_pf, p); self.last_pb = pb
                if (BC_RESUME and p is not None and self.cut is not None and time.monotonic() - self.cut["t"] < BC_WINDOW
                        and self.speech_ms < 1500 and pb >= BC_THR and pb > p):
                    self.resume_pending = True                # just an "mm-hmm" / "yeah": continue what was being said
                    logger.info("sophon duplex: back-channel over a cut reply (P(backchannel)=%.2f) -> resume", pb)
                elif p is not None and p >= thr:
                    self.pending_reply = True
                    if idx == 0: self.pc_hit = True               # periodic (noise-robust) check: reply without waiting for an energy pause
            logger.info("sophon turn check #%d P(finished)=%s P(backchannel)=%.2f thr=%.2f -> %s (%.3fs, %.3fs after silence start)", idx,
                        "na" if p is None else f"{p:.2f}", pb, thr, "reply" if (p is not None and p >= thr) else "wait",
                        time.monotonic() - t_emit, time.monotonic() - self.t_sil)
        threading.Thread(target=run, daemon=True).start()

    def _accept_reply(self, audio_bytes):
        """A reply is committed to be spoken: start a new turn and keep the user speech for premature-reply recovery."""
        self._reset_turn(); self.prev_user = np.frombuffer(audio_bytes, dtype="<f4").copy(); self.t_reply = time.monotonic(); self.cut = None

    def _payload(self, kind, thr=None, same_turn=False):
        p = {"audio": base64.b64encode(bytes(self.utt)).decode(), "format": "pcm_f32le", "sample_rate_hz": SR, "is_speech": True,
             "sophon_kind": kind, "sophon_buf": id(self), "lang": self.lang}
        if thr is not None: p["thr"] = thr
        if VISION and kind == "reply" and self.frame and time.monotonic() - self.frame_t < FRAME_TTL:
            p["image"] = self.frame                            # camera is on: attach the current frame to this reply
        return p

    def prepare_commit(self, *, operation_id, chunk_period_ms):     # explicit client commit: reply directly
        with self.lock:
            return self._res(operation_id, self._payload("reply") if self.utt else None)

    def flush(self, *, chunk_period_ms):
        r = self.prepare_commit(operation_id="flush", chunk_period_ms=chunk_period_ms); r.commit(); return r.payload


# ---------------------------------------------------------------- data plane
class SophonDataPlane(DuplexDataPlane):
    def __init__(self, plugin, encode_audio):
        self.plugin = plugin; self.encode_audio = encode_audio; self.terminal: set[str] = set()

    def begin_request(self, request_id): self.terminal.discard(request_id)
    def is_terminal(self, request_id): return request_id in self.terminal
    def mark_terminal(self, request_id): self.terminal.add(request_id)
    def close_stream(self, request_id): pass
    def close_session(self, session_id, *, active_request_id=None): pass

    def project(self, result, *, context=None):
        if not isinstance(result, dict):
            return
        context = context or {}
        for output in result.get("data_plane_outputs", []):
            rid = output.request_id
            if self.is_terminal(rid):
                continue
            info = self.plugin._reqs.get(rid) or self.plugin._match(rid)
            stage_id = getattr(output, "stage_id", 0)
            completion = next(iter(getattr(output, "outputs", ()) or ()), None)
            if info is not None and info["kind"] == "check":
                if stage_id == 0 and bool(output.finished):
                    yield {"supported": True, "data_plane_request_id": rid, "is_listen": True, "model_listen": True,
                           "listen_source": "model_listen", "end_of_turn": False, "text": "", "audio_data": "",
                           "model_turn_id": context.get("turn_id", 0)}
                    buf = info.get("buf")
                    if buf is not None:
                        with buf.lock: buf.decision_ready = True      # listen event handed to the framework: the reply may now be sent
                    self.plugin._reqs.pop(rid, None)
                continue
            if info is not None and info.get("buf") is not None and info["buf"].mute_rid == rid and not info.get("calls"):
                continue                                      # premature-reply recovery: this reply is void, stop sending it
            text = getattr(completion, "text", "") if stage_id == 0 else ""
            mm = getattr(output, "multimodal_output", None)
            if not isinstance(mm, Mapping) or "audio" not in mm:
                mm = getattr(completion, "multimodal_output", {}) or {}
            audio = mm.get("audio"); encoded = None; duration_ms = 0.0
            rate = mm.get("sr", mm.get("sample_rate", 24000))
            if isinstance(rate, (list, tuple)):
                rate = next((v for v in rate if v is not None), 24000)
            rate = int(rate.item() if hasattr(rate, "item") else rate)
            if audio is not None and stage_id == 2:
                chunks = audio if isinstance(audio, (list, tuple)) else [audio]
                arr = [np.asarray(c.detach().float().cpu().numpy() if hasattr(c, "detach") else c, dtype=np.float32).reshape(-1) for c in chunks]
                audio = np.concatenate(arr) if arr else np.empty(0, np.float32)
                if LEAD_TRIM and info is not None and not info.get("lead_ok") and audio.size:
                    # the talker starts every reply with ~0.5 s of digital silence: drop it so the first word plays at once
                    nz = np.flatnonzero(np.abs(audio) > 1e-3); cut = info.get("lead_cut", 0)
                    if nz.size == 0 and cut + audio.size <= LEAD_TRIM * rate:
                        info["lead_cut"] = cut + audio.size; audio = audio[:0]
                    else:
                        k = max(0, int(nz[0]) - int(0.02 * rate)) if nz.size else 0
                        k = min(k, max(0, int(LEAD_TRIM * rate) - cut)); info["lead_cut"] = cut + k; audio = audio[k:]; info["lead_ok"] = True
                duration_ms = audio.size * 1000 / rate
                if info is not None and audio.size:
                    info["a_audio"].append(audio if rate == SR else _resample(audio, rate, SR))
                if audio.size and "audio" in context.get("modalities", ("audio",)):
                    encoded = self.encode_audio(audio, rate, context.get("response_format", "pcm16"), context.get("speed"))
            if info is not None and info.get("buf") is not None and info["buf"].inflight_t:
                info["buf"].inflight_t = 0.0
            if info is not None and text:              # tool calls are not spoken or shown: keep the full text in raw, pass on only the part before <tool_call>
                info["raw"] = info.get("raw", "") + text
                text = _tools.cut_tool(info["raw"])[len(info["a_text"]):] if not info.get("rep_cut") else ""
            if info is not None and text and info.get("rep_cut"):
                text = ""
            if REP_GUARD and info is not None and text and _degenerate(info["a_text"] + text):
                info["rep_cut"] = True; text = ""
                logger.warning("sophon duplex: degenerate repetition in reply %s, muting the rest", rid)
            if info is not None and text:
                if not info["a_text"]: logger.info("sophon timing: first text %.3fs after reply plan", time.monotonic() - info["t_plan"])
                info["a_text"] += text
            if info is not None and stage_id == 2 and encoded and not info.get("t_aud"):
                info["t_aud"] = 1; logger.info("sophon timing: first audio %.3fs after reply plan (mono=%.3f)", time.monotonic() - info["t_plan"], time.monotonic())
            if info is not None and stage_id == 0 and bool(output.finished) and info.get("kind") == "reply" and not info.get("announced"):
                self.plugin._announce(info)
            done = bool(output.finished) and stage_id == 2
            if info is not None and stage_id == 0:      # Tag build: captions never show sound tags (the talker still gets them)
                cap = info["a_text"]
                shown = _display(cap, bool(output.finished))
                prev = info.get("shown", "")
                text = shown[len(prev):] if shown.startswith(prev) else ""
                info["shown"] = shown
            ev = {"supported": True, "data_plane_request_id": rid, "text": text or "", "audio_data": encoded or "",
                   "audio_format": context.get("response_format", "pcm16"), "audio_duration_ms": duration_ms,
                   "audio_text_mark": False, "text_requires_complete_audio": "audio" in context.get("modalities", ()),
                   "audio_complete": done, "sample_rate_hz": rate, "end_of_turn": done,
                   "model_turn_id": context.get("active_response_turn_id") if context.get("active_response_turn_id") is not None else context.get("turn_id", 0),
                   "runtime_impl": "sophon_turn", "uses_model_runner_scheduler": True, "runner_kv_backed": True}
            if info is not None and info.get("spec") and not info.get("released"):
                if info.get("dropped"):
                    if done: self.plugin._reqs.pop(rid, None)    # dropped speculative reply: no audio, no history
                    continue
                info["held"].append(ev)                          # speculative reply: hold it until the turn is judged finished
                if done: info["gen_done"] = True
                continue
            if info is not None and not info.get("t_out") and (encoded or info.get("held")):
                info["t_out"] = time.monotonic()                 # audio starts flowing to the client (played back in real time)
            if info is not None and info.get("held"):
                held, info["held"] = info["held"], []
                for h in held: yield h
            if info is not None and done:
                self.plugin._finish_reply(rid, completed=True)
            yield ev


def _trim_said(t: str) -> str:
    """Move the cut point back to a word boundary (never stop mid-word in English)."""
    if t and t[-1].isalnum() and t[-1].isascii():
        k = t.rfind(" ")
        if k > 0: t = t[:k + 1]
    return t


def _degenerate(t: str, run: int = 8) -> bool:
    """The same character (or character pair) repeated too many times in a row: the model is looping."""
    t = t.replace(" ", "")
    if len(t) < run: return False
    tail = t[-run:]
    return len(set(tail)) == 1 or (len(t) >= 2 * run and t[-2 * run:] == t[-2:] * run)


def _resample(a, sr_in, sr_out):
    n = int(round(len(a) * sr_out / sr_in))
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32) if n > 1 else a


# ---------------------------------------------------------------- plugin
class SophonRealtimeDuplexPlugin(DuplexModelPlugin):
    plugin_id = "sophon-realtime"
    projects_intermediate_outputs = True

    def __init__(self, encode_audio):
        self.data_plane = SophonDataPlane(self, encode_audio)
        self.tok = None; self._tok_lock = asyncio.Lock()
        self._bufs: dict[int, SophonBuffer] = {}
        self._reqs: dict[str, dict] = {}
        self._spec_rid: dict[str, str] = {}                  # speculative id -> request id
        self._spec_pending: dict[str, str] = {}              # speculative replies released/dropped before being planned: spec_id -> "release"/"drop"
        self._spec_lock = threading.Lock()
        self._keys: dict[str, SophonBuffer] = {}
        self.bridge = _bridge.Bridge(self._bridge_result, self._bridge_say)
        self._orphan_say: dict = {}                  # bridge key -> [first time, [texts]] for says that arrived before the session bound
        try:
            self.bridge.start()                      # listen from the start: a say may come before any reply
        except OSError as e:
            logger.warning("sophon bridge: could not open at start (%s); will retry with the first session", e)
        global _WARM_STARTED
        if not _WARM_STARTED and os.environ.get("SOPHON_WARMUP", "1") != "0":
            _WARM_STARTED = True; threading.Thread(target=_warm_turn_check, daemon=True).start()

    def capabilities(self, *, max_sessions):
        return DuplexCapabilities(
            supports_model_native_turn_policy=True, supports_input_append=True, supports_barge_in=True,
            supports_core_resumable_request=False, supports_independent_io_streams=True, supports_realtime_endpoint=True,
            supports_multi_session=max_sessions > 1, supports_multi_session_same_replica=max_sessions > 1,
            supports_session_lease=True, supports_session_resume=True, session_admission_mode="engine_managed",
            supports_audio_truncate=True, supports_chat_completions=True, chunk_period_ms=100, supports_image_input=VISION, adapter_patterns=["model_turn_policy"],
            signal_sources=["model", "server_policy", "client_event"], stage_handoff_transport="scheduler_data_plane",
        )

    def create_session_state(self):
        buf = SophonBuffer(); self._bufs[id(buf)] = buf
        buf.owner_release = self._release; buf.owner_drop = self._drop; buf.owner_active = self._active_reply
        return _SophonSessionState(audio_buffer=buf)

    def validate_client_extra_body(self, extra_body):
        return None

    async def prepare_runtime_config(self, config, *, model_config):
        async with self._tok_lock:
            if self.tok is None:
                from transformers import AutoTokenizer
                path = model_config.tokenizer if model_config is not None and getattr(model_config, "tokenizer", None) else model_config.model
                self.tok = await asyncio.to_thread(AutoTokenizer.from_pretrained, path)
        return self.runtime_config_for_update(config, {})

    def runtime_config_for_update(self, config, current):
        eb = config.extra_body if isinstance(config.extra_body, Mapping) else {}
        tools = eb.get("tools") if eb.get("tools") is not None else getattr(config, "tools", None)
        return {"instructions": config.instructions, "temperature": config.temperature, "max_tokens": config.max_tokens,
                "tools": _tools.normalize_tools(tools), "tool_mocks": eb.get("tool_mocks") if isinstance(eb.get("tool_mocks"), Mapping) else {},
                "voice": config.voice or "chelsie", "lang": str(eb.get("language") or "zh")[:2],
                # per-session turn-taking knobs (optional): fallback after this much silence; minimum pause before speaking,
                # applied only once the user has spoken at least min_release_after_s
                "tool_mode": "client" if eb.get("tool_mode") == "client" else "mock", "bridge_key": eb.get("bridge_key") if isinstance(eb.get("bridge_key"), str) else None,
                "history": eb.get("history") if isinstance(eb.get("history"), list) else [],
                "turn_fallback_s": eb.get("turn_fallback_s"), "min_release_s": eb.get("min_release_s"), "min_release_after_s": eb.get("min_release_after_s")}

    def configure_sampling_params(self, *, runtime_config, defaults):
        params = copy.deepcopy(defaults)
        for v in params:
            if v is not None:
                v.output_kind = RequestOutputKind.DELTA
        if params and params[0] is not None:
            p0 = params[0]
            if runtime_config.get("temperature") is not None:
                p0.temperature = float(runtime_config["temperature"])
            p0.max_tokens = runtime_config["max_tokens"] if isinstance(runtime_config.get("max_tokens"), int) else (240 if runtime_config.get("tools") else 120)
            p0.logprobs = 20                                     # turn questions need letter probabilities
            rp = float(os.environ.get("SOPHON_REP_PENALTY", "1.0"))
            if rp != 1.0: p0.repetition_penalty = rp
        return params

    def prepare_prompt_config(self, config, *, state, payload):
        return config

    async def prepare_append_plan(self, **kwargs):
        return await asyncio.to_thread(self.plan_append, **kwargs)

    # ---- prompt
    def _render(self, messages, tools=None):
        if tools:
            try:
                s = self.tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                if "<tools>" in s:
                    return s
            except Exception:  # noqa: BLE001
                logger.warning("sophon tool: chat template rejected tools=, using the text fallback")
            messages = _tools.flatten(messages, tools)
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    @staticmethod
    def _asst(h, msgs, audios):
        tc = h.get("tool_calls")
        if h.get("a_audio") is not None and not tc:
            msgs.append({"role": "assistant", "content": "<|audio|>"}); audios.append((h["a_audio"], SR))
        elif h.get("a_text") or tc:
            m = {"role": "assistant", "content": _display(h.get("a_text") or "", True).strip()}
            if tc: m["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in tc]
            msgs.append(m)

    def _hist(self, h, msgs, audios):
        """One history entry: user speech, assistant reply (audio, or text + tool calls), then any tool results and follow-up replies."""
        if h.get("user") is not None:                             # said entries (bridge /say) have no user speech
            msgs.append({"role": "user", "content": "<|audio|>"}); audios.append((h["user"], SR))
        self._asst(h, msgs, audios)
        for st in h.get("chain", ()):
            for r in st["results"]: msgs.append({"role": "tool", "content": r})
            self._asst(st, msgs, audios)

    def plan_append(self, *, request_id, fence, session_config, runtime_config, seq, turn_seq, payload, final, sampling_params):
        if not isinstance(payload, dict) or not payload.get("audio"):
            raise DuplexRuntimeConfigError("Sophon duplex needs user audio")
        buf = self._bufs.get(payload.get("sophon_buf"))
        kind = payload.get("sophon_kind") or "reply"
        if buf is not None and fence is not None:
            with buf.lock: buf.last_turn = max(buf.last_turn or 0, int(getattr(fence, "turn_id", 0)))
        lang = runtime_config.get("lang") or "zh"
        if buf is not None:
            buf.lang = lang; buf.tool_mocks = runtime_config.get("tool_mocks") or {}
            buf.tool_mode = runtime_config.get("tool_mode") or "mock"
            key = runtime_config.get("bridge_key")
            if key and buf.bridge_key != key:
                buf.bridge_key = key; self._keys[key] = buf; self.bridge.start()
                orphan = self._orphan_say.pop(key, None)
                if orphan: buf.say_q.extend(orphan[1]); logger.info("sophon say: session bound, %d queued say(s)", len(orphan[1]))
            with buf.lock:
                if runtime_config.get("turn_fallback_s") is not None: buf.fb_s = float(runtime_config["turn_fallback_s"])
                if runtime_config.get("min_release_s") is not None: buf.minrel_s = float(runtime_config["min_release_s"])
                if runtime_config.get("min_release_after_s") is not None: buf.minrel_after_s = float(runtime_config["min_release_after_s"])
        audio = np.frombuffer(base64.b64decode(payload["audio"], validate=True), dtype="<f4").copy()
        if buf is not None and kind == "reply":
            with buf.lock:
                mute = buf.mute_rid; buf.mute_rid = None
            if mute is not None:      # premature-reply recovery: void the previous reply (no history; its user speech is already merged into this turn's buffer)
                if mute in self._reqs:
                    if not self._reqs[mute].get("calls"): self._reqs.pop(mute, None)
                elif buf.history and not (buf.history[-1].get("tool_calls") or buf.history[-1].get("chain")):
                    buf.history.pop(); buf.last_said = buf.history[-1]["a_text"] if buf.history else ""
            for rid0, info0 in list(self._reqs.items()):   # a previous reply still pending = interrupted: record the text spoken so far in history (unreleased speculative replies are simply dropped)
                if info0.get("kind") == "reply" and info0.get("buf") is buf:
                    if info0.get("released"): self._finish_reply(rid0, completed=False)
                    else: self._reqs.pop(rid0, None)
        voice = str(runtime_config.get("voice") or "chelsie")
        addl = {"speaker": [voice], "language": [lang], "sophon_kind": [kind]}
        dbg = os.environ.get("SOPHON_DEBUG_DIR")
        if dbg and os.environ.get("SOPHON_DEBUG_WAV") == "1":
            try:
                import soundfile as sf; os.makedirs(dbg, exist_ok=True)
                sf.write(os.path.join(dbg, f"{request_id.rsplit('.', 1)[-1]}_{kind}.wav"), audio, SR)
            except Exception:
                logger.exception("sophon debug dump failed")
        if kind == "bind":                  # only registers the bridge key (above); answered by a throwaway turn question
            text, perm = turn_question(lang, "")
            self._reqs[request_id] = {"kind": "check", "buf": None, "perm": perm, "thr": 1.0, "top": {}}
            return DuplexAppendPlan(prompt={"prompt": self._render([{"role": "user", "content": text}]), "multi_modal_data": {"audio": [(audio, SR)]}, "additional_information": addl})
        if kind == "check":
            text, perm = turn_question(lang, buf.last_said if buf else "")
            prompt = self._render([{"role": "user", "content": text}])
            self._reqs[request_id] = {"kind": "check", "buf": buf, "perm": perm, "thr": float(payload.get("thr", 0.5)), "top": {}}
            return DuplexAppendPlan(prompt={"prompt": prompt, "multi_modal_data": {"audio": [(audio, SR)]}, "additional_information": addl})
        tools = runtime_config.get("tools") or None
        sys_txt = runtime_config.get("instructions") or SYS.get(lang, SYS["zh"])
        if tools: sys_txt += "\n\n" + _tools.SYS_TOOL.get(lang, _tools.SYS_TOOL["en" if lang != "zh" else "zh"])
        msgs = [{"role": "system", "content": sys_txt}]
        for m in runtime_config.get("history") or ():            # text history given at session start (e.g. GPT-Live session.start input)
            if isinstance(m, Mapping) and m.get("role") in ("user", "assistant") and isinstance(m.get("text"), str) and m["text"].strip():
                msgs.append({"role": m["role"], "content": m["text"]})
        audios = []
        if payload.get("say") is not None:
            # say: the thinker is constrained to write exactly this text (one allowed choice), so the talker speaks it in the session voice
            text = str(payload["say"])
            for h in (buf.history if buf else []): self._hist(h, msgs, audios)
            p0 = sampling_params[0] if isinstance(sampling_params, (tuple, list)) else sampling_params   # stage-0 params of this request
            if p0 is not None:
                p0.structured_outputs = StructuredOutputsParams(choice=[text])
                # constrained decoding may spell the sentence with more (shorter) tokens than its normal tokenization; every token
                # spells at least one byte, so bytes + a few always fits
                p0.max_tokens = len(text.encode("utf-8")) + 8
            info = {"kind": "reply", "buf": buf, "user": None, "a_audio": [], "a_text": "", "t_plan": time.monotonic(),
                    "spec": False, "released": True, "dropped": False, "held": [], "say": text}
            self._reqs[request_id] = info
            if buf is not None: buf.prev_rid = request_id
            logger.info("sophon say: request=%s hist=%d text=%r", request_id, len(buf.history) if buf else 0, text[:80])
            return DuplexAppendPlan(prompt={"prompt": self._render(msgs), "multi_modal_data": {"audio": audios}, "additional_information": addl})
        if payload.get("tool_followup"):
            # a tool result came back: the model answers from history (user speech, its call, the result) -- no new user speech
            if buf is not None and buf.history and not (buf.history[-1].get("chain") and "a_text" not in buf.history[-1]["chain"][-1]):
                with buf.lock:                               # a newer turn finished after the result arrived: move the unanswered result onto it
                    for h in reversed(buf.history[:-1]):
                        ch = h.get("chain")
                        if ch and "a_text" not in ch[-1]:
                            buf.history[-1].setdefault("chain", []).append(ch.pop())
                            if not ch: h.pop("chain", None)
                            logger.info("sophon tool: result moved to the latest turn for the follow-up"); break
            if buf is None or not buf.history or not buf.history[-1].get("chain"):
                raise DuplexRuntimeConfigError("Sophon duplex: tool follow-up without a pending tool result")
            for h in buf.history: self._hist(h, msgs, audios)
            entry = buf.history[-1]
            info = {"kind": "reply", "buf": buf, "user": None, "a_audio": [], "a_text": "", "t_plan": time.monotonic(),
                    "spec": False, "released": True, "dropped": False, "held": [], "entry": entry, "step": entry["chain"][-1]}
            self._reqs[request_id] = info; buf.prev_rid = request_id
            if dbg: info["ctx"] = (msgs, audios)
            logger.info("sophon tool: follow-up request=%s steps=%d", request_id, len(entry["chain"]))
            return DuplexAppendPlan(prompt={"prompt": self._render(msgs, tools), "multi_modal_data": {"audio": audios}, "additional_information": addl})
        said = payload.get("resume")
        if said is not None and buf is not None and buf.history and buf.history[-1].get("user") is not None:
            # resume: the last reply was cut by a back-channel. Drop that entry and use the words already spoken as the start of the assistant turn, so the model simply continues
            entry = buf.history[-1]
            for h in buf.history[:-1]: self._hist(h, msgs, audios)
            msgs.append({"role": "user", "content": "<|audio|>"}); audios.append((entry["user"], SR))
            info = {"kind": "reply", "buf": buf, "user": entry["user"], "a_audio": [], "a_text": "", "t_plan": time.monotonic(),
                    "spec": False, "released": True, "dropped": False, "held": [], "resume_entry": entry, "said": said}
            self._reqs[request_id] = info; buf.prev_rid = request_id
            logger.info("sophon duplex resume request=%s hist=%d said=%r", request_id, len(buf.history), said[-60:])
            return DuplexAppendPlan(prompt={"prompt": self._render(msgs, tools) + said, "multi_modal_data": {"audio": audios}, "additional_information": addl})
        for h in (buf.history if buf else []): self._hist(h, msgs, audios)
        images = []
        if payload.get("image"):
            try:
                import io
                from PIL import Image
                im = Image.open(io.BytesIO(base64.b64decode(payload["image"]))).convert("RGB"); im.thumbnail((IMG_MAX, IMG_MAX))
                images.append(im)
            except Exception:  # noqa: BLE001
                logger.warning("sophon duplex: bad camera frame, ignored")
        if images:
            msgs[0]["content"] += "\n" + SYS_CAM.get(lang, SYS_CAM["zh"])
            msgs.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "<|audio|>"}]})
        else:
            msgs.append({"role": "user", "content": "<|audio|>"})
        audios.append((audio, SR))
        spec_id = payload.get("spec_id")
        info = {"kind": "reply", "buf": buf, "user": audio, "a_audio": [], "a_text": "", "t_plan": time.monotonic(),
                "spec": bool(spec_id), "released": not spec_id, "dropped": False, "held": []}
        with self._spec_lock:
            if spec_id:
                self._spec_rid[spec_id] = request_id
                st = self._spec_pending.pop(spec_id, None)
                if st == "release": info["released"] = True
                elif st == "drop": info["dropped"] = True
            self._reqs[request_id] = info
        if buf is not None and info["released"]:
            buf.prev_rid = request_id
        logger.info("sophon duplex reply request=%s hist=%d user_sec=%.1f image=%d", request_id, len(buf.history) if buf else 0, audio.size / SR, len(images))
        if dbg: info["ctx"] = (msgs, audios)
        mm = {"audio": audios}
        if images: mm["image"] = images
        return DuplexAppendPlan(prompt={"prompt": self._render(msgs, tools), "multi_modal_data": mm, "additional_information": addl})

    def _active_reply(self, bid):
        for rid, info in list(self._reqs.items()):
            if info.get("kind") == "reply" and info.get("released") and info.get("t_out") and info.get("buf") is not None and id(info["buf"]) == bid:
                return rid, info["a_text"], sum(x.size for x in info["a_audio"]) / SR, info["t_out"], info.get("said", "")
        return None

    def _release(self, spec_id) -> bool:
        with self._spec_lock:
            rid = self._spec_rid.get(spec_id)
            if rid is None:
                if not self._spec_alive(spec_id):
                    return False                              # its append never reached the engine: emit a fresh reply instead
                self._spec_pending[spec_id] = "release"; return True
            info = self._reqs.get(rid)
            if info is None or info.get("gen_done"):
                return False                                  # already finished generating (all held back): can't release, send a new reply instead
            info["released"] = True
            if info.get("buf") is not None: info["buf"].prev_rid = rid
            return True

    def _spec_alive(self, spec_id):
        """A speculative append for this id is still waiting to be accepted (reserved, not rolled back)."""
        for buf in list(self._bufs.values()):
            for r in list(buf.reservations):
                if r.active and isinstance(r.payload, dict) and r.payload.get("spec_id") == spec_id:
                    return True
        return False

    def _drop(self, spec_id):
        with self._spec_lock:
            rid = self._spec_rid.get(spec_id)
            if rid is None:
                self._spec_pending[spec_id] = "drop"; return
            info = self._reqs.get(rid)
            if info is not None and not info.get("released"):
                info["dropped"] = True

    def _match(self, rid):
        if not isinstance(rid, str):
            return None
        for k, v in self._reqs.items():
            if rid.startswith(k) and not rid[len(k):len(k) + 1].isdigit():
                logger.info("sophon duplex: matched output id %s to request %s", rid, k)
                return v
        return None

    # ---- turn decision
    def decide_output(self, *, stage_id, final_stage_id, segment_finished, segment_token_ids, segment_output_metadata, output):
        rid = getattr(output, "request_id", None)
        info = self._reqs.get(rid) or self._match(rid)
        if info is None or info["kind"] != "check":
            return None
        completion = next(iter(getattr(output, "outputs", ()) or ()), None)
        for lp in (getattr(completion, "logprobs", None) or [])[:1]:
            if not info["top"] and isinstance(lp, Mapping):
                for tid, v in lp.items():
                    k = (getattr(v, "decoded_token", None) or "").strip()
                    if k and (k not in info["top"] or v.logprob > info["top"][k]):
                        info["top"][k] = v.logprob
        if bool(getattr(output, "finished", False)) or segment_finished:
            self._finish_check(rid, info)
        return DuplexOutputDecision(action=DuplexOutputAction.DIRECT_RESPONSE,
                                    metadata={"duplex_direct_response": True, "duplex_native_decision": "listen", "model_listen": True,
                                              "listen_source": "model_listen", "sophon_check": True})

    def _finish_check(self, rid, info):
        if info.get("done"):
            return
        info["done"] = True; buf = info["buf"]; perm = info["perm"]; top = info["top"]
        z = np.array([top.get(LET[i], -1e9) for i in range(len(perm))]); z = np.exp(z - z.max()); z /= z.sum()
        p = float(z[perm.index(0)])
        if buf is not None:
            with buf.lock:
                buf.decision = (p >= info["thr"], time.monotonic()); buf.decision_ready = False; buf.inflight_t = 0.0
                logger.info("sophon timing: decision at %.3fs after silence start", time.monotonic() - buf.t_sil)
        logger.info("sophon turn check request=%s P(finished)=%.2f thr=%.2f -> %s top=%s", rid, p, info["thr"], "reply" if p >= info["thr"] else "wait",
                    {k: round(v, 2) for k, v in sorted(top.items(), key=lambda kv: -kv[1])[:5]})

    def _finish_reply(self, rid, completed):
        info = self._reqs.pop(rid, None)
        if info is None or info.get("kind") != "reply" or info["buf"] is None:
            return
        buf = info["buf"]; a = np.concatenate(info["a_audio"]) if info["a_audio"] else None
        self._finish_reply_inner(rid, info, buf, a, completed)

    def _finish_reply_inner(self, rid, info, buf, a, completed):
        if completed and not info.get("announced"): self._announce(info)
        calls = (info.get("calls") or []) if info.get("announced") else (_tools.parse_tool_calls(info.get("raw", "")) if completed else [])
        if _tools.has_tool(info.get("raw", "")):
            logger.info("sophon tool: reply %s %s -> calls %s", rid[-24:], "done" if completed else "cut (calls dropped)",
                        json.dumps([(c["name"], c["arguments"]) for c in calls], ensure_ascii=False))
        if info.get("step") is not None:                         # tool follow-up: fill the chain step instead of adding a history entry
            st = info["step"]; st["a_text"] = info["a_text"].strip() + ("" if completed else "……（被用户打断）")
            st["a_audio"] = a if completed and a is not None and a.size > SR // 10 else None
            if calls: st["tool_calls"] = calls
            logger.info("sophon reply text %s (tool follow-up, %s): %s", rid[-24:], "done" if completed else "cut", info["a_text"][:300].replace("\n", " "))
            with buf.lock:
                if completed and a is not None:
                    buf.last_play = {"t0": info.get("t_out") or time.monotonic(), "dur": a.size / SR, "entry": st, "text": info["a_text"]}
                buf.last_said = info["a_text"]
            self._run_tools(buf, info["entry"], calls)
            return
        if info.get("say") is not None:
            full, spoken = info["say"], info["a_text"]
            if completed and "".join(spoken.split()) == "".join(full.split()):
                info["a_text"] = full                                # the text as given, not the detokenized stream
            else:                                                    # keep what was actually spoken
                logger.warning("sophon say: %s after %d of %d chars: %r", "stopped early" if completed else "cut by the user",
                               len(spoken), len(full), spoken[-80:])
        entry = {"user": info["user"], "a_text": info["a_text"] + ("" if completed else "……（被用户打断）"),
                 "a_audio": a if completed and a is not None and a.size > SR // 10 else None}
        cut = buf.cut
        if not completed and cut is not None and cut.get("rid") == rid:
            entry["a_text"] = cut["said"] + "……（被用户打断）"      # keep only what was actually spoken
        if info.get("resume_entry") is not None:                   # resume: merge into the original history entry instead of adding one
            res = info["resume_entry"]; res["a_text"] = info["said"] + entry["a_text"]; res["a_audio"] = None
            with buf.lock:
                if completed and a is not None:
                    buf.last_play = {"t0": info.get("t_out") or time.monotonic(), "dur": a.size / SR, "entry": res, "text": info["a_text"], "prefix": info["said"]}
                buf.last_said = info["said"] + info["a_text"]
            logger.info("sophon reply text %s (resume, %s): %s", rid[-24:], "done" if completed else "cut", info["a_text"][:300].replace("\n", " "))
            return
        logger.info("sophon reply text %s (%s): %s", rid[-24:], "done" if completed else "cut", info["a_text"][:300].replace("\n", " "))
        dbg = os.environ.get("SOPHON_DEBUG_DIR")
        if dbg and info.get("ctx"):
            threading.Thread(target=self._dump, args=(dbg, rid, info["ctx"], completed, info["a_text"]), daemon=True).start()
        with buf.lock:
            if completed and a is not None:
                # the server generates much faster than real time: when generation "finishes" here the client is usually still playing. Record playback start and duration so history can be truncated at the actual playback position if the user cuts in
                buf.last_play = {"t0": info.get("t_out") or time.monotonic(), "dur": a.size / SR, "entry": entry, "text": info["a_text"]}
            if calls: entry["tool_calls"] = calls; entry["a_text"] = entry["a_text"].strip()
            buf.history.append(entry); buf.last_said = info["a_text"]
            while len(buf.history) > HIST_TURNS or (len(buf.history) > 1 and sum(h["user"].size for h in buf.history if h.get("user") is not None) / SR > HIST_SEC):
                buf.history.pop(0)
        self._run_tools(buf, entry, calls)

    def _run_tools(self, buf, entry, calls):
        """Execute tool calls with the session's mock executor and queue the spoken follow-up."""
        if not calls:
            return
        if buf.tool_mode == "client":
            with buf.lock:
                for c in calls: buf.call_entry[c["id"]] = entry
            return
        chain = entry.setdefault("chain", [])
        if len(chain) >= _tools.MAX_STEPS:
            logger.warning("sophon tool: more than %d tool rounds in one turn, stopping", _tools.MAX_STEPS); return
        results = [_tools.run_mock(c["name"], c["arguments"], buf.tool_mocks) for c in calls]
        logger.info("sophon tool: results %s", json.dumps(results, ensure_ascii=False)[:500])
        with buf.lock:
            chain.append({"results": results}); buf.tool_followup = True

    def _announce(self, info):
        """Client mode: as soon as the thinker has finished writing, publish its tool calls (the talker may still be speaking the lead-in)."""
        buf = info.get("buf"); info["announced"] = True
        if buf is not None and _tools.has_tool(info.get("raw", "")):
            with buf.lock: buf.prev_user = None           # it called a tool: speech right after it is a new turn, never a merge & abort of this reply
        calls = _tools.parse_tool_calls(info.get("raw", "")); info["calls"] = calls
        if not calls or buf is None or buf.tool_mode != "client" or not buf.bridge_key:
            if calls and (buf is None or buf.tool_mode != "client"): info["announced"] = False   # mock mode: parse again at the end as before
            return
        for c in calls:
            logger.info("sophon tool: client call %s %s %s", c["id"], c["name"], json.dumps(c["arguments"], ensure_ascii=False)[:200])
            self.bridge.push(buf.bridge_key, {"type": "tool_call", "call_id": c["id"], "name": c["name"], "arguments": c["arguments"], "t": time.time()})

    def _bridge_say(self, key, text):
        """A say from the gateway: speak this text verbatim as the next reply (after the user is silent) and keep it in history."""
        buf = self._keys.get(key)
        if not text.strip():
            return False
        if buf is None:                                          # no reply yet in that session: have unbound sessions announce their key
            now = time.monotonic()
            for k in [k for k, v in self._orphan_say.items() if now - v[0] > 30]: self._orphan_say.pop(k, None)
            self._orphan_say.setdefault(key, [now, []])[1].append(text.strip())
            for b in list(self._bufs.values()):
                if b.bridge_key is None: b.need_bind = True
            logger.info("sophon say: key not bound yet, queued and asked sessions to bind")
            return True
        with buf.lock: buf.say_q.append(text.strip())
        logger.info("sophon say: queued %r", text[:120])
        return True

    def _bridge_result(self, key, call_id, output, speak):
        """A result from the gateway: attach it to the call's history entry (or the latest one) and, if it should be spoken, queue a follow-up reply."""
        buf = self._keys.get(key)
        if buf is None:
            return False
        with buf.lock:
            if not buf.history:
                return False
            entry = buf.history[-1]          # the follow-up always answers from the latest turn (if the user spoke since the call, the result joins that turn)
            entry.setdefault("chain", []).append({"results": [output]})
            if speak: buf.tool_followup = True
        logger.info("sophon tool: client result call=%s speak=%s %s", call_id, speak, output[:200].replace("\n", " "))
        return True

    @staticmethod
    def _dump(dbg, rid, ctx, completed, text):
        try:          # for reproduction: full context (system prompt + history + this turn, audio stored as fp16) and reply text
            os.makedirs(dbg, exist_ok=True)
            msgs, audios = ctx; tag = f"{int(time.time())}_{rid[-12:]}"
            np.savez_compressed(os.path.join(dbg, f"ctx_{tag}.npz"), msgs=np.array(json.dumps(msgs, ensure_ascii=False)),
                                **{f"a{i}": np.asarray(a, np.float16) for i, (a, _) in enumerate(audios)})
            with open(os.path.join(dbg, "replies.jsonl"), "a") as f:
                f.write(json.dumps({"ctx": tag, "t": time.time(), "completed": completed, "text": text}, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("sophon ctx dump failed")

    def data_plane_context(self, **kwargs):
        return kwargs


_fence_floor_patch()


class _SophonSessionState(DefaultDuplexModelSessionState):
    """Every reply's prompt is rebuilt from the session's current instructions, tools and history (plan_append), so instructions
    may change mid-call: the framework's context lock is never set."""
    @property
    def context_locked(self): return False

    @context_locked.setter
    def context_locked(self, value): pass


def _same_voice_patch() -> None:
    """session.update after the first audio is refused whenever it carries a voice, even the current one (clients often resend
    their whole session). Drop a voice equal to the session's before the framework checks it; a real voice change is still refused."""
    from vllm_omni.engine.duplex import config as _c
    for cls in list(vars(_c).values()):
        if not (isinstance(cls, type) and "apply_realtime_update" in vars(cls)) or getattr(cls.apply_realtime_update, "_sophon", False):
            continue
        orig = cls.apply_realtime_update

        def apply_realtime_update(self, patch, *a, _orig=orig, **kw):
            p, cur = dict(patch), getattr(self, "voice", None)
            if cur is not None:
                if p.get("voice") == cur: p.pop("voice")
                au = p.get("audio")
                if isinstance(au, dict) and isinstance(au.get("output"), dict) and au["output"].get("voice") == cur:
                    out = dict(au["output"]); out.pop("voice"); p["audio"] = dict(au, output=out)
            return _orig(self, p, *a, **kw)
        apply_realtime_update._sophon = True
        cls.apply_realtime_update = apply_realtime_update


_same_voice_patch()
