"""Engine adapters. Every adapter turns one engine session into the same small event stream the gateway understands:

  {"t": "audio", "pcm": np.float32 mono, "sr": int, "rid": str|None}   assistant audio
  {"t": "text", "delta": str, "rid": str|None}                         assistant transcript (spoken text only, no tool calls)
  {"t": "done", "rid": str|None, "status": str}                        a reply finished (completed / cancelled ...)
  {"t": "cancel"}                                                      barge-in: drop audio not yet played
  {"t": "tool_call", "call_id", "name", "arguments"}                   the model called a tool (e.g. delegate)
  {"t": "error", "message": str}

and offers: send_audio(f32 16 kHz), set_instructions(text) -> (ok, error), tool_result(call_id, output, speak), say(text), close()."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid

import aiohttp
import numpy as np
import websockets

log = logging.getLogger("sophon_live.adapter")


class VllmOmniAdapter:
    """Sophon-Realtime served by vLLM-Omni (/v1/realtime + the plugin's client tool bridge)."""

    def __init__(self, url: str, bridge_url: str):
        self.url, self.bridge_url = url, bridge_url.rstrip("/")
        self.key = "bk_" + uuid.uuid4().hex[:16]
        self.ws = None
        self.q: asyncio.Queue = asyncio.Queue()
        self.cfg: dict = {}
        self._tasks: list = []
        self._http: aiohttp.ClientSession | None = None
        self._closed = False
        self._acks: dict = {}                 # session.update event_id -> future (ok, error)

    async def start(self, *, instructions: str, voice: str, lang: str, tools: list, history: list) -> None:
        self.ws = await websockets.connect(self.url, max_size=None, compression=None, open_timeout=15)
        self.cfg = {"modalities": ["audio", "text"], "input_audio_format": "pcm_f32le", "output_audio_format": "pcm16", "voice": voice,
                    "instructions": instructions,
                    "turn_detection": {"type": "server_vad", "threshold": 0.5, "silence_duration_ms": 500, "min_speech_duration_ms": 250, "prefix_padding_ms": 300},
                    "extra_body": {"auto_response": True, "language": lang, "tools": tools, "tool_mode": "client", "bridge_key": self.key, "history": history}}
        await self.ws.send(json.dumps({"type": "session.update", "session": self.cfg}))
        while True:
            e = json.loads(await asyncio.wait_for(self.ws.recv(), 30))
            if e.get("type") == "session.created":
                break
            if e.get("type") == "error":
                raise RuntimeError(f"engine refused the session: {e.get('error')}")
        self._http = aiohttp.ClientSession()
        self._tasks = [asyncio.create_task(self._recv()), asyncio.create_task(self._poll())]

    async def send_audio(self, f32: np.ndarray) -> None:
        if self.ws is None or self._closed:
            return
        await self.ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(f32.astype("<f4").tobytes()).decode(),
                                       "format": "pcm_f32le", "sample_rate_hz": 16000}))

    async def set_instructions(self, text: str) -> tuple[bool, dict | None]:
        """Change instructions / tools mid-call. Sends only what may change (never the voice: the engine refuses any voice in
        session.update once audio has started) and waits for the engine's answer: (True, None) or (False, error)."""
        self.cfg["instructions"] = text
        eid = "evt_" + uuid.uuid4().hex[:16]
        fut = asyncio.get_running_loop().create_future(); self._acks[eid] = fut
        try:
            await self.ws.send(json.dumps({"type": "session.update", "event_id": eid,
                                           "session": {"instructions": text, "extra_body": self.cfg["extra_body"]}}))
            return await asyncio.wait_for(fut, 15)
        except asyncio.TimeoutError:
            return False, {"code": "engine_timeout", "message": "the engine did not confirm session.update within 15 s"}
        finally:
            self._acks.pop(eid, None)

    async def say(self, text: str) -> bool:
        """Speak this text verbatim in the session voice as the next reply (it is kept in the conversation history)."""
        try:
            async with self._http.post(f"{self.bridge_url}/bridge/{self.key}/say", json={"text": text}, timeout=aiohttp.ClientTimeout(total=10)) as r:
                return r.status == 200
        except Exception as e:  # noqa: BLE001
            log.warning("say failed: %s", e)
            return False

    async def tool_result(self, call_id: str | None, output: str, speak: bool = True) -> bool:
        try:
            async with self._http.post(f"{self.bridge_url}/bridge/{self.key}/result", json={"call_id": call_id, "output": output, "speak": speak},
                                       timeout=aiohttp.ClientTimeout(total=10)) as r:
                return r.status == 200
        except Exception as e:  # noqa: BLE001
            log.warning("tool_result failed: %s", e)
            return False

    async def _recv(self) -> None:
        try:
            async for raw in self.ws:
                e = json.loads(raw); t = e.get("type", "")
                rid = e.get("response_id") or (e.get("response") or {}).get("id")
                if t == "response.output_audio.delta":
                    b = base64.b64decode(e.get("delta") or "")
                    if b:
                        pcm = np.frombuffer(b, "<i2").astype(np.float32) / 32768.0
                        await self.q.put({"t": "audio", "pcm": pcm, "sr": int(e.get("sample_rate_hz") or 24000), "rid": rid})
                elif t in ("response.output_audio_transcript.delta", "response.output_text.delta", "response.text.delta"):
                    if e.get("delta"):
                        await self.q.put({"t": "text", "delta": e["delta"], "rid": rid})
                elif t == "response.done":
                    await self.q.put({"t": "done", "rid": rid, "status": (e.get("response") or {}).get("status", "completed")})
                elif t == "audio.cancelled":
                    await self.q.put({"t": "cancel"})
                elif t == "session.updated":
                    fut = next((f for f in self._acks.values() if not f.done()), None)
                    if fut is not None: fut.set_result((True, None))
                elif t == "error":
                    err = e.get("error") or {}
                    fut = self._acks.get(err.get("event_id") or e.get("event_id"))
                    if fut is not None and not fut.done():
                        fut.set_result((False, err)); continue
                    await self.q.put({"t": "error", "message": json.dumps(e.get("error"), ensure_ascii=False)[:300]})
        except websockets.ConnectionClosed:
            pass
        finally:
            if not self._closed:
                await self.q.put({"t": "error", "message": "engine connection closed"})

    async def _poll(self) -> None:
        while not self._closed:
            try:
                async with self._http.get(f"{self.bridge_url}/bridge/{self.key}/events?wait=20", timeout=aiohttp.ClientTimeout(total=30)) as r:
                    j = await r.json()
                for ev in j.get("events", []):
                    if ev.get("type") == "tool_call":
                        await self.q.put({"t": "tool_call", "call_id": ev.get("call_id"), "name": ev.get("name"), "arguments": ev.get("arguments") or {}})
            except asyncio.CancelledError:
                return
            except Exception as e:  # noqa: BLE001
                if not self._closed:
                    log.warning("bridge poll: %s", e); await asyncio.sleep(1)

    async def close(self) -> None:
        self._closed = True
        for t in self._tasks:
            t.cancel()
        try:
            if self.ws is not None:
                await self.ws.close()
        except Exception:  # noqa: BLE001
            pass
        if self._http is not None:
            await self._http.close()
