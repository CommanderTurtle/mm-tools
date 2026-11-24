"""Sophon Realtime on a Mac, served over a WebSocket: a network device supplies the microphone and the speaker.

The wire format is the simplest one a small device can speak. Binary frames are raw PCM16 mono audio at --rate, in both directions. A text frame {"type": "interrupt"} means "stop talking now": the server sends it when the user barges in, so the client can drop audio it has buffered, and a client may send it to cut the reply itself (a wake word spoken over the reply, for example). Reply audio is sent in 20 ms frames, no further ahead of real time than --lead: the engine produces speech faster than it is spoken, and a small device has no room to hold the difference. With --state it sends what the conversation is doing as {"type": "state", ...} frames, for a status display. With --events the server also sends the engine's own events (reply_start, first_audio, reply_done, barge_in, ...) as text frames.

The speaker model is WeSpeaker's voxceleb_resnet34_LM.onnx (CC BY 4.0, huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM), needing the onnxruntime package. When it is present, the first voice heard after connecting is taken as the person who woke the device, and speech in other voices (a television, bystanders) is ignored: it starts no reply, does not keep the conversation open, and a reply it cut off is resumed. The conversation ends by itself, after --first seconds of quiet with no reply yet or --followup seconds after a reply has been played: the server sends {"type": "disconnect"} and closes, and the client goes back to waiting for its wake word. While a reply is generated, sent or played, and --echo-tail after, the microphone audio is replaced by silence (--talk-over turns that off), because a small speaker next to its microphone is heard at a level and delay no canceller here follows; a client interrupts a reply with its wake word or a text frame. The reply audio is also the reference for WebRTC echo cancellation on the incoming audio (--no-aec turns that off for a headset), so the reply is not heard as the user speaking. One conversation at a time: a second client is refused while one is connected.

    python serve.py --host ADDRESS [--port 8765] [--rate 24000] [--lang zh|en] [--voice chelsie] [--events]"""
import argparse, asyncio, collections, json, os, sys
import numpy as np, websockets
from scipy.signal import resample_poly
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
from engine import Brain, Session, SR
from aec import Echo
from adapt import InputGain, Playout

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.environ.get("SOPHON_MODELS", os.path.join(os.path.dirname(HERE), "models")))
ap.add_argument("--lang", default="zh"); ap.add_argument("--voice", default="chelsie")
ap.add_argument("--host", required=True, help="address to listen on; there is no authentication, so name one interface's address")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--rate", type=int, default=24000, help="sample rate of the client's PCM16 audio, both directions")
ap.add_argument("--events", action="store_true", help="also send the engine's events to the client as text frames")
ap.add_argument("--followup", type=float, default=6.0, help="seconds of quiet after a reply has been played before the conversation ends (0: never)")
ap.add_argument("--first", type=float, default=10.0, help="seconds of quiet after connecting, with no reply yet, before the conversation ends (0: never)")
ap.add_argument("--device-buffer", type=float, default=1.0, help="seconds of reply audio the client can hold (its playback buffer plus its queue); the server never sends further ahead than a share of this. The Voice PE firmware holds about 0.4 s")
ap.add_argument("--max-hold", type=float, default=4.0, help="the longest, in seconds, a reply may be held back before it starts playing; how long is chosen from how fast the model has been and how the link has behaved")
ap.add_argument("--mic-gain", type=float, default=None, help="a fixed multiplier for the incoming microphone audio; by default the gain follows the speech level")
ap.add_argument("--state", action="store_true", help="also send {\"type\": \"state\", \"state\": ..., \"left\": ...} text frames saying what the conversation is doing (waiting, hearing, thinking, speaking, ignoring) and how much of the quiet before it ends is left, for a client with a status display")
ap.add_argument("--talk-over", action="store_true", help="keep listening while a reply plays, so speech over it interrupts it; only for a client that removes its own echo well")
ap.add_argument("--echo-tail", type=float, default=0.8, help="seconds after the last reply audio was sent during which the microphone is still ignored: the client plays it later than it receives it, and the room rings on")
ap.add_argument("--no-aec", action="store_true", help="with --talk-over, do not cancel echo here either (the client has a headset or does its own); without --talk-over the microphone is off while the device speaks and no echo is cancelled")
ap.add_argument("--delay-ms", type=int, default=250, help="how long after the server sends a reply frame its echo reaches the microphone, as a hint for the echo canceller: the send lead, the client buffer and the room")
ap.add_argument("--speaker-model", default=os.path.join(os.path.dirname(HERE), "models", "speaker", "voxceleb_resnet34_LM.onnx"), help="WeSpeaker ONNX model; when the file exists, only the voice that spoke first after connecting is answered")
ap.add_argument("--speaker-threshold", type=float, default=0.5, help="cosine similarity to the owner's voice below which speech is treated as someone else's")
ap.add_argument("--log-events", action="store_true", help="print every engine event to the server log")
ap.add_argument("--dump-dir", help="save the audio behind each speaker verdict here, to calibrate the threshold; it is audio of the room, so keep it local and delete it afterwards")
a = ap.parse_args()

FR = 320                      # the engine takes 20 ms frames at 16 kHz
OUT_RATE = 24000              # the engine's reply audio
INTERRUPT = json.dumps({"type": "interrupt"})
DISCONNECT = json.dumps({"type": "disconnect"})
MAX_UTT_S = 15.0             # speech running longer than this no longer keeps the conversation open
FRAME_S = 0.02               # reply audio goes out in frames this long
PL = Playout(a.device_buffer, a.max_hold)    # what the link and the model have been doing, kept across conversations
gain = InputGain()                            # the microphone's level is a property of the device and the room, so it is kept across conversations too
busy = False


def to_pcm(x):
    y = np.asarray(x, np.float32)
    if a.rate != OUT_RATE: y = resample_poly(y, a.rate, OUT_RATE)
    return (np.clip(y, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


async def handle(ws):
    global busy
    if busy:
        await ws.close(1013, "one conversation at a time"); return
    busy = True
    loop = asyncio.get_running_loop(); out = asyncio.Queue()
    st = {"ends": 0.0, "pending": 0, "replying": False, "replied": False}   # what the conversation is doing, for the idle watchdog
    played = collections.deque()                         # (loop time the client starts playing a frame, that frame at the engine's rate): the echo reference

    def play(x):
        """Called by the engine's worker thread with 24 kHz reply audio, or None to stop at once (barge-in)."""
        loop.call_soon_threadsafe(out.put_nowait, None if x is None else to_pcm(x))

    def log(m):
        e = json.loads(m); ev = e.get("ev")
        if a.log_events: print(m, flush=True)
        if ev == "reply_start": st["replying"] = st["replied"] = st["fresh"] = True
        elif ev == "reply_done":
            st["replying"] = False
            if e.get("audio_s") and e.get("sec"): loop.call_soon_threadsafe(PL.observe_reply, e["audio_s"], e["sec"], bool(e.get("cancelled")))    # on the loop: the controller is not thread safe
        if a.events: loop.call_soon_threadsafe(out.put_nowait, m)

    async def send():
        step = int(a.rate * FRAME_S) * 2                    # bytes in one frame
        pending = collections.deque(); ends = 0.0           # loop time at which the audio already sent finishes playing
        while True:
            now = loop.time()
            if ends and not pending and not st["replying"] and now > ends + 1.0:      # a reply has finished and been played out
                ends = 0.0; st["ends"] = 0.0
            held = bool(pending) and ends <= now and len(pending) * FRAME_S < PL.hold() and st["replying"]    # audio is not yet far enough ahead of its generation to start playing
            wait = (0.05 if held else ends - PL.lead() - now) if pending else None
            if wait is None or wait > 0:
                try: m = await asyncio.wait_for(out.get(), wait)
                except asyncio.TimeoutError: m = b""
            else:
                m = b"" if out.empty() else out.get_nowait()
            if m is None:
                pending.clear(); played.clear(); ends = 0.0; st["ends"] = 0.0; st["pending"] = 0; kept = []      # audio not yet sent is never sent; events are
                while not out.empty():
                    q = out.get_nowait()
                    if isinstance(q, str): kept.append(q)
                await ws.send(INTERRUPT)
                for q in kept: await ws.send(q)
            elif isinstance(m, str):
                await ws.send(m)
            else:
                pending.extend(m[i:i + step] for i in range(0, len(m), step))
            held = bool(pending) and ends <= loop.time() and len(pending) * FRAME_S < PL.hold() and st["replying"]
            if pending and not held and loop.time() >= ends - PL.lead():
                f = pending.popleft(); await ws.send(f)
                if st.pop("fresh", False): ends = 0.0       # the first frame of a new reply: the time since the last one is silence between replies, not an underrun
                gap = loop.time() - ends
                if ends > 0 and gap > 0.04:                  # the reply ran dry mid-sentence: the client ran out of audio this long ago, as far as the server can tell
                    PL.observe_stall(gap)
                    if a.log_events: print(json.dumps({"ev": "tx_gap", "gap_ms": round(gap * 1000), "next_hold_s": round(PL.hold(), 2)}), flush=True)
                start = max(ends, loop.time()); ends = start + len(f) / (2 * a.rate)
                ref = np.frombuffer(f, "<i2").astype(np.float32) / 32768.0
                played.append((start, ref if a.rate == SR else resample_poly(ref, SR, a.rate).astype(np.float32)))
            st["ends"] = ends; st["pending"] = len(pending)

    async def watchdog():
        """Ends the conversation once nothing has happened for a while, so the device is back to waiting for its wake word
        and room audio (a television, other people) is no longer streamed to the model. Busy means: a reply being generated,
        sent or played, or the user speaking (for at most MAX_UTT_S, so continuous speech cannot hold it open)."""
        idle_since = loop.time(); speaking_since = None
        while True:
            await asyncio.sleep(0.25); now = loop.time()
            with sess.lock: speaking = sess.speech_ms > 0
            speaking_since = (speaking_since or now) if speaking else None
            user_busy = speaking_since is not None and now - speaking_since < MAX_UTT_S
            limit = a.followup if st["replied"] else a.first
            st["left"] = max(0.0, 1.0 - (now - idle_since) / limit) if limit > 0 else 1.0       # how much of the quiet before the conversation ends is left
            if st["replying"] or st["pending"] or now < st["ends"] or user_busy: idle_since = now; st["left"] = 1.0; continue
            if limit > 0 and now - idle_since >= limit:
                await ws.send(DISCONNECT); await ws.close(); return

    echo = Echo(a.delay_ms) if a.talk_over and not a.no_aec else None          # needed only while the microphone is open during replies
    rec = ([], []) if a.dump_dir else None
    PL.new_link()

    async def probe_link():
        """Round trips to the client, twice a second: the jitter of the link, which sets how far ahead audio can safely be sent. Also keeps the input gain
        following the speech level, and with --log-events reports what the controllers have decided every ten seconds."""
        n = 0
        while True:
            try: PL.observe_rtt(await asyncio.wait_for(await ws.ping(), 5))
            except asyncio.TimeoutError: PL.observe_rtt(5.0)
            n += 1
            if n % 2 == 0: gain.update()
            if a.log_events and n % 20 == 0:
                print(json.dumps({"ev": "link", "rtt_p95_ms": round(sorted(PL.rtt)[int(0.95 * len(PL.rtt))] * 1000) if PL.rtt else None, "lead_s": round(PL.lead(), 2),
                                  "hold_s": round(PL.hold(), 2), "gain": round(a.mic_gain if a.mic_gain is not None else gain.gain, 2)}), flush=True)
            await asyncio.sleep(0.5)

    def ring_state(now):
        """What the conversation is doing, for a client's status display."""
        if st["ends"] > 0 and now < st["ends"] + PL.lead(): return "speaking"                # audio is being sent, or still playing on the client
        if st["replying"] or st["pending"]: return "thinking"                              # the model is working, or the reply is held until enough is ready
        with sess.lock: heard, other = sess.speech_ms > 0, sess.bystander
        return "ignoring" if other else "hearing" if heard else "waiting"

    async def state_frames():
        shown = None
        while True:
            state = ring_state(loop.time()); left = round(st.get("left", 1.0), 1) if state == "waiting" else 1.0
            if (state, left) != shown:
                shown = (state, left); await ws.send(json.dumps({"type": "state", "state": state, "left": left}))
            await asyncio.sleep(0.1)

    sess = Session(brain, play, log=log, speaker=speaker, speaker_thr=a.speaker_threshold, dump_dir=a.dump_dir); sender = asyncio.ensure_future(send()); dog = asyncio.ensure_future(watchdog()); prober = asyncio.ensure_future(probe_link()); stater = asyncio.ensure_future(state_frames()) if a.state else None; buf = np.zeros(0, np.float32)
    try:
        async for m in ws:
            if isinstance(m, str):
                try: kind = json.loads(m).get("type")
                except (ValueError, AttributeError): kind = None
                if kind == "interrupt":
                    with sess.lock: sess._onset()
                continue
            x = np.frombuffer(m[: len(m) // 2 * 2], "<i2").astype(np.float32) / 32768.0
            if a.rate != SR: x = resample_poly(x, SR, a.rate).astype(np.float32)
            buf = np.concatenate([buf, x])
            while len(buf) >= FR:
                f, buf = buf[:FR], buf[FR:]
                now = loop.time(); far = np.zeros(FR, np.float32)
                gated = not a.talk_over and (st["replying"] or st["pending"] or now < st["ends"] + a.echo_tail)
                if not gated: gain.observe(float(np.sqrt(np.mean(f * f))))             # the room and the user, not the device's own voice
                f = np.clip(f * (a.mic_gain if a.mic_gain is not None else gain.gain), -1.0, 1.0)
                while played and played[0][0] + FRAME_S <= now: played.popleft()
                if played and played[0][0] <= now: far[: min(FR, len(played[0][1]))] = played[0][1][:FR]
                if rec is not None: rec[0].append(f); rec[1].append(far)
                if gated: f = np.zeros(FR, np.float32)
                sess.feed(f if echo is None else echo.process(f, far), far)
    except websockets.ConnectionClosed:
        pass
    finally:
        with sess.lock:
            if sess.reply is not None: sess.reply["cancel"] = True
        if rec is not None and rec[0]:
            import time as _t; os.makedirs(a.dump_dir, exist_ok=True); stamp = _t.strftime("%H%M%S")
            for name, parts in (("mic", rec[0]), ("far", rec[1])):
                np.save(os.path.join(a.dump_dir, f"{stamp}-{name}.npy"), np.concatenate(parts).astype(np.float32))
        sender.cancel(); dog.cancel(); prober.cancel(); stater and stater.cancel(); busy = False


async def main():
    async with websockets.serve(handle, a.host, a.port, max_size=1 << 20):
        print(f"Listening on ws://{a.host}:{a.port} (PCM16 mono at {a.rate} Hz). Ctrl+C to quit.", flush=True)
        await asyncio.Future()


speaker = None
if os.path.exists(a.speaker_model):
    from speaker import Speaker
    speaker = Speaker(a.speaker_model)
print("Loading the model (first time ~1 min)...", flush=True)
brain = Brain(a.root, lang=a.lang, voice=a.voice, sys_prompt=os.path.join(HERE, "system_prompt.json")); brain.warmup()
try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("\nBye.", flush=True)
    os._exit(0)          # skip interpreter teardown (MLX streams owned by the worker thread would complain)
