"""Sophon Realtime on a Mac: talk to it with a microphone and speakers (or a headset).
Hearing, turn-taking and replying are all our own model, running locally on MLX. Echo cancellation is on by default, so you can
talk over it on speakers; with a headset you can turn it off (--no-aec).

    python live.py [--lang zh|en] [--voice chelsie] [--no-aec] [--in DEVICE] [--out DEVICE] [--list]"""
import argparse, collections, json, os, queue, sys, threading, time
import numpy as np, sounddevice as sd
from scipy.signal import resample_poly
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
from engine import Brain, Session, SR

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.environ.get("SOPHON_MODELS", os.path.join(os.path.dirname(HERE), "models")))
ap.add_argument("--lang", default="zh"); ap.add_argument("--voice", default="chelsie")
ap.add_argument("--no-aec", action="store_true"); ap.add_argument("--delay-ms", type=int, default=0, help="extra speaker->mic delay hint for the echo canceller")
ap.add_argument("--in", dest="din", default=None); ap.add_argument("--out", dest="dout", default=None)
ap.add_argument("--list", action="store_true"); ap.add_argument("--log", default=None)
a = ap.parse_args()
if a.list:
    print(sd.query_devices()); sys.exit(0)
try:
    sd.check_input_settings(device=a.din, samplerate=SR, channels=1)
except Exception as exc:  # noqa: BLE001
    sys.exit(f"No usable microphone ({exc}). Plug one in (a Mac Studio has none built in), or pick one with --in; see --list.")

FR = 320                                                    # 20 ms at 16 kHz
spk = collections.deque(); spk_lock = threading.Lock()      # 16 kHz samples still to be played (the reference for the echo canceller)
played = queue.Queue(maxsize=500)                           # what the speaker actually played, 20 ms at a time, in playback order
mic_q = queue.Queue(maxsize=500)


def play(x):
    """Called by the engine with 24 kHz reply audio, or None to stop at once (barge-in)."""
    with spk_lock:
        if x is None: spk.clear(); return
        spk.extend(resample_poly(np.asarray(x, np.float32), SR, 24000).astype(np.float32))


def out_cb(outdata, frames, t, status):
    with spk_lock:
        n = min(frames, len(spk)); y = np.fromiter((spk.popleft() for _ in range(n)), np.float32, n) if n else np.zeros(0, np.float32)
    y = np.pad(y, (0, frames - n)); outdata[:, 0] = y
    try: played.put_nowait(y.copy())
    except queue.Full: pass


def in_cb(indata, frames, t, status):
    try: mic_q.put_nowait(indata[:, 0].copy())
    except queue.Full: pass


log = open(a.log, "a") if a.log else None
def show(m):
    if log: log.write(m + "\n"); log.flush()
    e = json.loads(m); k = e["ev"]
    if k == "reply_start": print("\n[thinking]", flush=True)
    elif k == "first_audio": print(f"[speaking, {e['after_silence_s']:.2f}s after you stopped]", flush=True)
    elif k == "reply_done": print("Sophon:", e["text"], flush=True)
    elif k in ("barge_in", "merge_abort"): print("[you cut in]", flush=True)
    elif k == "error": print("[error]", e.get("err"), flush=True)


print("Loading the model (first time ~1 min)...", flush=True)
t0 = time.time(); brain = Brain(a.root, lang=a.lang, voice=a.voice, sys_prompt=os.path.join(HERE, "system_prompt.json")); brain.warmup()
print(f"Ready in {time.time() - t0:.0f}s. Speak whenever you like; Ctrl+C to quit.", flush=True)
sess = Session(brain, play, log=show)
echo = None
if not a.no_aec:
    from aec import Echo
    ins = sd.InputStream(device=a.din, samplerate=SR, channels=1, blocksize=FR, dtype="float32", callback=in_cb)
    outs = sd.OutputStream(device=a.dout, samplerate=SR, channels=1, blocksize=FR, dtype="float32", callback=out_cb)
    lat = int(round((ins.latency + outs.latency) * 1000)) + a.delay_ms
    echo = Echo(lat); print(f"Echo cancellation on (device latency {lat} ms{', over 500 ms: Bluetooth/AirPlay/HDMI output? a headset works best with --no-aec' if lat > 500 else ''}).", flush=True)
else:
    ins = sd.InputStream(device=a.din, samplerate=SR, channels=1, blocksize=FR, dtype="float32", callback=in_cb)
    outs = sd.OutputStream(device=a.dout, samplerate=SR, channels=1, blocksize=FR, dtype="float32", callback=out_cb)
buf = np.zeros(0, np.float32); fbuf = np.zeros(0, np.float32)
with ins, outs:
    try:
        while True:
            x = mic_q.get(); buf = np.concatenate([buf, x])
            while len(buf) >= FR:
                f, buf = buf[:FR], buf[FR:]
                if echo is not None:
                    while len(fbuf) < FR:
                        try: fbuf = np.concatenate([fbuf, played.get(timeout=0.05)])
                        except queue.Empty: fbuf = np.concatenate([fbuf, np.zeros(FR, np.float32)])
                    far, fbuf = fbuf[:FR], fbuf[FR:]
                    sess.feed(echo.process(f, far), far=far)
                else:
                    sess.feed(f)
    except KeyboardInterrupt:
        print("\nBye.", flush=True)
        os._exit(0)          # skip interpreter teardown (MLX streams owned by the worker thread would complain)
