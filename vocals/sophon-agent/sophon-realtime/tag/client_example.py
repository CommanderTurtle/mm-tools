"""Minimal client: send a WAV (default: examples/hello_zh.wav), stream back text + audio, save reply.wav.
usage: python client_example.py [--url http://127.0.0.1:8000] [--wav my.wav] [--lang zh|en] [--voice chelsie|vivian_zh|brand_zh|brand_en|ryan_en]"""
import argparse, base64, io, json, os, time, urllib.request
import numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--wav", default=os.path.join(HERE, "examples", "hello_zh.wav"))
ap.add_argument("--lang", default="zh"); ap.add_argument("--voice", default=""); ap.add_argument("--out", default="reply.wav"); ap.add_argument("--quiet", action="store_true")
A = ap.parse_args()
SYS = json.load(open(os.path.join(HERE, "config", "system_prompt.json"), encoding="utf-8"))
model = json.load(urllib.request.urlopen(A.url + "/v1/models"))["data"][0]["id"]
body = {"model": model, "stream": True, "modalities": ["text", "audio"], "language": A.lang, "max_tokens": 300, "temperature": 0.7, "top_p": 0.9,
        "messages": [{"role": "system", "content": SYS[A.lang]},
                     {"role": "user", "content": [{"type": "text", "text": "<|audio|>"}, {"type": "input_audio", "input_audio": {"data": base64.b64encode(open(A.wav, "rb").read()).decode(), "format": "wav"}}]}]}
if A.voice: body["voice"] = A.voice
req = urllib.request.Request(A.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time(); first = None; text = ""; pcm = []
with urllib.request.urlopen(req) as r:
    for line in r:
        s = line.decode().strip()
        if not s.startswith("data:") or s[5:].strip() == "[DONE]": continue
        d = json.loads(s[5:])
        for ch in d.get("choices", []):
            c = (ch.get("delta") or {}).get("content")
            if not c: continue
            if d.get("modality") == "audio":
                w, sr = sf.read(io.BytesIO(base64.b64decode(c)), dtype="float32"); pcm.append(w); first = first or time.time() - t0
            else:
                text += c
sf.write(A.out, np.concatenate(pcm) if pcm else np.zeros(1, np.float32), 24000)
if not A.quiet:
    print("reply text (sound tags such as [laughter] are performed, not read):", text)
    print(f"first audio {first:.2f}s, saved {A.out}" if first else "no audio received")
