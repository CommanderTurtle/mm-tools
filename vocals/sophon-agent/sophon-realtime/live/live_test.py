"""GPT-Live protocol test client (behaves like duplex-system's tcpV3): one session per case, real-time audio, mock backend answers."""
import argparse, asyncio, base64, json, os, time, wave
import numpy as np, websockets
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="ws://127.0.0.1:8780/v1/live/sessions"); ap.add_argument("--cases", default="cases/cases.json")
ap.add_argument("--out", default="out"); ap.add_argument("--only", default=""); A = ap.parse_args()
RESULTS = {"zh_time": "现在是下午三点二十分。", "zh_weather": "上海明天小雨，气温14到19度，出门记得带伞。", "zh_light": "客厅的灯已经关掉了。",
           "zh_memory": "用户上次说最喜欢吃火锅，尤其是麻辣锅底。", "zh_news": "今天的科技新闻：一家公司发布了新的开源语音模型，可以实时全双工对话。",
           "en_time": "It's 3:20 in the afternoon.", "en_weather": "Melbourne tomorrow: light rain, 14 to 19 degrees."}
PERSONA = {"zh": "你是小智，一个温暖、幽默的语音伙伴，说话简短自然。", "en": "You are Kai, a warm and witty voice companion. Keep replies short and natural."}
RATE = 16000

def load(path):
    w = wave.open(path); a = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32); sr = w.getframerate()
    n = int(len(a) * RATE / sr); return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.int16)

async def one(case):
    ws = await websockets.connect(A.url, max_size=None, additional_headers={"Authorization": "Bearer test"})
    await ws.send(json.dumps({"type": "session.start", "session": {"model": "gpt-live-1", "audio": {"format": {"type": "audio/pcm", "rate": RATE}, "output": {"voice": "marin"}},
                                                                    "delegation": {"type": "client"}, "store": False, "instructions": PERSONA[case["lang"]]}}))
    R = {"id": case["id"], "expect": case["expect"], "deleg": [], "out_text": "", "in_text": "", "ev": {}, "errors": []}
    out = bytearray(); t0 = None; eos = None; first_voice = None
    async def recv():
        nonlocal first_voice
        async for raw in ws:
            e = json.loads(raw); t = e["type"]; R["ev"][t] = R["ev"].get(t, 0) + 1
            if t == "session.output_audio.delta":
                b = base64.b64decode(e["delta"]); out.extend(b)
                if eos and first_voice is None and np.abs(np.frombuffer(b, "<i2")).max() > 600: first_voice = time.monotonic() - eos
            elif t == "session.output_transcript.delta": R["out_text"] += e["delta"]
            elif t == "session.input_transcript.delta": R["in_text"] += e["delta"]
            elif t == "session.delegation.created":
                d = e["delegation"]; R["deleg"].append({"t": round(time.monotonic() - eos, 2) if eos else None, "task": d.get("task"), "in_text_at": R["in_text"]})
                async def answer(did=d["id"]):
                    await asyncio.sleep(1.0)
                    await ws.send(json.dumps({"type": "session.commentary.append", "content": RESULTS.get(case["id"], "好的，已经处理好了。"), "delegation_id": did}))
                asyncio.create_task(answer())
            elif t == "error": R["errors"].append(e["error"])
            elif t == "session.started": R["session"] = e["session"]["id"]
    rt = asyncio.create_task(recv())
    while "session" not in R and not R["errors"]: await asyncio.sleep(0.05)
    a = load(f"cases/{case['id']}.wav"); CH = RATE // 10
    a = np.concatenate([np.zeros(RATE // 2, np.int16), a, np.zeros(RATE * 14, np.int16)])
    raw = load(f"cases/{case['id']}.wav"); fr = raw[: len(raw) // 320 * 320].reshape(-1, 320).astype(np.float32)
    lv = np.flatnonzero(np.sqrt((fr ** 2).mean(1)) > 300); last = (lv[-1] + 1) * 320 if lv.size else len(raw)
    t0 = time.monotonic(); speech_end = (RATE // 2 + last) / RATE
    for i in range(0, len(a), CH):
        d = t0 + i / RATE - time.monotonic()
        if d > 0: await asyncio.sleep(d)
        if eos is None and i / RATE >= speech_end: eos = time.monotonic()
        await ws.send(json.dumps({"type": "session.input_audio.append", "audio": base64.b64encode(a[i:i + CH].tobytes()).decode()}))
    await ws.send(json.dumps({"type": "session.close"})); await asyncio.sleep(0.5); rt.cancel()
    os.makedirs(A.out, exist_ok=True)
    w = wave.open(f"{A.out}/{case['id']}.wav", "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(bytes(out)); w.close()
    R["first_voice_s"] = round(first_voice, 2) if first_voice else None; R["got"] = "delegate" if R["deleg"] else "none"
    return R

async def main():
    cases = json.load(open(A.cases))
    if A.only: cases = [c for c in cases if c["id"] in A.only.split(",")]
    res = []
    for c in cases:
        r = await one(c); res.append(r)
        print(json.dumps({k: r[k] for k in ("id", "expect", "got", "first_voice_s", "in_text", "out_text", "deleg", "errors")}, ensure_ascii=False), flush=True)
    ok = sum(r["expect"] == r["got"] for r in res); print(f"DELEGATION ACC {ok}/{len(res)}")
    json.dump(res, open(f"{A.out}/results.json", "w"), ensure_ascii=False, indent=1)
asyncio.run(main())
