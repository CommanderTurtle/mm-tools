# SPDX-License-Identifier: AGPL-3.0-only
import asyncio
import math
import soxr
import os
import shutil
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from threading import RLock
from uuid import UUID, uuid4
from urllib.parse import urlsplit

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from inworld.voice_isolation import VoiceIsolator, audio
from live import LiveSession, _plot
from state import LANE_STAMP

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT.parents[1]/"models/inworld-isolator"
SAMPLES = ROOT/"voice-isolator/demo/examples"
models, lock = {}, RLock()
sessions = asyncio.Semaphore(2)
offline_lock = asyncio.Lock()
store = ROOT/".outputs"
store.mkdir(mode=0o700,exist_ok=True)
sweep_lock = RLock()
def sweep():
    with sweep_lock:
        for folder in store.iterdir():
            if folder.is_symlink() or not folder.is_dir(): continue
            try: UUID(folder.name)
            except ValueError: continue
            if folder.stat().st_mtime < time.time()-3600: shutil.rmtree(folder,ignore_errors=True)
inference_locks = {False:RLock(),True:RLock()}
mps_lock = RLock()
@contextmanager
def device_slot(engine,live):
    guard = mps_lock if engine.device.type=="mps" else inference_locks[live]
    if not guard.acquire(timeout=2 if live else 30): raise ValueError("Audio device busy. Retry when the current recording finishes.")
    try: yield
    finally: guard.release()
class LiveLock:
    def __init__(self,engine): self.engine=engine
    def __enter__(self):
        self.slot=device_slot(self.engine,True)
        return self.slot.__enter__()
    def __exit__(self,*exc): return self.slot.__exit__(*exc)
@asynccontextmanager
async def lifespan(_):
    sweep()
    yield
    models.clear()
app = FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None)
def model(live=False):
    with lock:
        if live not in models:
            path = WEIGHTS/("denoiser_realtime.safetensors" if live else "denoiser.safetensors")
            if not path.is_file(): raise ValueError("Select inworld-isolator in the model picker first.")
            models[live] = VoiceIsolator(str(path),mode="streaming" if live else "offline",lookahead_frames=2 if live else 0)
        return models[live]

@app.get("/")
def page(): return FileResponse(ROOT/"studio.html")
@app.get("/capture.js")
def worklet(): return FileResponse(ROOT/"capture.js",media_type="application/javascript")
@app.get("/samples")
def samples(): return sorted(p.stem for p in SAMPLES.glob("sample_*.wav"))
@app.get("/audio/{token}/{track}")
def asset(token: UUID,track: str):
    if track not in {n+s for n in ("original","voice","background") for s in (".wav","-float.wav")}: raise HTTPException(404)
    path = store/str(token)/track
    if not path.is_file(): raise HTTPException(404)
    return FileResponse(path,media_type="audio/wav",filename=track,content_disposition_type="inline")

def publish(original,voice):
    sweep()
    root = store
    token = str(uuid4())
    folder = root/token
    folder.mkdir()
    plots = {}
    for key,y in (("original",original),("voice",voice),("background",original-voice)):
        sf.write(folder/(key+".wav"),y,48000,subtype="PCM_16")
        sf.write(folder/(key+"-float.wav"),y,48000,subtype="FLOAT")
        plots[key] = _plot(y)
    return {"token":token,"plots":plots,"duration":len(voice)/48000}

def separate(handle):
    info = sf.info(handle)
    if not 0 < info.duration <= 600: raise ValueError("Select a recording up to ten minutes.")
    if hasattr(handle,"seek"): handle.seek(0)
    data,rate = sf.read(handle,dtype="float32",always_2d=True)
    mono = audio.to_mono(data)
    if not np.isfinite(mono.numpy()).all(): raise ValueError("Input contains non-finite audio.")
    original = (soxr.resample(mono.numpy(),rate,48000).astype(np.float32)
                if 48000//math.gcd(rate,48000)>1000 else audio.resample(mono,rate,48000).numpy())
    begin = time.perf_counter()
    engine = model()
    with device_slot(engine,False): voice = engine.isolate(original,48000)
    return {**publish(original,voice),"seconds":time.perf_counter()-begin,"device":str(engine.device)}

@app.post("/isolate")
async def isolate(file: UploadFile | None = File(None),sample: str = Form("")):
    if file is None and sample not in samples(): raise HTTPException(400,"Choose a sample or recording.")
    if file and (file.size or 0)>300_000_000: raise HTTPException(413,"Maximum upload: 300 MB.")
    try:
        async with offline_lock:
            return await asyncio.to_thread(separate,file.file if file else SAMPLES/(sample+".wav"))
    except (ValueError,RuntimeError) as e: raise HTTPException(422,str(e)) from e
    finally:
        if file: await file.close()

@app.websocket("/live")
async def live(ws: WebSocket):
    if urlsplit(ws.headers.get("origin","")).netloc!=ws.headers.get("host") or sessions.locked():
        await ws.close(code=1008); return
    async with sessions:
        await ws.accept()
        take = None
        try:
            rate = int(await asyncio.wait_for(ws.receive_text(),15))
            if not 8000<=rate<=192000: raise ValueError("Invalid capture rate.")
            engine = await asyncio.to_thread(model,True)
            take = await asyncio.to_thread(LiveSession,engine,model_lock=LiveLock(engine))
            await ws.send_json({"ready":True,"device":str(engine.device)})
            while not take.finished:
                msg = await asyncio.wait_for(ws.receive(),30)
                if msg["type"]=="websocket.disconnect": return
                if msg.get("text")=="stop": break
                block = msg.get("bytes")
                if not block or len(block)>rate*4 or len(block)%4: raise ValueError("Invalid PCM block.")
                update = await asyncio.to_thread(take.push,(rate,np.frombuffer(block,dtype="<f4")))
                await ws.send_json({"view":update.visualization,"seconds":update.input_seconds,"compute":update.processing_seconds})
            _,voice = await asyncio.to_thread(take.finish)
            _,original = take.recorded()
            if len(voice): await ws.send_json({"result":await asyncio.to_thread(publish,original,voice)})
        except WebSocketDisconnect: pass
        except (RuntimeError,ValueError,asyncio.TimeoutError) as e:
            try: await ws.send_json({"error":str(e) or "Microphone timed out."})
            except RuntimeError: pass
        finally:
            if take: take.reset()
            try: await ws.close()
            except RuntimeError: pass
print(f"source {LANE_STAMP}")
if __name__=="__main__": uvicorn.run(app,host="127.0.0.1",port=int(os.environ["GRADIO_SERVER_PORT"]))
