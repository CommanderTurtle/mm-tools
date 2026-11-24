# SPDX-License-Identifier: AGPL-3.0-only
import asyncio
import base64
import importlib.util
import io
import json
import os
import re
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit
import httpx
import soundfile as sf
import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, StreamingResponse
from websockets.asyncio.client import connect
from state import LANE_STAMP

ROOT = Path(__file__).resolve().parent
TAG = ROOT/"sophon-realtime/tag"
MODEL = ROOT.parents[1]/"models/sophon-agent"
LOCAL = ROOT/".state"
VOICES = LOCAL/"voices"
PORT = int(os.getenv("SOPHON_PORT","8498"))
BACK = f"http://127.0.0.1:{PORT}"
SYS = json.loads((TAG/"config/system_prompt.json").read_text())
@asynccontextmanager
async def lifecycle(app):
    quant = json.loads((MODEL/"thinker/config.json").read_text()).get("quantization_config") or {}
    index = MODEL/"thinker/model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values()) if index.is_file() else set()
    if (quant.get("quant_method") != "compressed-tensors" or quant.get("format") != "nvfp4-pack-quantized"
            or not shards or not all((MODEL/"thinker"/s).is_file() for s in shards)
            or not (MODEL/"config.json").is_file() or not (MODEL/"talker/flow.pt").is_file()):
        raise RuntimeError("Select sophon-agent in the model picker.")
    for port in (PORT,PORT+700):
        with socket.socket() as probe: probe.bind(("127.0.0.1",port))
    VOICES.mkdir(mode=0o700,parents=True,exist_ok=True)
    for p in (MODEL/"voices").iterdir():
        dest = VOICES/p.name
        if not dest.exists(): dest.symlink_to(p)
    cfg = yaml.safe_load((TAG/"config/sophon_realtime_tag_1gpu.yaml").read_text())
    cfg["duplex_session"]["server_vad_model_path"]=str(TAG/"config/silero_vad.onnx")
    if custom:=os.getenv("SOPHON_STAGE_MEMORY"):
        vals = list(map(float,custom.split(",")))
        if len(vals)!=3 or any(not 0<v<=1 for v in vals): raise ValueError("SOPHON_STAGE_MEMORY expects three fractions.")
        for stage,value in zip(cfg["stages"],vals): stage["gpu_memory_utilization"]=value
    deploy = LOCAL/"deploy.yaml"
    deploy.write_text(yaml.safe_dump(cfg,sort_keys=False))
    env = {**os.environ,"SOPHON_MODEL":str(MODEL),"SOPHON_PORT":str(PORT),"SOPHON_BRIDGE_PORT":str(PORT+700),
           "SOPHON_VOICES":str(VOICES),"FLASHINFER_WORKSPACE_BASE":str(LOCAL),"COSYVOICE3_BATCH_FLOW":"1",
           "SOPHON_FLOW_STEPS":"4","SOPHON_FLOW_CFG":"0","SOPHON_FLOW_ONNX":str(MODEL/"talker/flow.decoder.estimator.fp32.onnx"),
           "COSYVOICE3_TRT":str(int(importlib.util.find_spec("tensorrt") is not None))}
    proc = await asyncio.create_subprocess_exec(sys.executable,str(ROOT.parents[1]/"models/studio-child.py"),str(Path(sys.executable).with_name("vllm")),"serve",str(MODEL),
        "--omni","--host","127.0.0.1","--port",str(PORT),"--trust-remote-code","--init-timeout","3000",
        "--stage-init-timeout","3000","--deploy-config",str(deploy),env=env,cwd=TAG,stdin=asyncio.subprocess.PIPE,start_new_session=True)
    app.state.proc=proc
    try: yield
    finally:
        proc.stdin.close()
        await proc.wait()

app=FastAPI(lifespan=lifecycle,docs_url=None,redoc_url=None)
@app.middleware("http")
async def local_post(req:Request,call_next):
    from fastapi.responses import JSONResponse
    if req.method=="POST" and req.headers.get("origin") and urlsplit(req.headers["origin"]).netloc!=req.headers.get("host"):
        return JSONResponse({"detail":"Origin mismatch"},403)
    return await call_next(req)

def audio_info(buf):
    try: return sf.info(buf)
    except (RuntimeError,ValueError) as exc: raise HTTPException(400,"Unsupported or malformed audio.") from exc
@app.get("/")
def page(): return FileResponse(ROOT/"studio.html")
@app.get("/mic.js")
def mic(): return FileResponse(ROOT/"mic.js",media_type="application/javascript")
@app.get("/state")
async def state():
    ready=False
    if app.state.proc.returncode is None:
        try:
            async with httpx.AsyncClient(timeout=2,trust_env=False) as c: ready=(await c.get(BACK+"/health")).is_success
        except httpx.HTTPError: pass
    return {"state":"ready" if ready else "starting" if app.state.proc.returncode is None else "engine exited; see console",
            "voices":sorted({p.stem for p in VOICES.iterdir() if p.suffix in {".pt",".wav",".flac"}})}
@app.post("/voice")
async def voice(name:str=Form(...),file:UploadFile=File(...)):
    if not re.fullmatch("[a-z0-9][a-z0-9_-]{0,46}",name): raise HTTPException(400,"Use a short lowercase voice name.")
    if (file.size or 0)>10_000_000: raise HTTPException(413)
    buf=io.BytesIO(await file.read())
    info=audio_info(buf)
    if info.samplerate<16000 or not 3<=info.duration<=10: raise HTTPException(400,"Use 3–10 seconds of single-speaker audio, 16 kHz or higher.")
    buf.seek(0)
    data,rate=sf.read(buf,dtype="float32")
    path=VOICES/(name+".wav")
    if any(VOICES.glob(name+".*")): raise HTTPException(409,"Name already in use.")
    with path.open("xb") as out: sf.write(out,data,rate,format="WAV")
    return {"voice":name}
@app.websocket("/call")
async def relay(ws:WebSocket):
    if urlsplit(ws.headers.get("origin","")).netloc!=ws.headers.get("host"): await ws.close(code=1008); return
    await ws.accept()
    try:
        async with connect(f"ws://127.0.0.1:{PORT}/v1/realtime",max_size=16<<20,proxy=None) as native:
            async def up():
                while True: await native.send(await ws.receive_text())
            async def down():
                async for message in native: await ws.send_text(message)
            tasks=[asyncio.create_task(f()) for f in (up,down)]
            try: await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
            finally:
                for t in tasks: t.cancel()
                await asyncio.gather(*tasks,return_exceptions=True)
    except (OSError,TimeoutError) as exc: await ws.send_json({"type":"error","message":str(exc)})
    finally:
        try: await ws.close()
        except RuntimeError: pass
@app.post("/recording")
async def recording(file:UploadFile=File(...),voice:str=Form("chelsie"),language:str=Form("en")):
    if (file.size or 0)>20_000_000: raise HTTPException(413)
    b=io.BytesIO(await file.read())
    info=audio_info(b)
    if info.duration>180: raise HTTPException(400,"Select up to three minutes.")
    b.seek(0)
    audio,sr=sf.read(b,dtype="float32");wav=io.BytesIO();sf.write(wav,audio,sr,format="WAV",subtype="PCM_16")
    async def tokens():
        async with httpx.AsyncClient(timeout=httpx.Timeout(300,connect=8),trust_env=False) as client:
            model=(await client.get(BACK+"/v1/models")).json()["data"][0]["id"]
            msg=[{"role":"system","content":SYS.get(language,SYS["en"])},{"role":"user","content":[{"type":"text","text":"<|audio|>"},{"type":"input_audio","input_audio":{"format":"wav","data":base64.b64encode(wav.getvalue()).decode()}}]}]
            async with client.stream("POST",BACK+"/v1/chat/completions",json=dict(model=model,voice=voice,language=language,messages=msg,
                    modalities=["text","audio"],stream=True,max_tokens=300,temperature=.7,top_p=.9)) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line.startswith("data: "): yield line[6:]+"\n"
    return StreamingResponse(tokens(),media_type="application/x-ndjson")
print(f"source {LANE_STAMP}")
if __name__=="__main__": uvicorn.run(app,host="127.0.0.1",port=int(os.environ["GRADIO_SERVER_PORT"]))
