# SPDX-License-Identifier: AGPL-3.0-only
"""Own one patched AWQ engine and scope its media access to disposable studio files."""
import asyncio
from contextlib import asynccontextmanager, contextmanager, suppress
from collections import Counter
import fcntl
import json
import mimetypes
import os
from pathlib import Path
import secrets
import shutil
import socket
import sys
import tempfile
import time

import httpx
import uvicorn
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from state import LANE_STAMP

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT.parents[1] / "models/wemm-video/awq"
MEDIA = ROOT/"wemm-video/examples/media"
SAMPLES = {"movie":"apollo-11-first-steps.mp4","audio":"apollo-11-small-step.wav","image":"apollo-11-earth.jpg"}
EXT = {"video":{".mp4",".webm",".mov",".mkv",".avi"}, "audio":{".mp3",".m4a",".flac",".wav",".ogg"}, "image":{".png",".jpg",".jpeg",".webp"}}
LIMIT = {"video":2,"audio":2,"image":4}
sessions = {}
RUNS = ROOT/".state/uploads"

def sweep_abandoned():
    # Only purge recognizably owned, unlocked upload roots. Another live studio may share this checkout.
    for folder in RUNS.glob("run-*"):
        if folder.is_symlink() or not folder.is_dir(): continue
        try:
            marker=folder/".owner"
            if marker.is_symlink() or time.time()-marker.stat().st_mtime<60: continue
            with marker.open("rb") as handle:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                shutil.rmtree(folder)
        except (BlockingIOError,FileNotFoundError): continue

@contextmanager
def workspace():
    RUNS.mkdir(mode=0o700,parents=True,exist_ok=True)
    sweep_abandoned()
    with tempfile.TemporaryDirectory(prefix="run-",dir=RUNS) as tmp:
        with (Path(tmp)/".owner").open("xb") as owner:
            fcntl.flock(owner,fcntl.LOCK_EX)
            yield tmp

def discard(sid):
    s = sessions.pop(sid,None)
    if s: shutil.rmtree(s["dir"],ignore_errors=True)

async def reap():
    while True:
        await asyncio.sleep(120)
        await asyncio.to_thread(sweep_abandoned)
        for sid,s in list(sessions.items()):
            if not s["busy"] and not s["uploads"] and time.monotonic()-s["seen"]>3600: discard(sid)

@asynccontextmanager
async def lifespan(app):
    if not (WEIGHTS/"config.json").is_file(): raise RuntimeError("Select wemm-video in the model downloader first.")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]
    # Separate roots per process. Normal shutdown removes the entire upload namespace.
    with workspace() as tmp:
        app.state.tmp=Path(tmp)
        key=secrets.token_urlsafe(32)
        app.state.url=f"http://127.0.0.1:{port}"
        cmd=[str(Path(sys.executable).with_name("vllm")),"serve",str(WEIGHTS),"--served-model-name","wemm",
             "--host","127.0.0.1","--port",str(port),"--api-key",key,"--trust-remote-code","--dtype","bfloat16",
             "--max-model-len",os.environ.get("WEMM_CONTEXT","8192"),"--no-enable-prefix-caching","--mamba-ssm-cache-dtype","float32",
             "--allowed-local-media-path",tmp,"--limit-mm-per-prompt",json.dumps(LIMIT),
             "--media-io-kwargs",json.dumps({"video":{"fps":2,"min_frames":4,"max_frames":16,"total_pixels":262144}})]
        proc=await asyncio.create_subprocess_exec(sys.executable,str(ROOT.parents[1]/"models/studio-child.py"),*cmd,stdin=asyncio.subprocess.PIPE)
        app.state.proc=proc
        async with httpx.AsyncClient(base_url=app.state.url,headers={"Authorization":"Bearer "+key},trust_env=False,
                                     timeout=httpx.Timeout(600,connect=4)) as client:
            app.state.client=client; janitor=asyncio.create_task(reap())
            try: yield
            finally:
                janitor.cancel()
                with suppress(asyncio.CancelledError): await janitor
                proc.stdin.close(); await proc.wait(); sessions.clear()

app=FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None)

@app.get("/")
def index(): return FileResponse(ROOT/"index.html")

@app.get("/health")
async def health():
    proc=app.state.proc
    if proc.returncode is not None: return {"state":f"Engine exited ({proc.returncode}); see terminal."}
    try:
        r=await app.state.client.get("/v1/models",timeout=2)
        if r.is_success and any(m["id"]=="wemm" for m in r.json()["data"]): return {"state":"Ready"}
    except (httpx.HTTPError,ValueError,KeyError): pass
    return {"state":"Loading AWQ engine…"}

@app.post("/session")
async def new():
    sid=secrets.token_urlsafe(24); p=app.state.tmp/sid; p.mkdir()
    sessions[sid]=dict(dir=p,files={},history=[],used=[],busy=False,uploads=0,seen=time.monotonic())
    return {"session":sid}

def session(sid):
    if sid not in sessions: raise HTTPException(410,"Session expired. Start a new conversation.")
    s=sessions[sid]; s["seen"]=time.monotonic(); return s

@app.delete("/session")
async def clear(x_session: str=Header()):
    s=session(x_session)
    if s["busy"] or s["uploads"]: raise HTTPException(409,"Wait for the active operation to close.")
    discard(x_session); return {"ok":True}

def register(s,path,kind,name):
    fid=path.stem
    s["files"][fid]=(path,kind)
    return {"id":fid,"kind":kind,"name":name}

@app.post("/upload")
async def upload(file:UploadFile=File(...),x_session:str=Header()):
    s=session(x_session); ext=Path(file.filename or "").suffix.lower()
    kind=next((k for k,v in EXT.items() if ext in v),None)
    if not kind: raise HTTPException(415,"Unsupported media type.")
    if len(s["files"])+s["uploads"]>=12: raise HTTPException(422,"Start a new session before adding more files.")
    p=s["dir"]/(secrets.token_hex(16)+ext); size=0; s["uploads"]+=1
    try:
        with p.open("xb") as f:
            while chunk:=await file.read(1024*1024):
                size+=len(chunk)
                if size>512*1024**2: raise HTTPException(413,"Maximum upload is 512 MiB per file.")
                f.write(chunk)
        return register(s,p,kind,file.filename)
    except BaseException:
        p.unlink(missing_ok=True); raise
    finally: s["uploads"]-=1; await file.close()

@app.post("/sample/{key}")
async def sample(key:str,x_session:str=Header()):
    if key not in SAMPLES: raise HTTPException(404)
    s=session(x_session)
    if len(s["files"])>=12: raise HTTPException(422,"Start a new session to add more files.")
    src=MEDIA/SAMPLES[key]; dest=s["dir"]/(secrets.token_hex(16)+src.suffix)
    shutil.copyfile(src,dest)
    return register(s,dest,{"movie":"video","image":"image","audio":"audio"}[key],src.name)

@app.get("/sample-preview/{key}")
def sample_preview(key:str):
    if key not in SAMPLES: raise HTTPException(404)
    return FileResponse(MEDIA/SAMPLES[key])

class Question(BaseModel):
    prompt: str=Field(min_length=1,max_length=20000)
    files: list[str]=Field(default_factory=list,max_length=8)
    thinking: bool=False
    use_audio: bool=True
    fps: float=Field(1,ge=0.1,le=8)
    frames: int=Field(16,ge=4,le=256)
    tokens: int=Field(1024,ge=32,le=8192)

@app.post("/ask")
async def ask(q:Question,x_session:str=Header()):
    s=session(x_session)
    if s["busy"]: raise HTTPException(409,"There is already an answer in progress.")
    parts=[]; kinds=[]
    for fid in dict.fromkeys(q.files):
        if fid not in s["files"]: raise HTTPException(422,"Unknown attachment.")
        p,kind=s["files"][fid]; kinds.append(kind)
        src={"url":p.as_uri()}
        if kind=="video": src.update(use_audio=q.use_audio,fps=q.fps,max_frames=q.frames)
        parts.append({"type":kind+"_url",kind+"_url":src})
    if any(n>LIMIT[k] for k,n in Counter(s["used"]+kinds).items()): raise HTTPException(422,"Context limit: two videos, two audio files or four images. Start a new conversation.")
    parts.append({"type":"text","text":q.prompt})
    msg={"role":"user","content":parts}
    payload={"model":"wemm","messages":[*s["history"],msg],"temperature":0,"max_tokens":q.tokens,
             "chat_template_kwargs":{"enable_thinking":q.thinking},"stream":True}
    s["busy"]=True
    def emit(**data): return json.dumps(data,ensure_ascii=False)+"\n"
    async def answer():
        content=[]; finished=False; thinking=q.thinking; buffer=""; first=True; reason=None
        try:
            async with app.state.client.stream("POST","/v1/chat/completions",json=payload) as r:
                if not r.is_success:
                    err=(await r.aread()).decode(errors="replace")
                    try: err=json.loads(err)["error"]["message"]
                    except (KeyError,ValueError,TypeError): pass
                    yield emit(error=err[:1600]); return
                async for line in r.aiter_lines():
                    if not line.startswith("data:"): continue
                    val=line[5:].strip()
                    if val=="[DONE]": finished=True; break
                    evt=json.loads(val)
                    if evt.get("error"): yield emit(error=str(evt["error"])); return
                    for c in evt.get("choices",[]):
                        d=c.get("delta") or {}; txt=d.get("content"); reasoning=d.get("reasoning") or d.get("reasoning_content")
                        if reasoning:
                            # Backends with a reasoning parser have already removed the wire delimiters.
                            thinking=False
                            yield emit(thought=reasoning)
                        if txt and thinking:
                            buffer+=txt
                            if first:
                                if "<think>".startswith(buffer) and buffer != "<think>": continue
                                buffer=buffer.removeprefix("<think>"); first=False
                            body, marker, rest=buffer.partition("</think>")
                            if marker:
                                if body: yield emit(thought=body)
                                thinking=False; buffer=""; txt=rest
                            else:
                                # Keep the longest suffix that may start a split closing delimiter.
                                keep=next((n for n in range(min(7,len(buffer)),0,-1) if "</think>".startswith(buffer[-n:])),0)
                                cut=len(buffer)-keep
                                if cut: yield emit(thought=buffer[:cut])
                                buffer=buffer[cut:]; txt=""
                        if txt: content.append(txt); yield emit(text=txt)
                        if c.get("finish_reason"): finished=True; reason=c["finish_reason"]
            if not finished or thinking or not "".join(content).strip():
                if buffer: yield emit(thought=buffer)
                yield emit(error="No final answer. Increase reply tokens or switch off Thinking." if thinking or reason=="length" else "Incomplete engine response.")
                return
            text="".join(content).strip()
            s["history"].extend([msg,{"role":"assistant","content":text}]); s["used"].extend(kinds)
            yield emit(done=True,finish_reason=reason)
        except (httpx.HTTPError,ValueError) as exc: yield emit(error=str(exc) or "Connection to the engine closed.")
        finally: s["busy"]=False; s["seen"]=time.monotonic()
    return StreamingResponse(answer(),media_type="application/x-ndjson")

print(f"source {LANE_STAMP}")
if __name__=="__main__": uvicorn.run(app,host="127.0.0.1",port=int(os.environ["GRADIO_SERVER_PORT"]))
