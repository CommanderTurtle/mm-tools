# SPDX-License-Identifier: AGPL-3.0-only
import base64
import importlib.util
import io
import os
import sys
import tempfile
import threading
import time
from urllib.parse import unquote
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, Form, File, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from gradio_client.utils import encode_file_path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "jevertion"
WEIGHTS = ROOT.parents[1] / "models/jeveration"
TEMP = Path(os.environ["GRADIO_TEMP_DIR"]).resolve()
TEMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location("jev_native", SRC / "app.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
from jeveration import JeverationPipeline
from state import LANE_STAMP

app = FastAPI(docs_url=None, redoc_url=None)
app.mount("/examples", StaticFiles(directory=SRC / "examples"), name="examples")
lock = threading.Lock()
pipeline = None

def infer(fn, *args):
    global pipeline
    with lock:
        if pipeline is None:
            w = WEIGHTS / "weights"
            pipeline = JeverationPipeline(str(w/"jeveration.pt"), decoder=str(w/"decoder512.pt"),
                source_decoder=str(w/"source_decoder512.pt"), backbone=str(WEIGHTS/"qwen"),
                tatitok=str(WEIGHTS/"tokens"), clip=str(WEIGHTS/"clip"))
        return fn(pipeline, *args)

def upright(file):
    with Image.open(file) as im:
        if im.width * im.height > 40_000_000: raise ValueError("Image exceeds 40 megapixels.")
        return ImageOps.exif_transpose(im).convert("RGB")

def seed_value(value):
    n = int(value)
    if not 0 <= n < 2**32: raise ValueError("Choose a seed between 0 and 4294967295.")
    return n

def edit(image, task, prompt, photo, seed):
    if task not in native.config.I2I_TASKS: raise ValueError("Unknown edit.")
    return infer(native.image_to_image, image, task, prompt, photo, seed_value(seed))

def png(arr):
    stream = io.BytesIO()
    Image.fromarray(arr).save(stream, format="PNG")
    return stream.getvalue()

@app.get("/")
def index(): return FileResponse(ROOT/"index.html")

# Workflow's upload node posts to the origin root, even when the graph is mounted below it.
@app.post("/gradio_api/upload")
@app.post("/upload")
def graph_upload(request: Request):
    query = request.url.query
    return RedirectResponse("/workflow/gradio_api/upload"+("?" + query if query else ""),status_code=307)

@app.get("/settings")
def settings():
    return dict(prompts=native.T2I_EXAMPLES, edits=native.I2I_EXAMPLES, dense=native.DENSE_EXAMPLES, tasks=native.config.I2I_TASKS)

@app.post("/render")
def render(mode: str=Form(...), prompt: str=Form(""), seed: int=Form(42), task: str=Form("hed"),
           photo: bool=Form(False), picture: UploadFile | None=File(None)):
    try:
        if mode == "Generate": rows = infer(native.text_to_image, prompt, seed_value(seed))
        elif picture is None: raise ValueError("Choose or sketch an image.")
        elif mode == "Edit": rows = edit(upright(picture.file), task, prompt, photo, seed)
        elif mode == "Dense": rows = infer(native.dense_predict, upright(picture.file))
        else: raise ValueError("Unknown mode.")
        return dict(images=["data:image/png;base64,"+base64.b64encode(png(a)).decode() for a in rows[:-1]], note=rows[-1])
    except (ValueError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc

def read_node(value):
    if isinstance(value, dict):
        url = value.get("url") or ""
        value = value.get("path") or (unquote(url.split("/gradio_api/file=",1)[1]) if "/gradio_api/file=" in url else None)
    if not isinstance(value, str): raise ValueError("Connect an image or upload one.")
    p = Path(value).resolve()
    if not p.is_relative_to(TEMP): raise ValueError("Choose a workflow upload.")
    return upright(p)

def publish(arr):
    cutoff = time.time() - 86400
    for old in TEMP.glob("jev-*.png"):
        try:
            if old.stat().st_mtime < cutoff: old.unlink(missing_ok=True)
        except FileNotFoundError: pass
    with tempfile.NamedTemporaryFile(dir=TEMP,prefix="jev-",suffix=".png",delete=False) as f:
        f.write(png(arr)); path = f.name
    return {"path": path, "url": "/workflow/gradio_api/file="+encode_file_path(path), "is_file": True}

def generate_node(prompt, seed: str="42") -> dict:
    return publish(infer(native.text_to_image, prompt, seed_value(seed))[0])
def edit_node(image, task: str="anime", caption: str="", seed: str="42") -> dict:
    return publish(edit(read_node(image), task, caption, False, seed)[1])
def dense_node(image) -> tuple[dict, dict, dict]:
    return tuple(publish(item) for item in infer(native.dense_predict,read_node(image))[:3])

graph = gr.Workflow(graph=str(WEIGHTS/"space/workflow.json"), bind={
    "jeveration_generate": generate_node, "jeveration_edit": edit_node, "jeveration_dense": dense_node})
app = gr.mount_gradio_app(app,graph,path="/workflow",ssr_mode=False,allowed_paths=[str(TEMP)])
print(f"source {LANE_STAMP}")
if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1",port=int(os.environ["GRADIO_SERVER_PORT"]))
