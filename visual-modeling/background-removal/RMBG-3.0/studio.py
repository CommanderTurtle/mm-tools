# SPDX-License-Identifier: AGPL-3.0-only
"""Local RMBG 3 matte service. The model's training instruction is not an editable prompt."""
import importlib.util
import io
import json
import os
import threading
from pathlib import Path
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageChops, ImageOps, UnidentifiedImageError
from state import LANE_STAMP

ROOT = Path(__file__).resolve().parent
MODELS = ROOT.parents[2] / "models/bria-rmbg3"
BASE = MODELS / "fibo"
spec = importlib.util.spec_from_file_location("bria_recipe", ROOT / "rmbg/remove_bg.py")
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)
app = FastAPI(docs_url=None,redoc_url=None)
lock = threading.Lock()
pipe = None

def load():
    from diffusers import AutoModel, BriaFibo2Pipeline
    transformer = AutoModel.from_pretrained(str(MODELS/"matte"), subfolder="transformer", dtype=torch.bfloat16, local_files_only=True)
    model = BriaFibo2Pipeline.from_pretrained(str(BASE), transformer=transformer, dtype=torch.bfloat16, local_files_only=True)
    mode = os.environ.get("RMBG_MEMORY", "auto")
    if mode == "auto":
        if torch.cuda.mem_get_info()[0] < 30 * 1024**3: mode = "sequential"
        else: mode = "model" if os.environ.get("RMBG_CPU_OFFLOAD") == "1" else "resident"
    if mode == "sequential": model.enable_sequential_cpu_offload()
    elif mode == "model": model.enable_model_cpu_offload()
    elif mode == "resident": model.to("cuda")
    else: raise ValueError("RMBG_MEMORY must be auto, sequential, model or resident.")
    print(f"RMBG-3.0 · BF16 · {mode} placement",flush=True)
    return model

@app.get("/")
def index(): return FileResponse(ROOT / "index.html")

@app.get("/example")
def example(): return FileResponse(MODELS/"matte/assets/rmbg3_example_input_dog.jpg")

@app.post("/matte")
def matte(photo: UploadFile=File(...), preserve: bool=Form(True), seed: int=Form(0)):
    global pipe
    if not 0 <= seed < 2**32: raise HTTPException(422,"Seed must be an unsigned 32-bit integer.")
    try:
        with Image.open(photo.file) as raw:
            if raw.width*raw.height>48_000_000: raise HTTPException(413,"Image exceeds 48 megapixels.")
            upright = ImageOps.exif_transpose(raw).convert("RGBA")
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as exc: raise HTTPException(415,"Unreadable image.") from exc
    rgb = upright.convert("RGB")
    with lock, torch.inference_mode():
        if pipe is None: pipe = load()
        pred = pipe(prompt=json.dumps({"edit_instruction":recipe.RMBG3_INSTRUCTION}), image=rgb,
                    num_inference_steps=recipe.RMBG3_STEPS, guidance_scale=recipe.RMBG3_GUIDANCE,
                    generator=torch.Generator("cuda").manual_seed(seed),fonts=[str(BASE/"fonts"/s) for s in recipe.RMBG3_FONTS]).images[0]
    alpha = pred.convert("L").resize(rgb.size,Image.Resampling.LANCZOS)
    if preserve: alpha = ImageChops.multiply(alpha,upright.getchannel("A"))
    rgb.putalpha(alpha)
    data = io.BytesIO(); rgb.save(data,"PNG")
    return Response(data.getvalue(),media_type="image/png")

print(f"source {LANE_STAMP}")
if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1",port=int(os.environ["GRADIO_SERVER_PORT"]))
