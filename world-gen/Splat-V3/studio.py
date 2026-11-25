# SPDX-License-Identifier: AGPL-3.0-only
import json
import time
from uuid import uuid4
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from inference import main as reconstruct
from src.utils.panorama import convert_panorama
from state import LANE_STAMP

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "infinisplat/InfiniSplat_v3"
MODELS = ROOT.parents[1] / "models/splat-v3"
EX = MODELS / "demo/examples"
SCENES = {f"{p.parent.name}/{p.name}": p for p in sorted(EX.glob("*/*")) if (p/"images").is_dir()}
POSTERS = MODELS / "demo/assets/scene-previews"
FORMATS = {".png", ".webp", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
KINDS = ["Images", "Camera set", "Video", "Panorama", "Example"]
CATALOG = json.loads((POSTERS / "catalog.json").read_text())["scenes"] if (POSTERS/"catalog.json").is_file() else {}
app = FastAPI(docs_url=None, redoc_url=None)
def embed(url):
    return f'<iframe sandbox="allow-scripts allow-pointer-lock" allowfullscreen style="width:100%;height:480px;border:0" src="{url}" title="Scene walk-through"></iframe>'

@app.get("/walk/{name}")
def walk(name: str):
    if name in CATALOG:
        page = (MODELS/"showcase"/CATALOG[name]["path"]).resolve()
        if (MODELS/"showcase").resolve() not in page.parents: raise HTTPException(404)
    elif len(name)==32 and all(c in "0123456789abcdef" for c in name):
        page = OUTPUT / ("scene-"+name) / "scene.html"
    else: raise HTTPException(404)
    if not page.is_file(): raise HTTPException(404)
    def chunks():
        # Supersplat reads preferences before drawing; opaque frames cannot access Web Storage.
        shim = b"""<script>{const m=new Map;Object.defineProperty(window,'localStorage',{value:{get length(){return m.size},key:i=>[...m.keys()][i]??null,getItem:k=>m.get(String(k))??null,setItem:(k,v)=>m.set(String(k),String(v)),removeItem:k=>m.delete(String(k)),clear:()=>m.clear()}})}</script>"""
        with page.open("rb") as stream:
            prefix = stream.read(32768)
            end = prefix.lower().find(b"<head>")
            pos = end + 6 if end >= 0 else (prefix.find(b">")+1 if prefix.lstrip().lower().startswith(b"<!doctype") else 0)
            yield prefix[:pos]+shim+prefix[pos:]
            yield from iter(lambda: stream.read(256*1024), b"")
    return StreamingResponse(chunks(),media_type="text/html",headers={"Content-Security-Policy":"sandbox allow-scripts allow-pointer-lock; connect-src data: blob:"})

OUTPUT = ROOT / ".outputs"
OUTPUT.mkdir(exist_ok=True)
if not all((MODELS / f).is_file() for f in ("SplatAnything.ckpt","worldmirror2.safetensors","skyseg.onnx")):
    raise SystemExit("Select splat-v3 in the model picker first.")


def scene_defaults(key):
    scene = SCENES[key]
    kind = scene.parent.name
    res = 504 if kind in {"aerial", "street", "rig"} else 756
    prior = (scene/"prior_camera.json").is_file() and kind != "indoor"
    poster = POSTERS / (key.replace("/", "-") + ".webp")
    return ("Example", "pose" if prior else "rgb", res, res, kind in {"street","rig"},
            {"rig":150, "street":100, "aerial":100}.get(kind,24), str(poster) if poster.is_file() else None)


def stage_images(files, directory, limit):
    paths = sorted(map(Path, files or []), key=lambda p:p.name)
    if limit:
        paths = paths[:int(limit)]
    if not paths:
        raise gr.Error("Supply at least one view.")
    if len({p.stem for p in paths}) != len(paths):
        raise gr.Error("Each image needs a distinct filename for camera matching.")
    directory.mkdir(parents=True)
    for p in paths:
        if p.suffix.lower() not in FORMATS:
            raise gr.Error(f"Unsupported image: {p.name}")
        shutil.copyfile(p,directory/p.name)


def extract(video, count, root):
    cap = cv2.VideoCapture(video)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 2:
            raise gr.Error("No decodable video sequence.")
        root.mkdir(parents=True)
        for i, frame in enumerate(np.linspace(0,total-1,min(int(count),total),dtype=int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame))
            ok, pixels = cap.read()
            if not ok or not cv2.imwrite(str(root/f"frame_{i:04}.png"), pixels):
                raise gr.Error(f"Could not decode frame {frame}.")
    finally:
        cap.release()


def run(kind, files, captures, video, count, pano, key, mode, camera, w, h, sky,
        align, limit, voxel, opacity, pct, flying, progress=gr.Progress()):
    token = uuid4().hex
    job = OUTPUT / ("scene-"+token)
    job.mkdir(mode=0o700)
    scene = job / "capture"
    try:
        if kind == "Panorama":
            if not pano: raise gr.Error("Supply a 2:1 panorama.")
            convert_panorama(pano,scene,max(int(w), int(h)),60,False)
            if limit:
                for path in sorted((scene/"images").iterdir())[int(limit):]:
                    path.unlink()
            mode = "pose"
        elif kind == "Video":
            if not video: raise gr.Error("Upload a video.")
            extract(video, min(count,limit) if limit else count, scene/"images")
        elif kind == "Camera set":
            stage_images(captures,scene/"images",limit)
        elif kind == "Example":
            if key not in SCENES: raise gr.Error("Select a scene.")
            src = SCENES[key]
            stage_images([p for p in (src/"images").iterdir() if p.suffix.lower() in FORMATS],scene/"images",limit)
            camera = src/"prior_camera.json"
        else:
            stage_images(files,scene/"images",limit)
        if mode == "pose" and kind != "Panorama":
            if not camera or not Path(camera).is_file(): raise gr.Error("Camera JSON is required in pose mode.")
            shutil.copyfile(camera,scene/"prior_camera.json")
        cfg = OmegaConf.load(SOURCE/"configs/inference.yaml")
        cfg.wrapper.model.gs_voxel_size = voxel
        cfg.wrapper.model.max_scale_pct = pct
        cfg.wrapper.model.filter_flying_points_infer = flying
        config = job/"settings.yaml"
        OmegaConf.save(cfg,config)
        progress(.25, desc="Geometry and Gaussian reconstruction")
        reconstruct(["--config_path",str(config),"--input_path",str(scene),"--output_path",str(job/"export"),
                     "--resolution",str(int(w)),str(int(h)),"--mode",mode,"--opacity_thresh",str(opacity),
                     "--ckpt_path",str(MODELS/"SplatAnything.ckpt"),"--geometry_ckpt_path",str(MODELS/"worldmirror2.safetensors"),
                     *(["--no_cam_align"] if not align else []),
                     *(["--sky_seg","--sky_onnx_path",str(MODELS/"skyseg.onnx")] if sky else [])])
        out = job/"export/capture"
        ply = out/"gaussians.ply"
        progress(.86, desc="Packaging scene")
        html = job/"scene.html"
        # Remove machine-local paths from the portable archive.
        portability = json.loads((out/"run.json").read_text())
        for field in ("input_path","output_path","images_path","cameras","config_path","sky_onnx_path","ckpt_path","geometry_ckpt_path"):
            if portability.get(field): portability[field] = Path(portability[field]).name
        portability["images"] = [Path(s).name for s in portability["images"]]
        portability["source"] = LANE_STAMP
        (out/"run.json").write_text(json.dumps(portability,indent=2))
        published = OmegaConf.load(out/"config.yaml")
        for field in ("SplatAnything_pretrained","worldmirror2_pretrained"):
            published.wrapper[field] = Path(published.wrapper[field]).name
        OmegaConf.save(published,out/"config.yaml")
        try:
            subprocess.run([str(ROOT/"node_modules/.bin/splat-transform"),"-w",str(ply),"-H","0","-g","cpu",
                        "--max-workers","2",str(html)],check=True,timeout=600)
        except (subprocess.SubprocessError,OSError):
            html.unlink(missing_ok=True)
            gr.Warning("HTML preview unavailable; the native PLY and scene archive are ready.")
        archive = shutil.make_archive(str(job/"reconstruction"),"zip",out)
        return str(ply), sorted(str(x) for x in (out/"depth").glob("*.png")), [str(ply),archive]+([str(html)] if html.is_file() else []), embed("/walk/"+token) if html.is_file() else ""
    except BaseException:
        shutil.rmtree(job,ignore_errors=True)
        raise


def capture(image, items):
    if not image: raise gr.Error("Take a photo first.")
    # Gradio's captured filenames can repeat; keep camera IDs deterministic.
    items = list(items or [])
    if len(items) >= 150: raise gr.Error("Start a new camera set.")
    with tempfile.NamedTemporaryFile(prefix=f"view-{len(items):04}-",suffix=".png",delete=False,dir=OUTPUT) as f:
        with Image.open(image) as im: im.convert("RGB").save(f,format="PNG")
    items.append(f.name)
    return items,items


with gr.Blocks(title="SplatAnything",analytics_enabled=False,delete_cache=(3600,86400)) as studio:
    gr.Markdown("# SplatAnything")
    kind = gr.Radio(KINDS,value="Images",label="Input")
    with gr.Row():
        with gr.Column():
            files = gr.File(file_count="multiple",file_types=["image"],label="Overlapping views")
            with gr.Accordion("Capture",open=False):
                cam = gr.Image(sources=["webcam"],type="filepath",label="Camera")
                items = gr.State([])
                capture_btn = gr.Button("Add view")
                clear = gr.Button("New set")
                captured = gr.Gallery(label="Camera set",height=160)
                capture_btn.click(capture,[cam,items],[items,captured]).then(lambda:"Camera set",outputs=kind)
                clear.click(lambda:([],[]),outputs=[items,captured])
            video = gr.Video(sources=["upload"],label="Scene video")
            count = gr.Slider(2,32,8,step=1,label="Video views")
            pano = gr.Image(type="filepath",label="360° panorama")
            video.upload(lambda:"Video",outputs=kind)
            pano.upload(lambda:"Panorama",outputs=kind)

        with gr.Column(scale=2):
            viewer = gr.Model3D(label="Reconstruction",height=480)
            with gr.Accordion("Scene library",open=True):
                key = gr.Dropdown(list(SCENES),label="Object · Indoor · Street · Aerial · Rig · Panorama")
                cover = gr.Image(label="Reference scene",height=180)
    with gr.Accordion("Reconstruction",open=True):
        mode = gr.Radio(["rgb","pose"],value="rgb",label="Camera conditioning")
        camera = gr.File(file_types=[".json"],label="Camera poses and intrinsics")
        with gr.Row():
            w = gr.Number(756,precision=0,minimum=14,step=14,label="Width")
            h = gr.Number(756,precision=0,minimum=14,step=14,label="Height")
            limit = gr.Number(0,precision=0,minimum=0,label="Max images · 0 = all")
        with gr.Row():
            sky = gr.Checkbox(False,label="Remove sky")
            align = gr.Checkbox(True,label="Align to first camera")
            flying = gr.Checkbox(False,label="Filter flying points")
        voxel = gr.Slider(.0015,.002,.0018,step=.0001,label="Voxel size")
        opacity = gr.Slider(0,1,.05,step=.01,label="Minimum opacity")
    def uploaded(paths):
        width = height = 896
        for path in paths or []:
            with Image.open(path) as img:
                width,height = min(width,img.width),min(height,img.height)
        if min(width,height) < 14: raise gr.Error("Views must be at least 14 pixels wide and high.")
        return "Images",width//14*14,height//14*14,0
    files.upload(uploaded,files,[kind,w,h,limit])
    pct = gr.Number(99.5,minimum=.01,maximum=100,label="Maximum Gaussian scale percentile")
    walkview = gr.HTML(label="Walk-through")
    key.input(scene_defaults,key,[kind,mode,w,h,sky,limit,cover])
    def explore(key):
        name = (key or "").replace("/","-")
        if name not in CATALOG or not (MODELS/"showcase"/CATALOG[name]["path"]).is_file():
            raise gr.Error("Select splat-v3:1+2 in the picker to explore prebuilt examples.")
        return embed("/walk/"+name)
    gr.Button("Explore example in 3D").click(explore,key,walkview)
    depth = gr.Gallery(label="Depth",height=180)
    downloads = gr.File(file_count="multiple",label="Standalone fly-through · native Gaussians · complete reconstruction")
    gr.Button("Reconstruct",variant="primary").click(run,[kind,files,items,video,count,pano,key,mode,camera,w,h,sky,
        align,limit,voxel,opacity,pct,flying],[viewer,depth,downloads,walkview],concurrency_limit=1)
for entry in OUTPUT.iterdir():
    if not entry.is_symlink() and entry.stat().st_mtime < time.time()-86400:
        if entry.name.startswith("scene-") and entry.is_dir(): shutil.rmtree(entry)
        elif entry.name.startswith("view-") and entry.is_file(): entry.unlink()
gr.mount_gradio_app(app,studio.queue(max_size=4),path="/",allowed_paths=[str(OUTPUT),str(POSTERS)])
if __name__ == "__main__":
    import os
    uvicorn.run(app,host="127.0.0.1",port=int(os.environ["GRADIO_SERVER_PORT"]))
