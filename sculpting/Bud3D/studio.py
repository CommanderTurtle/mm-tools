# SPDX-License-Identifier: AGPL-3.0-only
import os
import secrets
import shutil
from contextlib import contextmanager
import tempfile
import time
from pathlib import Path

import gradio as gr
import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image, ImageDraw
from budget3d.models.load import decode_gaussians, load_budget3d, load_decoders
from budget3d.render.splat import Splats, orbit_positions, render
from budget3d.sampling.fast import optimize_for_inference
from budget3d.sampling.sampler import ImagePreprocessor, sample
from budget3d.sampling.turbo import sample_turbo
from state import LANE_STAMP

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
WEIGHTS = str(MODELS / "idol/weights")
CHECKPOINTS = {"Idol": "budget3d_idol", "Idol-Turbo": "budget3d_idol_turbo"}
AVAILABLE = {name: path for name, stem in CHECKPOINTS.items()
             if (path := MODELS / "bud3d" / name / (stem + ".safetensors")).is_file()}
if not AVAILABLE:
    raise SystemExit("Select bud3d (or bud3d:2 for turbo) in models/download_models.py first.")


@contextmanager
def sampling_options(model, fast):
    # Upstream's compiler switches process-wide attention and matmul flags.
    from budget3d.upstream import idol_layers
    previous = idol_layers.ATTN_BACKEND, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        if fast:
            optimize_for_inference(model)
        yield
    finally:
        idol_layers.ATTN_BACKEND, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


@torch.inference_mode()
def generate(image, variant, budget, mode, seed, shuffle, steps, guidance, shift, count,
             fast, warm, every, until, size, views, bg, yaw, pitch, radius, fov,
             orbit, frames, fps, progress=gr.Progress()):
    if not image or variant not in AVAILABLE:
        raise gr.Error("Select an image and an installed checkpoint.")
    prune_exports()
    out = Path(tempfile.mkdtemp(prefix="budget-", dir=OUTPUT))
    turbo = variant == "Idol-Turbo"
    seed = secrets.randbits(31) if shuffle else int(seed)
    progress(.05, desc="Matting and encoding")
    prep = ImagePreprocessor(WEIGHTS, "cuda", mode)
    prepared, cond = prep(image)
    del prep
    prepared.save(out / "input.png")
    model, _ = load_budget3d(str(AVAILABLE[variant]), turbo=turbo, weights_dir=WEIGHTS,
                             lora=str(AVAILABLE[variant].with_name("idol_turbo_lora.safetensors")) if turbo else None,
                             device="cuda")
    progress(.2, desc="Sampling token layout")
    start = time.monotonic()
    with sampling_options(model, fast):
        if turbo:
            z, _, layout = sample_turbo(model, cond, budget, int(steps or 5), shift, seed=seed)
        else:
            z, _, layout = sample(model, cond, budget, int(steps or 50), guidance, shift,
                                  seed=seed, warmup_steps=int(warm), relayout_every=int(every),
                                  relayout_until_step=int(until))
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    del model, cond
    progress(.67, desc="Decoding Gaussians")
    dec, gs, config = load_decoders(WEIGHTS, "cuda")
    cloud = decode_gaussians(z, dec, gs, config, int(count))
    ply = out / "splats.ply"
    cloud.save_ply(str(ply))
    del z, dec, gs
    splats = Splats(cloud)

    def pixels(camera):
        im = render(splats, camera, fov, size=size, bg=float(bg == "White"))
        return (im.cpu().numpy() * 255).round().astype(np.uint8)

    gallery = []
    for i, pos in enumerate(orbit_positions(yaw, pitch, radius, int(views))):
        frame = out / f"view-{i+1:02}.png"
        Image.fromarray(pixels(pos)).save(frame)
        gallery.append(str(frame))
    tokens = layout.n_tokens if layout is not None else 8192
    cols = min(3, len(gallery))
    sheet = Image.new("RGB", (size * cols, 32 + (size+22) * ((len(gallery)+cols-1)//cols)), "#17191c")
    brush = ImageDraw.Draw(sheet)
    brush.text((8,9), f"{variant}  |  {tokens} tokens  |  seed {seed}  |  {elapsed:.2f} s", fill="white")
    for i, path in enumerate(gallery):
        x, y = size*(i%cols), 32+(size+22)*(i//cols)
        with Image.open(path) as frame:
            sheet.paste(frame, (x,y))
        brush.text((x+7,y+size+3), f"{i+1:02}  /  yaw {np.degrees(yaw)+360*i/len(gallery):.0f} deg", fill="white")
    sheet.save(out / "views.png")
    gallery.insert(0,str(out / "views.png"))
    video = None
    if orbit:
        video = str(out / "orbit.mp4")
        writer = imageio_ffmpeg.write_frames(video, (size, size), fps=fps, codec="libx264")
        writer.send(None)
        try:
            for i, pos in enumerate(orbit_positions(yaw, pitch, radius, int(frames))):
                progress(.74 + .25 * i / frames, desc="Rendering orbit")
                writer.send(pixels(pos).tobytes())
        finally:
            writer.close()
    return str(ply), str(out / "input.png"), gallery, video, [str(ply), str(out / "input.png"), *gallery,
            *([video] if video else [])], f"{tokens:,} tokens · {elapsed:.2f} s sampling · seed {seed}"


OUTPUT = Path(__file__).resolve().parent / ".outputs"
OUTPUT.mkdir(mode=0o700, exist_ok=True)
def prune_exports():
    cutoff = time.time() - 86400
    for entry in os.scandir(OUTPUT):
        if entry.name.startswith("budget-") and entry.is_dir(follow_symlinks=False) and entry.stat().st_mtime < cutoff:
            shutil.rmtree(entry.path)

prune_exports()
with gr.Blocks(title="Budget3D", analytics_enabled=False, delete_cache=(3600, 86400)) as studio:
    gr.Markdown("# Budget3D")
    with gr.Row():
        with gr.Column(scale=2):
            source = gr.Image(type="filepath", image_mode="RGBA", sources=["upload", "clipboard"], label="Reference")
            examples = ROOT / "sculpting/idol/idol-sdk/demo/examples"
            variant = gr.Dropdown(list(AVAILABLE), value=next(iter(AVAILABLE)), label="Checkpoint")
            budget = gr.Slider(.0625, 1, value=.5, step=.0125, label="Token budget · fraction of 8,192")
            mode = gr.Radio(["synthetic", "real"], value="synthetic", label="Framing")
            run = gr.Button("Generate splats", variant="primary")
        with gr.Column(scale=3):
            viewer = gr.Model3D(label="Orbit · splats", height=460, clear_color=(.06,.07,.08,1))
            status = gr.Markdown()
    with gr.Accordion("Sampling", open=False):
        seed = gr.Number(42, precision=0, label="Seed")
        shuffle = gr.Checkbox(False, label="Random seed")
        steps = gr.Slider(0,100,0,step=1,label="Steps · 0 uses checkpoint default")
        guidance = gr.Slider(0,10,3,label="Guidance · Idol")
        shift = gr.Slider(1,6,3,label="Time shift")
        count = gr.Dropdown([65536,131072,262144,524288,1048576],value=262144,label="Gaussians")
        fast = gr.Checkbox(False,label="Compiled BF16 sampling")
        warm = gr.Slider(0,20,0,step=1,label="Dense warm-up · Idol")
        every = gr.Slider(1,20,5,step=1,label="Re-layout interval · Idol")
        until = gr.Slider(-1,40,5,step=1,label="Last re-layout step · Idol")
    with gr.Accordion("Camera and turntable", open=False):
        size = gr.Dropdown([256,384,448,512,640,768],value=512,label="Resolution")
        views = gr.Slider(1,12,4,step=1,label="Still views")
        bg = gr.Radio(["White","Black"],value="White",label="Background")
        yaw = gr.Slider(-3.2,3.2,.5,label="Start yaw")
        pitch = gr.Slider(-1.4,1.4,.2,label="Pitch")
        radius = gr.Slider(.8,4,1.55,label="Camera radius")
        fov = gr.Slider(10,90,42,label="Field of view")
        orbit = gr.Checkbox(True,label="Render turntable")
        frames = gr.Slider(16,240,72,step=8,label="Orbit frames")
        fps = gr.Slider(8,60,24,step=1,label="Frame rate")
    presets = [("fig-knight",.5,7), ("food-avocado",.4,7), ("fig-axolotl",.5,42), ("fig-robot-cute",.25,7),
               ("fur-chesterfield",.6,7), ("obj-power-drill",.5,7), ("obj-backpack",1.,42)]
    gr.Examples([[str(examples / (name+".webp")), b, s] for name,b,s in presets], [source,budget,seed],
                cache_examples=False)
    with gr.Tab("Prepared input"): prepared = gr.Image()
    with gr.Tab("Views"): gallery = gr.Gallery()
    with gr.Tab("Turntable"): video = gr.Video()
    files = gr.File(file_count="multiple",label="Exports")
    run.click(generate, [source,variant,budget,mode,seed,shuffle,steps,guidance,shift,count,fast,warm,every,until,
                         size,views,bg,yaw,pitch,radius,fov,orbit,frames,fps],
              [viewer,prepared,gallery,video,files,status],concurrency_limit=1)
print(f"source {LANE_STAMP}")
studio.queue(max_size=4).launch(share=False, allowed_paths=[str(OUTPUT), str(ROOT/"sculpting/idol/idol-sdk/demo/examples")])
