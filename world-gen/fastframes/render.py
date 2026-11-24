# SPDX-License-Identifier: AGPL-3.0-only
import io
import json
import struct
import sys
from dataclasses import replace
from pathlib import Path
import imageio_ffmpeg as ff
import torch
from PIL import Image
from scripts.generate import load_model
from fastframes.pipeline import to_uint8


def render(job):
    spec = json.loads((job/"spec.json").read_text())
    model = load_model("ff-l",Path(__file__).resolve().parents[2]/"models/fastframes")
    count = 4*spec["seconds"]+1
    # Preserve the release's full seeded noise sequence, even on short clips.
    noise = model._noise(spec["seed"])
    model.config = replace(model.config,num_frames=count,num_pixel_frames=4*count-3)
    model._noise = lambda _: noise
    predict = model._predict
    features, mods = [], []
    shown = 0
    def view(x,t,frame,*args):
        nonlocal shown
        result = predict(x,t,frame,*args)
        schedule = model.config.timesteps if frame else model.config.first_frame_timesteps
        if t == schedule[-1]:
            features.append(result[1])
            mods.append(result[2])
            if (frame+1)%3==0 or frame+1==count:
                first = max(0,shown-6)
                context = frame+1-first
                h,w = model.config.latent_size
                grid = torch.tensor([[context,h//2,w//2]],device=model.device)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    pixels = model.decoder(torch.cat(features[first:],1),torch.cat(mods[first:],1),grid)[0]
                begin = 4*(shown-first) if shown else 3
                partial = job/"partial.bin"
                with partial.open("wb") as stream:
                    for img in to_uint8(pixels[begin:]).cpu().numpy():
                        jpeg = io.BytesIO()
                        Image.fromarray(img).save(jpeg,format="JPEG",quality=82)
                        data = jpeg.getvalue()
                        stream.write(struct.pack(">I",len(data)))
                        stream.write(data)
                partial.replace(job/f"part-{frame:03}.bin")
                shown = frame+1
        return result
    model._predict = view
    frames = model.generate(spec["prompt"],spec["seed"]).numpy()
    writer = ff.write_frames(str(job/"movie.mp4"),(frames.shape[2],frames.shape[1]),fps=16,codec="libx264",output_params=["-movflags","+faststart"])
    writer.send(None)
    try:
        for frame in frames: writer.send(frame.tobytes())
    finally: writer.close()

if __name__ == "__main__":
    render(Path(sys.argv[1]))
