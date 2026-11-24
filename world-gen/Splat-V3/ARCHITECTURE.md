# Splat-V3

SplatAnything reconstruction studio: overlapping views, a scene video or a 360°
panorama in, a walkable Gaussian scene back out. The browser walk-through is
served from the lane; reconstruction writes a portable `run.json` that names
every input it used.

## Setup and start

```bash
./setupwithuv.sh
./startwithuv.sh
```

Choose `splat-v3` in the model downloader (`splat-v3-showcase` adds the
optional offline fly-through gallery). Setup installs the source committed in
this lane, `infinisplat/InfiniSplat_v3`, into a CUDA 12.8 uv environment, plus
the npm packages for the walk-through viewer. Open `http://127.0.0.1:8295`;
`GRADIO_SERVER_PORT` changes the port.

## Checkpoints

`models/splat-v3` holds the three runtime weights; the studio checks for them
before accepting a job and never resolves an upstream repository name over the
network.

## Lane contents

- `infinisplat/InfiniSplat_v3` — SplatAnything v3, `bf5d3b0a88d4385baaf419d8f9b433eb7a6f6fbf`
- `state.py` — records this lane's source; every exported `run.json` carries that
  identity under `source`, so a scene names the exact source that produced it.
