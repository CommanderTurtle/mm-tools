# Bud3D

Budget3D image-to-splats studio: one reference image in, a navigable Gaussian
scene out, with the Idol-style view sweep, a turntable video and the exported
splat set back out. Sampling runs compiled BF16 with an optional dense warm-up
and the turbo schedule.

## Setup and start

```bash
./setupwithuv.sh
./startwithuv.sh
```

Choose `bud3d` in the model downloader (`bud3d-turbo` adds the turbo weights).
Setup installs the source committed in this lane, `budget3d/` and `idol-sdk/`,
into a CUDA 12.8 uv environment; nothing is fetched from the network at run
time. Open `http://127.0.0.1:8294`; `GRADIO_SERVER_PORT` changes the port.

## Checkpoints

Checkpoint resolution is shared with the shipped Idol lane, so one download
feeds both studios: `IDOL_ROOT` is this lane's `idol-sdk/`, while
`BUDGET3D_DINOV3` and `BUDGET3D_FLUX2_VAE` point at `models/idol/dinov3` and
`models/idol/flux2`. The demo example images are served from the shipped Idol
lane at `sculpting/idol/idol-sdk/demo/examples`.

## Lane contents

- `budget3d/` — Budget3D, `9f78f302f25cc3107c6107d22d5dc83979a51471`
- `idol-sdk/` — Idol's released model code, `956439f5e664a7e8c5db25a8055e72d6365aafa5`
- `state.py` — records this lane's source; the studio prints that identity at start,
  so a log names the exact source that served a result.
