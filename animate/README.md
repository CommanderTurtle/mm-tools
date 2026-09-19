# Wan Animate 2 Motion Studio

Wan-Animate-2 turns a reference character image plus a driving video into a
performance, and can replace a character inside an existing video while
keeping its motion, lighting, and background intact. The studio runs the
official Comfy workflow on the PR-pinned shared runtime as a job-scoped
subprocess that infers on the 5090 and owns its own `.venv`.

Two production lanes plus a diagnostic:

- Motion transfer: character image + pose/driving video, with the full
  official conditioning surface (separate prompts, CLIP vision on both
  inputs, independent strengths, continuation frames, sampler overrides).
- Character replacement: source-preserving swap driven by a local BiRefNet
  matte and ViTPose/YOLO pose retargeting.
- Pose lab: inspectable skeleton movie, synchronized face crops, and body
  point JSON for difficult source material.

 Runs are automated the way the model expects: the input is probed, the
 reference loads at its native resolution, and generation runs on the
 selected canvas class - a 720p-class default, an opt-in 1080p-class canvas
 for sharper detail, and a 480p class matching the LightX2V LoRA's training
 resolution. Delivery matches the requested resolution, up to the 2160x1440
 ceiling. The LightX2V profiles load the base INT8 model plus the
 acceleration LoRA for six-step or four-step distilled inference.

## Install and start

From the project folder:

```bash
../workflows/download_models.py animate
./setupwithuv.sh     # delegates to the shared video-studio setup, then builds this venv
./startwithuv.sh     # serves http://127.0.0.1:8264
```

Setup links the repository-pinned Kijai preprocess nodes into the shared
Comfy tree and verifies every weight before touching the environment.
ARCHITECTURE.md documents both lanes, the 4n+1 frame accounting, and the
CUDA allocator policy.

## Upstream

- Model: https://github.com/Wan-Video/Wan-Animate-2
- Preprocess nodes: https://github.com/kijai/ComfyUI-WanAnimatePreprocess
