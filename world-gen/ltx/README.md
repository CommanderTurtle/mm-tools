# LTX-2.5 Studio

LTX-2.5 is a distilled 22B audio-and-video diffusion model: one pass emits a
video clip with its synchronized sound track. The studio exposes the three
official ComfyUI recipes against the shared pinned Comfy checkout: text to
video, image to video (a still opens the clip), and first + last frame
(both endpoints are pinned natively while the middle is invented).

The profile follows the published templates: text/image generation samples at
half resolution, runs the x2 latent spatial upscaler between the two sigma
stages, and decodes through tiled VAE passes; first + last frame brackets both
frames at full canvas resolution in a single stage. Frame counts follow the
native `seconds x FPS + 1` rule and stay under the 1,000-frame audio latent
ceiling. INT8 ConvRot is the verified default; the NVFP4 Blackwell twin from
the same bundle is a one-click switch that frees activation headroom on RTX
50-series GPUs. Everything stays local, with inference on the 5090.

## Install and start

From the project folder:

```bash
../../models/download_models.py ltx   # pulls the official LTX-2.5 checkpoints from the ungated comfyicu mirror
./setupwithuv.sh     # verifies weights, then builds the isolated venv
./startwithuv.sh     # serves http://127.0.0.1:8269
```

ARCHITECTURE.md documents the 5090 profile, the graph layout, and the
delivery rules.

## Upstream

- Code: https://github.com/Comfy-Org/ComfyUI (official LTX-2.5 blueprints)
- Weights: https://huggingface.co/comfyicu/LTX-2.5 (ungated mirror of https://huggingface.co/Lightricks/LTX-2.5)
- Model card: https://lightricks.github.io/ltx2/
