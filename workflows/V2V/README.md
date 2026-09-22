# ID-V2V Film Lab

ID-V2V ("Identity Diffusion Video-to-Video", Eyeline Labs) restyles a driving
video while preserving identity, pose, and motion. The studio runs the
official graph on the PR-pinned local Comfy runtime and loads the Kijai INT8
ConvRot checkpoint directly on the GPU, with an optional
normal-plus-depth-augmented variant for tighter scene guidance.

## Install and start

From the project folder:

```bash
./setupwithuv.sh     # needs NVIDIA driver >= 570 and ffmpeg; verifies every weight first
./startwithuv.sh     # serves http://127.0.0.1:8265
```

Setup fails with exact hints when a weight or tool is missing. The shared
Comfy tree under `ComfyUI/` is the pinned Comfy-Org checkout (PR #15139,
SVI-style identity padding) that Animate 2 also runs; each project keeps its
own `.venv` and never installs into the other's.

Models come from the central downloader:

```bash
../../models/download_models.py v2v
```

Foreground matting reuses the BiRefNet checkpoint already owned by the
Sculpting Studio, so no second segmentation model is staged. The
conditioning contract, the normal/depth variant, and job-scoped shutdown
behavior are documented in ARCHITECTURE.md.

## Upstream

- Model and paper: https://github.com/Eyeline-Labs/ID-V2V
- Weights: https://huggingface.co/Kijai/Wan_ID_V2V_comfy
- Runtime pin: https://github.com/Comfy-Org/ComfyUI/pull/15139
