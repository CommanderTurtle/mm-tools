# Shared Comfy Video Lanes

The video stack lives in one place: a shared PR-pinned Comfy checkout plus
three preserved video studios, the importable workflow JSONs they were built
from, and a standalone model downloader for the weights those graphs load.

**The studios are optional.** Each lane's workflow JSONs are complete
API-format graphs; import them into any native ComfyUI install (or POST them
to `/prompt`) and run the pipeline without the studio. The studios were a
refined convenience layer — they queue jobs, stage outputs, and gate on
hardware — but they are deliberately not configurable, so for custom work
the native Comfy editor is the better surface. These lanes were moved here
because the full project trees were too bulky for this linux box; the owner
works on them in ComfyUI proper.

## Layout

| Folder / file | Contents |
| --- | --- |
| `ComfyUI/` | Shared PR-pinned Comfy checkout (PR #15139, SVI-style identity padding). Ultra-minimal by design: only the runtime code and committed blueprints are tracked; everything else is reconstructed locally. Not a full upstream clone. |
| `V2V/` | ID-V2V film studio (preserved) — identity-preserving restyle, relight, normal+depth control |
| `animate/` | Wan Animate 2 motion studio (preserved) plus exported recipes: motion transfer (distilled + LightX2V), scene-preserving character replacement, pose & face lab |
| `ltx/` | LTX-2.5 audio-and-video studio (preserved) plus distilled recipes: text to video, image to video, first + last frame |
| `build.sh` | Studio builder for people who want the studios: `./build.sh [v2v \| animate \| ltx \| all]` |
| `download_models.py` | Standalone downloader for the `ltx` and `animate` bundles |
| `models/` | Download destination (gitignored) |

## Weights

Two downloaders feed one models directory (`workflows/ComfyUI/models/`):

```bash
# from the repo root
models/download_models.py v2v          # ID-V2V DiTs + the shared Wan support set
# from this folder
.venv/bin/python download_models.py ltx animate   # LTX bundle + Animate lane-specific weights
```

Shared weights (UMT5 FP8, CLIP Vision H, Wan VAE) are downloaded exactly
once, by the main `v2v` bundle; the Animate bundle carries only its own
DiTs, LoRA, NVFP4 twins, and ONNX pose detectors. Snapshots land in resumable
`imports/` folders and are **hard-linked** (never copied) into
`workflows/ComfyUI/models/` on the same filesystem, so a weight occupies disk
once no matter how many lanes reference it. Point any ComfyUI instance's
models folder at `workflows/ComfyUI/models/` and the exported graphs resolve
their weights unchanged.

## Install and run

### Native ComfyUI (no studio required)

Run any ComfyUI instance against these weights by pointing its models folder
at `workflows/ComfyUI/models/` or by copying the downloaded files into your
own models directory. Open a workflow JSON in the ComfyUI editor or POST it
to `/prompt`; replace the placeholder asset names (`first_frame.png`,
`driving.mp4`, `mask.mp4`) with real files in ComfyUI's input folder.

### The studios (optional)

From this folder:

```bash
uv venv .venv --python 3.12 --seed --managed-python   # once, for the downloader
uv pip install --python .venv/bin/python huggingface_hub hf_transfer
./build.sh                       # or: ./build.sh v2v | animate | ltx
./V2V/startwithuv.sh             # http://127.0.0.1:8265
./animate/startwithuv.sh         # http://127.0.0.1:8264
./ltx/startwithuv.sh             # http://127.0.0.1:8269
```

Each studio keeps its own `.venv` and spawns a job-scoped private Comfy
runtime from the shared checkout; none of them exposes an editor port.

## Lane notes

The LTX lane uses stock ComfyUI nodes only. The Animate lane needs two
repo-pinned custom packs (linked into `custom_nodes` by the project setup):
`animate/ComfyUI-MMToolsAnimate` for the GPU-only loaders and
`animate/ComfyUI-WanAnimatePreprocess` for the ONNX pose pipeline. The
ID-V2V lane is pure core nodes (PR #15139).

Both the Animate and LTX lanes ship INT8 ConvRot defaults with an NVFP4
Blackwell twin in the same bundle; switch the loader node's model name to
the `.nvfp4.` file to use it.
