# ComfyUI Pipeline Exports and Native Video Studios

Committed copies of the exact ComfyUI API graphs the in-repo video studios run
against the shared pinned Comfy checkout, plus a standalone model downloader
for the weights those graphs load. The studio adapters build these same
graphs at request time; the exports here are the static, editable form of each
recipe. The `V2V/` lane hosts the PR-pinned Comfy checkout every other video
lane runs against, so the whole video stack lives in one place.

## Layout

| Folder | Contents |
| --- | --- |
| `animate/` | Wan Animate 2 motion studio plus its exported recipes: motion transfer (distilled + LightX2V), scene-preserving character replacement, pose & face lab |
| `V2V/` | ID-V2V film studio plus the shared PR-pinned Comfy checkout (`V2V/ComfyUI`) that all lanes resolve their weights from |
| `ltx/` | LTX-2.5 audio-and-video studio plus its distilled recipes: text to video, image to video, first + last frame |
| `download_models.py` | Standalone downloader for the `ltx` and `animate` bundles |
| `models/` | Download destination (gitignored) |

Why the studios live here: they prefer running directly against native
ComfyUI; the mm-tools studios are an optional convenience layer on top. They
were moved under this lane because the full project trees were too bulky for
the current linux box - the owner will work on these lanes in ComfyUI proper.

## Install and run

From this folder:

```bash
uv venv .venv --python 3.12 --seed --managed-python   # once
uv pip install --python .venv/bin/python huggingface_hub hf_transfer
.venv/bin/python download_models.py ltx animate       # resumable snapshot downloads
```

Run any ComfyUI instance against these weights by pointing its models folder
at `workflows/V2V/ComfyUI/models/` or by copying the downloaded files into your own
models directory. Open a workflow JSON in the ComfyUI editor or POST it to
`/prompt`; replace the placeholder asset names (`first_frame.png`,
`driving.mp4`, `mask.mp4`) with real files in ComfyUI's input folder.

The LTX lane uses stock ComfyUI nodes only. The Animate lane needs two
repo-pinned custom packs (linked into `custom_nodes` by the project setup):
`animate/ComfyUI-MMToolsAnimate` for the GPU-only loaders and
`animate/ComfyUI-WanAnimatePreprocess` for the ONNX pose pipeline.

Both lanes ship INT8 ConvRot defaults with an NVFP4 Blackwell twin in the same
bundle; switch the loader node's model name to the `.nvfp4.` file to use it.
