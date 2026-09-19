# ComfyUI Pipeline Exports

Committed copies of the exact ComfyUI API graphs the in-repo video studios run
against the shared pinned Comfy checkout. The studio adapters build these same
graphs at request time; the exports here are the static, editable form of each
recipe.

## Layout

| Folder | Contents |
| --- | --- |
| `LTX/` | LTX-2.5 distilled recipes: text to video, image to video, first + last frame |
| `animate/` | Wan Animate 2 recipes: motion transfer (distilled + LightX2V), scene-preserving character replacement, pose & face lab |

## Install and run

From the repository root, pull the allowlisted weights with the central
downloader:

```bash
cd models && uv run download_models.py ltx animate
```

Run any ComfyUI instance against these weights by pointing its models folder
at `V2V/ComfyUI/models/` or by copying the downloaded files into your own
models directory. Open a workflow JSON in the ComfyUI editor or POST it to
`/prompt`; replace the placeholder asset names (`first_frame.png`,
`driving.mp4`, `mask.mp4`) with real files in ComfyUI's input folder.

The LTX lane uses stock ComfyUI nodes only. The Animate lane needs two
repo-pinned custom packs (linked into `custom_nodes` by the project setup):
`animate/ComfyUI-MMToolsAnimate` for the GPU-only loaders and
`animate/ComfyUI-WanAnimatePreprocess` for the ONNX pose pipeline.

Both lanes ship INT8 ConvRot defaults with an NVFP4 Blackwell twin in the same
bundle; switch the loader node's model name to the `.nvfp4.` file to use it.
