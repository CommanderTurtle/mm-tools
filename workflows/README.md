# ComfyUI Pipeline Exports

Committed copies of the exact ComfyUI API graphs the in-repo video studios run
against the shared pinned Comfy checkout. The studio adapters build these same
graphs at request time; the exports here are the static, editable form of each recipe.

## Layout

| Folder | Contents |
| --- | --- |
| `LTX/` | LTX-2.5 distilled recipes: text to video, image to video, first + last frame |

## Install and run

From the repository root, pull the allowlisted weights with the central
downloader:

```bash
cd models && uv run download_models.py ltx
```

Run any ComfyUI instance against these weights by pointing its models folder
at `V2V/ComfyUI/models/` or by copying the downloaded files into your own
models directory. Open a workflow JSON in the ComfyUI editor or POST it to
`/prompt`; replace the placeholder asset names (`first_frame.png`,
`last_frame.png`) with real files in ComfyUI's input folder.

The exported LTX pipelines use stock ComfyUI nodes only. They ship INT8
ConvRot defaults with an NVFP4 Blackwell twin in the same bundle; switch the
loader node's model name to the `.nvfp4.` file to use it.
