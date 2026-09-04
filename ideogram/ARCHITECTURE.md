# Ideogram Architecture and Operations

This document covers the runtime shipped in this monorepo: local Ideogram 4 generation, the editing workbench, its private masked-edit engine, and the optional external ComfyUI adapter.

## Architecture

```mermaid
flowchart LR
    Image["Input image"] --> Painter["Browser painter and fuzzy mask"]
    Painter --> ObjectClear["ObjectClear reconstruction"]
    Painter --> OpenCV["OpenCV deterministic repair"]
    Painter --> Crop["Mask bounds + context, explicit processing size"]
    Crop --> Caption["Optional local Qwen3-VL edit caption"]
    Caption --> Private["Private Comfy: Ideogram + differential diffusion"]
    Crop --> Private
    Private --> Composite["Composite masked pixels into original-size image"]
    Composite --> Edited
    Image --> BiRefNet["BiRefNet background cutout"]
    ObjectClear --> Edited["Local PNG"]
    OpenCV --> Edited
    BiRefNet --> Edited
    Painter --> ComfyPNG["Image plus alpha mask PNG"]
    Prompt["Structured caption"] --> Generator["Local Ideogram 4 FP8 pipeline"]
    Generator --> Generated["Generated image"]
```

ObjectClear, BiRefNet and OpenCV keep their existing implementations and defaults. The optional **Ideogram 4 masked edit** engine uses the same local base checkpoint for image-conditioned masked sampling, not a hosted edit API or the generation-only CLI. The painter still exports a standard mask or one PNG that Comfy reads as both IMAGE and MASK.

## Masked editing in the WebUI

1. Open an image and paint the object/region to change, or use **Lasso** to draw a freehand outline (release to fill it); choose **Ideogram 4 masked edit**. Multiple lasso/brush selections accumulate in the same mask, and Erase refines it.
2. Describe the desired result. **Draft edit caption locally** lets you review/edit the JSON before sampling. Leave JSON empty to draft automatically. Supplied valid JSON bypasses the caption model completely.
3. Choose the crop processing size, context padding and mask feather. Apply the masked edit.
4. Download the full-size lossless PNG. The source canvas is not replaced by the result.

For **background replacement**, paint the foreground to preserve, select **Edit outside the selection**, and describe the new background. For a **transparent background**, either use the unchanged automatic BiRefNet button or paint the foreground and choose **Cut out selected foreground** (mask-only, no model, no resizing). Ideogram inpainting does not return a segmentation alpha mask; these are deliberately different operations.

Coordinate boxes are only crop/caption hints, not rectangular edit masks. Include a related shadow or mirror reflection with another brush/lasso selection when it should disappear too. Strict final compositing cannot remove something outside the mask; automatic semantic discovery of related regions is not implemented or claimed. A crop must contain enough context, and the caption must describe the desired remaining scene.

### Resolution and preservation

The server bounds the selected region, adds context and feather support, and scales only that crop to the selected maximum edge. Model dimensions are multiples of 16 and at least 256. A 4K/8K source is not silently reduced to 512 and upscaled as a whole. The returned crop is blended into the original: all pixels outside the feathered selection and existing alpha are preserved. A feather may intentionally modify pixels just outside the hard brush edge. Metadata/profile preservation is not promised.

512–2048 are practical processing choices; 4096/8192 are explicit experimental options, not a quality or VRAM guarantee. A large or inverted selection can cover the entire image, in which case that entire selected region is processed at the chosen cap. The unchanged ObjectClear lane still uses its original 512-pixel short-side implementation. Undo retains at most 256 MiB worth of masks (one full mask minimum), rather than twenty unbounded 8K snapshots.

### Private engine, local weights

`private_comfy.py` owns its subprocess, startup checks, deadline and cleanup. It uses the **vendored runtime source** at `../minimax/runtime`, not the Minimax application or its running service. This avoids duplicating Comfy in the runtime-only monorepo. It binds only to loopback (default `8175`); the workbench remains LAN-accessible at `0.0.0.0:8174`. No external Comfy URL is accepted. An occupied port is an error, not permission to use or kill that service.

The existing Minimax virtualenv is reused when present; `IDEOGRAM_COMFY_PYTHON` can instead point to an independently provisioned interpreter. All input/output/temp/user/database/log files are confined to `ideogram/.runtime/editing`, not the shared runtime tree. Uploaded crop files and returned output files are deleted after use. Failed/crashed runs may leave diagnostics or partial artifacts there. No source image or downloaded result is deleted. Stop/shutdown kills only the owned process group. A force-kill of the web server cannot execute normal Python cleanup.

Both successful and failed caption/edit requests stop the private engine afterward. Captioning stops before loading the large diffusion models. Switching to Ideogram unloads the other **workbench-owned** models to release VRAM; unrelated services are never stopped. “Start engine” checks availability and starts Comfy, but does not preload 27 GB of weights.

Captioning uses a concise edit-specific schema, not the generation prompt's instructions to invent visual embellishments. The upstream validator checks the response before sampling. An invalid response receives at most one local formatting-repair attempt; persistent invalid JSON is an explicit error, not an infinite retry or a fallback to a hosted model. The mask/cutout routes reject decoded images above `OBJECT_REMOVER_MAX_PIXELS` (100 million by default) before processing; ordinary 4K/8K images fit.

Weights:

- `IDEOGRAM4_FP8_MODEL`: the existing `models/ideogram-ai--ideogram-4-fp8` folder, containing the conditional/unconditional DiTs, text encoder and VAE.
- `IDEOGRAM_CAPTION_MODEL`: by default the existing `models/qwen/text-encoder-vl-nvfp4/qwen3_vl_4b_nvfp4_full.safetensors`. This is a local vision-language drafting helper, not a new dependency on a music model. Its input is the unpainted crop, the selection bounds and your instructions. The mask itself controls diffusion, not the caption model. Instructions and editable JSON describe the intended result; drafting is not guaranteed to infer the correct removal from an ambiguous mask.

`comfy_nodes/` preserves the official checkpoint's **per-output-row FP8 scales**. It does not rename them into a different quantization format, requantize, or download converted checkpoints. Storage remains FP8; each linear operation casts and applies its row scales before matrix multiplication, matching the native loader. Ordinary Comfy offloading manages the models; dynamic-weight patching is disabled for these custom loaders. Layout mismatches fail explicitly. Conditioning uses the native 8B language tensors and 13 hidden-layer taps; the unused vision tensors are not loaded for conditioning. Captioning instead uses Comfy's existing Qwen3-VL generation node.

The graph is `VAEEncodeTiled → SetLatentNoiseMask → DifferentialDiffusion → DualModelGuider → Ideogram4Scheduler/Euler → VAEDecodeTiled`. Both diffusion models are used. Denoising strength trims the schedule; the UI guidance is a constant CFG, not the upstream generation preset's multi-stage CFG schedule. No LoRA or patched Comfy core is required for this path.

### Research basis and boundaries

- [OzzyGT's Ideogram 4 inpainting implementation](https://huggingface.co/blog/OzzyGT/ideogram-4-inpainting) demonstrates JSON-conditioned image/soft-mask differential diffusion. This workbench implements that sampling approach using our native Comfy runtime and existing FP8 files; it does not download/execute the article's remote Diffusers blocks.
- [Official Ideogram 4 source](https://github.com/ideogram-oss/ideogram4) supplies the checkpoint architecture, native FP8 math and caption validation.
- [BitPoet's reference-editing workflow](https://github.com/BitPoet/ComfyUI-bitpoet-IG4Inpaint) is a separate LoRA/core-modification approach, not silently mixed into this one.

This integration does not claim a published automatic Ideogram alpha-matting model, guaranteed superior removal, or benchmarked native 8K quality. Follow the checkpoint's bundled model license, including its non-commercial terms where applicable.

## First setup (fresh environment only)

```bash
cd ~/multimedia/ideogram
uv venv --python 3.12.10 --seed --managed-python .venv
source .venv/bin/activate
uv pip install --torch-backend auto torch torchvision
uv pip install -r requirements-local.txt
cp .env.local.example .env
```

The shared model downloader's Ideogram selection supplies FP8, ObjectClear, BiRefNet and the local Qwen caption helper. The helper is shared with Minimax and deduplicated when both projects are selected. Runtime code does not download checkpoints.

Existing installations should **not** recreate their virtualenv or overwrite `.env`. The already-provisioned Minimax runtime/interpreter is sufficient for the private Comfy lane. If using a separate interpreter, install `../minimax/runtime/requirements.txt` into that environment with `uv pip install --python /path/to/python --torch-backend YOUR_BACKEND -r ../minimax/runtime/requirements.txt`, then set `IDEOGRAM_COMFY_PYTHON`. No model files are installed by that command. The optional local caption model is the same one used by Minimax's prompt helper; set its path or supply caption JSON manually.

New settings are documented in `.env.local.example`; omitted settings have defaults. Restart only the editing workbench after updating its Python code. No Minimax service restart is needed.

## Checks

```bash
cd ~/multimedia/ideogram
.venv/bin/python -m unittest discover -s tests -v
../minimax/.venv/bin/python tests/comfy_contract.py
bun tests/selection_contract.js
```

The first suite tests crop/alpha preservation, masks, caption validation, bounded repair and lifecycle guardrails. API-client checks additionally use the optional `httpx` test dependency, and skip if it is absent. The second uses CPU mode, tiny tensors and checkpoint **headers**, then starts/stops a private CPU Comfy instance to validate node contracts; it does not run large-model inference. The JavaScript contract checks the real painter's event handlers with mock canvas contexts (lasso commit/cancel, additive regions, pointer isolation, brush/erase and undo); it does not launch a browser. Temporary test directories are separate from production state.

During integration a real local Qwen caption + 20-step Ideogram smoke test removed a brown square from a synthetic 512×512 image. A separate 4-step test exercised the graph but was insufficient to remove the square. These are integration checks, not comparisons against ObjectClear or evidence of photographic/4K/8K quality. No model weights were downloaded, no production virtualenv was reinstalled, and no other service was restarted.

## Runtime lanes

- Object-removal browser: `./start-object-remover.sh` on port `8174` by default.
- Object-removal CLI:

  ```bash
  python -m object_remover.cli image.png mask.png cleaned.png \
    --method telea --radius 5 --grow 5 --feather 4
  ```

- Ideogram generation: set `PYTHONPATH="$PWD/src"`, then run `python local_generate.py --help`.

  ```bash
  export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
  python local_generate.py \
    --prompt-file caption.json --preset V4_DEFAULT_20 --output design.png
  ```

- ComfyUI adapter: place `ComfyUI-Ideogram4` under an existing ComfyUI `custom_nodes` directory and import `workflows/ideogram-local-workflow.json`.

All weights stay under `~/multimedia/models`; outputs and local `.env` values remain untracked.
