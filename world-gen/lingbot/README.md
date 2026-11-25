# LingBot World Studio

LingBot-World 2.0 is a real-time interactive world model: open-ended
camera-driven video generation with an unbounded interaction horizon,
diverse interactive elements, and an agentic pilot/director harness. The
studio runs the pinned native causal inference code with KV caching,
processing frames chunk-by-chunk on GPU 0.

The studio tier runs the official 14B causal-fast package with the DiT
converted to BF16 at load and the UMT5 encoder pinned to system RAM; it
targets one 64 GB+ GPU (the fp32 DiT is roughly 37 GiB resident in BF16 plus
the KV window). Consumer cards from 8 GB run the same 14B model as GGUF
through the bundled `ComfyUI_Rebels_LingBotWorld/` custom nodes in any
ComfyUI install: the quantized DiTs stream from RAM and the GGUF UMT5 encoder
replaces the large T5 checkpoint. Frame counts normalize to the native 4n+1
causal timeline, and DiT/VAE residency stays on CUDA during inference.

Camera control speaks the native contract: OpenCV camera-to-world matrices
and pinhole intrinsics. The UI generates locked, dolly, truck, crane, pan,
orbit, handheld, or arbitrary keyframed paths, previews them, and
exports/imports reusable ZIP or NPZ bundles. Every run keeps its complete
request, camera arrays, metadata, path preview, and MP4 together in the
output library.

## Install and start

From the project folder:

```bash
../../models/download_models.py lingbot:1   # native 14B tier (the studio)
./setupwithuv.sh     # verifies weights, then builds the local venv
./startwithuv.sh     # serves http://127.0.0.1:8267
```

For consumer GPUs, run `../../models/download_models.py lingbot` and pick a
GGUF tier (2–5); copy `ComfyUI_Rebels_LingBotWorld/` into any ComfyUI
`custom_nodes/` and load the bundled `LINGBOT WORLD GGUF workflow.json`.
Tier 6 is the NVFP4 export for SGLang serving on 72 GB+ Blackwell cards.

Direct chunked inference also runs through `run_fast.sh` with a weights
directory and frame count. ARCHITECTURE.md documents the model tiers,
the camera contract, and the retention policy.

## Upstream

- Code: https://github.com/robbyant/lingbot-world-v2
- Weights: https://huggingface.co/collections/robbyant/lingbot-world-v2
- Project page: https://technology.robbyant.com/lingbot-world-v2
- Paper: https://arxiv.org/abs/2607.07534
- GGUF nodes (vendored at `f310acc1109e7af447d5c6efb94f29fe746408a1`):
  https://github.com/RealRebelAI/Rebels_LingBot-World-V2_GGUF_ComfyUI
- GGUF quants (downloader tiers 2–5): https://huggingface.co/realrebelai/LingBot_World_V2_ComfyUI
- NVFP4 export (tier 6): https://huggingface.co/igorls/LingBot-World-V2-14B-Causal-Fast-NVFP4
