# LingBot World Studio

LingBot-World 2.0 is a real-time interactive world model: open-ended
camera-driven video generation with an unbounded interaction horizon,
diverse interactive elements, and an agentic pilot/director harness. The
studio runs the pinned native causal inference code with KV caching,
processing frames chunk-by-chunk on GPU 0.

The workstation profile pairs the official 1.3B causal-fast DiT with the
shared UMT5 encoder, tokenizer, and Wan VAE from the 14B package; the 14B
BF16 DiT (about 70 GB) is documented but not retained because it does not
fit one 32 GB card. Frame counts normalize to the native 4n+1 causal
timeline, and all residency stays on CUDA during inference.

Camera control speaks the native contract: OpenCV camera-to-world matrices
and pinhole intrinsics. The UI generates locked, dolly, truck, crane, pan,
orbit, handheld, or arbitrary keyframed paths, previews them, and
exports/imports reusable ZIP or NPZ bundles. Every run keeps its complete
request, camera arrays, metadata, path preview, and MP4 together in the
output library.

## Install and start

From the project folder:

```bash
../../models/download_models.py lingbot
./setupwithuv.sh     # verifies weights, then builds the local venv
./startwithuv.sh     # serves http://127.0.0.1:8267
```

Direct chunked inference also runs through `run_fast.sh` with a weights
directory and frame count. ARCHITECTURE.md documents the 5090 profile,
the camera contract, and the retention policy.

## Upstream

- Code: https://github.com/robbyant/lingbot-world-v2
- Weights: https://huggingface.co/collections/robbyant/lingbot-world-v2
- Project page: https://technology.robbyant.com/lingbot-world-v2
- Paper: https://arxiv.org/abs/2607.07534
