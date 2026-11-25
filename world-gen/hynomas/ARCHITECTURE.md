# Hy-Nomas · studio

Continuous world generation with text, reference images and recorded camera motion.

```bash
cd models
uv run download_models.py hynomas
# Optional cinematic and dual-arm robotics adapters:
uv run download_models.py hynomas:1+2
cd ../world-gen/hynomas
./setupwithuv.sh
./startwithuv.sh
```

Open `http://127.0.0.1:8286`. Setup requires Linux, CUDA 13 toolkit, `uv`,
`bun` and `ffmpeg`. Paths derive from the checkout, including relative model links.

Load the model, describe a world or upload a reference image, then create its
first turn. Continue with directions or record a camera path using WASD, arrows
and right-drag. Import/export paths, tune translation/rotation, scrub the segment
timeline or roll back a turn. Timed native segment JSON can bypass prompt enhancement.

Save world state before releasing the model to resume later with the same adapter.
Generated videos, prompt JSON and local gallery entries persist independently.
MP4 export can include prompt/camera overlays; share links point to this local server.
Stop releases the model, prompt enhancer and active preparation process.

The default is the native single-GPU profile: INT8 DiT and KV, BF16 compute and
BF16 PixelUMM decoder, 1280 × 704 at 24 fps. CPU offload and native history handling
remain enabled. The FP8 decoder's SM89-only kernel is excluded from this profile.

Nomas-PE runs on demand in a separate CPU environment and exits after each request.
Its approximately 56 GB of weights add to offloaded model memory; a 64 GB host can
use substantial swap. An existing local Nomas-PE service can be selected explicitly:

```bash
./startwithuv.sh --pe-url http://127.0.0.1:8001/v1/chat/completions
```

For LoRAs, stop the model, select the adapter and choose **Prepare selected LoRA as
INT8**. Native CPU merge uses the optional FP32 base, then native quantization caches
an INT8 derivative. Allow temporary disk space and host-memory headroom for merging.

Source paths resolve through `paths.py`; checkpoints come from the central
[inventory](../../models/which-ones.txt).
