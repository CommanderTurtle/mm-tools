# LingBot World Studio architecture

The product surface is the private shared Studio server. Jobs are durable,
queued one-at-a-time on GPU 0, and run the pinned native causal inference code.
There are no cloud calls, telemetry hooks, hosted model fallbacks, or browser
automation paths.

## Model tiers

The studio tier runs the official 14B causal-fast package: the fp32 DiT is
converted to BF16 at load (`--convert_model_dtype`), the UMT5 encoder is built
in system RAM (`--t5_cpu`), and `--offload_model false` keeps the DiT and VAE
resident on CUDA for the whole job, one process, Ulysses degree one. The DiT
is about 37 GiB resident and the upstream 18+6 KV window at 480×832 adds
roughly 21 GiB, so the profile targets single-GPU cards of 64 GiB and up;
shrinking the window releases KV memory, not quality of residency.

Consumer cards run the same 14B model through the vendored
`ComfyUI_Rebels_LingBotWorld/` custom nodes: GGUF DiTs (Q4_K_S through Q8_0)
stream from pinned system RAM and a UMT5-XXL encoder GGUF replaces the
11 GiB T5 checkpoint, from 8 GiB of VRAM upward. The node folder drops into
any ComfyUI `custom_nodes/`; the matching weights come from downloader tiers
2–5 and the reference workflow ships with them. The node README documents the
quant, window, and resolution matrix.

The community hybrid-NVFP4 export (SGLang serving, SM120 Blackwell, 72 GiB+)
is downloader tier 6; the snapshot carries its patch, apply script, and
SHA256SUMS, and it is served outside the studio. The retired 1.3B single-5090
experiment remains reachable through the FastFrames lane.

## Camera contract

LingBot consumes OpenCV camera-to-world matrices in `poses.npy` and rows of
`fx, fy, cx, cy` in `intrinsics.npy`. The UI can generate locked, dolly,
truck, crane, pan, orbit, handheld, or arbitrary keyframed paths, preview them,
and export/import a reusable ZIP or NPZ bundle. Frame counts are normalized to
the native `4n+1` causal timeline.

The complete request, camera arrays, human-readable metadata, path preview,
and MP4 are retained together in the output library for reproducibility.
