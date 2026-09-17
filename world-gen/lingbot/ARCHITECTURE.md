# LingBot World Studio architecture

The product surface is the private shared Studio server. Jobs are durable,
queued one-at-a-time on GPU 0, and run the pinned native causal inference code.
There are no cloud calls, telemetry hooks, hosted model fallbacks, or browser
automation paths.

## RTX 5090 profile

The released 14B BF16 checkpoint is documented but its roughly 70 GB DiT is
not retained because it is not runnable on one 32 GB card. Its shared UMT5
encoder, tokenizer, and Wan VAE remain local. A community hybrid-NVFP4 export
was also rejected for this profile: its own measured peak is 52–56 GiB with no
CPU offload. The canonical workstation path uses the official 1.3B causal-fast
DiT (about 6.8 GB) with the official 14B package's shared UMT5,
tokenizer, and VAE. All model residency remains on CUDA during inference;
`--offload_model false`, `--t5_cpu` disabled, one process, Ulysses degree one.

## Camera contract

LingBot consumes OpenCV camera-to-world matrices in `poses.npy` and rows of
`fx, fy, cx, cy` in `intrinsics.npy`. The UI can generate locked, dolly,
truck, crane, pan, orbit, handheld, or arbitrary keyframed paths, preview them,
and export/import a reusable ZIP or NPZ bundle. Frame counts are normalized to
the native `4n+1` causal timeline.

The complete request, camera arrays, human-readable metadata, path preview,
and MP4 are retained together in the output library for reproducibility.
