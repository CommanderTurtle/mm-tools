# Animate 2 Motion Studio

The product surface is the shared mm-tools Studio; native inference is a
job-scoped instance of the PR-pinned Comfy runtime under `../V2V/ComfyUI`.
Comfy listens on an ephemeral loopback port, its editor is never exposed, API
nodes are disabled, metadata is disabled, and every model stays on CUDA. The
only optional custom node allowed to load is the repository-pinned
`ComfyUI-WanAnimatePreprocess` pack used by the diagnostic mode.

`motion_transfer` constructs the current Comfy-Org distilled Wan Animate 2
graph directly. It retains the complete native conditioning surface: separate
character and pose prompts, CLIP vision on both inputs, independent reference
and motion strengths, denoising-time motion windows, continuation frames and
offsets, manual context windows, FreeNoise/fusion options, ConvRot GPU caches,
sampler/scheduler/shift/CFG, cropping, and encoded delivery.

`pose_lab` is deliberately separate. Animate 2 can consume video end-to-end,
but difficult source material benefits from an inspectable CUDA ViTPose/YOLO
pass. The lab emits the full skeleton movie, synchronized face crops, and body
point JSON with optional proportion retargeting.

Uploads, queue state, logs, settings receipts, and outputs stay inside
`animate/.runtime/studio`. The default listener is `127.0.0.1`; no telemetry,
cloud API, remote asset, or CORS policy is enabled.
