# Animate 2 Motion Studio

The product surface is the shared mm-tools Studio; native inference is a
job-scoped instance of the PR-pinned Comfy runtime under `../../V2V/ComfyUI`.
Comfy listens on an ephemeral loopback port, its editor is never exposed, API
nodes are disabled, metadata is disabled, and inference runs on CUDA while
Comfy's native memory management places idle weights between the GPU and host
RAM. Weight staging is disk-backed (`--fast-disk`): staged checkpoints stay
mmap-resident behind the page cache instead of a pinned host buffer, because
the target host's RAM is below the pinned-ring budget dynamic VRAM would
otherwise reserve. Each lane whitelists exactly the repository-pinned packs it needs:
`mmtools_animate` for the creation lanes, plus `ComfyUI-WanAnimatePreprocess`
for the pose lab and character replacement.

`motion_transfer` constructs Wan Animate 2 directly and has two local INT8
lanes. Native distilled inference uses the upstream 10-step Euler/no-CFG
recipe. LightX2V loads the base INT8 model plus the rank-64 acceleration LoRA;
its recommended profile mirrors Comfy's published six-step LCM graph, with a
separate literal four-step speed profile. Only the selected model set loads;
inference stays on CUDA and idle weights follow Comfy's native placement. The graph retains the complete conditioning
surface: separate character and pose prompts, CLIP vision on both inputs,
independent reference and motion strengths, denoising-time motion windows,
continuation frames and offsets, manual context windows, FreeNoise/fusion
options, optional ConvRot GPU caches, explicit manual sampler overrides,
cropping, and encoded delivery.
 Resolution is two-stage: the generation canvas is fitted to the selected
 pixel budget (a 720p-class default, an opt-in 1080p class, and a 480p class
 matching the LightX2V rank-64 distill LoRA's 480p training resolution), and
 the delivered video matches the requested resolution up to the 2160x1440
 delivery ceiling, scaled once with lanczos when needed.
The studio exposes two independent Blackwell switches, each defaulting to the
verified non-NVFP4 file and effective on its own. A DiT weights switch swaps
the base INT8 ConvRot DiT, and the scene-replacement DiT, for LHQAQ-Li's
fp16/NVFP4 twin; a text-encoder switch swaps the FP8-scaled UMT5-XXL for
rst220's NVFP4 twin, so the lighter encoder can free VRAM while the DiT stays
INT8. The distilled profile remains INT8-only for the DiT and rejects the DiT
switch with a precise error, but accepts either encoder. Only the selected
files load; inference stays on CUDA and idle weights follow Comfy's native placement.

`character_replace` preserves the source scene: the driving video supplies the
stage, lighting, and camera while a replacement character performs inside it.
A local ViTPose pass tracks the original performer, the shared sculpting
BiRefNet model isolates them frame by frame into a white-on-black matte, and
the graph conditions the unmasked pixels of every frame back into the result.
The lane runs the Wan 2.2 Animate Mix model on the official 77-frame
single-pass window at the 480p LightX2V distill canvas and delivers at the
source resolution up to the 2160x1440 ceiling. A prepared mask video can
replace the automatic matte when it is already available.

`pose_lab` is deliberately separate. Animate 2 can consume video end-to-end,
but difficult source material benefits from an inspectable CUDA ViTPose/YOLO
pass. The lab emits the full skeleton movie, synchronized face crops, and body
point JSON with optional proportion retargeting.

Uploads, queue state, logs, settings receipts, and outputs stay inside
`workflows/animate/.runtime/studio`. The default listener is `127.0.0.1`; no telemetry,
cloud API, remote asset, or CORS policy is enabled.
