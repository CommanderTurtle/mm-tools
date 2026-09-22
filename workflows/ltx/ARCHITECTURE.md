# LTX-2.5 Studio Architecture

The product surface is the shared mm-tools Studio; native inference is a
job-scoped instance of the PR-pinned Comfy runtime under `../V2V/ComfyUI`.
Comfy listens on an ephemeral loopback port, its editor is never exposed, API
nodes are disabled, metadata is disabled, and inference runs on CUDA while
Comfy's native memory management places idle weights between the GPU and host
RAM. Weight staging is disk-backed (`--fast-disk`): staged checkpoints stay
mmap-resident behind the page cache instead of a pinned host buffer, because
the target host's RAM is below the pinned-ring budget dynamic VRAM would
otherwise reserve. No custom node is loaded for this lane.

`local_app/adapter.py` builds the three official graphs directly instead of
importing the blueprint JSON, so the studio pins every recipe constant by
name: the two-stage sigma schedules, the Euler ancestral sampler pair, the
dual-CFG guider at 1.0/1.0, the 0.7/1.0 first-frame pin strengths, the 1536
longer-side image resize, the compression-18 preprocessing, the 512/64 tiled
decode with 64/16 temporal tiling, and the official negative prompts.

`t2v` and `i2v` sample at half the requested canvas, run the x2 latent
spatial upscaler between stages, and deliver at the largest multiple of 32
inside the doubled half-canvas (1280x720 delivers 1280x704). `flf2v` samples
at the full canvas through `LTXVAddGuide` pins at frame 0 and frame -1, crops
the guides after sampling, and delivers at the largest multiple of 32 inside
the requested size. `CreateVideo` attaches the decoded audio track; delivery
container, codec, CRF, bit depth, and color space use the shared SaveVideo
flat-key pattern.

Prompt enhancement mirrors the optional template path: the small E2B encoder
rewrites the prompt (image-conditioned when a frame is present) before the
Gemma 4 12B projection encoder conditions sampling. It stays off by default.

Frame counts are whole seconds times FPS plus one, capped at 1,000 frames by
the `LTXVEmptyLatentAudio` ceiling; the adapter raises a precise error naming
the maximum duration at the selected frame rate instead of surfacing a schema
failure mid-run. Widths and heights are multiples of 8 between 256 and 2160.

Weights: the INT8 ConvRot distilled DiT is the verified default; the NVFP4
twin from the same bundle swaps in when selected and needs the matching files
from the ltx downloader bundle. The temporal upscaler and duration head ship
in the bundle for the long-form lane but are unused by these distilled
graphs. Only the selected model set loads; inference stays on CUDA and idle weights follow Comfy's native placement.

Uploads, queue state, logs, settings receipts, and outputs stay inside
`workflows/ltx/.runtime/studio`. The default listener is `127.0.0.1:8269`;
no telemetry, cloud API, remote asset, or CORS policy is enabled.
