# LTX-2.5 Pipelines

Distilled 22B audio-and-video recipes, exported from the studio adapter in
`./local_app/adapter.py` (the graphs it builds at request time).
Each file pins the official two-stage sampling schedule; node ids mirror the
adapter's creation order so the exports diff cleanly against the source.

| File | Recipe | Canvas rule |
| --- | --- | --- |
| `t2v.json` | Text to video | Half canvas (640x360) + x2 latent spatial upscale between stages |
| `i2v.json` | Image to video | Half canvas; first frame pinned at strength 0.7 in stage 1, re-pinned at 1.0 after the upscale |
| `flf2v.json` | First + last frame | Full canvas (1280x720) in a single stage; both endpoints pinned via guide nodes at strength 0.7 |

Pinned constants shared by all three lanes:

- Sigma schedules: stage 1 `1.0, 0.99375, ..., 0.421875, 0.0`; stage 2 `0.85, 0.7250, 0.4219, 0.0` (FLF2V runs stage 1 only, then crops the guides).
- Dual CFG 1/1 with the published negative prompts verbatim; FLF2V uses the long cinematic negative and `SamplerEulerAncestral` (eta 0, s_noise 1) per the official template.
- Frame rule `seconds x FPS + 1` under the 1,000-frame audio latent ceiling; defaults are 5 s @ 24 FPS = 121 frames, seed 42.
- Tiled VAE decode (512/64 spatial, 64/16 temporal), MP4 H.264 CRF 18 delivery.
- Image conditioning enters through `LTXVPreprocess` at compression 18.

Weights default to INT8 ConvRot. The NVFP4 Blackwell twin ships in the same
bundle; point the `UNETLoader` at `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` to switch. The E2B prompt-enhancer encoder
(`TextGenerateLTX2Prompt`) is a studio-side feature and is intentionally not
part of these exports.
