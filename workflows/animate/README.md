# Wan Animate 2 Pipelines

Exported from the studio adapter in `animate/local_app/adapter.py` (the graphs
it builds at request time); node ids mirror the adapter's creation order so
the exports diff cleanly against the source. All lanes pin the official 21/8
context-window schedule (static pyramid, FreeNoise, causal fix), 480p canvas
(480x832), and MP4 H.264 CRF 18 delivery.

| File | Recipe | Profile |
| --- | --- | --- |
| `motion-transfer-distilled.json` | New prompt-directed scene from a driving video | Distilled INT8 DiT, 10-step Euler, no LoRA |
| `motion-transfer-lightx2v.json` | Same graph, LightX2V recipe | Base INT8 DiT + rank-64 distill LoRA, 6-step LCM |
| `character-replacement.json` | Swap the performer, keep the stage | Wan 2.2 Animate 14B INT8 + rank-64 LoRA, fixed 4-step LCM, prepared character-mask video |
| `pose-lab.json` | Inspection only | ONNX ViTPose/YOLOv10m detection, pose movie + face crops + body-points JSON |

Notes on the exported defaults:

- Placeholder assets: `reference.png`, `driving.mp4`, `mask.mp4` in the ComfyUI input folder; the `Video Slice` windows assume a 24 FPS source and are sized to the lane length (81 or 77 frames).
- The motion-transfer lanes round the frame count to 4n+1 and cap it at 1,921 frames; replacement is fixed at 77 frames by the Wan 2.2 Animate recipe.
- Character replacement takes a prepared grayscale mask video (red channel). The studio's automatic matte (local BiRefNet pass over the driving video) happens before the graph and is not part of these exports.
- Prompt text in `CLIPTextEncode` nodes is an example; the replacement lane zero-outs the negative conditioning per the upstream recipe.
- NVFP4 switch: point the loader at `wan2.2_animate_14b_fp16_nvfp4_comfy_V2.safetensors` and the UMT5 loader at `UMT5_XXL_NVFP4.safetensors`. The distilled profile ships INT8-only.

Custom nodes: everything except two repo-pinned packs is stock in the pinned
ComfyUI runtime. The packs (linked into `custom_nodes` by the project setup)
provide the GPU-only loaders (`animate/ComfyUI-MMToolsAnimate`:
`MMToolsGpuOnlyWanLoader`, `MMToolsGpuOnlyWan22Loader`) and the ONNX pose
pipeline (`animate/ComfyUI-WanAnimatePreprocess`: `OnnxDetectionModelLoader`,
`PoseAndFaceDetection`, `DrawViTPose`).
