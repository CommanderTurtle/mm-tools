# Sculpting Studio

Image-to-3D and multiview sculpting on a single 5090, with four production
lanes:

- TRELLIS.2 single-reference reconstruction, with every sparse occupancy,
  structured shape, texture, cascade, seed, and token control exposed
- Pixal3D camera-aware reconstruction with automatic MoGe-2 calibration or
  exact manual lens/distance controls
- Shape-conditioned PBR retexturing for an uploaded mesh and reference image
- A model-free mesh lab for inspection, repair, simplification,
  normalization, and GLB/OBJ/PLY/STL delivery

Every generative lane supports UV/remesh parameters, 1K-4K texture baking,
deterministic multi-candidate seeds, local PBR turntables, and a
dependency-free WebGL2 viewer (orbit, zoom, wire inspection, A/B comparison)
that needs no CDN. The sparse pipeline runs on locally built FlexGEMM/SDPA
backends with o-Voxel delivery, and the verified 32 GiB profile is the 1024
cascade with a 65536-token ceiling; heavier profiles are rejected rather
than silently offloaded to system memory.

The WorldSculpt sub-studio under `world/` covers scene-to-models generation
on its own virtualenv at http://127.0.0.1:8263.

## Install and start

From the project folder:

```bash
../models/download_models.py sculpting
./setupwithuv.sh     # builds the sparse backends and verifies weights
./startwithuv.sh     # main studio at http://127.0.0.1:8262
cd world && ./setupwithuv.sh && ./startwithuv.sh   # WorldSculpt at :8263
```

ARCHITECTURE.md documents the lanes, the 5090 contract, and the viewer.

## Upstream

- Reconstruction: https://github.com/microsoft/TRELLIS.2
- Comfy nodes: https://github.com/dreamrec/ComfyUI-Pixal3D
- Scene lane: https://github.com/AlayaLab/WorldSculpt
