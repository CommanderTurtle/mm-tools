# Fire3D Scene Reconstruction

Fire3D reconstructs simulation-ready 3D scenes from a single RGB image or
casual RGB video in a single feed-forward pass: object-level 6-DoF pose,
oriented boxes, textured meshes, and a composed scene GLB, with no
test-time optimization. The studio runs the released pipeline locally; the
`fire3d` CLI additionally exposes the frozen release protocols for the
iTHOR, Imaginarium, ScanNet++, and single-image datasets.

Reconstruction cascades sparse-structure, shape, and PBR flow-matching
models over a shared point cloud; HC-VAE compression lets many scene
instances batch through flow sampling and VAE decode. Sparse VAE decoders
and the O-Voxel/CuMesh postprocessor (built by setup into this project's
`.venv`) produce individually transformable textured meshes.

## Install and start

From the project folder (needs a CUDA-capable NVIDIA driver and a C++
compiler):

```bash
../../models/download_models.py fire3d
./setupwithuv.sh     # builds the O-Voxel/CuMesh CUDA extensions and verifies weights
./startwithuv.sh     # serves http://127.0.0.1:8268
```

CLI usage for the frozen dataset protocols:

```bash
.venv/bin/fire3d download --models
.venv/bin/fire3d infer --dataset ithor --scene-id iTHOR_FloorPlan312_physics --gpu 0
```

Outputs land under `results/<dataset>/<scene-id>/`; every run records its
resolved protocol and exact subprocess commands. ARCHITECTURE.md documents
the protocol table, the HC-VAE design, and the repository layout.

## Upstream

- Code: https://github.com/xiahongchi/Fire3D
- Weights: https://huggingface.co/hongchi/Fire3D
- Paper: https://arxiv.org/abs/2609.08848
