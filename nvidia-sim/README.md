# NVIDIA Motion + Body Studio

Two local systems, preserved as-is behind one studio:

- ARDY turns text descriptions into articulated motion: constrained
  diffusion over autoregressive history, foot cleanup, and skeleton output
  as NPZ arrays plus G1 MuJoCo qpos.
- SOMA-X provides parametric body and hand meshes: identity PCA, procedural
  twist joints, LOD topology, skinning, and pose correctives, exported as
  GLB with the native faces alongside lossless NPZ arrays.

All four released ARDY checkpoints and the quality-preserving merged INT8
LLM2Vec encoder are local and checked before setup. The merged encoder
replaces the old gated BF16 assembly without changing ARDY's 4096-dimensional
conditioning contract. Models load only on CUDA device 0 with no offload
path, Hugging Face and Transformers run forced-offline, the listener is
loopback, and the browser contains no remote assets.

## Install and start

From the project folder:

```bash
../models/download_models.py nvidia-sim
./setupwithuv.sh     # verifies checkpoints and the SOMA-X runtime, then builds the venv
./startwithuv.sh     # serves http://127.0.0.1:8266
```

ARCHITECTURE.md documents the ARDY/SOMA split, the GPU and privacy contract,
and the native output formats.

## Upstream

- Motion (ARDY): https://research.nvidia.com/labs/sil/projects/ardy/
- Bodies: https://github.com/NVlabs/SOMA-X
