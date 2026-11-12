# NVIDIA Motion + Body Studio

The studio preserves the two native systems rather than translating either
into a generic abstraction. ARDY owns text conditioning, constrained diffusion,
autoregressive history, foot cleanup, skeleton arrays, and G1 MuJoCo qpos.
SOMA-X owns identity PCA, procedural twist joints, LOD topology, skinning, and
pose correctives.

`local_app/adapter.py` is the narrow bridge into the shared local studio. ARDY
runs through its pinned `scripts/generate.py`; the adapter adds a compact motion
document consumed by the dependency-free browser skeleton player. SOMA body and
hand modes call the pinned library directly, export GLB with the native faces,
and retain NPZ arrays for lossless downstream work.

## GPU and privacy contract

- The four released ARDY checkpoints and the quality-preserving merged INT8
  LLM2Vec encoder are local and explicitly checked before setup.
- The merged encoder replaces the old gated BF16 assembly without changing the
  4096-dimensional ARDY conditioning contract.
- Models are loaded only on CUDA device 0. CPU conversion is limited to output
  serialization after inference; there is no model offload path.
- Hugging Face and Transformers are forced offline at runtime. The listener is
  `127.0.0.1` by default and the browser contains no remote assets or analytics.

## Native outputs

- ARDY: NPZ (`posed_joints`, rotations, root positions, contacts, text, FPS),
  optional G1 MuJoCo qpos CSV, and `.motion.json` for the studio player.
- SOMA-X: GLB mesh, vertices/faces/joints/pose/identity arrays, joint hierarchy,
  and a human-readable build receipt.

The `.motion.json` file is a preview index, never the authority. The original
NPZ is always retained unchanged.
