# WorldSculpt Scene Foundry

## Product boundary

This application is the complete **released** WorldSculpt inference path, not a
thin command launcher. A job safely expands a calibrated scene package, audits
its topology, builds camera-aligned instance crops, runs the released sparse and
1024-shape multi-view finetunes, composes every object back into metric world
space, renders calibrated proof frames, exports interactive GLBs, plots spatial
diagnostics, and optionally preserves a resumable project archive.

The published checkpoint currently contains sparse-structure and shape stages.
It does not contain WorldSculpt's texture finetune. The studio therefore labels
its outputs honestly as geometry/normal assets and never substitutes an
unrelated texture model or attempts a network download at runtime.

## Local data flow

1. The shared Studio server stores uploads under the ignored runtime asset
   vault and queues jobs in SQLite.
2. The adapter expands ZIP/TAR input into a job-scoped directory. Absolute
   paths, parent traversal, symlinks, hard links, devices, FIFOs, excessive
   member counts, and expansion bombs are rejected.
3. `prepare_crops_scene.py` preserves the supplied cameras, creates canonical
   object cubes, and generates projected instance crops.
4. `reconstruct_batch.py` loads the Pixal3D stack once per scene, swaps in both
   released WorldSculpt denoisers/IBR aggregators, and emits atomic `mesh.pt`
   states for every object. Upstream `--low_vram` is intentionally never used.
5. `compose_scene.py --normal` cleans/decimates each geometry, applies its
   `T_canon_to_metric`, exports multi-object and merged GLBs, and creates
   original/predicted camera proofs.
6. `visualize_pointcloud.py` exposes mesh and fused backprojection clouds,
   anchors, camera directions, canonical cubes, and top-down maps.
7. Selected delivery files move into the durable output library. The temporary
   expanded package and work tree are deleted even after cancellation/failure.

## Shared dependency contract

WorldSculpt owns `world/.venv` while using the project-local Pixal3D checkpoint,
DINOv3, BiRefNet, MoGe-2, NAF, O-Voxel, CuMesh, and FlexGEMM runtimes. Its setup
installs the core and WorldSculpt modules into that environment. Models are not copied.

The canonical 5090 path is PyTorch fused SDPA plus FlexGEMM sparse convolution.
The grouped ragged-SDPA shim batches equal-length views, so FlashAttention and
xFormers are not runtime requirements. `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, and telemetry opt-outs are set by both launchers and
every inference subprocess.

## Scene package contract

Each scene root must contain a `transforms.json` with `frames` and `instances`.
Every frame path is resolved relative to that scene. The upstream layout also
expects the masks referenced by its instance metadata beneath `masks/`. A
package may contain several scene roots; Inspect package reports their exact
relative paths for deterministic selection.

## Operations

```bash
cd ~/multimedia/sculpting/world
./setupwithuv.sh
./startwithuv.sh
```

Default listener: `http://127.0.0.1:8263`. Copy `.env.local.example` to
`.env.local` only when changing the private binding or token.
