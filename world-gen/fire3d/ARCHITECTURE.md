# Fire3D Scene Foundry

This folder preserves Fire3D's released inference code and adds a private,
loopback-only mm-tools studio.  The adapter does not pretend Fire3D is a raw
RGB-to-3D model.  Its single-image contract is an RGB frame plus an organized
point cloud with exactly `(height / 2) × (width / 2)` vertices.  The preparation
mode can create that contract from calibrated metric depth.

## RTX 5090 execution contract

The published reference configuration was demonstrated on substantially larger
VRAM.  The local profile makes the release pipeline bounded for a 32 GiB RTX
5090 without model CPU offload:

- perception, geometry, and PBR remain separate native stages;
- SS, Shape, and PBR flow parameters use BF16 CUDA residency, matching the
  release's BF16 autocast inference path;
- SS/Shape, PBR, decode, mesh, and raster batches are fixed to one object;
- model caching is disabled, so geometry is released before PBR is loaded;
- condition-token limits remain configurable but default to the release values;
- PyTorch uses the CUDA 13.0 wheel on the CUDA 13.3-capable host and builds
  native extensions for Blackwell `sm_120`.

This is a truthful compatibility profile, not a claim that the upstream authors
validated 32 GiB hardware.  Every request stores the exact modified protocol,
command, environment contract, input audit, and native summaries.  Those
artifacts make any runtime boundary diagnosable without weakening the input or
quality contract.

## Modes

- **RGB + aligned point cloud** validates and stages native single-image input.
- **Prepared bundle** safely extracts a reusable Fire3D ZIP.
- **Local release dataset** runs an already-installed official dataset layout.
- **RGB + metric depth** creates organized PLY geometry, camera metadata, and a
  reusable input bundle without loading CUDA models.
- **Inspect GLB** opens existing scenes in the shared dependency-free viewer.

Blender is optional.  The normal product path returns textured GLBs and uses the
native browser viewer.  Set `FIRE3D_WITH_BLENDER=1` while running `setupwithuv.sh`
only when release-style proof-sheet rendering is wanted.

## Privacy and persistence

The server binds to `127.0.0.1` by default, enables Hugging Face offline and
do-not-track flags, performs no telemetry or authentication calls, and stores
uploads, jobs, logs, protocols, and outputs beneath `.runtime/studio/`.
