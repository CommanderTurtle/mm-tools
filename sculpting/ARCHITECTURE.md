# Sculpting Studio architecture

## Complete local product surface

`startwithuv.sh` opens a private asset workstation rather than the upstream
demo. It includes four production lanes:

- TRELLIS.2 single-reference reconstruction with every sparse occupancy,
  structured shape, structured texture, cascade, seed, and token control;
- Pixal3D camera-aware reconstruction with automatic local MoGe-2 calibration
  or exact manual lens/distance controls, four DINOv3 projection stages, and
  local NAF upsampling;
- shape-conditioned PBR retexturing for an uploaded mesh and reference image;
- a model-free mesh lab for inspection, repair, simplification, normalization,
  archive-safe reports, and GLB/OBJ/PLY/STL delivery.

Every generative lane exposes UV/remesh parameters, 1K–4K texture baking,
optional companion geometry, deterministic multi-candidate seeds, local PBR
turntables, diagnostic passes, still views, saved settings, and the shared
dependency-free WebGL2 GLB room. The viewer supports orbit, zoom, auto-orbit,
wire inspection, reset, fullscreen, and A/B comparison without a CDN.

## RTX 5090 contract

The verified workstation path is 1024 cascade and a 65,536 sparse-token hard
ceiling. Models are switched explicitly and remain GPU-resident while active.
The visible 1536 profile is rejected on a 32 GiB card because silently moving
model layers to system memory would violate the mm-tools runtime contract.

Dense and sparse attention use PyTorch SDPA. Variable-length sparse windows
are grouped by equal shape and processed in bounded fused batches, avoiding
both padded attention masks and the extra FlashAttention/xFormers dependency.
FlexGEMM remains the sparse-convolution backend. CuMesh, FlexGEMM,
nvdiffrast, nvdiffrec-render, and o-voxel are compiled locally with
`TORCH_CUDA_ARCH_LIST=12.0` against the pinned CUDA 12.8 PyTorch environment.

## Pinned auxiliary stack

`setupwithuv.sh` materializes exact source revisions beneath ignored
`.runtime/vendor`, removes their nested Git metadata, records each revision,
and installs them without build isolation. The model downloader separately
owns the allowlisted DINOv3, BiRefNet, MoGe-2, NAF, sparse decoder, TRELLIS.2,
and Pixal3D artifacts. Runtime loading is offline-only; localized pipeline
JSON points only at those durable paths.

## Privacy and durable state

Uploads, queue history, recipes, logs, and outputs live under the ignored
`.runtime/studio` directory. There are no remote fonts, analytics, model hub
lookups, or cloud fallbacks. The server binds to loopback by default. When an
operator deliberately binds to a private-LAN interface, a long
`MM_STUDIO_TOKEN` should be set in `.env.local`.
