# Hy3D

Local image-conditioned geometry studio for Hi3D 1.0 / Sparc3D. Generate a coarse
mesh from an image, or refine an existing GLB, PLY, OBJ or STL against a reference.
Inspect coarse and refined geometry in clay, normals or wireframe; export GLB,
PLY and OBJ. Original images, conditioning images, settings and stages stay in
`workspace/`. Stopped runs resume from the last completed stage.

## Setup

In the [model picker](../../models/download_models.py), select **hy3d** and, for image-only
generation, **hy3d-coarse**. Then, from this folder:

```bash
./setupwithuv.sh
./startwithuv.sh
```

Open `http://127.0.0.1:8284`. `HY3D_PORT` selects another localhost port.
Setup requires `uv`, `git`, `bun` and a CUDA toolkit with `nvcc`. Python 3.10 and
the native source install locally. The model link is relative to
this checkout and works from any directory name or depth.

The default `cu124` profile targets the Torch 2.5 / CUDA 12.4 stack.
For CUDA 12.8 systems, including Blackwell, choose the explicit kernel profile:

```bash
./setupwithuv.sh cu128
```

That profile uses Torch 2.7.1, spconv-cu126 with PTX compilation and FlashAttention
2.8.3, built against the local toolkit. Model precision stays native in either
profile. Move an existing `.venv` before changing profiles.

## Geometry

512, 1024 and 1536 extraction grids use one, two and three refinement passes.
Steps and seed apply to native refinement; the optional coarse stage keeps its
native 30-step / 256-grid settings. Large grids require substantial sparse
activation memory. Sequential CPU offload applies to refinement weights;
preparation and coarse generation each release their process before refinement.

Transparent references retain their alpha. Optional U2Net background removal
uses the picker-managed local model on CPU. The conditioning view shows the
native crop, white background and padding. Exported meshes use native unit
normalization and mesh cleanup; this pipeline produces geometry rather than
textures. Orbit by dragging, pan with the right button and scroll to zoom.

Model revisions are recorded in each saved run; weight and dependency conditions are
listed in the [model inventory](../../models/which-ones.txt).
