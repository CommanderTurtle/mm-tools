# PhysAg3D

Image-to-interactive-object studio. TRELLIS.2 produces the textured candidate;
your selected OMP model builds and refines its Blender/MuJoCo assembly.

```bash
# From the repository root, after preparing the model downloader environment:
cd models && uv run download_models.py physag3d
cd ../sculpting/PhysAg3D
./setupwithuv.sh
./startwithuv.sh
```

Open `http://127.0.0.1:8282`. Requires Linux, `uv`, `bun`, configured `omp`
and Blender. Native compilation uses the existing studio's CUDA 13 / PyTorch 2.11
Blackwell stack, a C++ compiler and `nvcc`. CPU physics uses MuJoCo 3.12; EGL and
GLU system libraries support offscreen rendering/meshing. An existing TRELLIS
environment is reused; `PHYS_TRELLIS_PYTHON` can select another prepared one.

1. Add an object image. Open **Runtime**, refresh models and select an exact OMP
   provider/model. NVIDIA mode first asks you to load your model on port 7999;
   its discovered ID must also exist in your OMP configuration.
2. **Plan & scaffold** saves the observation plan and provisional Blender scene.
3. Unload the external agent model, then **Generate**. The default is native
   1024-cascade, 24 steps, 2048 texture, seed 42. The native worker exits when done.
4. Reload the selected agent model, then **Refine & simulate**. Add feedback to
   continue saved work. OMP runs local commands under your existing permissions.
5. Inspect clips, direct part controls, comparison images, motion video and
   reported physics. Download the GLB, Blender, MJCF or complete asset bundle.

Each stage is explicit and bounded. Stop or close the studio to terminate its
owned process group. Images, attempts, logs and OMP sessions remain in
`workspace/`; restart and select the case to continue. Earlier attempts stay
intact. **Package** rebuilds the viewer bundle from a saved refinement without
contacting a model. The studio leaves independently managed model servers alone.

Part controls are geometric; animation may be simulated replay or kinematic.
The delivery record distinguishes these from a static candidate and records
unresolved physics/appearance limitations. Each refinement has at most two
acceptance passes and two repair rounds. A source image cannot determine exact
hidden geometry or physical properties.

The ZIP contains local viewer dependencies and editable assets. Extract it and
serve its root with `python -m http.server 8080 --bind 127.0.0.1`, then open
`http://127.0.0.1:8080/presentation/`. No CDN is required.

`PHYSAG3D_PORT` changes the studio port; `PHYS_OMP` selects an OMP executable.
The native runner lives in `PhysAg3D/`; weights come from the central picker listed in
`models/which-ones.txt`.
