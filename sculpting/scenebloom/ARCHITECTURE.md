# SceneBloom studio

Prompt-to-room generation with native FreeFuse proposals, geometric fitting, wall and surface
furnishings, material selection and textured GLB export. Bedroom, living room, dining room,
office and bathroom presets share the native sampling controls. Inspect zones and objects in
orbit, plan or walking view; save a view, download the scene or reopen a previous run.

## Setup

Linux (glibc 2.35+), an NVIDIA GPU, `uv`, Bun and a CUDA development toolkit (`nvcc`, C++
compiler and headers) are required. Genesis rendering uses the host's OpenGL/EGL libraries. The
default CUDA 12.8 build targets RTX 5090; install a matching toolkit. `SCENEBLOOM_CUDA=cu121
./setupwithuv.sh` selects the CUDA 12.1 / Torch 2.5.1 profile for supported older GPUs.
`TORCH_CUDA_ARCH_LIST` can explicitly select the extension targets for another supported device.

Accept FLUX.1-dev's access terms and authenticate with `hf auth login` first. From this
directory:

```bash uv run --with huggingface_hub ../../models/download_models.py scenebloom --yes
./setupwithuv.sh ./startwithuv.sh ```

Open `http://127.0.0.1:8285`. `SCENEBLOOM_PORT` changes the local port. The picker includes
checkpoints, the full object/LoRA library, layout previews and cached 2K material packs. Setup
links the central model store relatively, uses the native source committed beside it and builds
its extensions. It uses a separate `.vl-venv` for the shared local Qwen3-VL NVFP4 encoder.
Native FLUX, lifting and rendering retain their own precision and sampling settings.

## Runs

The first run builds the two layout-retrieval indexes; later runs reuse them. Generation moves
through retrieval, zoning, proposals/lifting, fitting, furnishing, materials, rendering and
export. Each stage saves its output in `workspace`. Workers exit between stages; vision calls
also exit after replying. Cancel stops the active stage. Resume continues from the first
incomplete stage, including after restarting the studio. Closing the terminal stops the server
and its workers while retaining saved runs.

Downloads include the textured GLB, full scene JSON, zone plan, materials and run settings.
Proposal images and Genesis camera renders remain in each run. Walking view uses WASD or arrow
keys and drag-to-look; walls can be hidden for inspection. Serving uses local files, including
browser assets.

Weights and their conditions are recorded in [the model
inventory](../../models/which-ones.txt).
