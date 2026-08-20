# ACE-Step local runtime

This folder exposes ACE-Step 1.5 as a private, offline Gradio music workbench. The source tree, Python environment, and model tree remain independent: project code lives here, `.venv` is disposable, and all checkpoints live under `../models/Ace-Step--Ace-Step1.5`.

## Start

```bash
cd acestep
./setupwithuv
./startwithuv.sh
```

The setup prompt selects a GPU-aware or CPU-only PyTorch environment. The launcher serves on `http://127.0.0.1:8250` by default, preloads the turbo DiT and 1.7B 5 Hz language model, and never enables a Gradio share link. `Ctrl+C` ends the owning Python process and releases its weights.

All runtime settings are copied from `.env.local.example` into the ignored `.env` on first setup. `ACESTEP_CHECKPOINTS_DIR`, `ACESTEP_DEVICE`, model names, offloading, host, and port can be changed there.

## Runtime boundary

The native interface provides ACE-Step generation, editing, repainting, extension, and audio-to-audio controls. It uses only the `ACE-Step/Ace-Step1.5` snapshot downloaded by the shared model script. The separately published ComfyUI-only ACE and MiniMax repackages are not read by this native Python runtime and are therefore not part of its portable model set.

No checkpoint is downloaded at launch. Offline Hugging Face and Transformers flags are mandatory in the default environment; a missing artifact stops startup with its exact expected path.
