# ACE-Step local runtime

This folder exposes ACE-Step 1.5 as a private, offline Gradio music workbench. The source tree, Python environment, and model tree remain independent: project code lives here, `.venv` is disposable, and all checkpoints live under `../models/Ace-Step--Ace-Step1.5`.

## Start

```bash
cd acestep
./setupwithuv
./startwithuv.sh
```

The setup prompt selects a GPU-aware or CPU-only PyTorch environment. The launcher serves on `http://127.0.0.1:8250` by default, preloads the highest-quality native XL-SFT DiT and 4B 5 Hz language model, and never enables a Gradio share link. `Ctrl+C` ends the owning Python process and releases its weights.

All runtime settings are copied from `.env.local.example` into the ignored `.env` on first setup. `ACESTEP_CHECKPOINTS_DIR`, `ACESTEP_DEVICE`, model names, offloading, host, and port can be changed there.

## Runtime boundary

The native interface provides ACE-Step generation, editing, repainting, extension, and audio-to-audio controls. Its common embedding and VAE assets come from `ACE-Step/Ace-Step1.5`; the native XL-SFT DiT and 4B language model come from their official ACE-Step repositories and are placed beneath the same checkpoint root by the shared model script. The separately published ComfyUI-only ACE and MiniMax repackages are not read by this native Python runtime.

No checkpoint is downloaded at launch. Offline Hugging Face and Transformers flags are mandatory in the default environment; a missing artifact stops startup with its exact expected path.
