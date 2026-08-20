# Stable Audio Foundation local runtime

This folder packages the enhanced Stable Audio browser interface around RoyalCities Foundation-1. Code, `.venv`, and model artifacts are independent; the ignored environment can be rebuilt while the shared checkpoints remain under `../models`.

## Start

```bash
cd stableaudio
./setupwithuv
./startwithuv.sh
```

The setup prompt installs either GPU-aware or CPU-only PyTorch. Python 3.10 is intentional because the complete interface includes Basic Pitch. The launcher serves `http://127.0.0.1:8251`, loads Foundation-1 plus its local `t5-base` conditioner, and never requests a Gradio share link. `Ctrl+C` ends the model-owning process and releases its weights.

`.env.local.example` is copied to the ignored `.env` on first setup. It contains only portable paths relative to this checkout and local host/port settings.

## Model boundary

The Foundation checkpoint contains the diffusion transformer and audio autoencoder. Its model configuration names `t5-base`, so the shared downloader also materializes `google-t5/t5-base`; the conditioner patch resolves that local directory with `local_files_only=True`. Startup fails with an explicit path if either tree is incomplete.

The interface keeps its existing prompt modes, waveform and spectrogram results, MIDI analysis, and optional TorchAO controls. Default environment flags disable Hugging Face, Gradio, and Weights & Biases telemetry, and no model download occurs at launch.
