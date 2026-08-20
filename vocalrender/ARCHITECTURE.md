# VocalRender local runtime

This folder adds a private browser studio to VocalRender’s highest-quality released single-sample singing synthesis path. It uses only `VocalRender-Pro`, with an explicit model load/unload lifecycle, structured SVS JSON, clean prompt singing, browser playback, and permanent local WAV downloads.

## Start

```bash
cd vocalrender
./setupwithuv
./startwithuv.sh
```

The setup prompt selects GPU-aware or CPU-only PyTorch. The UI is served at `http://127.0.0.1:8253`. Checkpoint, output, device, autoload, upload, host, and port settings are copied into the ignored `.env` from `.env.local.example`.

The server begins unloaded by default. **Load** prepares the selected variant, **Unload** releases it, and generation automatically loads the requested variant if necessary. The FastAPI lifespan always unloads the model during `Ctrl+C` shutdown.

## Input contract

VocalRender consumes the same JSON entry used by its canonical single-sample CLI: `word` plus optional `pitch`, `note`, `pitch2word`, and `bpm`. A clean 2–8 second prompt-singing clip is required because the released checkpoints were trained with prompt conditioning. WAV, FLAC, and OGG prompts are accepted.

Equivalent CLI:

```bash
uv run --active --no-sync python scripts/infer_vocalrender_svs_single.py \
  --ckpt_dir ../models/pymaster--VocalRender/VocalRender-Pro \
  --json_file phrase.json --item_name phrase-01 \
  --prompt_audio prompt.wav --output outputs/phrase.wav
```

The inference install keeps dataset writers, SingMOS, and other training/evaluation extras lazy rather than importing them into the serving process. Default environment flags force local Transformers/Hugging Face reads and disable telemetry.
