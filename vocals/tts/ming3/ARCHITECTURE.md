# Ming3

```bash
uv run models/download_models.py ming3-0.5b   # dense, ~3 GB
uv run models/download_models.py ming3-tiny   # MoE + podcast, ~18 GB; optional
cd vocals/tts/ming3
./setupwithuv.sh
./startwithuv.sh                             # http://127.0.0.1:8291
```

Choose either model or both. Setup creates separate environments only for downloaded models, uses the native source committed beside the studio, and links `models` to the repository's central model directory. Run setup again after adding the other model. `MING3_PORT` selects another loopback port.

Clone voices, design a voice, direct delivery, select built-in voices, control emotion or attributes, speak seven languages and two dialects, and insert vocal tags. `tiny` also generates podcast dialogue: each line starts with `speaker_N:`, with one ordered recording and exact transcript per speaker. Its native automatic frame budget has a 48-second ceiling. Script turns are shown as text cues, rather than estimated audio timestamps.

Upload or record references; transcribed presets are included by the model picker. Take history restores scripts, controls and recordings. Playback includes waveform seeking and rate controls. Export float32 WAV, PCM24 FLAC, or a bundle with recordings and generation settings. The model is loaded in an owned worker per take and released after completion, cancellation or server shutdown. Drafts stay in the browser; takes live in `.work/runs/`.

Both paths use native BF16 inference, task serialization, normalization and text splitting. `tiny` retains its release FP32 recurrent state and SDPA configuration. Separate dependency sets preserve each native stack; CUDA 12.8 PyTorch 2.7.1 wheels support Blackwell. Raw text bypasses preparation and splitting. Advanced fields left blank use native defaults, including seed 1895 for dense, 1234 for tiny and 1235 for podcast; `-1` selects a fresh seed, recorded with the take. Dense voice cloning accepts an optional reference transcript; tiny requires it.

Weights and their conditions are listed in the central model inventory.
