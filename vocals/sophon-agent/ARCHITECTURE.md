# Sophon Agent

Full-duplex speech studio: talk over the microphone and the agent listens,
talks back and interrupts, with the endpointing driven by the committed TAG
runtime. Voices are the clips recorded in the studio (`.state/voices`), and a
finished call can be saved as a recording.

## Setup and start

```bash
./setupwithuv.sh
./startwithuv.sh
```

Choose `sophon-agent` in the model downloader. Setup installs the plugin
committed at `sophon-realtime/tag/plugin` plus the vLLM and duplex runtime into
a CUDA 12.8 uv environment, and then the TensorRT wheels unless `NO_TRT=1` is
set; without them the studio falls back to torch flow decoding.
`GRADIO_SERVER_PORT` changes the browser port; `SOPHON_PORT` (default 8498) is
the private engine port the studio proxies to. Open `http://127.0.0.1:8298`.

## Checkpoints

The NVFP4A16 thinker is read from `models/sophon-agent`, and
`VLLM_NO_USAGE_STATS` keeps the engine quiet.

## Lane contents

- `sophon-realtime/` — Sophon Realtime Tag, `e187202824738290d76dde27adb748fe3c14da6e`
- `sophon-realtime/tag/config/silero_vad.onnx` — the 2.2 MB endpointing model.
  It stays in the lane because the committed runtime resolves it relative to
  its own config directory; the rest of the weights live in
  `models/sophon-agent`.
- `sophon-realtime/tag/plugin/.../deploy/warmup.wav` — same reason: the plugin
  plays it during startup warm-up.
- `state.py` — records this lane's source; the studio prints that identity at start,
  so a log names the exact source that served a call.
