# Audio8

Local Realistic Air speech studio. Linux, NVIDIA CUDA and FFmpeg on PATH.

```bash
uv run models/download_models.py audio8
cd vocals/tts/audio8
./setupwithuv.sh
./startwithuv.sh
```

Open `http://127.0.0.1:8290`. `AUDIO8_PORT` changes the port.

Write text, choose either reference preset or upload/record 3–15 seconds with
its exact transcript. No-reference generation is also available. Languages
follow the input text. Blank-line splitting creates separate takes; batching,
seed, sampling and frame budgets retain the native controls.

Waveforms support seeking and playback speed. Sessions retain inputs, partial
takes and logs under `.work/runs`; export float WAV, 24-bit FLAC, codec tokens
or a session ZIP. An unfinished end-token sequence is visibly marked. Retry
runs once with the larger budget; shorten text/reference if context runs out.

The published Air model is 0.6B with a separate 44.1 kHz codec. The native
Transformers path uses CUDA BF16 and a 2048-position packed context.
Transformers remains on 4.57.3 to match this checkpoint's generation API.
Each job owns its model process; completion, cancellation or server shutdown
releases it. Setup links the shared model store relative to this folder.
Weights and their conditions are in the central model inventory.
