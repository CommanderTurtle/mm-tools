# Underdog

Bark 3.0 English speech-to-text studio. Upload audio or record from a microphone,
select a model, transcribe, compare results, and copy or export TXT/JSON.

```bash
cd models
uv run download_models.py underdog:4
# Optional independent selections:
uv run download_models.py underdog:1+2+3   # or bare 'underdog' for all four variants
cd ../vocals/stt/underdog
./setupwithuv.sh                  # NVIDIA CUDA; use `cpu` for CPU
./startwithuv.sh                  # studio: http://127.0.0.1:8293
```

The studio starts unloaded. Each model has its own load/unload control;
transcribing also loads the selected model if needed. Existing loaded models
stay resident until explicitly unloaded or the service closes. Operations are
serialized within each service. Start only the service(s) you need.

| Selection | Output | Native PyTorch weights |
| --- | --- | --- |
| `base` | Verbatim | Exact unpacked FP32 |
| `plus` | Verbatim | Exact unpacked FP32 |
| `clean` | Fillers, restarts and repeats removed | Exact unpacked FP32 |
| `pro` | Verbatim | BF16 |

The packed download sizes differ from runtime memory use. CPU setup selects
FP32 for Pro explicitly. Existing `.env` settings are preserved on subsequent
setup runs. `BARK_DEVICE` and `BARK_PRO_DTYPE` control the device and Pro precision.

## HTTP-only

Set `BARK_MODEL=base`, `plus`, `clean` or `pro` in `.env`, then:

```bash
./starthttp.sh                    # API: http://127.0.0.1:8292
curl -sS http://127.0.0.1:8292/api/transcribe -F 'file=@recording.m4a'
curl -sS http://127.0.0.1:8292/api/health
curl -sS -X POST http://127.0.0.1:8292/api/unload
curl -sS -X POST http://127.0.0.1:8292/api/load
```

This service exposes the API only and loads its configured model on startup.
Set `BARK_HTTP_AUTOLOAD=0` for lazy loading. The HTTP service and studio have
separate model lifecycles; shutting either down releases its own models.

The transcription request uses Whisper's multipart `file` / `language` /
`operation` shape. Defaults are `language=en`, `operation=auto`,
`chunk_duration=30`. `auto` resolves to `verbatim`, or `intended` for Clean.
The response is `{ok, operation, results: {verbatim: {...}}}` (or `intended`),
with text, duration, processing time, language, model revision and chunk segments.
Timings describe audio chunks; `words` is empty. Unsupported language/operation
requests receive a clear error. Busy operations return 423.

The studio API additionally accepts `variant` on load, unload and transcribe.
HTTP-only mode always uses the `.env` selection. `POST /api/cancel` cancels at
the next chunk boundary. The studio enables cancellation after upload acceptance
and identifies its request explicitly. Audio uploads and decoded files are temporary.

## Local configuration

FFmpeg decodes input to mono 16 kHz. The native greedy decoding and per-model
leading silence are preserved: 0.25 seconds for base/Plus/Pro, zero for Clean.
Long audio is processed in independent chunks, following the Pro playground.
`BARK_MAX_AUDIO_SECONDS` defaults to 600; `BARK_MAX_UPLOAD_MB` defaults to 1024.

Setup creates a relative `models` symlink to the shared downloader store.
Repository names and nesting depth may vary. All checkpoints, processor files
and loader code are declared in `models/manifests/underdog.json`; inference uses
local files. Model conditions are in the central inventory.

Both services bind to loopback by default. For trusted LAN access, set
`BARK_HOST`; `BARK_CORS_ORIGINS` optionally lists allowed browser origins,
comma-separated. Microphone capture requires localhost or HTTPS; set both
`BARK_TLS_CERTFILE` and `BARK_TLS_KEYFILE` to serve HTTPS directly.

Launchers target Linux/WSL. Ctrl+C stops subsequent chunks and the owned FFmpeg
decoder; a bounded shutdown releases the process even if a native call stalls.
