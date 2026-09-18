# CrisperWhisper Transcription

Fast multilingual speech recognition built on the CrisperWhisper 2.0
checkpoint and a local CTranslate2 runtime, with a dependency-free native
Rust core under `native/`. Two surfaces share the inference code but own
separate model lifecycles:

- Browser workbench for interactive transcription
- Machine HTTP service for pipelines and other tools

Known-language requests take one full pass. The optional detect route
generates low-budget language candidates, arbitrates locally, then runs one
final pass with the selected language. Every result carries literal and
intended transcript variants with word-level timing.

## Install and start

From the project folder:

```bash
../models/download_models.py whisper
./setupwithuv.sh gpu   # checks the checkpoint, backend, device, and dtype values in .env
./startwithuv.sh       # browser workbench at http://127.0.0.1:8173
./starthttp.sh         # machine service at http://127.0.0.1:8172
```

Setup does not download weights; review the paths in `.env` before starting.
ARCHITECTURE.md documents the decode flow, normalization, and the
service/browser split.

## Upstream

- Runtime lineage: https://github.com/SYSTRAN/faster-whisper
- Weights: https://huggingface.co/nyralabs
