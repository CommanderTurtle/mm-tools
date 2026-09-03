# Local translation and spoken-language arbitration

This repository exposes one private HTTP service for Vox and other sovereign
LAN clients. It runs no agent, ComfyUI process, diffusion pipeline, cloud API,
or telemetry.

Three local models have deliberately separate jobs:

```text
ordinary text translation
  -> EraX Translator V1.0 Q8_0 (llama.cpp/GPU)
  -> INT4 EraX-VL fallback only if the dedicated model no-ops

image explanation / OCR translation / visual questions
  -> INT4 EraX-VL V1.5 (OpenVINO/CPU)

optional unknown spoken language
  -> XLM-R cheap batch narrowing (CPU)
  -> INT4 EraX-VL fragment/coherence ranking (CPU)
  -> optional translated-finalist comparison
  -> one ISO token returned to CrisperWhisper
```

Translation itself is multilingual and needs only a target. A supplied source
language is retained as API metadata for compatibility; it does not gate the
model. The ISO token selected by `/arbitrate` belongs only to the final full
CrisperWhisper pass.

## Install and start

```bash
cd ~/multimedia/translate
./setupwithuv.sh
./starthttp.sh
```

`setupwithuv.sh` defaults to the workstation's CUDA build. Set
`TRANSLATE_ACCELERATOR=cpu` only for a CPU-only host. Models remain outside the
repository at the paths in `.env`; setup never downloads weights.

With `TRANSLATE_AUTOLOAD=1`, all three local runtimes load before the service
accepts work. `Ctrl+C` closes the service and releases them. Re-running
`starthttp.sh` reuses an existing healthy process instead of binding or loading
twice.

```bash
curl http://127.0.0.1:8176/health
curl -X POST http://127.0.0.1:8176/load -d '{}'
curl -X POST http://127.0.0.1:8176/unload -d '{}'
```

## Translation

Only the target is required:

```bash
curl -X POST http://127.0.0.1:8176/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Guten Morgen","target_language":"English"}'
```

The dedicated Q8 model uses its model-card sampling settings. If it returns the
input unchanged or emits a confidently wrong target language, the already
resident INT4 EraX-VL model performs one narrow fallback. This keeps normal
translation fast while making less common source/target pairs reliable.

The diagnostic classifier remains available independently:

```bash
curl -X POST http://127.0.0.1:8176/detect \
  -H 'Content-Type: application/json' \
  -d '{"text":"Guten Morgen"}'
```

## Image understanding and OCR translation

The same resident EraX-VL model used for speech arbitration is a real
multimodal OpenVINO pipeline. The service accepts image bytes, never a remote
URL, so it cannot become an HTTP fetch proxy into the LAN.

```bash
{
  printf '%s' '{"image_data_url":"data:image/png;base64,'
  base64 -w0 page.png
  printf '%s' '","mode":"explain"}'
} | curl -X POST http://127.0.0.1:8176/vision \
  -H 'Content-Type: application/json' --data-binary @-

{
  printf '%s' '{"image_data_url":"data:image/png;base64,'
  base64 -w0 page.png
  printf '%s' '","mode":"translate","source_language":"auto","target_language":"English"}'
} | curl -X POST http://127.0.0.1:8176/vision \
  -H 'Content-Type: application/json' --data-binary @-
```

Supported modes are `explain`, `translate`, and `custom`; custom mode requires
`prompt`. The browser workbench exposes the same three modes and can either own
its local models or explicitly reuse the service on port 8176. Text requests
retain their 1 MiB body limit. Image requests default to a 20 MiB decoded-file
limit inside a 32 MiB JSON-body limit; the larger body allowance accounts for
Base64 expansion. Pixel, token, and external-timeout limits are also independent
in `.env`.

## Optional Crisper MITM

`POST /arbitrate` accepts the low-budget rows produced by CrisperWhisper's
`/api/transcribe-candidates`. It is invoked only when the caller selected
`detect`; known-language routes skip it completely.

The service batch-classifies all nonempty rows with XLM-R to keep the expensive
prompt small. EraX-VL then ranks complete, grammatical transcripts using the
named prompt language, punctuation, end-of-transcript state, and coherence. A
Whisper acoustic prior above `0.90` resolves obvious cases. When several rows
remain, only those finalists are translated into a common language and one
small final EraX-VL comparison selects the most complete result. The response
contains the selected Crisper ISO token; no language is persisted.

## API compatibility and privacy

Vox can use `/v1/models` and `/v1/chat/completions` at
`http://alien.local:8176/v1`. There is no cloud fallback. Leave the bearer token
empty only on the trusted private LAN, or configure the same
`TRANSLATE_API_KEY` in each client.

Local model ownership:

- `EraX-Translator-V1.0.Q8_0.gguf` — dedicated translation, llama.cpp/GPU.
- `papluca/xlm-roberta-base-language-detection` — 20-language classifier, CPU.
- `EraX-VL-7B-V1.5-Openvino-INT4` — image explanation, OCR translation,
  custom visual questions, fragment/ambiguity arbitration, and rare text
  translation fallback, OpenVINO/CPU.

Upstream references: [EraX Translator](https://huggingface.co/erax-ai/EraX-Translator-V1.0),
[EraX-VL](https://huggingface.co/erax-ai/EraX-VL-7B-V1.5),
[XLM-R language detector](https://huggingface.co/papluca/xlm-roberta-base-language-detection),
and [llama-cpp-python](https://github.com/abetlen/llama-cpp-python).
