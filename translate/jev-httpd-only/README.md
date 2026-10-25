# jev-httpd-only

An optional translator backend for an **already-running NInfer multimodal model**.
Python 3.11+ standard library only. No setup, weights, GPU runtime, model loading,
resident agent, scheduled polling, or cloud fallback.

The existing EraX backend remains available and unchanged. Choose either backend
on port 8176, or use a different port and point Vox's `backends.translator_url` at
it. The MV3 extension and Underbelly do not need code changes.

## Run

Keep your usual NInfer server running with `--jev` enabled. Then, from mm-tools:

```bash
JEV_BASE_URL=http://127.0.0.1:8000 bash translate/jev-httpd-only/starthttp.sh
```

Or, without a shell wrapper (also works on Windows):

```bash
python translate/jev-httpd-only/server.py --host 127.0.0.1 --port 8176
```

For persistent configuration, copy this directory's `.env.example` to `.env`.
The launcher reads **only its own** `.env`, not the dedicated backend's model
configuration. `JEV_BASE_URL` accepts either the server root or a trailing `/v1`.
`JEV_MODEL` may be empty when `/v1/models` advertises exactly one model.

Do not run both backends on the same port. A bind failure never kills or replaces
the other process. Stopping this adapter never stops or unloads NInfer.

## Existing clients

The normal route is unchanged:

```text
MV3 extension → Vox :8180 → translator :8176 → existing NInfer :8000
Firecrawl / Underbelly → translator :8176 → existing NInfer :8000
```

Retain Vox's existing `translator_url` and `translator_api_key` when replacing the
dedicated backend on port 8176. `TRANSLATE_API_KEY` protects this adapter;
`JEV_API_KEY` authenticates to NInfer. They are independent. Listen on loopback
unless your trusted LAN/container clients need access; the default `0.0.0.0`
matches the existing translator.

Text and image-only extension use can also point `VOX_HTTP_BASE_URL` directly at
this adapter: `/v1/detect`, `/v1/translate`, and `/v1/vision` are aliases. It is
**not** a replacement for Vox's speech, playback, or routing features.

```bash
curl http://127.0.0.1:8176/health
curl http://127.0.0.1:8176/detect -H 'Content-Type: application/json' \
  -d '{"text":"Guten Morgen"}'
curl http://127.0.0.1:8176/translate -H 'Content-Type: application/json' \
  -d '{"text":"Guten Morgen","source_language":"German","target_language":"English"}'
```

## What runs when

* Known-source translation: one ordinary `/v1/chat/completions` request per bounded
  text fragment. No Jev decision is made.
* `/detect`, or translation with source `auto` / `Detect`: one on-demand
  `/v1/systemone` choice over the shared language catalog. The returned classifier
  confidence is the selected probability, not a number invented in chat output.
  Insufficient evidence returns `und`. Whitespace/numeric input needs no decision.
* Translation output: a forced `submit_translation({"translation":"…"})` function
  call. The adapter validates and unwraps it into the existing `translation`
  response field. This is an output contract, not a tool capable of browser,
  filesystem, shell, or network actions. One fresh format-repair attempt is allowed;
  malformed, empty, or token-truncated results fail instead of becoming page text.
* `/vision`: forwards the existing `image_data_url` and `explain` / `translate` /
  `custom` prompt to the same model using ordinary OpenAI multimodal messages.
  No Jev decision or second vision model. The running model must support images.
* Health/discovery: metadata GET requests only, **no inference probes**. Health
  reports availability of NInfer, not proof that its model is good at translation
  or that `--jev` is enabled; a missing Jev route gives an actionable detection error.
* Idle: no requests at all. Active work does consume the existing model's inference
  capacity, but never attaches to or modifies an agent's conversation/history.

The extension already translates text nodes and restores the original DOM;
Underbelly already protects Markdown/code/URLs. Those layers stay in charge of
structure. This adapter translates the supplied text, not a newly rewritten DOM
or executable HTML. Long text is split on UTF-8-safe line/word boundaries;
separators and outer whitespace are retained. The language decision is made once
for the request, not once per fragment. Context metadata conservatively reduces
fragment/output budgets; token-limit failures are explicit, never silent clipping.

`JEV_CONCURRENCY=1` limits this adapter to one active model request. Up to 16 extra
HTTP requests can wait (useful for MV3's four parallel fragments), sharing the
110-second default deadline with inference. Excess work receives `429` with
`Retry-After: 1`; timeout receives `504`. Increase concurrency only to suit the
capacity of your existing NInfer lanes. There is no automatic capacity benchmark,
background detection, or retry storm. No text, image, or authorization logging.

## Routes and limits

| Route | Result |
| --- | --- |
| `GET /health` | Existing `status`, `loaded`, `model`, `cloud:false` fields; additionally `owns_model:false` |
| `GET /v1/models` | The actual selected NInfer model, not an EraX alias |
| `POST /detect` | `{language, confidence, detector}` |
| `POST /translate` | `{translation, source_language, source_confidence, model}` plus target/route metadata |
| `POST /vision` | `{output, mode, model}` |
| `POST /v1/chat/completions` | Existing nonstreaming translation compatibility shape / `<source_text>` prompt parsing; not a general chat proxy |
| `POST /load`, `POST /unload` | Resume/pause **this adapter only**, never model lifecycle operations; in-flight work may finish |
| `POST /arbitrate` | Explicit `501`: Crisper speech-candidate arbitration is not implemented here |

Text bodies default to 1 MiB, vision JSON to 32 MiB, decoded image bytes to 20 MiB.
Vision accepts base64 PNG/JPEG/WebP/GIF data URLs, never remote image fetches.
Image decoding, pixel/context limits and format support remain NInfer's job.
Upstream redirects and ambient HTTP proxies are disabled; only the configured
endpoint receives the configured upstream bearer token. Neither mode downloads
models. See `.env.example` for the small set of tunables.

## Check

```bash
python -m unittest discover -s translate/jev-httpd-only/tests -v
```

Tests use a local fake NInfer HTTP endpoint with the real SystemOne and OpenAI
wire shapes; they require no model or third-party Python packages. They validate
protocol behavior, not translation quality. A live language-quality smoke test
still requires your running model.

If you also have Vox checked out and Bun installed, set `VOX_MV3_DIR` to its
`mv3-extension` directory when running the same test command. This additionally
runs Vox's actual, unmodified `translateWithVox` against the adapter.
