# AuK Speech Studio architecture

## Product surface

`startwithuv.sh` launches the shared mm-tools studio shell with AuK's native
adapter.  The browser exposes every task in the upstream capability table:

- reference and instruction-only TTS;
- speech-content and lyric editing;
- signed pitch, speed, and dB edits;
- the published happy, angry, sad, fearful, surprised, disgusted, calm, and
  excited vocabulary as an emoji-driven emotion constellation;
- timbre, de-accenting, whisper conversion, and stackable nonverbal events;
- denoise/dereverberate/quality restoration, speaker extraction, and vocal
  separation;
- an open native-instruction lane for valid one-step combinations.

The performance sliders do not pretend to be undocumented tensor controls.
They compile to an inspectable natural-language instruction—the interface AuK
was trained to consume—and that exact instruction is saved next to every take.

## Native inference

`local_app/adapter.py` imports `AukInfer` directly.  It keeps exactly one model
variant resident, runs one FIFO GPU job at a time, and never enables the
upstream CPU-offload path.  Base uses the visible NFE/CFG/sway controls.
AuK-Flash, when separately installed, remains pinned to its upstream four-step
time grid and CFG-off recipe inside `AukInfer`.

The adapter writes lossless WAV or FLAC, a plain-text compiled direction, and a
JSON reproducibility record.  Peak safety only attenuates a generated waveform
that exceeds the selected ceiling; it does not normalize ordinary output.

## Prompt Enhancer without a privacy surprise

Prompt Enhancer is optional and off by default.  It only accepts a loopback
OpenAI-compatible endpoint unless the operator explicitly enables a private
LAN address.  Audio-backed requests require a visible, user-supplied source
transcript.  A tiny transcript provider feeds that text into the upstream
Prompt Enhancer, preventing its cloud-ASR and CPU-ASR fallback paths.

## Vox/API lane

`starthttp.sh` runs the same adapter and queue with the HTML surface disabled.
In addition to `/api/jobs`, it exposes `POST /v1/audio/speech`.  The common
OpenAI fields (`input`, `voice`, `model`, `speed`, and `response_format`) work;
AuK extensions add `reference_asset`, `emotions`, expressive controls,
duration, seed, NFE, CFG, and an asynchronous mode.  Upload a reference through
`POST /api/assets` first and pass the returned id as `reference_asset`.

## Local state and boundaries

Uploads, SQLite job state, logs, recipes, and generated artifacts live in the
ignored `.runtime/studio` tree.  The UI has no remote assets, analytics, CORS,
or cloud fallback.  The listener defaults to loopback.  Set a long
`MM_STUDIO_TOKEN` before deliberately binding to a private-LAN interface.
