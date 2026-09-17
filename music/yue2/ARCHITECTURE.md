# YuE2 Composition Studio architecture

## Native stages, not a prompt wrapper

The WebUI maps directly to YuE2's released stages:

1. `plan()` creates a full melody/chord score, a melody-only score, or the
   direct-generation prefix.
2. `generate_semantic()` writes the complete semantic music sequence.
3. `synthesize()` performs native AR–NAR flow matching.
4. the pinned `YuE2-Vae` decodes 48 kHz stereo audio.

The full, melody-only, and direct paths remain separate visible workflows.
Every AR and ABC sampling field in the native protocol is exposed under
Advanced, together with CFG, ODE steps, CUDA-graph/eager execution, native
BF16 or the upstream FP8-AR option, VAE tile size, candidates, and deterministic
seed stepping.

Each candidate keeps `audio.flac`, `score.abc` when applicable, exact planner
tokens/prefix, semantic tokens, acoustic latents, effective configuration,
timings, model identities, integrity hashes, and `plan-bundle.zip`.  The bundle
can be resumed only while unchanged; edited ABC enters through the score lanes
and therefore receives a new identity.

## GPU residency

YuE2's AR/NAR model remains on CUDA while a short-lived decoder instance is
created on CUDA.  The adapter deliberately does not invoke the upstream
AR-to-CPU transition before decoding and never enables `offload_ar`.  Decoded
audio and persisted NumPy artifacts return to host memory, but model inference
and model weights are not offloaded for CPU execution.  One FIFO job owns the
single GPU at a time.

## Covers and transcription

SheetSage2 is isolated in `.venv-sheetsage2` because its Transformers/Numpy
contract differs from YuE2.  Both environments use uv's shared wheel cache and
hardlinks, so isolation does not imply duplicated package downloads.

`local_app/transcribe.py` passes the downloaded `MERT-v2-FullSong` directory as
an explicit local parent.  It always sets local-only loading: the model cannot
silently fetch its configured Hugging Face parent.  It exposes full versus
melody-only score generation, overlapping-window controls, decoder prompts,
and optional logits, scores, embeddings, and all 24 MERT hidden layers.  PDF
rendering's Playwright dependency is intentionally absent; the native ABC,
MIDI, separated MIDI, events, and `.lab` outputs are retained without a browser
automation runtime.

The one-click cover workflow unloads YuE2, transcribes on the same GPU, exits
that process, then loads YuE2.  The default melody-only route preserves both
Vocal and Ins melodies while freeing accompaniment; enabling source harmony
uses full-score mode.

## Score lab and listening room

The repository's pinned strict ABC implementation validates the exact two-voice
dialect, ties, pitch, duration, meter, key changes, chord locations, and nominal
duration.  It can remove chord symbols without changing melody and compare an
edited score against an original for exact sounding-note/meter invariants.

The shared output room adds a live spectrum and lyric-follow view to every
audio render, synchronized A/B playback with an equal-power crossfade, durable
recipes, metadata, downloads, and exact recipe reopening.  The visualization
uses only browser Web Audio against the local output URL.

## Privacy boundary

The launch contract forces Hugging Face and Transformers offline, disables
telemetry, contains no remote fonts/scripts/images, and defaults to loopback.
Uploads, job SQLite state, logs, and outputs live in the ignored `.runtime`
tree.  The HTTP API is the same queue used by the UI; set `MM_STUDIO_TOKEN`
before deliberately exposing the listener on a trusted private network.
