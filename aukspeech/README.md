# AuK Speech Studio

AuK (Tencent Hunyuan) is a foundational speech generation and editing model.
The studio exposes the full published capability table on one local page:

- Text to speech, reference-voice TTS, and instruction-only TTS
- Speech-content and lyric editing
- Signed pitch, speed, and loudness edits
- The eight published emotions as an emoji-driven constellation
- Timbre control, de-accenting, whisper conversion, and stackable nonverbal
  events
- Denoise, dereverberation, quality restoration, speaker extraction, and
  vocal separation
- An open native-instruction lane for valid one-step combinations

Performance sliders do not pretend to be undocumented tensor controls: they
compile to the natural-language instruction AuK was trained on, and that
exact instruction is saved next to every take. Inference is the native
`AukInfer` with one model variant resident, a single FIFO GPU job at a time,
and no CPU-offload path.

A machine-only HTTP route with the same backend serves non-UI clients as a
Vox-compatible speech service.

## Install and start

From the project folder:

```bash
../models/download_models.py aukspeech
./setupwithuv.sh     # verifies weights, then builds the local venv
./startwithuv.sh     # browser studio at http://127.0.0.1:8260
./starthttp.sh       # Vox-compatible HTTP service instead of the UI
```

ARCHITECTURE.md covers the capability mapping, the AuK-Flash recipe, and the
HTTP contract.

## Upstream

- Model: https://github.com/Tencent-Hunyuan/AuK
