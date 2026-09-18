# MuScriptor Score Transcription

Audio in, symbolic score out: MuScriptor converts music recordings into note
events and MIDI, served through a local streaming service and a piano-roll
workbench. The CLI and the browser share the same local decoder, and sound
preview uses local SF2/SF3 soundfonts configured in `.env`. The web build is
bun output; the Python inference path and the configured model and
soundfont files are authoritative.

## Install and start

From the project folder (requires `uv`, `ffmpeg`, and `bun`):

```bash
../models/download_models.py muscriptor
./setupwithuv.sh gpu
./startwithuv.sh     # serves http://127.0.0.1:8222
```

Review `MUSCRIPTOR_MODEL_PATH`, `MUSCRIPTOR_SF2_PATH`, and
`MUSCRIPTOR_SF3_PATH` in `.env` before starting. Model variants (small,
medium, large) are listed in `../models/which-ones.txt`.

## Upstream

- Models and assets: https://huggingface.co/MuScriptor
