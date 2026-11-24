# Inworld Voice Isolator

Speech isolation studio: upload or record up to ten minutes of audio and get
the vocal and accompaniment tracks back, or run the microphone path live over
a websocket with the same model, one take at a time.

## Setup and start

```bash
./setupwithuv.sh
./startwithuv.sh
```

Choose `inworld-isolator` in the model downloader. Setup installs the package
committed in this lane, `voice-isolator/`, into a Python 3.11 uv environment
together with the FastAPI serving stack. Open `http://127.0.0.1:8297`;
`GRADIO_SERVER_PORT` changes the port.

## Checkpoints

The weights are read from `models/inworld-isolator` and nothing is downloaded
at run time.

## Live audio device

The live path holds the audio device under a lock: a second request waits
briefly and then reports that the device is busy rather than mixing two
captures.

## Lane contents

- `voice-isolator/` — Inworld Voice Isolator, `c73b3fab81a870410cbac50d2f05f25edc4d10aa`
- `voice-isolator/demo/examples` — the eighteen example clips the studio offers
  as one-click samples; they are served from the lane, so a fresh checkout works
  offline
- `state.py` — records this lane's source; the studio prints that identity at start,
  so a log names the exact source that served a result.
