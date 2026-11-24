# Ask Video

WeMM-Video desk: hand the 35B-A3B AWQ engine a clip, an image or a track and
ask questions about it. The studio owns exactly one patched engine process and
scopes its media access to disposable files, so uploads cannot reach the rest
of the disk.

## Setup and start

```bash
./setupwithuv.sh          # needs ffmpeg and ffprobe on PATH
./startwithuv.sh
```

Choose `wemm-video` in the model downloader. Setup runs the patched installer
committed at `wemm-video/patch/install.sh`, which pins the vLLM build that the
AWQ checkpoint needs, and adds the NVIDIA 2D image-processing library whose
libraries `startwithuv.sh` puts on `LD_LIBRARY_PATH`.
Open `http://127.0.0.1:8302`; `GRADIO_SERVER_PORT` changes the port.

## Checkpoints

Weights are read from `models/wemm-video/awq`; `/health` reports whether the
owned engine is up.

## Engine lifecycle

The engine is launched once, with the studio, and closed when it stops. Each
session copies its input into a private working directory, and a reaper
discards those directories once a session expires, so the upload namespace
stays small.

## Lane contents

- `wemm-video/` — WeMM-Video, `6cefad232b94e3e190d747251f35799b8b142213`,
  including the `patch/sglang` and `patch/vllm` manifests the installer applies
- `wemm-video/examples/media` — the three Apollo examples the page offers; they
  are served from the lane, so a fresh checkout works offline
- `state.py` — records this lane's source; the studio prints that identity at start,
  so a log names the exact source that answered.
