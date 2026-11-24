# RMBG-3.0

BRIA RMBG-3.0 cutout service: an image in, an alpha matte and a clean
background back out. The model is instruction-tuned, so the editing
instruction is fixed by the upstream recipe rather than exposed as a free-form
prompt.

## Setup and start

```bash
./setupwithuv.sh
./startwithuv.sh
```

Choose `bria-rmbg3` in the model downloader. Setup installs torch from the
CUDA 12.8 index plus the requirements committed at
`rmbg/requirements-rmbg3.txt`. Open `http://127.0.0.1:8301`;
`GRADIO_SERVER_PORT` changes the port.

## Checkpoints

The pipeline itself is fetched by `models/download_models.py` from ModelScope
into `models`, so the studio loads it without resolving an upstream repository
name over the network.

## Lane contents

- `rmbg/` — the upstream BRIA RMBG-3.0 branch, `2ddee117f34ce8aa6966bfa1e9c1e7c2aa9342d8`:
  `remove_bg.py` (loaded by the studio through `importlib`, and the source of
  the fixed instruction), `birefnet.py` and `BiRefNet_config.py` (the
  architecture), and the two processor config files
- `state.py` — records this lane's source; the studio prints that identity at start,
  so a log names the exact source that produced a matte.

The Ideogram/ObjectClear editor sits beside this lane at
`visual-modeling/background-removal/ideogram`.
