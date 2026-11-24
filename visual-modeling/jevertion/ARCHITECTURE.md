# Jeveration

Image desk for Jeveration: generate, edit and dense-capability runs behind one
workflow page. The browser UI is this lane's own `index.html`; the graph it
drives is the workflow file that ships with the model, bound to the controls.

## Setup and start

```bash
./setupwithuv.sh
./startwithuv.sh
```

Choose `jeveration` in the model downloader. Setup installs torch from the
CUDA 12.8 index and then the requirements list that the model bundle places at
`models/jeveration/space/requirements.txt`, so the environment matches the
upstream space. `GRADIO_TEMP_DIR` keeps workflow scratch in `.state/workflow`.
Open `http://127.0.0.1:8300`; `GRADIO_SERVER_PORT` changes the port.

## Checkpoints

Two annotator checkpoints are pinned by commit into the lane's Hugging Face
cache (`models/jeveration/hub`), and `HF_HUB_CACHE` points there, so the
studio never writes to `~/.cache`.

## Lane contents

- `jevertion/` — Jeveration, `cf74be36efe522bdf49adc3326fe233190731812`; its
  `jeveration/third_party/tatitok` tree is real runtime code (the tokenizer is
  imported from it), so it stays where upstream put it.
- `state.py` — records this lane's source; the studio prints that identity at start,
  so a log names the exact source that served a render.
