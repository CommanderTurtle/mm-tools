# FastFrames

Playable world studio: a prompt starts a LingBot-Wan world you drive from the
browser over a websocket, with a waiting room that hands the GPU to the next
person, plus the FastFrames-L clip path for short text-to-video renders.

## Setup and start

```bash
./setupwithuv.sh
./startwithuv.sh
```

Choose `fastframes:1` in the model downloader (tier `fastframes:2` adds the 14B
LingBot world and the Wan2.1 T2V weights). Setup runs `uv sync --frozen`
against the `pyproject.toml` and `uv.lock` committed inside `fastframes/`,
installing the `lingbot` and `demo` extras into the lane environment; nothing
else is fetched. Open `http://127.0.0.1:8296`; `GRADIO_SERVER_PORT` changes
the port.

## Checkpoints

Checkpoints live in `models/fastframes` and the studio refuses a world until
the matching bundle is present.

## Lane contents

- `fastframes/` — FastFrames, `a65c59240e01cf1de86197fa771ca09b38cf592a`
- `lingbot/` — LingBot-World v2, `1895d300d8ac936401689b26389f51cbd36530eb`.
  The studio passes this tree as `lingbot_root`: FastFrames imports its `wan/`
  modules directly and reads the example camera paths (`poses.npy`,
  `intrinsics.npy`, `image.jpg`) out of `lingbot/examples`.
- `state.py` — records this lane's source; the `spec.json` handed to `render.py`
  for every clip carries that identity under `source`, so a render is
  attributable to the exact source that produced it.
