# Idol

Image-to-Gaussian-splat studio.

Choose `idol` in the model downloader, then:

```bash
./setupwithuv.sh
./startwithuv.sh
```

Open **http://127.0.0.1:8283**. Setup uses uv and Bun, Python 3.12 and
CUDA 13 PyTorch wheels. `IDOL_PORT` changes the port; `IDOL_DEVICE` selects the
native device, such as `cuda:0`. Paths follow this checkout, including its
relative `models` link.

Upload, drop or paste an image; choose synthetic or real photo. Settings expose
the native Gaussian count, steps, guidance, time shift and seed. Examples load
their original settings. Drag to orbit, scroll to zoom, and download the native
3DGS PLY from the viewer. The opening backpack is a cached example.

Saved runs retain the original, oriented RGBA input, prepared model input,
camera, settings, PLY and log under `workspace/`. Select one to inspect or repeat
it. Cancel and shutdown release the worker and its compiler children; interrupted
runs remain available. Native CUDA compilation stays enabled, with its normal
first-run compile cost. Model processes exit after each generation.

Model revisions and encoder access conditions are listed in the
[model inventory](../../models/which-ones.txt).
