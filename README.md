# mm-tools

`mm-tools` is a pruned, local-first multimedia workstation monorepo. It contains the runtime source, per-project setup entrypoints, and browser/CLI frontends used by this tool suite. It does not contain checkpoints, virtual environments, caches, generated media, training/evaluation material, upstream documentation, or nested Git repositories.

All tracked source is one AGPLv3 monorepo under the root `LICENSE`. There are no nested project licenses.

Runtime inference is local. Network access is needed only while installing dependencies and downloading the model artifacts you choose to use.

## What you actually have to do

Install the host tools required by the projects you plan to run:

- `uv` for Python environments and packages
- `ffmpeg` and `ffprobe` for audio/video projects
- `bun` for MuScriptor's browser frontend
- Rust/Cargo for `img2svg` and the optional CrisperWhisper launcher
- `pdftoppm` from Poppler for MuSViT PDF input
- a supported Torch compute stack, or CPU mode

Then download the checkpoints once:

```bash
cd ~/multimedia/models
uv venv --python 3.12.10 --seed --managed-python .venv
source .venv/bin/activate
uv pip install huggingface_hub hf_transfer
python download_models.py
deactivate
```

The downloader is resumable. Run it again after an interrupted transfer. Model downloads remain governed by each model's native license and any access terms accepted at its host.

For Python projects, choose the setup entrypoint in that project only. No root setup wrapper exists.

### Ideogram / ObjectClear

Local Ideogram 4 generation source plus the fuzzy-mask ObjectClear/BiRefNet object-removal browser.

```bash
cd ~/multimedia/ideogram
./setupwithuv
./startwithuv                             # http://127.0.0.1:8174
```

The full Ideogram generator is `local_generate.py`; add `src` to `PYTHONPATH` or install the local package when using that lane. The object-removal lane does not load the large Ideogram generator.

### img2svg

Native VTracer-based raster-to-SVG CLI and browser studio.

```bash
cd ~/multimedia/img2svg
./setupwithuv
./startwithuv                             # http://127.0.0.1:4170
```

### LongCat

Local LongCat voice synthesis and cloning. The HTTP service and browser workbench own independent model lifecycles.

```bash
cd ~/multimedia/longcat
./setupwithuv
./starthttp.sh                            # machine API: :8230
./startwithuv.sh                          # browser UI: :8231
```

### MuScriptor

Audio-to-MIDI/score transcription with a browser piano roll.

```bash
cd ~/multimedia/muscriptor
./setupwithuv
./startwithuv                             # http://127.0.0.1:8222
```

Single-file CLI:

```bash
./startwithuv INPUT.mp3
```

### MuSViT

Sheet-music image/PDF encoding and SMT-backed beKern/MIDI/SVG conversion.

```bash
cd ~/multimedia/musvit
./setupwithuv
./startwithuv SCORE_IMAGE_OR_PDF
```

Use `musvit_embed.py` directly for the embedding lane.

### ReDesign

Flat-image decomposition into editable layers using local detection, segmentation, inpainting, OCR, and Qwen Image Layered components.

```bash
cd ~/multimedia/redesign
./setupwithuv
./startwithuv                             # http://127.0.0.1:8173
```

Review `.env` before launch. Native Diffusers is the primary image lane; an existing private ComfyUI listener is optional.

### Translate

Local EraX translation, language classification, and the optional INT4 arbitration lane.

```bash
cd ~/multimedia/translate
./setupwithuv.sh
./starthttp.sh                            # http://127.0.0.1:8176
```

The setup prompt accepts GPU or CPU; `TRANSLATE_ACCELERATOR` remains available for noninteractive runs.

### Video Compact

Private VideoSmaller-style FFmpeg compression.

```bash
cd ~/multimedia/video-compact
./setupwithuv
./startwithuv                             # http://127.0.0.1:8240
```

### Video to GIF/AVIF

Local trim/crop/resize and animated GIF or AVIF conversion.

```bash
cd ~/multimedia/video-to-gif-avif
./setupwithuv
./startwithuv                             # http://127.0.0.1:8241
```

The host FFmpeg build must expose the requested GIF/AVIF encoder and AVIF muxer.

### CrisperWhisper

Local CrisperWhisper 2.0 transcription in normalized or literal mode. The HTTP service and browser workbench use separate ports and model lifecycles.

```bash
cd ~/multimedia/whisper
./setupwithuv
./starthttp.sh                            # machine API: :8172
./startwithuv.sh                          # browser UI: :8173
```

ReDesign also defaults to `8173`; change `CW2_UI_PORT` when both browser services run together.

## Port map

| Port | Runtime |
| --- | --- |
| `4170` | img2svg |
| `8172` | CrisperWhisper HTTP |
| `8173` | CrisperWhisper UI or ReDesign |
| `8174` | Ideogram/ObjectClear editor |
| `8176` | Translate |
| `8222` | MuScriptor |
| `8230` | LongCat HTTP |
| `8231` | LongCat UI |
| `8240` | Video Compact |
| `8241` | Video to GIF/AVIF |

Services bind to the configured private-LAN interface and do not add public-facing authentication by default. Keep them behind the host firewall or set the supported bearer token where provided.
