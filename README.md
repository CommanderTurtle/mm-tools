# mm-tools

`mm-tools` is a pruned, local-first multimedia workstation monorepo. It contains the runtime source, per-project setup entrypoints, and browser/CLI frontends used by this tool suite. It does not contain checkpoints, virtual environments, caches, generated media, or training/evaluation material.

Pruning from tracked sources lives one AGPLv3 monorepo under the root `LICENSE`. There are no nested project licenses.

The all-in-one frontend for local multimedia inference. Cutting out the cloud.

![feelsyogaman](https://huggingface.co/sHEL1562/shelling/resolve/main/src/1ddd0f54-cbea-4e71-a472-fc594d83f489-cropped.png)

Runtime inference is local. Network access is needed only while installing or downloading the model artifacts required.

This repo was created for protecting open-source runtimes under a documentable heirarchy with AGPL.

## What you actually have to do

Install the host tools required by the projects you plan to run:

- `uv` for Python environments and packages
- `ffmpeg` and `ffprobe` for audio/video projects
- `bun` - [sandwich](https://github.com/CommanderTurtle/sandwich) optional
- Rust/Cargo for `img2svg` and the optional CrisperWhisper launcher
- a GPU or CPU, and likely a monitor

Then download the checkpoints once (the dependencies):

> Due to the size of the model weights. Make sure you're sitting on a speedy drive.

```bash
cd ~ && git clone https://github.com/CommanderTurtle/mm-tools multimedia && cd multimedia
cd models
uv venv --python 3.12.10 --seed --managed-python
source .venv/bin/activate
uv pip install huggingface_hub hf_transfer

### EDIT THE CPU CORES INSIDE `download_models.py` (16 threads by default)
uv run download_models.py
# (This might take a while)
deactivate # when done
```

The downloader is resumable. Run it again after an interrupted transfer. Model downloads remain governed by each model's native license and any access terms accepted for their usage. The ability to self-host a tool rivaling Adobe in 2026 was the purpose for this repo.

Assuming one has a nice GPU, you can host a wide variety of tools. Emulating the top-tier closed source software out there, solely with open-source runtimes. It wouldn't be without the community that building something like this would be possible.

~16-20gb vram recommended. BUT. Each project has their upper unbounded cap. None is authoritative if you run them one at a time. (only ~8-9gb for LongCat)

> So. What can you do?

# Voice Cloning

https://github.com/user-attachments/assets/cb2c3b6d-60e5-4599-9be7-ad5472342b7e

# Translating

*compatible with vox*

![translate](https://huggingface.co/sHEL1562/shelling/resolve/main/src/2-translate.avif)

- an in-house built translation pipeline with native web-ui. Compatibile with [vox](https://github.com/CommanderTurtle/vox) system-audio driver. (This exact fork)

# Speech-to-Text

*compatible with vox*

https://github.com/user-attachments/assets/944a9918-28dd-4a29-8e2b-76d1d80cd4e7

# Video Compression

![videosmaller](https://huggingface.co/sHEL1562/shelling/resolve/main/src/2-videosmaller-clip.avif)

# Video to GIF

https://github.com/user-attachments/assets/5bbf433b-e625-43c0-95ae-b639768bebb1

*because who says ffmpeg is fine in cli?*

# Background Removal (Locally Hosted) -> AI Perfect SVG

*view what a combined workflow looks like:*

[2-ideogram%2Bimg2svg.mp4](https://huggingface.co/sHEL1562/shelling/resolve/main/src/2-ideogram%2Bimg2svg.mp4)

# Redesign's asset-separation (qwen layered):

https://github.com/user-attachments/assets/a74bba02-e665-41fe-b380-1824e4882f68

# Music Tools (MusVIT & Muscriptor Frontends)

https://github.com/user-attachments/assets/263b2134-0891-411b-932a-f0e5ead6b077

![musvit](https://huggingface.co/sHEL1562/shelling/resolve/main/src/2-musvit.png)

## Have a 5090? 

![startupsh](https://huggingface.co/sHEL1562/shelling/resolve/main/src/2-startupsh.avif)

> Run all at once.

The install sequence is simple for each project. Requiring WSL if on Windows.

Jump into any project folder. Run the setup: `./setupwithuv` or similar. 

`cp .env.local.example .env` --edit for config beforehand.

Source into the venv for py projects. With info above. Run setups first.

Tutorial:

```bash
ls -a # (see venv?)
source .venv/bin/activate # Activate it (`deactivate` after)
# Start (once ur in venv):
./startwithuv.sh # or similar
```

### [Vox](https://github.com/CommanderTurtle/vox) - Native Audio/Translate Routing for *this* repo.

<details>
  <summary>See other projects by me!</summary>

### Agent Infrastructure Projects:

https://github.com/CommanderTurtle/persephone just my integration of own internal config + continuity with omp. 
gpu & local-only for just about everything so far

- [Diogenes](https://github.com/CommanderTurtle/diogenes) - hosting (linux, service manager), maintained fork of Odysseus for backends and control center
— this can install and integrate the following if not already installed:

- [sandwich](https://github.com/CommanderTurtle/sandwich) - node package management/auditing for all these package.jsons 
- [libriarian](https://github.com/CommanderTurtle/librarian) - wikifier (makes wikis, dreaming agents)
- [retrieval](https://github.com/CommanderTurtle/retrieval) - skill labrador that fetches necessary skills from archive on disk, rather than bloating agent prompt with full list
- [persephone](https://github.com/CommanderTurtle/persephone) - adds omp (pi agent) gateway layer
- [leetcoder](https://github.com/CommanderTurtle/leetcoder) - allows hermes to puppeteer pi’s for small coding tasks (delegation)

Lots of ideas from stablyai/orca, which does all these things, and where i got a lot of inspiration for the ecosystem

Still WIP. Doing lots of editing as I continue development. Especially with Diogenes automatically setting up my config upon install

### Web-Development:

- [orc](https://github.com/CommanderTurtle/orc) - a multisite/multiframework F# engine for modular websites in py, ts, c#, js, and ruby.
- [reactor](https://github.com/CommanderTurtle/librarian) - a lightweight rust-poller that allows for a live fsharp repo in 7+ languages. Side-hosts vite dev server, zensical serve, jekyll serve, netdocs serving (C#), and more with [preview](https://github.com/CommanderTurtle/preview)
- [tools](https://github.com/CommanderTurtle/retrieval) - just some lightweight tools that help making maintaining websites easy in modular languages

### Libraries:

- [regedited](https://github.com/CommanderTurtle/regedited) - for databases
- [macrohelp](https://github.com/CommanderTurtle/macrohelp) - for macros
- [firebending](https://github.com/CommanderTurtle/firebending) - an Anything-MCP built on macrohelp (TBD)

### Apps:

- [app/adspace](https://app.shel.sh/adspace/templates/1) - ad simulator
- [app/countku](https://app.shel.sh/countku) - count in haiku
- [app/webclip](https://app.shel.sh/webclip) - scrape the web to markdown


</details>


<details>
  <summary>extra</summary>

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
./setupwithrust
./startwithrust                           # http://127.0.0.1:417
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
./startwithuv                             # browser studio: :8223
./startwithuv SCORE_IMAGE_OR_PDF          # direct CLI
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
./startwithuv.sh                          # standalone browser UI: :8177
```

The setup prompt accepts GPU or CPU; `TRANSLATE_ACCELERATOR` remains available for noninteractive runs. The browser UI owns an independent model lifecycle. Its explicit external-load button can attach to an already-loaded `starthttp.sh` instance, but it never falls back between them automatically.

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
| `417` | img2svg |
| `8172` | CrisperWhisper HTTP |
| `8173` | CrisperWhisper UI or ReDesign |
| `8174` | Ideogram/ObjectClear editor |
| `8176` | Translate |
| `8177` | Translate standalone UI |
| `8222` | MuScriptor |
| `8223` | MuSViT local score studio |
| `8230` | LongCat HTTP |
| `8231` | LongCat UI |
| `8240` | Video Compact |
| `8241` | Video to GIF/AVIF |

Services bind to the configured private-LAN interface and do not add public-facing authentication by default. Keep them behind the host firewall or set the supported bearer token where provided.

</details>
