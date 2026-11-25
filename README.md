# mm-tools

`mm-tools` is a pruned, local-first multimedia workstation monorepo. It contains the runtime source, per-project setup entrypoints, and browser/CLI frontends used by this tool suite. It does not contain checkpoints, virtual environments, caches, generated media, or training/evaluation material.

Root-authored mm-tools code is distributed under the root AGPLv3 `LICENSE`.
Imported upstream attributions and revisions are recorded in [the commit ledger](https://github.com/CommanderTurtle/mm-tools/commits/main/).

The all-in-one frontend for local multimedia inference. Cutting out the cloud.

![feelsyogaman](https://huggingface.co/sHEL1562/shelling/resolve/main/src/1ddd0f54-cbea-4e71-a472-fc594d83f489-cropped.png)

Runtime inference is local. Network access is needed only while installing or downloading the model artifacts required.

This repo was created for protecting open-source runtimes under a documentable heirarchy with AGPL.

## What you actually have to do

| Directory | Runtimes |
| --- | --- |
| `vocals/tts`, `vocals/stt`, `vocals/singing` | Speech synthesis, transcription, singing |
| `music` | Composition, generation, score transcription |
| `motion-sim` | Character animation and motion reconstruction |
| `sculpting` | Mesh generation and editing |
| `world-gen` | World generation |
| `visual-modeling` | Image understanding and decomposition |

`translate`, `img2svg`, `video-compact`, `models`, `workflows`, and
`tests` remain at the root. GIF/AVIF conversion lives in `video-compact/gif-avif`.

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
uv pip install -U huggingface_hub hf_transfer

uv run download_models.py
# Choose workers first, then select project bundles or press Enter for all.
# Non-interactive full install: uv run download_models.py all --yes --workers 24
deactivate # when done
```

The downloader is resumable. Run it again after an interrupted transfer. Use
`uv run download_models.py --list` for the complete current menu.
`uv run download_models.py --audit` inventories the store without downloading:
what each bundle still needs, plus files claimed by no bundle, with an offer
to sweep them (`--prune` automates the sweep).
The imports include model files that carefully get cloned from hf/ms sources with multi-worker speed. Each project builds with some type of `setupwith**.sh` file. These prompt for single-GPU adapters, setup/start contracts, and
browser studio builds; model artifacts remain external to the downloader and
resumable. Some deliberately separate shared encoder/VAE bundles from
the official runtimes for quants that run on consumer hardware. 

Model downloads remain governed by each model's native license and any access
terms accepted for their usage. ARDY uses an ungated, pre-merged INT8 LLM2Vec
encoder that preserves the exact Llama 3 + MNTP + supervised-adapter pipeline;
the original gated BF16 assembly is not downloaded. 4DAnyone's neutral SMPL-X
body model remains a separate manual download. See
[the model inventory](models/which-ones.txt) for
every included artifact, output path, and explicit exclusion. The ability to
self-host a tool rivaling Adobe in 2026 was the purpose for this repo.

Assuming one has a nice GPU, you can host a wide variety of tools. Emulating the top-tier closed source software out there, solely with open-source runtimes. It wouldn't be without the community that building something like this would be possible.

~16-20gb vram recommended for most tools. The largest quality-first music paths benefit from 24-32 GB. Each project has its own upper bound, and none is authoritative if you run them one at a time. (only ~8-9gb for LongCat)

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

# Motion authoring

Motion authoring: [MotionBricks studio](motion-sim/nv-motionbricks/ARCHITECTURE.md)
provides SOMA and Unitree G1 inbetweening, Smart Primitive authoring, scene editing
and interactive character preview through the pinned native runtime.

[FlowHMR studio](motion-sim/reftoanimesh/ARCHITECTURE.md) reconstructs SMPL-H motion
from reference video, with person selection, reusable intermediates and an
optional side-by-side PHC humanoid rollout.

[PhysAg3D studio](sculpting/PhysAg3D/ARCHITECTURE.md) builds articulated objects from
images through separately saved TRELLIS generation and OMP/Blender/MuJoCo stages,
with part controls, motion playback and portable asset bundles.

[Idol studio](sculpting/idol/ARCHITECTURE.md) turns images into Gaussian splats, with
the original Space presets and sampling controls, an orbit viewer and PLY export.

[SceneBloom studio](sculpting/scenebloom/ARCHITECTURE.md) generates furnished rooms
with local vision, saved stages, spatial inspection and textured GLB export.

[Hy-Nomas studio](world-gen/hynomas/ARCHITECTURE.md) generates continuous worlds with
text, image starts and recorded camera motion, rollback, saved state and MP4 exports.

# Video Compression

![videosmaller](https://huggingface.co/sHEL1562/shelling/resolve/main/src/2-videosmaller-clip.avif)

# Video to GIF

https://github.com/user-attachments/assets/5bbf433b-e625-43c0-95ae-b639768bebb1

*because who says ffmpeg is fine in cli?*

# Background Removal (Locally Hosted) -> AI Perfect SVG

*view what a combined workflow looks like:*

https://github.com/user-attachments/assets/2b6de3b7-21a0-4354-9fad-9dc20af4cade

# Redesign's asset-separation (qwen layered):

https://github.com/user-attachments/assets/a74bba02-e665-41fe-b380-1824e4882f68

# Music Tools (generation, singing, orchestration, OMR, and transcription)

https://github.com/user-attachments/assets/263b2134-0891-411b-932a-f0e5ead6b077

![musvit](https://huggingface.co/sHEL1562/shelling/resolve/main/src/2-musvit.png)

# Update 8-20, added minimax music custom stdio

> (uses minimized, stripped comfyUI pipeline directly, but custom WebUI)

https://github.com/user-attachments/assets/278fbb3b-93c9-4d4a-a4d8-0acaff4bc30a


## Have a 5090? 

![startupsh](https://huggingface.co/sHEL1562/shelling/resolve/main/src/2-startupsh.avif)

> Run all at once.

The install sequence is simple for each project. Requiring WSL if on Windows.

Jump into a project folder, run its setup once, and start it. Setup creates the ignored `.env` and isolated `.venv` automatically.

```bash
cd ~/multimedia/PROJECT
./setupwithuv.sh
./startwithuv.sh
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
- [reactor](https://github.com/CommanderTurtle/reactor) - a lightweight rust-poller that allows for a live fsharp repo in 7+ languages. Side-hosts vite dev server, zensical serve, jekyll serve, netdocs serving (C#), and more with [preview](https://github.com/CommanderTurtle/preview)
- [tools](https://github.com/CommanderTurtle/tools) - just some lightweight tools that help making maintaining websites easy in modular languages

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

Local Ideogram 4 generation source plus the fuzzy-mask editing browser: ObjectClear, BiRefNet, OpenCV, and optional private-Comfy Ideogram masked editing.

```bash
cd ~/multimedia/visual-modeling/background-removal/ideogram
./setupwithuv.sh
./startwithuv.sh                             # http://127.0.0.1:8174
```

The standalone Ideogram generator is `local_generate.py`; add `src` to `PYTHONPATH` or install the local package when using that lane. The editor loads Ideogram only when explicitly selected. Its private engine reuses bundled Comfy code (not the Minimax service), supports local caption drafting and configurable crop processing, and preserves the original image outside the feathered mask. See [Ideogram architecture and setup](visual-modeling/background-removal/ideogram/ARCHITECTURE.md).

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
cd ~/multimedia/vocals/tts/longcat
./setupwithuv.sh
./starthttp.sh                            # machine API: :8230
./startwithuv.sh                          # browser UI: :8231
```

### ACE-Step

Native ACE-Step 1.5 music generation, editing, repainting, extension, and audio-to-audio controls.

```bash
cd ~/multimedia/music/acestep
./setupwithuv.sh
./startwithuv.sh                          # http://127.0.0.1:8250
```

The browser process preloads the native XL-SFT DiT and 4B 5 Hz language model: the quality-first ACE combination for a 24 GB+ workstation. It reads the shared native ACE-Step model tree directly and releases all weights on `Ctrl+C`.

### MiniMax Music 3

Standalone MiniMax Music 3 text-to-song production desk using the official graph: full FP16 DiT, pruned INT8 text encoder, DAV decoder, real CFG/top-k/sampler controls, and an animated lyrics-and-spectrum performance view. It does not require a separately running ComfyUI.

```bash
cd ~/multimedia/music/minimax
./setupwithuv.sh
./startwithuv.sh                          # http://127.0.0.1:8254
```

The browser owns a private loopback song engine. Its separate Prompt Guide tab uses the bundled Comfy `CLIPLoader(type=krea2) → TextGenerate` path with the local `SergiusFlavius/Qwen3-VL-4B-Instruct-heretic-NVFP4` checkpoint, but begins fully off and unloaded; enabling it starts a second loopback-only engine. A Guided Brief Lab adds one-click recipes plus expandable genre, instrumentation, key/mode, BPM, meter, groove, harmony, vocal, form, production, and listening-context vocabulary. It creates plain-English copypasta without invoking a model or silently changing controls. Guide results remain detached, with manual copy controls for MiniMax's Global Metadata, Vocal Details, Arrangement, and tuning notes. `Load models` and `Unload` control song-model residency explicitly, while `Ctrl+C` unloads all weights and closes the web app plus both private engines. Links in the studio hand covers/editing to ACE-Step, loops to Foundation-1, and audio reverse engineering to MuScriptor.

### Stable Audio Foundation

Foundation-1 text-to-audio with the enhanced Stable Audio Gradio controls, local T5 conditioning, MIDI analysis, and optional TorchAO controls.

```bash
cd ~/multimedia/music/stableaudio
./setupwithuv.sh
./startwithuv.sh                          # http://127.0.0.1:8251
```

### SymphonyGen

Two-stage symbolic composition: generate harmony MIDI, then orchestrate a MIDI sketch with baseline or GRPO checkpoints.

```bash
cd ~/multimedia/music/symphony
./setupwithuv.sh
./startwithuv.sh                          # http://127.0.0.1:8252
```

The web process is model-free while idle. Each generation request owns and releases its model in a short-lived worker.

### VocalRender

Prompt-conditioned local singing synthesis with the highest-quality Pro weights only, explicit load/unload controls, browser playback, and WAV downloads.

```bash
cd ~/multimedia/vocals/singing/vocalrender
./setupwithuv.sh
./startwithuv.sh                          # http://127.0.0.1:8253
```

### MuScriptor

Audio-to-MIDI/score transcription with a browser piano roll.

```bash
cd ~/multimedia/music/muscriptor
./setupwithuv.sh
./startwithuv.sh                             # http://127.0.0.1:8222
```

Single-file CLI:

```bash
./startwithuv.sh INPUT.mp3
```

### MuSViT

Sheet-music image/PDF encoding and SMT-backed beKern/MIDI/SVG conversion.

```bash
cd ~/multimedia/music/musvit
./setupwithuv.sh
./startwithuv.sh                             # browser studio: :8223
./startwithuv.sh SCORE_IMAGE_OR_PDF          # direct CLI
```

Use `musvit_embed.py` directly for the embedding lane.

### Unlimited OCR 2

[Unlimited OCR 2 studio](visual-modeling/OCR/ARCHITECTURE.md) parses ordered PDF/image
pages into a structured document with layout overlays, tables, equations and
figure crops. Select `unlimited-ocr2` in the downloader; serve on `:8287`.

### Pixel Wizard

[Pixel Wizard](visual-modeling/pixel-wizard/ARCHITECTURE.md) uses WeVisOne-VL-4B for
grounding, detection, counting, pointing, OCR, captions, segmentation and alpha
matting. Select `pixel-wizard` in the downloader; serve on `:8288`.

### ReDesign

Flat-image decomposition into editable layers using local detection, segmentation, inpainting, OCR, and Qwen Image Layered components.

```bash
cd ~/multimedia/visual-modeling/redesign
./setupwithuv.sh
./startwithuv.sh                             # http://127.0.0.1:8173
```

Review `.env` before launch. Native Diffusers is the primary image lane; an existing private ComfyUI listener is optional.

### Translate

Local EraX text translation, language classification, and a shared INT4
multimodal lane for image explanation, OCR translation, custom visual prompts,
and spoken-language arbitration.

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
./setupwithuv.sh
./startwithuv.sh                             # http://127.0.0.1:8240
```

### Video to GIF/AVIF

Local trim/crop/resize and animated GIF or AVIF conversion.

```bash
cd ~/multimedia/video-compact/gif-avif
./setupwithuv.sh
./startwithuv.sh                             # http://127.0.0.1:8241
```

The host FFmpeg build must expose the requested GIF/AVIF encoder and AVIF muxer.

### Nemotron

Local Nemotron 3.5 transcription, speaker timelines, karaoke, subtitles and audio questions.

```bash
uv run models/download_models.py nemotron
cd vocals/stt/nemotron
./setupwithuv.sh
./startwithuv.sh                             # http://127.0.0.1:8289
```

Native NVFP4/FP8 on Blackwell; microphone, media files and explicit URL imports,
word seeking, CPU vocal separation, saved conversations and timed exports.

### Audio8

Local Realistic Air speech, reference voices and paragraph takes.

```bash
uv run models/download_models.py audio8
cd vocals/tts/audio8
./setupwithuv.sh
./startwithuv.sh                             # http://127.0.0.1:8290
```

Native BF16 and 44.1 kHz codec, microphone/file references, waveform seeking,
saved sessions, WAV/FLAC and codec-token exports.

### Ming3

Local voice design, cloning, multilingual delivery and multi-speaker podcasts.

```bash
uv run models/download_models.py ming3:1  # dense, ~3 GB
uv run models/download_models.py ming3:2  # optional MoE + podcast, ~18 GB (ming3 = both tiers)
cd vocals/tts/ming3
./setupwithuv.sh
./startwithuv.sh                             # http://127.0.0.1:8291
```

Separate native BF16 environments, reference recording, director controls,
vocal tags, saved takes, waveform seeking and WAV/FLAC exports.

### CrisperWhisper

Local CrisperWhisper 2.0 transcription in normalized or literal mode. The HTTP service and browser workbench use separate ports and model lifecycles.

```bash
cd ~/multimedia/vocals/stt/whisper
./setupwithuv.sh
./starthttp.sh                            # machine API: :8172
./startwithuv.sh                          # browser UI: :8173
```

ReDesign also defaults to `8173`; change `CW2_UI_PORT` when both browser services run together.

### Underdog Bark

English transcription with independent Bark 3.0, Plus, Clean and Pro loaders.
Whisper-style studio, microphone recording, transcript comparison and TXT/JSON export.

```bash
uv run models/download_models.py underdog  # all four Bark variants; one tier only: underdog:4 (1 base, 2 plus, 3 clean, 4 pro)
cd vocals/stt/underdog
./setupwithuv.sh
./startwithuv.sh                           # studio: :8293
./starthttp.sh                             # optional API: :8292, BARK_MODEL in .env
```

## Port map

| Port | Runtime |
| --- | --- |
| `417` | img2svg |
| `8172` | CrisperWhisper HTTP |
| `8173` | CrisperWhisper UI or ReDesign |
| `8174` | Ideogram/ObjectClear editor |
| `8176` | Translate |
| `8177` | Translate standalone UI |
| `8221` | MotionBricks character studio |
| `8222` | MuScriptor |
| `8223` | MuSViT local score studio |
| `8230` | LongCat HTTP |
| `8231` | LongCat UI |
| `8240` | Video Compact |
| `8241` | Video to GIF/AVIF |
| `8250` | ACE-Step 1.5 |
| `8251` | Stable Audio Foundation-1 |
| `8252` | SymphonyGen local studio |
| `8253` | VocalRender local studio |
| `8254` | MiniMax Music 3 local studio |
| `8281` | FlowHMR reference-to-motion studio |
| `8282` | PhysAg3D staged asset studio |
| `8283` | Idol Gaussian studio |
| `8284` | Hy3D mesh studio |
| `8285` | SceneBloom scene studio |
| `8286` | Hy-Nomas world studio |
| `8287` | Unlimited OCR 2 document studio |
| `8288` | Pixel Wizard vision studio |
| `8289` | Nemotron transcription studio |
| `8290` | Audio8 speech studio |
| `8291` | Ming3 speech and podcast studio |
| `8292` | Underdog Bark HTTP |
| `8293` | Underdog Bark studio |
| `8294` | `sculpting/Bud3D` · Budget3D image-to-splats (`bud3d` / `bud3d-turbo`) |
| `8295` | `world-gen/Splat-V3` · SplatAnything reconstruction (`splat-v3`; optional `splat-v3-showcase`) |
| `8296` | `world-gen/fastframes` · FastFrames (`fastframes-live / fastframes-l`) |
| `8297` | `vocals/inworld-isolator` · Voice Isolator (`inworld-isolator`) |
| `8298` | `vocals/sophon-agent` · Sophon duplex speech studio (`sophon-agent`) |
| `8299` | `motion-sim/FAB` · FAB motion studio (`fab`) |
| `8300` | `visual-modeling/jevertion` · Jeveration image desk (`jeveration`) |
| `8301` | `visual-modeling/background-removal/RMBG-3.0` · BRIA matte studio (`bria-rmbg3`) |
| `8302` | `visual-modeling/ask-video` · WeMM video desk (`wemm-video`) |

The new studios default to loopback; existing services retain their configured private-LAN interfaces. Keep network-exposed services behind the host firewall or set the supported bearer token where provided.

### Update 9-17, new, local creative studios

The current expansion uses one shared, dependency-light product surface with
project-native adapters. Every studio has responsive dark/light themes,
searchable task cards, complete native controls, presets and drafts, raw
streaming uploads, a persistent single-GPU queue, cancellation, SSE progress,
model load/unload health, an output library, side-by-side comparison, and API
examples. There are no hosted assets, analytics, CORS opt-ins, or automatic
model downloads at runtime.

| Project | Canonical surface | Default port | VRAM (approx) |
| --- | --- | ---: | ---: |
| AuK | 16 speech generation, cloning, editing, emotion, timing, and nonverbal workflows; Vox-compatible speech route | 8260 | ~10 GB |
| 4DAnyone | monocular 4D human reconstruction and reusable result inspection | 8261 | ~14 GB |
| TRELLIS.2 + Pixal3D | image/multiview sculpting, local refinement, and interactive GLB inspection | 8262 | ~20 GB |
| WorldSculpt | complete multi-object scene reconstruction and scene inspection | 8263 | ~18 GB |
| ID-V2V | identity-preserving transfer, normal/depth guidance, and reusable preprocessing | 8265 | ~22 GB |
| ARDY + SOMA-X | text-to-motion, motion inspection, parametric bodies, and hands | 8266 | ~8 GB |
| LingBot World V2 | 14B causal-fast world generation and visual camera-path authoring; ComfyUI GGUF tiers for smaller cards | 8267 | 64 GB native |
| Fire3D | RGB/RGBD scene reconstruction, native-release evaluation, and GLB inspection | 8268 | ~20 GB |
| YuE2 | composition, planning, continuation/cover, score realization/editing, transcription, resume, and VAE decode | 8270 | ~24 GB |

### Update 11-12, more local creative studios

This expansion folds thirteen additional runtimes into the same shared,
dependency-light product surface with project-native adapters: motion repair
and video-to-character animation, geometry, scene, and world generation,
optical character recognition and natural-language image editing, plus speech
synthesis, transcription, and podcast generation. Every studio keeps
responsive dark/light themes, searchable task cards, complete native controls,
presets and drafts, raw streaming uploads, the persistent single-GPU queue,
cancellation, SSE progress, model load/unload health, an output library,
side-by-side comparison, and API examples. Every runtime source rides
in-tree, so a fresh checkout restores any runtime with dependency
installation alone; only checkpoint bytes route through the shared downloader.

| Project | Canonical surface | Default port | VRAM (approx) |
| --- | --- | ---: | ---: |
| Audio8 | speech synthesis | 8179 | ~8 GB |
| MotionBricks | motion repair, inpainting, outpainting, and trajectory steering | 8272 | ~10 GB |
| FlowHMR | video and reference to animated character meshes | 8273 | ~12 GB |
| PhysAg3D | physics-grounded 3D human scenes | 8275 | ~16 GB |
| Idol | video to 3D character generation and inspection | 8276 | ~22 GB |
| Hy3D | image and video to high-fidelity 3D geometry | 8277 | ~22 GB |
| SceneBloom | procedural 3D scene generation with Blender-MCP refinement | 8278 | ~12 GB |
| Hy-Nomas | interactive generated worlds with first-person play | 8279 | ~28 GB |
| Unlimited OCR 2 | document and scene text extraction | 8280 | ~12 GB |
| Pixel Wizard | natural-language image editing | 8281 | ~24 GB |
| Nemotron | long-form transcription | 8282 | ~6 GB |
| Ming3 | speech synthesis and podcast generation | 8291 | ~16 GB |
| Underdog | Bark speech behind plain HTTP and studio surfaces | 8292/8293 | ~6 GB |

### Update 11-24, lanes that run their committed source

Nine more studios join the same surface, with one change: each lane runs the
source committed beside it, so a fresh checkout needs dependency installation
only and no network access at run time. Every lane carries its own
`ARCHITECTURE.md` and a `state.py` that records the lane source in job records
and startup logs, so a result says which source produced it. Checkpoints still
come from `models/download_models.py`.

| Project | Canonical surface | Default port | VRAM (approx) |
| --- | --- | ---: | ---: |
| FAB | one take to tracked body, fitted mesh, contacts and a retargeted G1 reference | 8299 | ~12 GB |
| Budget3D | reference image to navigable Gaussians, Idol-style view sweep and turntable | 8294 | ~22 GB |
| SplatAnything v3 | views, scene video or panorama to a walkable Gaussian scene | 8295 | ~24 GB |
| FastFrames | driveable LingBot worlds with a waiting room, plus FastFrames-L clips | 8296 | 32–40 GB |
| Inworld Voice Isolator | upload, file and realtime microphone vocal separation | 8297 | ~4 GB |
| Sophon Realtime Tag | full-duplex voice calls with recorded-clip voices | 8298 | ~24 GB |
| Jeveration | image generate, edit and dense runs behind one workflow page | 8300 | ~24 GB |
| BRIA RMBG-3.0 | instruction-fixed alpha matte and clean background export | 8301 | ~8 GB |
| WeMM-Video | clip, image or track question answering on one owned AWQ engine | 8302 | ~28 GB |

</details>

# More?

### Trellis & Pixal3D - 3d modeling/scuplting pipelines

https://github.com/user-attachments/assets/513e5f2e-09d8-4823-9799-5d73c494842c

### YuE2

The only Yue2 frontend with stem splitting!

![yuesplitter](https://huggingface.co/sHEL1562/shelling/resolve/main/src/stems.avif)

<a href="https://a.shel.sh/#ss:~9h=@:aQCb685#]:[a@)a;YaCeHZ[[K4([v)*p&OJB274p;Qifr" title="a.shel.sh superlink">Live Viewer Exporting</a>

- (Share with Friends via app.shel.sh/make & a.shel.sh superlinks, entirely local)

Another:

https://a.shel.sh/#media:rZ9jBas~g*mK]Q-()SB$IFDo]Xl-b5y[P,/@f/FwenhdLmh~Aaw
