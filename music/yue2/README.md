# YuE2 Composition Studio

YuE2 turns lyrics and a style prompt into an editable melody-and-chord
plan, then realizes that plan as a complete song with vocals and
accompaniment. The WebUI runs YuE2's released stages natively rather than
wrapping a prompt endpoint:

- `plan()` writes the full score, a melody-only score, or a direct-generation
  prefix, keeping each path a separate visible workflow
- `generate_semantic()` writes the complete semantic music sequence
- `synthesize()` runs native AR-NAR flow matching
- the pinned YuE2-Vae decodes 48 kHz stereo audio

Covers work by transcribing a source recording through the SheetSage2
score lane and re-rendering the melody in a new style; edited ABC scores
enter the same lanes as any other score and receive a fresh identity.
Every candidate keeps its FLAC, score, planner tokens, semantic tokens,
acoustic latents, effective configuration, timings, model identities, and
integrity hashes in a resumable plan bundle. All AR and ABC sampling
fields (CFG, ODE steps, CUDA-graph/eager execution, BF16 or FP8-AR, VAE
tile size, candidates, deterministic seeds) stay exposed under Advanced.

Inference is one FIFO GPU job at a time with the AR/NAR model resident on
CUDA; the upstream CPU-offload transition is never enabled.

## Install and start

From the project folder:

```bash
../models/download_models.py yue2
./setupwithuv.sh     # verifies weights, then builds the local venv
./startwithuv.sh     # serves http://127.0.0.1:8270
```

ARCHITECTURE.md documents the stage mapping, artifact contract, and the
cover/transcription lane.

## Upstream

- Model and code: https://github.com/multimodal-art-projection/YuE (YuE-v1 branch)
- Weights: https://huggingface.co/m-a-p/YuE2-3B
- Project page: https://map-yue2.github.io/
