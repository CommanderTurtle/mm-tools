# SymphonyGen local runtime

SymphonyGen is exposed here as a private two-stage browser studio. The lightweight FastAPI process never holds a model while idle. Each request launches the project’s canonical generator with one released checkpoint, writes MIDI artifacts, and exits; process exit releases all associated CPU and GPU memory.

## Start

```bash
cd symphony
./setupwithuv
./startwithuv.sh
```

The setup prompt selects GPU-aware or CPU-only PyTorch. The UI is available at `http://127.0.0.1:8252`. Settings and relative checkpoint/output paths live in the ignored `.env` copied from `.env.local.example`.

## Browser routes

- **Generate harmony sketches** invokes `arch/harmo/generator.py` with `stage_one_pretrained.pt`.
- **Orchestrate MIDI** invokes `arch/symph/generator.py` with the stage-two baseline, GRPO clamp, or GRPO clamp+track checkpoint. Harmony analysis, variation count, piano exclusion, dissonance sampling, and register decay are explicit controls.
- Completed MIDI files remain under `symphony/outputs` and are served as direct local downloads.

Equivalent CLI examples:

```bash
uv run --active --no-sync python -m arch.harmo.generator \
  ../models/SymphonyGen--SymphonyGen/stage_one_pretrained.pt \
  --batch_size 4 --save_dir outputs/harmony

uv run --active --no-sync python -m arch.symph.generator \
  ../models/SymphonyGen--SymphonyGen/grpo_clamp+track_epoch_6.pt input.mid \
  --analyze_harmo --group_size 2 --save_dir outputs/orchestration
```

The shared downloader retains exactly the four packed generation checkpoints. Training reward models and evaluation assets are not runtime dependencies. Default environment flags force offline Transformers/Hugging Face behavior and disable telemetry.
