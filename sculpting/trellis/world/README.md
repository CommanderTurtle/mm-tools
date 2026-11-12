# WorldSculpt Scene Foundry

WorldSculpt turns grounded multi-view scenes into compositional mesh
representations: hundreds of individual objects with metric transforms,
not one watertight soup. This studio is the complete released inference
path, run locally:

- expands and audits a calibrated scene package (ZIP/TAR) with strict
  member checks (no absolute paths, symlinks, devices, or expansion bombs)
- preserves the supplied cameras and builds canonical object cubes with
  projected instance crops
- runs the released sparse-structure and 1024-shape multi-view finetunes on
  the local Pixal3D stack, loading it once per scene with no low-VRAM path
- composes every object back into metric world space and exports
  multi-object and merged GLBs with original/predicted camera proofs
- renders calibrated proof frames and plots spatial diagnostics, with an
  optional resumable project archive

The published checkpoint covers the sparse-structure and shape stages; it
does not include the texture finetune. Outputs are therefore labeled as
geometry/normal assets, and nothing is downloaded or substituted at
runtime.

## Install and start

From the project folder:

```bash
../../models/download_models.py worldsculpt
./setupwithuv.sh     # verifies weights, then builds the local venv
./startwithuv.sh     # serves http://127.0.0.1:8263
```

The main Sculpting Studio at `..` covers single-reference and
camera-aware reconstruction on port 8262. ARCHITECTURE.md documents the
data flow, the audit rules, and the delivery contract.

## Upstream

- Models: https://huggingface.co/AlayaLab/WorldSculpt
- Project page: https://alaya-lab.github.io/WorldSculpt
- Paper: https://arxiv.org/abs/2609.05416
