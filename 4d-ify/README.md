# 4DAnyone Capture Studio

Monocular 4D human capture: ordinary video in, layered 4D point captures out.
The studio wraps the pinned native 4DAnyone pipeline rather than translating
it through a second framework, and keeps the complete output tree of every
run (resolved settings, camera calibration, reusable GVHMR motion, skeleton
conditioning, proposal and dense target views) so results can be inspected,
reused, and re-rendered without re-running inference.

Camera recovery is selectable per job: static tripod, SimpleVO, or DPVO for
moving-camera footage. DPVO, GVHMR, and their Eigen runtime are integrated
directly into the project tree with no nested Git metadata. The SMPL-X body
model ships from a pinned archive; setup fetches only its neutral parameter
asset and verifies it by checksum. No account credential ever enters the
page.

`4DAnyone-Turbo` is the default because it keeps the upstream quality target
in four denoising steps; Base remains selectable. The server binds to
loopback and the page loads no remote scripts, analytics, or telemetry.

## Install and start

From the project folder:

```bash
../models/download_models.py 4d
./setupwithuv.sh     # verifies weights and the SMPL-X asset, then builds the venv
./startwithuv.sh     # serves http://127.0.0.1:8261
```

ARCHITECTURE.md documents the pipeline stages, the camera-recovery modes,
and the output contract.

## Upstream

- Model: https://github.com/ant-research/4DAnyone
- Weights: https://huggingface.co/4DAnyone/4danyone
