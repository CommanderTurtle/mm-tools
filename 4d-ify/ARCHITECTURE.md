# 4DAnyone Capture Studio

The studio wraps the pinned native 4DAnyone pipeline rather than translating it
through a second inference framework. Jobs run in a cancellable child process,
share the root studio's durable FIFO GPU queue, and retain the complete native
output tree: resolved settings, camera calibration, reusable GVHMR motion,
skeleton conditioning, proposal views, and every dense target view.

GVHMR and DPVO are integrated directly into the 4D project tree. Runtime cache identity is
therefore checked
against its archived, pinned commit rather than requiring nested Git metadata.

The SMPL-X Python runtime is integrated directly from
`CommanderTurtle/archive--smplx@1265df7ba545e8b00f72e7c557c766e15c71632f`;
there is no PyPI or nested Git dependency. Its neutral parameter asset remains
separate from code. Setup accepts the official ZIP layout or exact neutral NPZ,
installs only that file, and creates the compatibility link expected by GVHMR.
No account credential enters the WebUI.

The pipeline is GPU-only for model inference. `4DAnyone-Turbo` is the default
because it retains the upstream quality target with four denoising steps; Base
remains selectable. Outputs are local, the server binds to loopback by default,
and the page has no remote scripts, analytics, or telemetry.
