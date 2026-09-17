# 4DAnyone Capture Studio

The studio wraps the pinned native 4DAnyone pipeline rather than translating it
through a second inference framework. Jobs run in a cancellable child process,
share the root studio's durable FIFO GPU queue, and retain the complete native
output tree: resolved settings, camera calibration, reusable GVHMR motion,
skeleton conditioning, proposal views, and every dense target view.

The imported GVHMR checkout intentionally has no nested `.git` directory in the
single-repository mm-tools export. Runtime cache identity is therefore checked
against its archived, pinned commit rather than requiring nested Git metadata.

SMPL-X has separate license terms and is never downloaded by the general model
downloader. The setup mode accepts only the official ZIP layout or the exact
neutral NPZ, installs that one file, and creates the compatibility link expected
by GVHMR. No account credential enters the WebUI.

The pipeline is GPU-only for model inference. `4DAnyone-Turbo` is the default
because it retains the upstream quality target with four denoising steps; Base
remains selectable. Outputs are local, the server binds to loopback by default,
and the page has no remote scripts, analytics, or telemetry.
