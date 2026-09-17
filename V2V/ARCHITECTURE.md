# ID-V2V Film Lab architecture

The product surface is `local_app/studio.json`; `local_app/adapter.py` is the
only inference adapter.  The shared Studio owns FIFO queueing, cancellation,
uploads, persistent run history, previews, output downloads, and its local API.

Every diffusion job starts a job-scoped ComfyUI process on an ephemeral
`127.0.0.1` port with API nodes and all custom nodes disabled.  The process is
pinned to the PR #15139 checkout and is launched with `--gpu-only`.  It loads
the Kijai INT8 ConvRot checkpoint directly; no model is staged in system RAM or
offloaded to the CPU.

## Conditioning contract

The normal ID-V2V path reproduces the released graph:

1. the stylized frame is encoded by CLIP Vision;
2. `WanImageToVideo` pins it as frame zero and uses the same image for
   `ref_pad_image`, the SVI-style identity padding added by PR #15139;
3. `WanVaceToVideo` attaches the pixel-control video;
4. the normal+depth checkpoint appends synchronized normal and depth VACE
   branches in their documented order;
5. native Wan sampling, VAE decode, and Comfy video encoding remain inside the
   private runtime.

Automatic foreground preparation uses the already allowlisted BiRefNet model
from `sculpting/pretrained/deps`; it is released before Comfy loads the 20 GB
diffusion model.  This avoids a duplicate segmentation checkpoint and avoids
the native project's gated SAM3 dependency.  A prepared SAM3-compatible video
can always be uploaded instead.  Relighting deliberately skips segmentation
and uses raw source pixels, matching the upstream recipe.

Long footage is split into overlapping `4n+1` clips.  Each next clip starts on
the prior clip's final generated frame unless the user supplied an explicit
style keyframe at that index.  During assembly, the preceding duplicate frame
is discarded so the newly pinned keyframe wins.  The final package includes a
trimmed source, optional generated/source split, alternating flip test,
controls, per-clip evidence, and a resolved JSON receipt.

No cloud endpoint, telemetry client, remote model lookup, Node runtime, or
public Comfy listener is present.
