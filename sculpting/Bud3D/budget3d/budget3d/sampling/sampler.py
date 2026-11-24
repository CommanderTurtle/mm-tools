"""Sampling Budget3D-Idol at a token budget (the CLI is scripts/sample.py).

50 Euler steps with classifier-free guidance.  The layout is built at step 0 from pure noise -- there are no dense
warm-up steps -- and refreshed once at step 5 from the running clean estimate, by which point the object's
macro-occupancy has settled; it is then frozen for the remaining steps, so no bisection or argsort runs inside the
Euler loop.  The 5-step guidance-free sampler of the Idol-turbo variant is ``budget3d.sampling.turbo.sample_turbo``.
"""
import os

import numpy as np
import torch
from PIL import Image

from ..octree import N_SOBOL, build_layout, dense_layout, detail_scores, snap_realizable_tokens


class ImagePreprocessor:
    """Idol's own framing of an input image (BiRefNet matting + crop) and its two frozen encoders, kept resident.

    ``mode``: "synthetic" for renders and generated images, "real" for photographs (Idol's crop margins).
    Calling it with an image path returns (the prepared 1024x1024 PIL image, the conditioning TensorDict).
    """

    def __init__(self, weights_dir, device="cuda", mode="synthetic"):
        from ..models.encoders import build_encoder                      # first: it puts Idol's code on sys.path
        from idol.models import BiRefNet
        from idol.preprocess import CROP_MARGIN, preprocess_image
        from safetensors.torch import load_file
        birefnet = BiRefNet()
        birefnet.load_state_dict(load_file(os.path.join(weights_dir, "birefnet.safetensors")), strict=True)
        birefnet = birefnet.eval().requires_grad_(False).to(device)
        self.birefnet = birefnet.half() if torch.device(device).type == "cuda" else birefnet
        self.encoder = build_encoder(device=device)
        self.pre, self.margin, self.dev = preprocess_image, CROP_MARGIN[mode], device

    @torch.no_grad()
    def __call__(self, path):
        prepared = self.pre(Image.open(path), background_remover=self.birefnet, image_size=1024,
                            crop_margin=self.margin)
        pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255.0)[None].to(self.dev)
        return prepared, self.encoder(pixels)


def time_grid(steps, rescale_t):
    """Idol's shifted time grid: ``steps + 1`` values from t = 1 down to 0."""
    t = np.linspace(1, 0, steps + 1)
    return rescale_t * t / (1 + (rescale_t - 1) * t)


@torch.no_grad()
def sample(student, cond, budget=1.0, steps=50, cfg=3.0, rescale_t=3.0, relayout_every=5,
           relayout_until_step=5, seed=42, device="cuda", warmup_steps=0, noise=None):
    """cond: TensorDict of image features -> (latent [B, 8192, 16], camera [B, 1, 5]).

    ``warmup_steps``: how many initial Euler steps run at the dense 8192-token layout before
    the budget layout takes over (0 = the budget is enforced from the very first step, and the
    first layout is built from pure noise).  The relayout schedule counts from the first
    budgeted step, so warmup_steps=0 is exactly the original behaviour.
    """
    b = cond.batch_size[0]
    n_tokens = snap_realizable_tokens(budget * N_SOBOL)
    h_pos = student.embed_context(cond)
    h_neg = student.embed_context(cond.apply(torch.zeros_like))

    if noise is not None:              # caller-fixed starting noise (z, cam), e.g. per object
        z, cam = noise
    else:
        torch.manual_seed(seed)
        z = torch.randn(b, student.q_token_length, student.in_channels, device=device)
        cam = torch.randn(b, 1, student.cam_channels, device=device)
    grid = time_grid(steps, rescale_t)
    layout, z0 = None, z
    dense = dense_layout(b, device) if warmup_steps > 0 else None
    for i in range(steps):
        t, t_prev = float(grid[i]), float(grid[i + 1])
        j = i - warmup_steps                      # index among the budgeted steps
        if j < 0:
            step_layout = dense
        else:
            # the layout is built at the first budgeted step (from pure noise when there is
            # no warm-up) and refreshed from the running clean estimate until
            # relayout_until_step, then frozen (-1 keeps refreshing to the end)
            refresh = relayout_until_step < 0 or j <= relayout_until_step
            if layout is None or (refresh and j % relayout_every == 0):
                ref = z if i == 0 else z0
                s4, s2 = detail_scores(ref[:, student.perm_to_octree].reshape(b, 16, 16, 16, 2, -1))
                layout = build_layout(s4, s2, n_tokens)
            step_layout = layout
        tt = torch.full((b,), 1000 * t, device=device)
        v, v_cam = student.forward_cfg(z, cam, tt, h_pos, h_neg, step_layout, cfg=cfg)
        z0 = z - t * v
        z = z - (t - t_prev) * v
        cam = cam - (t - t_prev) * v_cam
    return z, cam, layout
