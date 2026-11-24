"""Sampling Budget3D-Idol-Turbo at a token budget (the CLI is ``scripts/sample.py --turbo``): Idol-turbo's ~5
guidance-free Euler steps, the expert following t (the phase-LoRA model is ``budget3d.models.turbo``)."""
import torch
from tensordict import TensorDict

from ..models.turbo import PHASE_THRESHOLD
from ..octree import N_SOBOL, build_layout, detail_scores, snap_realizable_tokens
from .sampler import time_grid


@torch.no_grad()
def embed_context_phase(model, cond, slow):
    st = model.phase_state
    st.set_uniform(slow, cond.batch_size[0], cond["feature1"].device)
    return model.embed_context(cond)


@torch.no_grad()
def sample_turbo(student, cond, budget=1.0, steps=5, rescale_t=3.0, relayout_until_step=-1, seed=42, device="cuda",
                 noise=None):
    """Guidance-free few-step sampling at a token budget (the idol-turbo sampler with Budget3D layouts).

    One forward per step; the expert follows t; the layout is built from pure noise at step 0 and rebuilt from
    the running clean estimate at every later step (cheap next to a forward), or until ``relayout_until_step``.
    """
    b = cond.batch_size[0]
    n_tokens = snap_realizable_tokens(budget * N_SOBOL)
    h_slow = embed_context_phase(student, cond, True)
    h_fast = embed_context_phase(student, cond, False)
    if noise is not None:
        z, cam = noise
    else:
        torch.manual_seed(seed)
        z = torch.randn(b, student.q_token_length, student.in_channels, device=device)
        cam = torch.randn(b, 1, student.cam_channels, device=device)
    grid = time_grid(steps, rescale_t)
    layout, z0 = None, z
    for i in range(steps):
        t, t_prev = float(grid[i]), float(grid[i + 1])
        if layout is None or relayout_until_step < 0 or i <= relayout_until_step:
            ref = z if i == 0 else z0
            s4, s2 = detail_scores(ref[:, student.perm_to_octree].reshape(b, 16, 16, 16, 2, -1))
            layout = build_layout(s4, s2, n_tokens)
        tt = torch.full((b,), 1000 * t, device=device)
        out = student(TensorDict({"latent": z, "camera": cam}, batch_size=b), tt,
                      h_cond=h_slow if t >= PHASE_THRESHOLD else h_fast, layout=layout)
        v, v_cam = out["latent"], out["camera"]
        z0 = z - t * v
        z = z - (t - t_prev) * v
        cam = cam - (t - t_prev) * v_cam
    return z, cam, layout
