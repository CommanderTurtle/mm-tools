"""The persistent streaming teacher rollout buffer.

Idol released its denoiser and its 3DGS decoders but not the encoder that turns a 3D asset
into a ``pos_pe``-aligned [8192, 16] latent, so there is no ground-truth z_0 to train on.
Instead the frozen teacher supplies the targets, and it does so *on policy*: the student is
asked to match the teacher at states the teacher actually visits along its own CFG-3 Euler
trajectories, which is exactly the manifold sampling will put the student on.

Rolling a trajectory from t = 1 to a random step on every training step would cost about
R/2 teacher passes per update.  Instead ``RolloutStream`` keeps the trajectories *alive*
across optimizer steps, staggered over the R rollout phases, so each training step costs
exactly one teacher pass and only ~1/R of the trajectories need a new image encoded.
"""
import torch
from tensordict import TensorDict

from ..octree import N_SOBOL


def warped_grid(n, steps, rescale_t=3.0, jitter=0.4, device="cpu", generator=None):
    """[n, steps + 1] decreasing time grids in [0, 1], Idol's warp with interior jitter.

    s goes 0 -> 1 over the rollout, u = 1 - s, and t = rescale_t u / (1 + (rescale_t - 1) u),
    which spends more steps near t = 1 where the global shape is decided.  Jittering the
    interior nodes makes the union of the grids cover t continuously instead of R values.
    """
    s = torch.linspace(0, 1, steps + 1, device=device).expand(n, steps + 1).clone()
    if jitter > 0 and steps > 1:
        noise = (torch.rand(n, steps - 1, device=device, generator=generator) * 2 - 1) * (jitter / steps)
        s[:, 1:steps] = (s[:, 1:steps] + noise).clamp(0, 1).sort(dim=1).values
    u = 1 - s
    return rescale_t * u / (1 + (rescale_t - 1) * u)


class RolloutStream:
    """``n_traj`` live teacher trajectories on one device.

    Each trajectory holds its own cached image conditioning, its noisy state (z, cam), its
    rollout index k and its jittered time grid.  ``targets`` evaluates the teacher once on
    a slice of trajectories; ``advance`` moves that slice one Euler step.
    """

    def __init__(self, teacher, encoder, images, n_traj, latent_tokens=N_SOBOL, latent_ch=16, cam_ch=5,
                 steps=25, cfg=3.0, rescale_t=3.0, grid_jitter=0.4, k0_stay_p=0.70,
                 device="cuda", dtype=torch.bfloat16, seed=0, model_channels=1024):
        self.teacher, self.encoder, self.images = teacher, encoder, images
        self.n, self.steps, self.cfg = n_traj, steps, cfg
        self.rescale_t, self.jitter, self.k0_stay_p = rescale_t, grid_jitter, k0_stay_p
        self.device, self.dtype = torch.device(device), dtype
        self.lat, self.ch, self.cam_ch = latent_tokens, latent_ch, cam_ch
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        self.z = torch.zeros(n_traj, latent_tokens, latent_ch, device=device)
        self.cam = torch.zeros(n_traj, 1, cam_ch, device=device)
        self.z0 = torch.zeros_like(self.z)
        self.f1 = torch.zeros(n_traj, 4101, 1280, device=device, dtype=dtype)
        self.f2 = torch.zeros(n_traj, 4101, 128, device=device, dtype=dtype)
        # the teacher is frozen and its context refiner reads neither t nor z_t, so each
        # trajectory's refined condition is constant for its whole rollout -- and the
        # unconditional branch, whose features are all zero, is constant for the whole run
        # fp32: the student computes its own h_cond in fp32, so storing the teacher's in
        # bf16 would put a systematic per-image offset into every target that the student
        # has no way to reproduce
        self.hc = torch.zeros(n_traj, 4101, model_channels, device=device)
        with torch.no_grad():
            zero = TensorDict({"feature1": torch.zeros(1, 4101, 1280, device=device),
                               "feature2": torch.zeros(1, 4101, 128, device=device)}, batch_size=1)
            self.hc_neg = teacher.embed_context(zero)
        # staggered phases, so only ~n/R trajectories restart on any one step
        self.k = (torch.arange(n_traj, device=device) * steps) // max(1, n_traj) % steps
        self.grid = warped_grid(n_traj, steps, rescale_t, grid_jitter, device, self.gen)
        self.restarts = 0
        for i in range(n_traj):
            self._new_image(torch.tensor([i], device=device))
            self._new_noise(torch.tensor([i], device=device))

    # -----------------------------------------------------------------------------------
    def _new_image(self, idx):
        """Encode a fresh batch of object images into the slots ``idx`` (the only DINOv3 cost)."""
        imgs = next(self.images)[: len(idx)].to(self.device, non_blocking=True)
        if len(imgs) < len(idx):
            idx = idx[: len(imgs)]
        feats = self.encoder(imgs)
        self.f1[idx] = feats["feature1"].to(self.dtype)
        self.f2[idx] = feats["feature2"].to(self.dtype)
        with torch.no_grad():
            # read the features back exactly as the student will, so teacher and student
            # condition on bit-identical inputs
            same = TensorDict({"feature1": self.f1[idx].float(), "feature2": self.f2[idx].float()},
                              batch_size=len(idx))
            self.hc[idx] = self.teacher.embed_context(same)
        self.restarts += len(idx)

    def _new_noise(self, idx):
        self.z[idx] = torch.randn(len(idx), self.lat, self.ch, device=self.device, generator=self.gen)
        self.cam[idx] = torch.randn(len(idx), 1, self.cam_ch, device=self.device, generator=self.gen)
        self.z0[idx] = self.z[idx]
        self.grid[idx] = warped_grid(len(idx), self.steps, self.rescale_t, self.jitter,
                                     self.device, self.gen)

    def cond(self, idx, drop=None):
        """The cached conditioning of ``idx``; rows where ``drop`` is true are zeroed (CFG branch)."""
        f1, f2 = self.f1[idx].float(), self.f2[idx].float()
        if drop is not None and bool(drop.any()):
            keep = (~drop).float().reshape(-1, 1, 1)
            f1, f2 = f1 * keep, f2 * keep
        return TensorDict({"feature1": f1, "feature2": f2}, batch_size=len(idx))

    def state(self, idx):
        """(x, t, t01) of the trajectories ``idx`` at their current rollout index."""
        k = self.k[idx]
        t01 = self.grid[idx].gather(1, k.unsqueeze(1)).squeeze(1)
        x = TensorDict({"latent": self.z[idx], "camera": self.cam[idx]}, batch_size=len(idx))
        return x, t01 * 1000.0, t01

    # -----------------------------------------------------------------------------------
    @torch.no_grad()
    def targets(self, idx):
        """One packed [2B] teacher pass -> the conditional and unconditional targets.

        Returns the stop-gradient velocity, camera velocity and per-point trunk features of
        both CFG branches, plus the guided velocity used to advance the trajectory.
        """
        x, t, t01 = self.state(idx)
        tg = dict(self.teacher.forward_dual(x["latent"], x["camera"], t, self.hc[idx],
                                            self.hc_neg, return_features=True))
        tg["v_cfg"] = (1 + self.cfg) * tg["v_pos"] - self.cfg * tg["v_neg"]
        tg["cam_cfg"] = (1 + self.cfg) * tg["cam_pos"] - self.cfg * tg["cam_neg"]
        tg["t01"], tg["x"] = t01, x
        # the running clean estimate, which is what the layout is normally built from
        self.z0[idx] = x["latent"] - t01.reshape(-1, 1, 1) * tg["v_cfg"]
        return tg

    @torch.no_grad()
    def advance(self, idx, targets):
        """Take the teacher's Euler step, oversampling the pure-noise phase k = 0."""
        k = self.k[idx]
        t_cur = self.grid[idx].gather(1, k.unsqueeze(1)).squeeze(1)
        t_next = self.grid[idx].gather(1, (k + 1).unsqueeze(1)).squeeze(1)
        dt = (t_cur - t_next).reshape(-1, 1, 1)
        self.z[idx] = self.z[idx] - dt * targets["v_cfg"]
        self.cam[idx] = self.cam[idx] - dt * targets["cam_cfg"]

        # at k = 0 (t = 1) stay put with fresh noise on the same image, with probability k0_stay_p:
        # 0-warmup inference builds its layout from pure noise, so that state is worth oversampling
        at0 = k == 0
        stay = at0 & (torch.rand(len(idx), device=self.device, generator=self.gen) < self.k0_stay_p)
        self.k[idx] = torch.where(stay, k, k + 1)
        if bool(stay.any()):
            self._new_noise(idx[stay])
        done = self.k[idx] >= self.steps
        if bool(done.any()):
            where = idx[done]
            self.k[where] = 0
            self._new_image(where)
            self._new_noise(where)
