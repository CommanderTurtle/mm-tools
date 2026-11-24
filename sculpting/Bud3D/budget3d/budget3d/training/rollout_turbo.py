"""The streaming rollout buffer for the idol-turbo teacher (guidance-free, phase-LoRA experts).

Same idea as rollout.RolloutStream -- live trajectories kept across optimizer steps, one teacher pass per
training step -- with three changes:
  * the teacher runs a single conditional pass (no CFG branch): its targets are what the turbo sampler follows;
  * the refined image condition is cached once per phase (the context refiner carries the LoRAs too);
  * the teacher's expert is chosen per row from each trajectory's own t.
"""
import torch
from tensordict import TensorDict

from ..models.turbo import PHASE_THRESHOLD
from .rollout import RolloutStream


class TurboRolloutStream(RolloutStream):
    def __init__(self, teacher, encoder, images, n_traj, **kw):
        kw["cfg"] = 0.0
        # the parent computes an unconditional context once; with LoRAs in the context refiner a phase must be set
        teacher.phase_state.set_uniform(True, 1, kw.get("device", "cuda"))
        super().__init__(teacher, encoder, images, n_traj, **kw)

    def _new_image(self, idx):
        if not hasattr(self, "hc_fast"):
            self.hc_fast = torch.zeros_like(self.hc)
        imgs = next(self.images)[: len(idx)].to(self.device, non_blocking=True)
        if len(imgs) < len(idx):
            idx = idx[: len(imgs)]
        feats = self.encoder(imgs)
        self.f1[idx] = feats["feature1"].to(self.dtype)
        self.f2[idx] = feats["feature2"].to(self.dtype)
        with torch.no_grad():
            same = TensorDict({"feature1": self.f1[idx].float(), "feature2": self.f2[idx].float()},
                              batch_size=len(idx))
            st = self.teacher.phase_state
            st.set_uniform(True, len(idx), self.device)
            self.hc[idx] = self.teacher.embed_context(same)
            st.set_uniform(False, len(idx), self.device)
            self.hc_fast[idx] = self.teacher.embed_context(same)
        self.restarts += len(idx)

    @torch.no_grad()
    def targets(self, idx):
        """One guidance-free dense teacher pass: velocity, camera velocity and trunk features."""
        x, t, t01 = self.state(idx)
        slow = (t01 >= PHASE_THRESHOLD).reshape(-1, 1, 1)
        h = torch.where(slow, self.hc[idx], self.hc_fast[idx])
        out = self.teacher(x, t, h_cond=h, return_features=True)      # bound forward: expert per row from t
        tg = {"v_pos": out["latent"], "cam_pos": out["camera"], "h_pos": out["feature"]}
        tg["v_cfg"], tg["cam_cfg"] = tg["v_pos"], tg["cam_pos"]
        tg["t01"], tg["x"] = t01, x
        self.z0[idx] = x["latent"] - t01.reshape(-1, 1, 1) * tg["v_cfg"]
        return tg
