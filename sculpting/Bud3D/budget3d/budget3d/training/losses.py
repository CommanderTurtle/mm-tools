"""The distillation objective.

Three terms, all against the frozen teacher's dense prediction at the same state:
the velocity the decoders ultimately read, the camera token that fixes the input view, and
the trunk features of the last joint block, which say *how* the teacher represented the
object rather than only what it predicted.  Points that a merged token had to paint are
weighted up, because those are the predictions the refiner is responsible for.
"""
import torch.nn.functional as F


def distill_loss(out, targets, point_scale, coarse_boost=0.5, cam_weight=0.2, feat_weight=0.2,
                 feat_cos_weight=0.15, branch="pos"):
    v_t = targets[f"v_{branch}"]
    cam_t = targets[f"cam_{branch}"]
    h_t = targets[f"h_{branch}"]

    w = 1.0 + coarse_boost * (point_scale > 1).to(v_t.dtype)
    w = w / w.mean(dim=1, keepdim=True)                       # per example, so the scale is budget-free
    l_vel = (w * (out["latent"].float() - v_t.float()).pow(2).mean(-1)).mean()
    l_cam = (out["camera"].float() - cam_t.float()).pow(2).mean()

    h_s, h_t = out["feature"].float(), h_t.float()
    rms = h_t.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(0.1)
    l_nmse = ((h_s - h_t) / rms).pow(2).mean()
    l_cos = (1 - F.cosine_similarity(h_s, h_t, dim=-1)).mean()

    total = l_vel + cam_weight * l_cam + feat_weight * (l_nmse + feat_cos_weight * l_cos)
    return total, {"vel": l_vel.detach(), "cam": l_cam.detach(), "feat_nmse": l_nmse.detach(),
                   "feat_cos": l_cos.detach(), "total": total.detach()}
