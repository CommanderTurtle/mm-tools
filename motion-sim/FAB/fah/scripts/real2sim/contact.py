"""Hand-object contact refinement for the physics replay: a smooth per-frame translation of the hand (world frame) so
that, while the object is being manipulated (it has left its start pose), each MANO part that is near the object touches
it with a small squeeze, and no vertex sinks deep into it; outside the manipulation window only penetration counts. shape=True also fits a per-axis scale of the object
(object frame, prior around 1): single-view reconstructions guess the unseen depth, and the grasp measures it.
Signed distance = union of the object's convex parts (the same CoACD hulls the simulator collides with):
sd(x) = min over hulls of max over facets (n . x + b), negative inside.

    from scripts.real2sim.contact import refine
    delta, axis_scale, info = refine(Vw, valid, Tw, hull_paths, labels)   # Vw + delta[:, None] = refined hand vertices
"""
import numpy as np
import torch
import trimesh
from scipy.spatial import ConvexHull


def hull_planes(paths):
    eq = [ConvexHull(np.asarray(trimesh.load(p).vertices)).equations for p in paths]
    m = max(len(e) for e in eq)
    out = np.full((len(eq), m, 4), [0, 0, 0, -1.0])  # padding facets never win the max
    for k, e in enumerate(eq):
        out[k, :len(e)] = e
    return out


def signed_distance(X, planes):
    """X (..., 3) in the object frame, planes (H, F, 4) -> (...,)"""
    s = torch.einsum("...d,hfd->...hf", X, planes[..., :3]) + planes[..., 3]
    return s.max(-1).values.min(-1).values


def manipulation_window(Tw, move=0.02, lead=6):
    d = np.linalg.norm(Tw[:, :3, 3] - Tw[0, :3, 3], axis=1)
    on = d > move
    if not on.any():
        return on
    first = int(np.argmax(on))
    on[max(first - lead, 0):] = True
    return on


def refine(Vw, valid, Tw, hulls, labels, squeeze=0.003, near=0.03, pen_allow=0.006, sigma=0.03, smooth=0.004, steps=600,
           shape=False, shape_sigma=0.2, chunk=16, device="cuda"):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    P = torch.tensor(hull_planes(hulls), dtype=torch.float32, device=dev)
    sub = np.arange(0, Vw.shape[1], 2)  # every 2nd MANO vertex (memory: frames x vertices x hulls x facets)
    V = torch.tensor(Vw[:, sub], dtype=torch.float32, device=dev)
    labels = np.asarray(labels)[sub]
    R = torch.tensor(Tw[:, :3, :3], dtype=torch.float32, device=dev)
    t = torch.tensor(Tw[:, :3, 3], dtype=torch.float32, device=dev)
    ok = torch.tensor(valid, device=dev)
    win = torch.tensor(manipulation_window(Tw), device=dev) & ok
    lab = torch.tensor(labels, device=dev)
    onehot = torch.nn.functional.one_hot(lab, 16).bool()  # (778, 16)
    delta = torch.zeros(len(V), 3, device=dev, requires_grad=True)
    ls = torch.zeros(3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([dict(params=[delta], lr=1e-3)] + ([dict(params=[ls], lr=5e-3)] if shape else []))

    def sd_of(dl, ls_=None):
        out = []
        for a in range(0, len(V), chunk):  # frames in chunks (memory)
            X = torch.einsum("nvd,nde->nve", V[a:a + chunk] + dl[a:a + chunk, None] - t[a:a + chunk, None], R[a:a + chunk])
            if ls_ is not None:
                X = X / torch.exp(ls_.clamp(-0.4, 0.4))  # the object scaled per axis about its frame origin
            out.append(signed_distance(X, P))
        return torch.cat(out)  # (n, vertices)

    with torch.no_grad():
        sd0 = sd_of(torch.zeros_like(delta))
        part0 = torch.where(onehot[None], sd0[..., None], torch.full_like(sd0[..., None], 9.0)).min(1).values  # (n, 16)
        act = (part0 < near) & win[:, None]  # parts expected to touch
    n_act = act.sum().clamp_min(1)
    for _ in range(steps):  # frames in chunks (memory: frames x vertices x hulls x facets), gradients accumulated
        opt.zero_grad()
        for a in range(0, len(V), chunk):
            sl = slice(a, a + chunk)
            X = torch.einsum("nvd,nde->nve", V[sl] + delta[sl, None] - t[sl, None], R[sl])
            if shape:
                X = X / torch.exp(ls.clamp(-0.4, 0.4))
            sd = signed_distance(X, P)
            part = torch.where(onehot[None], sd[..., None], torch.full_like(sd[..., None], 9.0)).min(1).values
            e_att = ((part + squeeze) ** 2 * act[sl]).sum() / n_act
            e_pen = ((torch.relu(-sd - pen_allow) ** 2) * ok[sl, None]).sum(1).sum() / len(V)
            (e_att + 10 * e_pen).backward()
        e_reg = ((delta / sigma) ** 2).sum(1).mean() * 1e-4
        e_smo = (((delta[1:] - delta[:-1]) / smooth) ** 2).sum(1).mean() * 1e-4
        (e_reg + e_smo + (((ls / shape_sigma) ** 2).sum() * 1e-4 if shape else 0)).backward()
        opt.step()
    with torch.no_grad():
        sd = sd_of(delta, ls if shape else None)
        part = torch.where(onehot[None], sd[..., None], torch.full_like(sd[..., None], 9.0)).min(1).values
        dn = delta.norm(dim=1)
    info = dict(window_frames=int(win.sum()), parts_active_median=float(act.sum(1)[win].float().median()) if win.any() else 0.0,
                delta_mm_median=round(float(dn[win].median()) * 1e3, 1) if win.any() else 0.0,
                delta_mm_max=round(float(dn.max()) * 1e3, 1),
                part_gap_mm_before=round(float(part0[act].abs().median()) * 1e3, 1) if act.any() else None,
                part_gap_mm_after=round(float((part[act] + squeeze).abs().median()) * 1e3, 1) if act.any() else None,
                max_pen_mm_before=round(float((-sd0[ok]).max()) * 1e3, 1), max_pen_mm_after=round(float((-sd[ok]).max()) * 1e3, 1))
    axis_scale = torch.exp(ls.clamp(-0.4, 0.4)).detach().cpu().numpy().astype(np.float64)
    info["axis_scale"] = np.round(axis_scale, 3).tolist()
    return delta.detach().cpu().numpy().astype(np.float64), axis_scale, info
