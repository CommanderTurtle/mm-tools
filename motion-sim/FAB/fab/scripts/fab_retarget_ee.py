"""FAB human motion -> Unitree G1 reference motion with the end-effector positions as the primary objective (fab-retarget-ee).

The same solver, stance pin and modes as fab-retarget (scripts/fab_retarget.py), with fab/retarget/release_config.py
solve_cfg_ee(): the hand and foot position terms (EE_W) first, the bone-direction and dense-point terms x EE_SECONDARY.
  - Targets (default): the person's own hands (SMPL hand joints) and soles (each foot's lowest vertices), at their positions in
    the input's world: the G1 reaches and stands where the person did. --ee_targets: given tracks instead (e.g. VIVE trackers).
  - World: with --ee_scale none (default) the G1 world keeps the input's horizontal coordinates (the root path is not scaled
    by the G1 / human scale s; heights are, about z = 0, as in fab-retarget); --ee_scale s: fab-retarget's world (everything
    x s, the targets too).
  - Flat floor only: the solve always stands on a flat floor at the person's ground (the median over frames of the lowest foot
    vertex); a reconstructed scene (heightfield / mesh) in the input is ignored.
  - Hand-floor windows (--hand_windows auto, the default): frames where the lowest hand vertex is within EE_HAND_H of that
    floor, debounced as the stance pin's runs; there the hand's support points are held on the floor by the contact / skate
    terms and the hand's position target is off.
  - Steps 1-2 of fab-retarget only (solve + stance pin; no terrain clean).
-> <out>/<clip>_g1.npz (as fab-retarget's; g1_root_pos in the world above, the floor at s x the person's ground) and
<out>/<clip>_g1.json (inputs, settings, s, the floor height).

    fab-retarget-ee out/fab_world/clip.npz [--out DIR] [--mode points|mesh] [--ee_targets npz] [--ee_scale none|s]
        [--hand_windows auto|off|<json>]
--ee_targets npz: frames (the input's frame ids) and any of left_hand / right_hand / left_foot / right_foot (T, 3) in the input's
world (metres) with <name>_valid (T,) bool; the solver's end-effector points are the midpoint of the G1 hand's thumb and pinky
points and the centre of the four sole points. docs/retarget_ee.md.
"""
import argparse
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

R_ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R_))
sys.path.insert(0, str(R_ / "scripts"))


def g1_scale(J, fps, sigma_in):
    """The G1 / human scale as bench/videomimic/scene_retarget.py computes it: the joints resampled to 50 fps and smoothed
    (--sigma_in), the bone-length ratio's median over frames (float32, on the solver's device)."""
    import torch
    from scipy.ndimage import gaussian_filter1d

    from bench.videomimic.scene_retarget import URDF, resample
    from fab.retarget.g1 import BONES, G1Kinematics

    n = int(np.floor((len(J) - 1) / fps * 50.0)) + 1
    J50 = resample(np.asarray(J, np.float64), n, fps, 50.0)
    if sigma_in > 0:
        J50 = gaussian_filter1d(J50, sigma_in, axis=0, mode="nearest")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kin = G1Kinematics(URDF).to(dev)
    Jt = torch.as_tensor(J50, dtype=torch.float32, device=dev)
    with torch.no_grad():
        hum_len = torch.stack([(Jt[:, a] - Jt[:, b]).norm(dim=-1) for a, b in BONES], -1)
        return float((kin.rest_lengths().sum() / hum_len.sum(-1).clamp_min(1e-3)).median())


def runs(m, fps, gap_s, run_s):
    """The stance pin's debounce: gaps shorter than gap_s filled, runs shorter than run_s dropped -> [(first, last)] indices."""
    m = np.asarray(m, bool).copy()
    idx = np.flatnonzero(m)
    for a_, b_ in zip(idx[:-1], idx[1:]):
        if 1 < b_ - a_ <= int(round(gap_s * fps)):
            m[a_:b_] = True
    out, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            if (j - i + 1) >= int(round(run_s * fps)):
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def ground_z(z, part):
    """The person's ground: the median over the valid frames of the lowest foot vertex (input world, metres)."""
    from fab.retarget.contact import HUMAN_PARTS

    V = np.asarray(z["verts"], np.float64)
    valid = np.asarray(z["valid"], bool) if "valid" in z else np.ones(len(V), bool)
    vi = np.flatnonzero(np.isin(part, HUMAN_PARTS["lfoot"] + HUMAN_PARTS["rfoot"]))
    return float(np.median(V[valid][:, vi, 2].min(1)))


GROUND_SPAN_WARN = 0.25  # m: the person's ground spans more than this -> a warning (slopes / stairs are not supported)


def ground_span(z, part):
    """p95 - p5 of the lowest foot vertex over the feet's contact frames (the input's contact labels; every valid frame without
    them), input world, metres: how far the person's ground is from one level."""
    from fab.retarget.contact import HUMAN_PARTS

    V = np.asarray(z["verts"], np.float64)
    valid = np.asarray(z["valid"], bool) if "valid" in z else np.ones(len(V), bool)
    con = json.loads(str(z["contacts"])) if "contacts" in z else {}
    lows = []
    for g in ("lfoot", "rfoot"):
        low = V[:, np.flatnonzero(np.isin(part, HUMAN_PARTS[g])), 2].min(1)
        lab = np.zeros(len(V), bool)
        if g in con:
            m_ = np.asarray(con[g], bool)[: len(V)]
            lab[: len(m_)] = m_
        else:
            lab[:] = True
        lows.append(low[valid & lab])
    x = np.concatenate(lows)
    return float(np.percentile(x, 95) - np.percentile(x, 5)) if len(x) >= 10 else 0.0


def flat_hfield(J, z0, pad=3.0, res=0.04):
    """A flat heightfield at z0 over the pelvis path +- pad (the solver's {x0, y0, res, height})."""
    lo, hi = J[:, 0, :2].min(0) - pad, J[:, 0, :2].max(0) + pad
    nx, ny = (np.ceil((hi - lo) / res).astype(int) + 1)
    return {"x0": float(lo[0]), "y0": float(lo[1]), "res": float(res), "height": np.full((ny, nx), z0, np.float32)}


def human_ee(z, part):
    """The person's end effectors in the input's world: hands = the SMPL hand joints (22 / 23); feet = each foot's sole, the
    lowest vertex height and the mean xy of the foot vertices within 2 cm of it."""
    from fab.retarget.contact import HUMAN_PARTS

    J, V = np.asarray(z["joints"], np.float64), np.asarray(z["verts"], np.float64)
    out = {"left_hand": J[:, 22], "right_hand": J[:, 23]}
    for key, g in (("left_foot", "lfoot"), ("right_foot", "rfoot")):
        Vf = V[:, np.flatnonzero(np.isin(part, HUMAN_PARTS[g]))]
        zmin = Vf[:, :, 2].min(1)
        near = Vf[:, :, 2] <= zmin[:, None] + 0.02
        xy = (Vf[:, :, :2] * near[..., None]).sum(1) / near.sum(1, keepdims=True)
        out[key] = np.concatenate([xy, zmin[:, None]], 1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("input", help="fab-world's <out>/<clip>.npz (or a recon <clip>.npz; its scene is not used)")
    p.add_argument("--out", default="output/fab_retarget_ee")
    p.add_argument("--mode", default="points", choices=("points", "mesh"), help="as fab-retarget's --mode")
    p.add_argument("--ee_targets", default="", help="end-effector tracks (npz, see above) instead of the person's own")
    p.add_argument("--ee_scale", default="none", choices=("none", "s"),
                   help="none: the input's horizontal coordinates, unscaled (default); s: fab-retarget's world scaled by s")
    p.add_argument("--hand_windows", default="auto", help="auto (the height rule), off, or a json {windows: {left | right: "
                   "[{frames: [first, last]}]}} of the input's frame ids")
    p.add_argument("--w_ee_hand", type=float, default=None, help="the hand position weight (default release_config.EE_W)")
    p.add_argument("--w_ee_foot", type=float, default=None, help="the foot position weight (default release_config.EE_W; 0: hands only)")
    p.add_argument("--secondary", type=float, default=None, help="the factor on the direction / dense-point weights "
                   "(default release_config.EE_SECONDARY; 1: fab-retarget's weights, the end-effector terms added)")
    p.add_argument("--ground_aware", action="store_true", help="opt-in: as fab-retarget's --ground_aware (fab-world's video foot "
                   "contacts; docs/retarget.md)")
    p.add_argument("--no_graph", action="store_true", help="the plain solver / pin loops instead of CUDA graphs")
    a = p.parse_args()
    import fab_retarget as F
    from bench.body_models import SMPL
    from bench.videomimic.scene_retarget import SMPL_PKL
    from fab.retarget import release_config as C

    src = Path(a.input).resolve()
    c = src.stem
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rd = src.parent if "verts" in np.load(src, allow_pickle=True).files else src.parent / "work" / "joint" / "recon"
    z = dict(np.load(rd / f"{c}.npz", allow_pickle=True))
    fps = float(z["fps"])
    J = np.asarray(z["joints"], np.float64)
    valid = np.asarray(z["valid"], bool) if "valid" in z else np.ones(len(J), bool)
    s = g1_scale(J, fps, C.SIGMA_IN)
    part = SMPL(SMPL_PKL).weights.argmax(1).numpy()
    z0 = ground_z(z, part)
    span = ground_span(z, part)
    if span > GROUND_SPAN_WARN:  # docs/retarget_ee.md: one flat floor; slopes / stairs are not supported
        print(f"[fab_retarget_ee] WARNING {c}: the person's ground spans {span:.2f} m (p95 - p5 of the lowest foot vertex in "
              f"contact, > {GROUND_SPAN_WARN} m): not a level floor (a slope, stairs or a tilted world). The motion is placed on one "
              f"flat floor at z = {z0:.3f} m, so the G1 will float or sink where the ground differs; level the input first", flush=True)
    if (rd / f"{c}_hfield.npz").exists() or (rd / f"{c}_scene.ply").exists():
        print(f"[fab_retarget_ee] {c}: flat floor only: the reconstructed scene is ignored; the floor is z = {z0:.3f} m "
              "(the person's ground)", flush=True)
    # the end-effector targets in the solver's world (input world x: none = xy kept, z x s; s = everything x s)
    if a.ee_targets:
        ez = dict(np.load(a.ee_targets, allow_pickle=True))
        tg = {k: np.asarray(ez[k], np.float64) for k in ("left_hand", "right_hand", "left_foot", "right_foot") if k in ez}
        tv = {k: np.asarray(ez[k + "_valid"], bool) for k in tg}
        tf = np.asarray(ez["frames"], int)
        src_t = str(Path(a.ee_targets).resolve())
    else:
        tg = human_ee(z, part)
        tv = {k: valid for k in tg}
        tf = np.asarray(z["frames"], int)
        src_t = "the person's own hands / soles"
    sc = np.array([1.0, 1.0, s]) if a.ee_scale == "none" else np.array([s, s, s])
    # the input for the solver: with --ee_scale none each frame's body moved horizontally so that the solver's x s puts the root
    # path at the person's own xy (body shape and heights unchanged); the flat floor under the moved body
    wk = out / "ee_work"
    rr = wk / "joint" / "recon"
    rr.mkdir(parents=True, exist_ok=True)
    (wk / "ours" / "fab").mkdir(parents=True, exist_ok=True)
    zz = dict(z)
    if a.ee_scale == "none":
        d = J[:, 0, :2] * (1.0 / s - 1.0)  # (T, 2): pelvis xy -> pelvis xy / s
        for k in ("joints", "verts"):
            x = np.asarray(z[k], np.float64).copy()
            x[..., :2] += d[:, None]
            zz[k] = x.astype(np.asarray(z[k]).dtype)
        if "trans" in z:
            x = np.asarray(z["trans"], np.float64).copy()
            x[:, :2] += d
            zz["trans"] = x.astype(np.asarray(z["trans"]).dtype)
    hh = flat_hfield(np.asarray(zz["joints"], np.float64), z0)
    zz["ee_world"] = json.dumps({"ee_scale": a.ee_scale, "s": s, "floor_z": z0, "source": str(rd / f"{c}.npz")})
    np.savez(rr / f"{c}.npz", **zz)
    np.savez(rr / f"{c}_hfield.npz", **hh)
    raw = rd.parent.parent / "ours" / "fab" / f"{c}_raw.npz"
    if raw.exists() and not (wk / "ours" / "fab" / raw.name).exists():
        (wk / "ours" / "fab" / raw.name).symlink_to(raw.resolve())
    # hand-floor windows
    cfg = dict(C.solve_cfg_ee(a.mode, secondary=a.secondary), ee_targets=str(out / f"{c}_ee_targets.npz"))
    for k in ("w_ee_hand", "w_ee_foot"):
        if getattr(a, k) is not None:
            cfg[k] = float(getattr(a, k))
    win = None
    if a.hand_windows == "auto":
        from fab.retarget.contact import HUMAN_PARTS

        V = np.asarray(z["verts"], np.float64)
        win = {}
        for side, g in (("left", "lhand"), ("right", "rhand")):
            Vh = V[:, np.flatnonzero(np.isin(part, HUMAN_PARTS[g]))]
            ht = Vh[:, :, 2].min(1) - z0
            win[side] = [{"frames": [int(z["frames"][i]), int(z["frames"][j])]}
                         for i, j in runs((ht < C.EE_HAND_H) & valid, fps, C.EE_GAP_S, C.EE_RUN_S)]
        json.dump({"windows": win, "rule": f"lowest hand vertex within {C.EE_HAND_H} m of the floor, debounced "
                   f"(gaps < {C.EE_GAP_S} s filled, runs < {C.EE_RUN_S} s dropped)"}, open(out / f"{c}_hand_windows.json", "w"), indent=1)
        cfg["hand_windows"] = str(out / f"{c}_hand_windows.json")
    elif a.hand_windows != "off":
        cfg["hand_windows"] = str(Path(a.hand_windows).resolve())
        win = json.loads(Path(a.hand_windows).read_text())["windows"]
    if cfg.get("hand_windows"):
        cfg["contact_groups"] = cfg["contact_groups"] + ",lhand,rhand"
        # inside a hand's floor windows the contact / skate terms hold it on the floor: its position target is off there
        for side in ("left", "right"):
            k = side + "_hand"
            if k in tv:
                for w in (win or {}).get(side, []):
                    tv[k] = tv[k] & ~((tf >= w["frames"][0]) & (tf <= w["frames"][1]))
    np.savez(out / f"{c}_ee_targets.npz", frames=tf, **{k: v * sc for k, v in tg.items()}, **{k + "_valid": v for k, v in tv.items()})
    a2 = Namespace(input=str(rr / f"{c}.npz"), mode=a.mode, no_clean=True, no_graph=a.no_graph, release=None,
                   ground_aware=a.ground_aware, ground_work=str(rd.parent.parent))
    f, info = F.scene(a2, c, out, extra_cfg=cfg)
    assert f.exists(), f"no output {f}"
    shutil.copy(f, out / f"{c}_g1.npz")
    json.dump({"input": str(src), "output": str(f), "retargeter": "fab-retarget-ee", "ee_scale": a.ee_scale, "s": s,
               "world": ("the input's horizontal coordinates (root path unscaled), heights x s about z = 0" if a.ee_scale == "none"
                         else "fab-retarget's G1 world (the input x s)"), "floor_z_input": z0, "floor_z_g1": z0 * s,
               "ee_targets": src_t, "hand_windows": win, **info}, open(out / f"{c}_g1.json", "w"), indent=1, default=str)
    print(f"[fab_retarget_ee] {c}: s {s:.4f}, ee_scale {a.ee_scale}, hand windows "
          f"{ {k: len(v) for k, v in (win or {}).items()} } -> {out / f'{c}_g1.npz'}", flush=True)


if __name__ == "__main__":
    main()
