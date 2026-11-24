"""Robot-side contact check (a physical-plausibility audit of G1 motions): is the G1 sole on the terrain when the human
contact head says the foot is in contact? G1 motions:
  references  the retargeter's <run>/<ref>/<clip>_motion.npz (g1_root_pos, g1_root_quat_xyzw, g1_dof (29, g1_joint_names),
              frames = the recon row per motion frame, fps 50, contacts = the labels the solve used) with its terrain
              heightfield (<clip>_terrain_g1.npz: x0, y0, res, height (ny, nx), rows along y)
  sim         tracker rollouts <run>/run_terrain/<ref>__<clip>/raw/<tracker>/P0/trajectories/<tracker>/seed_0/
              *.npz (robot_qpos (T, 36): root xyz + wxyz + the 29 dof in <run>/view/view/manifest.json's order;
              motion_t = the reference frame tracked per sample), on the sim's terrain (<run>/view/view/terrain/
              <ref>__<clip>.npz), up to the fall time (stoop_board.json)
  port        policy rollouts <run>/port_<tag>[_limB]/traj/<clip>_motion.npz (VideoMimic's terrain
              policy; qpos + qpos_names, sample k = reference frame k) on the reference's terrain_g1, from
              0.5 s (respawn drop) to results.json's hard_fall_time_s
against the clip's contact file (<contacts>/<clip>__0.contact.npz, keyed by the recon's source frame
ids; window suffixes _w06 / _w20 / _full stripped; video head first). Recon rows -> source frames through the recon npz
(<run>/recon/<clip>.npz or --recon_dirs) when it has "frames", else the identity.
  G1 sole contact   per foot: the lowest of its 5 sole points (4 contact spheres + the toe, URDF, r 5 mm) within --h_thr of the terrain
                    below it; "planted" additionally the lowest sphere moving < --v_thr m/s
  human contact     per foot: heel or toe static (planted) probability >= 0.5 (and any-contact for the "any" rows)
  scores            per foot and frame, the head as reference: P / R / F1 of G1 planted vs head static and of G1 contact
                    vs head any; contact_on_static = G1 sole down on head-static frames, speed_on_static = the sole's
                    speed there (median / p90, m/s); onset / offset timing = the median signed lag (ms) of each head
                    static onset / offset to the nearest G1 planted one within 0.5 s (+ = the robot later); sole
                    penetration (p95 depth below the terrain); stance drift = per head-static run >= 0.2 s, the max
                    distance (cm) of the nearest sole sphere from the run's onset contact point (roll-compensated;
                    slides and early lift-off count); with the motion's own labels: G1 planted
                    vs labels (the solve's / tracker's fidelity) and labels vs head static (the input's agreement)
    python bench/g1_contact_check.py --motions "<runs>/*/<tag>/*_motion.npz" \
        --sims "<runs>/*/run_terrain/*/raw/*/P0/trajectories/*/seed_0/*.npz" --ports "<runs>/*/port_*/traj/*.npz" --out <json>
One definition for other code: score_motion(motion_npz, terrain_npz, contact_npz, recon, qpos, ...) -> dict.
Also stance_runs (the stance pin's stance definition) and strict_gate (the strict MuJoCo validity gate bench/stance_pin.py
scores with).
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fab.retarget.g1 import G1Kinematics  # noqa: E402

from fab.retarget import g1_files  # noqa: E402

URDF = g1_files.urdf()
SOLE = {s: [f"{s}_sole_{i}" for i in range(1, 5)] + [f"{s}_toe"] for s in ("left", "right")}  # scripts/build_g1.py
R_SPH = 0.005
WINDOWS = ("_w06", "_w20", "_full")
PAIRS = ("planted_vs_static", "contact_vs_any", "planted_vs_labels", "labels_vs_static")


def quat_xyzw_to_R(q):
    x, y, z, w = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


def terrain_h(t, xy):
    """Bilinear terrain height at world xy (N, 2); rows along y, columns along x (the retargeter's heightfield)."""
    H = t["height"].astype(np.float64)
    c = (xy[:, 0] - float(t["x0"])) / float(t["res"])
    r = (xy[:, 1] - float(t["y0"])) / float(t["res"])
    c = np.clip(c, 0, H.shape[1] - 1.001)
    r = np.clip(r, 0, H.shape[0] - 1.001)
    c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
    fc, fr = c - c0, r - r0
    return (H[r0, c0] * (1 - fc) * (1 - fr) + H[r0, c0 + 1] * fc * (1 - fr) + H[r0 + 1, c0] * (1 - fc) * fr
            + H[r0 + 1, c0 + 1] * fc * fr)


def prf(p, g):
    tp, fp, fn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
    return dict(P=round(tp / max(tp + fp, 1), 3), R=round(tp / max(tp + fn, 1), 3), F1=round(2 * tp / max(2 * tp + fp + fn, 1), 3),
                n=int(len(p)), rate_g1=round(float(p.mean()), 3) if len(p) else None,
                rate_ref=round(float(g.mean()), 3) if len(g) else None, tp=tp, fp=fp, fn=fn)


def lags(p, g, fps, win=0.5):
    """Median signed lag (ms) of g's onsets / offsets to the nearest p ones within win s (+ = p later)."""
    out = {}
    for nm, d in (("onset", 1), ("offset", -1)):
        eg = np.flatnonzero(np.diff(g.astype(int)) == d) + 1
        ep = np.flatnonzero(np.diff(p.astype(int)) == d) + 1
        ls = []
        for e in eg:
            if len(ep):
                k = ep[np.argmin(np.abs(ep - e))]
                if abs(k - e) <= win * fps:
                    ls.append((k - e) / fps * 1000)
        out[f"{nm}_lag_ms"] = round(float(np.median(ls)), 1) if ls else None
        out[f"{nm}_matched"] = f"{len(ls)}/{len(eg)}"
    return out


def soles(kin, root_pos, quat_xyzw, dof, names):
    """-> {side: (T, 5, 3)} sole sphere centres, world (z up)."""
    order = [names.index(j) for j in kin.revolute]  # the motion's dof order -> the URDF's
    q = torch.as_tensor(np.asarray(dof)[:, order], dtype=torch.float32)
    R = torch.as_tensor(quat_xyzw_to_R(np.asarray(quat_xyzw, np.float64)), dtype=torch.float32)
    with torch.no_grad():
        pos, _ = kin(R, torch.as_tensor(np.asarray(root_pos), dtype=torch.float32), q)
    return {s: np.stack([pos[n].numpy().astype(np.float64) for n in SOLE[s]], 1) for s in SOLE}


def base_clip(clip):
    for suf in WINDOWS:
        clip = clip[: -len(suf)] if clip.endswith(suf) else clip
    return clip


def recon_frames(clip, run_dir, a):
    """The recon's source frame id per row (None: identity)."""
    for d in [Path(run_dir) / "recon"] + [Path(x) for x in a.recon_dirs.split(",") if x]:
        f = d / f"{clip}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True)
            return np.asarray(z["frames"], int) if "frames" in z.files else None
    return None


def category(run_dir, clip):
    m = Path(run_dir) / "meta.json"
    if m.exists():
        return json.loads(m.read_text()).get(clip, {}).get("category", "-")
    return "lab" if clip.startswith("lab") else "-"


def score(S, t, rows, labels, c, fps, a, meta):
    """S {side: (T, 5, 3)} sole points at motion frames whose recon rows are rows (T,); t terrain; labels {lfoot, rfoot}
    (T,) the motion's own contact labels (or {}); c the contact file."""
    head = next(h for h in ("video_", "image_") if f"{h}static_prob" in c.files)
    fr = meta.pop("recon_frames")
    src = rows if fr is None else fr[np.clip(rows, 0, len(fr) - 1)]
    row = {int(f): i for i, f in enumerate(c["frame"])}
    keep = np.array([int(f) in row for f in src])
    hi = np.array([row[int(f)] for f in src[keep]], int)
    res = dict(meta, frames=int(keep.sum()), head=head.rstrip("_"), frames_mapped=fr is not None)
    if keep.sum() < 2:
        return None
    pen = []
    for k, (s, g) in enumerate((("left", "lfoot"), ("right", "rfoot"))):
        P = S[s]
        hgt = P[..., 2] - R_SPH - terrain_h(t, P[..., :2].reshape(-1, 2)).reshape(P.shape[:2])
        low, am = hgt.min(1), hgt.argmin(1)
        sp = np.linalg.norm(np.gradient(P, axis=0), axis=2)[np.arange(len(P)), am] * fps  # the lowest sphere's speed
        g1_c = low < a.h_thr
        g1_p = g1_c & (sp < a.v_thr)
        st = c[f"{head}static_prob"][hi][:, 2 * k: 2 * k + 2].max(1) >= 0.5
        an = c[f"{head}any_prob"][hi][:, 2 * k: 2 * k + 2].max(1) >= 0.5
        g1_c, g1_p, low, sp = g1_c[keep], g1_p[keep], low[keep], sp[keep]
        on = sp[st & g1_c]
        d = dict(planted_vs_static=prf(g1_p, st), contact_vs_any=prf(g1_c, an), n_static=int(st.sum()),
                 n_contact_on_static=int((g1_c & st).sum()),
                 speed_on_static_m_s=[round(float(np.median(on)), 3), round(float(np.quantile(on, 0.9)), 3)] if len(on) else None,
                 **lags(g1_p, st, fps))
        if g in labels:
            lb = np.asarray(labels[g], bool)[keep]
            d.update(planted_vs_labels=prf(g1_p, lb), labels_vs_static=prf(lb, st))
        d["stance_drift_cm"] = stance_drift(P[keep], am[keep], st, getattr(a, "min_run", 10))
        res[s] = d
        pen.append(np.clip(-low, 0, None))
    res["sole_penetration_mm_p95"] = round(float(np.quantile(np.concatenate(pen), 0.95) * 1000), 1)
    return res


def stance_drift(P, am, st, min_run=10):
    """Stance drift (cm) per head-static run of >= min_run frames: the largest distance, over the run, from the run's
    onset contact point (the lowest sphere at its first frame) to the nearest of the 5 sole points. A foot rolling
    heel -> toe keeps a sphere near the onset point; sliding or lifting off moves all of them away ("the sole stays on its
    onset dot"). P (T, 5, 3), am (T,) lowest sphere, st (T,) head static
    -> [drift per run]."""
    out = []
    e = np.flatnonzero(np.diff(np.r_[0, st.astype(int), 0]))
    for b, f in zip(e[::2], e[1::2]):
        if f - b < min_run:
            continue
        piv = P[b, am[b]]
        out.append(round(float(np.linalg.norm(P[b:f] - piv, axis=2).min(1).max() * 100), 2))
    return out


def stance_runs(contact, rows=None, recon=None, fps=50.0, on=0.6, off=0.4, min_run=0.2, fill_gap=0.1, head="video"):
    """Stance runs from the contact head, one definition for the retargeter's solver_stance mask (the stance pin).
    contact: <clip>__0.contact.npz (path or loaded); rows: the recon row per output frame (a motion's
    "frames"; None = the contact file's own frames), recon: that recon's "frames" array / npz path (None = identity).
    Per foot, p = max(heel, toe) static probability; hysteresis (stance starts at p >= on, ends at p < off); gaps shorter
    than fill_gap s inside a stance are filled, runs shorter than min_run s dropped. Pin per stance frame: 1 heel / 2 toe /
    3 both (heel / toe p >= 0.5), else the more likely point; a 5-frame median removes flicker.
    -> {"left" / "right": dict(stance (T,) bool, pin (T,) uint8 0 none / 1 heel / 2 toe / 3 both, run (T,) int, -1 none)}"""
    from scipy.ndimage import median_filter

    c = np.load(contact) if isinstance(contact, (str, Path)) else contact
    key = f"{head}_static_prob" if f"{head}_static_prob" in c.files else "image_static_prob"
    prob, fr_c = c[key].astype(np.float64), c["frame"].astype(int)
    if rows is None:
        src = fr_c
    else:
        fr = recon
        if isinstance(recon, (str, Path)):
            z = np.load(recon, allow_pickle=True)
            fr = np.asarray(z["frames"], int) if "frames" in z.files else None
        rows = np.asarray(rows, int)
        src = rows if fr is None else fr[np.clip(rows, 0, len(fr) - 1)]
    idx = {int(f): i for i, f in enumerate(fr_c)}
    have = np.array([int(f) in idx for f in src])
    P = np.zeros((len(src), 4))
    P[have] = prob[[idx[int(f)] for f in src[have]]]
    gap, mrun = int(round(fill_gap * fps)), int(round(min_run * fps))
    out = {}
    for k, side in enumerate(("left", "right")):
        ph, pt = P[:, 2 * k], P[:, 2 * k + 1]
        p = np.maximum(ph, pt)
        st = np.zeros(len(p), bool)
        cur = False
        for t in range(len(p)):
            cur = p[t] >= on if not cur else p[t] >= off
            st[t] = cur
        e = np.flatnonzero(np.diff(np.r_[0, st.astype(int), 0]))
        b_, f_ = e[::2], e[1::2]
        for j in range(len(b_) - 1):  # fill short gaps between runs
            if b_[j + 1] - f_[j] < gap:
                st[f_[j]: b_[j + 1]] = True
        e = np.flatnonzero(np.diff(np.r_[0, st.astype(int), 0]))
        run = np.full(len(p), -1, int)
        nr = 0
        for b, f in zip(e[::2], e[1::2]):
            if f - b < mrun:
                st[b:f] = False
                continue
            run[b:f] = nr
            nr += 1
        pin = np.where(ph >= 0.5, 1, 0) + np.where(pt >= 0.5, 2, 0)
        pin = np.where(pin == 0, np.where(ph >= pt, 1, 2), pin)
        pin = median_filter(pin, size=5, mode="nearest").astype(np.uint8)
        pin[~st] = 0
        out[side] = dict(stance=st, pin=pin, run=run)
    return out


GATE = dict(sole_speed_p99=0.10, sole_pen_cm=1.2, body_pen_cm=0.5, root_jump_cm=5.0)
MJCF_G1 = str(g1_files.mjcf())


def mj_gate(root_pos, quat_xyzw, dof, names, terrain, stance, fps=50.0, mjcf=MJCF_G1, thr=None):
    """A MuJoCo gate on the G1 MJCF (Model 15 collision geoms) on the clip's terrain as a MuJoCo hfield (rows along y,
    the terrain_g1 grid), kinematic
    replay at the motion rate (mj_forward per frame; MuJoCo contacts with the hfield, not mj_geomDistance):
      sole_speed_p99  p99 over the stance frames (stance (T, 2) left / right) of that foot's lowest sole sphere speed (m/s)
      sole_pen_cm     the deepest contact of a sole geom (an ankle-roll link geom: its contact spheres) into the hfield
      body_pen_cm     the deepest contact of any other robot geom
      root_jump_cm    the largest frame-to-frame root displacement
    -> dict with the values, their frame counts and pass flags against thr (default GATE)."""
    import mujoco

    thr = dict(GATE, **(thr or {}))
    t = np.load(terrain) if isinstance(terrain, (str, Path)) else terrain
    H = t["height"].astype(np.float64)
    ny, nx = H.shape
    res, x0, y0 = float(t["res"]), float(t["x0"]), float(t["y0"])
    zmin, span = float(H.min()), max(float(H.max() - H.min()), 1e-3)
    sp = mujoco.MjSpec.from_file(mjcf)
    for g in sp.geoms:
        if g.name in ("ground", "floor"):
            g.contype, g.conaffinity = 0, 0
    rx, ry = (nx - 1) * res / 2, (ny - 1) * res / 2
    sp.add_hfield(name="terrain", size=[rx, ry, span, 0.1], nrow=ny, ncol=nx,
                  userdata=((H - zmin) / span).astype(np.float32).ravel().tolist())
    sp.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain",
                          pos=[x0 + rx, y0 + ry, zmin], contype=1, conaffinity=1)
    m = sp.compile()
    d = mujoco.MjData(m)
    tg = m.geom("terrain").id
    bname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""  # noqa: E731
    sole = {g for g in range(m.ngeom) if "ankle_roll" in bname(g)}
    free = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    qa = m.jnt_qposadr[free]
    adr = [m.jnt_qposadr[m.joint(n).id] for n in names]
    sph = {s: [m.body(n).id for n in SOLE[s]] for s in SOLE}
    T = len(root_pos)
    sole_pen, body_pen = np.zeros(T), np.zeros(T)
    P = {s: np.zeros((T, 5, 3)) for s in SOLE}
    for k in range(T):
        d.qpos[qa: qa + 3] = root_pos[k]
        d.qpos[qa + 3: qa + 7] = np.asarray(quat_xyzw[k])[[3, 0, 1, 2]]
        d.qpos[adr] = dof[k]
        mujoco.mj_forward(m, d)
        for c in d.contact[: d.ncon]:
            if tg not in (c.geom1, c.geom2) or c.dist >= 0:
                continue
            o = c.geom2 if c.geom1 == tg else c.geom1
            if o in sole:
                sole_pen[k] = max(sole_pen[k], -c.dist)
            else:
                body_pen[k] = max(body_pen[k], -c.dist)
        for s in SOLE:
            P[s][k] = d.xpos[sph[s]]
    spd = []
    for j, s in enumerate(("left", "right")):
        h = P[s][..., 2] - R_SPH - terrain_h(t, P[s][..., :2].reshape(-1, 2)).reshape(T, 5)
        am = h.argmin(1)
        v = np.linalg.norm(np.gradient(P[s], axis=0), axis=2)[np.arange(T), am] * fps
        spd.append(v[np.asarray(stance)[:, j].astype(bool)])
    spd = np.concatenate(spd)
    rj = np.linalg.norm(np.diff(np.asarray(root_pos, np.float64), axis=0), axis=1)
    out = dict(sole_speed_p99=round(float(np.quantile(spd, 0.99)), 3) if len(spd) else None, stance_frames=int(len(spd)),
               sole_pen_cm=round(float(sole_pen.max() * 100), 2), sole_pen_frames_gt1cm=int((sole_pen > 0.01).sum()),
               body_pen_cm=round(float(body_pen.max() * 100), 2), body_pen_frames_gt5mm=int((body_pen > 0.005).sum()),
               root_jump_cm=round(float(rj.max() * 100), 2) if len(rj) else 0.0, frames=int(T))
    out["pass"] = {k: (out[k] is not None and out[k] <= v) for k, v in thr.items()}
    out["pass_all"] = all(out["pass"].values())
    return out


GATE_SCENE = str(g1_files.gate_scene())  # FAB's gate scene (scripts/build_g1.py: the collision-capsule G1 on a floor plane)
GATE_V1 = dict(slide=0.10, root_step=0.05, feet_pen=0.012, body_pen=0.005)  # the gate's thresholds (m, m/s)


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def strict_gate(clip_dir, stance=None, base_xml=GATE_SCENE, every=2, ref="gate_ref", contacts="contacts.npz", raycheck=0,
            terrain="terrain/terrain_hfield_2cm.npz", allow=None, allow_tol=0.02):
    """A strict MuJoCo validity gate for a whole-body-tracking training slot: slide, root step, feet / body penetration.
    The clip (refs/<ref>.npz joint_pos at 50 fps, terrain/
    terrain_hfield_2cm.npz) turned onto the route's main axis and moved into the scene slot (x -2.5..11.5, y -2..2, the
    start at x = 0, the lowest terrain at z = 0), the terrain resampled on a 2 cm grid (nearest; outside: the lowest) as
    the hfield "terrain_grid" of the gate scene (its floor plane kept). Stance: `stance` (T, 2) if given, else
    contacts.npz solver_stance, else video_static heel | toe.
      slide      p99, both feet pooled, of |sole(t) - sole(t-1)| x 50 over frames in stance at t - 1 and t; the sole point
                 = the foot's collision footprint centre at its lowest point, fixed in the ankle-roll link frame
      root_step  the largest |root xyz(t) - root xyz(t-1)|
      feet_pen   max over every `every`-th frame of the depth (-dist) of MuJoCo contacts between an ankle-roll-link geom
                 and the hfield, or mj_geomDistance (< 0) against the other scene geoms (the plane)
      body_pen   the same for every other robot collision geom
    `ref` / `contacts`: the reference under refs/ (e.g. the stance-pinned refs) and the contacts file; `terrain`: the
    hfield file (e.g. terrain/terrain_hfield_2cm_clean.npz).
    The hfield rows are written +y first (MJCF elevation order; writing y = -2 m first mirrors the terrain). raycheck = n: n mj_rayHfield casts vs the
    grid -> ray_p90_mm / ray_max_mm.
    allow = {robot body name: (T,) bool} (contact-aware rule): a labelled support contact
    (a knee / hand / seat on the surface it rests on) may sink up to allow_tol into a support-like surface (contact normal
    within ~45 deg of vertical) on the frames its label is on; everything else stays strict. -> out["contact_aware"]
    (body_pen of the rest, body_pen_allowed, pass_all with that body_pen); the plain values are unchanged.
    -> dict(values (m, m/s), pass flags against GATE_V1, pass_all)."""
    import tempfile
    import xml.etree.ElementTree as ET

    import mujoco

    d = Path(clip_dir)
    mo = np.load(d / f"refs/{ref}.npz")
    q = np.asarray(mo["joint_pos"], np.float64).copy()
    T = len(q)
    # the scene slot
    X0, X1, Y0, Y1, RES = -2.5, 11.5, -2.0, 2.0, 0.02
    xy = q[:, :2]
    c = xy.mean(0)
    _, V = np.linalg.eigh((xy - c).T @ (xy - c))
    ax = V[:, -1]
    if (xy[-1] - xy[0]) @ ax < 0:
        ax = -ax
    th = float(np.arctan2(ax[1], ax[0]))
    Rm = np.array([[np.cos(-th), -np.sin(-th)], [np.sin(-th), np.cos(-th)]])
    u = (xy - c) @ Rm.T
    shift = np.array([-u[0, 0], -u[:, 1].mean()])
    uv = u + shift
    t = np.load(d / terrain)
    H, gx0, gy0, gres = np.asarray(t["height"], float), float(t["x0"]), float(t["y0"]), float(t["res"])
    xs, ys = np.arange(X0, X1 + RES / 2, RES), np.arange(Y0, Y1 + RES / 2, RES)
    XX, YY = np.meshgrid(xs, ys)
    P = np.stack([XX.ravel() - shift[0], YY.ravel() - shift[1]], 1) @ Rm + c
    ix, iy = np.round((P[:, 0] - gx0) / gres).astype(int), np.round((P[:, 1] - gy0) / gres).astype(int)
    inside = (ix >= 0) & (ix < H.shape[1]) & (iy >= 0) & (iy < H.shape[0])
    z = np.full(len(P), np.nan)
    z[inside] = H[iy[inside], ix[inside]]
    f0 = float(np.nanmin(z))
    z = np.maximum(np.where(np.isnan(z), f0, z).reshape(XX.shape) - f0, 0.0)
    z = np.round(z * 1000.0) / 1000.0  # the grid's int16 mm encoding (scene.encode_grid)
    q[:, :2] = uv
    q[:, 2] -= f0
    q[:, 3:7] = _qmul(np.array([np.cos(-th / 2), 0.0, 0.0, np.sin(-th / 2)]), q[:, 3:7])
    # the scene MJCF (scene.Scene / GridScene.to_mjcf)
    tree = ET.parse(base_xml)
    root = tree.getroot()
    bdir = Path(base_xml).parent
    for inc in root.iter("include"):
        inc.set("file", str((bdir / inc.get("file")).resolve()))
    for comp in root.iter("compiler"):
        if comp.get("meshdir"):
            comp.set("meshdir", str((bdir / comp.get("meshdir")).resolve()) + "/")
    ny, nx = z.shape
    top = max(float(z.max()), 0.01)
    sx, sy = (nx - 1) * RES / 2, (ny - 1) * RES / 2
    root.find("asset").append(ET.Element("hfield", {"name": "terrain_grid", "nrow": str(ny), "ncol": str(nx),
                                                    "size": f"{sx:.4f} {sy:.4f} {top:.4f} 0.1",
                                                    "elevation": " ".join(f"{v:.4f}" for v in np.clip(z[::-1] / top, 0.0, 1.0).ravel())}))
    root.find("worldbody").append(ET.Element("geom", {"name": "terrain_grid", "type": "hfield", "hfield": "terrain_grid",
                                                      "pos": f"{X0 + sx:.4f} {Y0 + sy:.4f} 0.0000", "friction": "0.8 0.005 0.0001",
                                                      "solref": "0.01 1"}))
    with tempfile.TemporaryDirectory() as td:
        xp = Path(td) / "scene.xml"
        tree.write(xp)
        m = mujoco.MjModel.from_xml_path(str(xp))
    dd = mujoco.MjData(m)
    ray = {}
    if raycheck:  # the MuJoCo surface vs the grid z[i, j] at (xs[j], ys[i]), away from the border
        mujoco.mj_forward(m, dd)
        rng = np.random.default_rng(0)
        ii, jj = rng.integers(2, ny - 2, raycheck), rng.integers(2, nx - 2, raycheck)
        hg = m.geom("terrain_grid").id
        e = [top + 1.0 - mujoco.mj_rayHfield(m, dd, hg, np.array([xs[j], ys[i], top + 1.0]), np.array([0.0, 0.0, -1.0])) - z[i, j]
             for i, j in zip(ii, jj)]
        e = np.abs(np.array(e)) * 1000
        ray = dict(ray_p90_mm=round(float(np.quantile(e, 0.9)), 2), ray_max_mm=round(float(e.max()), 2))
    feet_b = [m.body(n).id for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    # the sole point (kin.G1Kin._sole_point): standing pose, the lowest collision points of the foot's geoms
    dd.qpos[:] = m.qpos0
    dd.qpos[2] = 1.0
    mujoco.mj_kinematics(m, dd)
    sole = []
    for b in feet_b:
        R_, p_ = dd.xmat[b].reshape(3, 3), dd.xpos[b]
        pts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
                continue
            gt, size = m.geom_type[g], m.geom_size[g]
            down = size[0] if gt == mujoco.mjtGeom.mjGEOM_SPHERE else (size[2] if gt == mujoco.mjtGeom.mjGEOM_BOX else size[0])
            pts.append(dd.geom_xpos[g] - np.array([0.0, 0.0, down]))
        pts = np.array(pts) if pts else np.array([p_ + R_ @ np.array([0.04, 0.0, -0.035])])
        foot = pts[np.abs(pts[:, 2] - pts[:, 2].min()) < 0.01].mean(0)
        sole.append(R_.T @ (foot - p_))
    if stance is None:
        ct = np.load(d / contacts)
        if "solver_stance" in ct.files:
            stance = np.asarray(ct["solver_stance"], bool)[:T]
        else:
            vs = np.asarray(ct["video_static"], bool)[:T]
            stance = np.stack([vs[:, 0] | vs[:, 1], vs[:, 2] | vs[:, 3]], 1)
    stance = np.asarray(stance, bool)
    hf = m.geom("terrain_grid").id
    robot = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and (m.geom_contype[g] or m.geom_conaffinity[g])]
    scene = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0 and (m.geom_contype[g] or m.geom_conaffinity[g]) and g != hf]
    rset, fset = set(robot), set(feet_b)
    S_ = np.zeros((T, 2, 3))
    feet_pen = body_pen = 0.0
    ab = {m.body(n).id: np.asarray(v, bool) for n, v in (allow or {}).items()}
    pen_ca = {"strict": 0.0, "allowed": 0.0}

    def body_hit(g, dep, nz, k):
        b_ = int(m.geom_bodyid[g])
        ok_ = b_ in ab and len(ab[b_]) and ab[b_][min(k, len(ab[b_]) - 1)] and dep <= allow_tol and abs(nz) >= 0.7
        key = "allowed" if ok_ else "strict"
        pen_ca[key] = max(pen_ca[key], dep)
    ft = np.zeros(6)
    for k in range(T):
        dd.qpos[:] = q[k]
        mujoco.mj_forward(m, dd)
        for j, b in enumerate(feet_b):
            S_[k, j] = dd.xpos[b] + dd.xmat[b].reshape(3, 3) @ sole[j]
        if k % every:
            continue
        for i in range(dd.ncon):
            cc = dd.contact[i]
            g1, g2 = int(cc.geom1), int(cc.geom2)
            g, s_ = (g1, g2) if g2 == hf else (g2, g1)
            if s_ == hf and g in rset and cc.dist < 0:
                if m.geom_bodyid[g] in fset:
                    feet_pen = max(feet_pen, -float(cc.dist))
                else:
                    body_pen = max(body_pen, -float(cc.dist))
                    body_hit(g, -float(cc.dist), float(cc.frame[2]), k)
        for g in robot:
            for s_ in scene:
                dist = mujoco.mj_geomDistance(m, dd, g, s_, 0.05, ft)
                if dist < 0:
                    if m.geom_bodyid[g] in fset:
                        feet_pen = max(feet_pen, -dist)
                    else:
                        body_pen = max(body_pen, -dist)
                        nv = ft[3:6] - ft[:3]
                        body_hit(g, -dist, float(nv[2] / max(np.linalg.norm(nv), 1e-9)) if np.linalg.norm(nv) > 1e-9 else 1.0, k)
    vel = np.linalg.norm(np.diff(S_, axis=0), axis=-1) * 50.0
    planted = stance[1:] & stance[:-1]
    out = dict(slide=float(np.percentile(vel[planted], 99)) if planted.any() else 0.0,
               root_step=float(np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1).max()), feet_pen=feet_pen, body_pen=body_pen,
               frames=int(T), stance_pairs=int(planted.sum()))
    out["pass"] = {k: out[k] <= v for k, v in GATE_V1.items()}
    out["pass_all"] = all(out["pass"].values())
    if allow is not None:
        ca = dict(out, body_pen=pen_ca["strict"])
        out["contact_aware"] = {"body_pen": pen_ca["strict"], "body_pen_allowed": pen_ca["allowed"], "allow_tol": allow_tol,
                                "pass_all": all(ca[k] <= v for k, v in GATE_V1.items()),
                                "labelled_frames": {n: int(np.asarray(v, bool).sum()) for n, v in allow.items()}}
    out.update(ray)
    return out


_KIN = []


def score_motion(motion_npz, terrain_npz, contact_npz, recon=None, qpos=None, qpos_names=None, motion_idx=None, t0=0,
                 t1=None, h_thr=0.02, v_thr=0.2, meta=None):
    """The one definition: a G1 motion vs the contact head -> score()'s dict (per foot
    P / R / F1 of planted vs static and contact vs any, contact_on_static, speed_on_static_m_s, onset / offset lags,
    labels agreement; sole penetration p95), or None when fewer than 2 frames match the contact file.
      motion_npz   the retargeter's reference <clip>_motion.npz (G1, frames = recon rows, contacts, fps)
      terrain_npz  the heightfield the G1 stands on (x0, y0, res, height; rows along y)
      contact_npz  the contact head's <clip>__0.contact.npz
      recon        the recon npz path (or its "frames" array) that `frames` index; None = identity
      qpos         optional (T, 36) sim states (root xyz, quat wxyz, the dof in qpos_names order) instead of the
                   reference's G1; sample k tracks motion frame motion_idx[k] (default k)
      t0, t1       the scored samples [t0, t1) (skip a respawn drop, cut at a fall)"""
    if not _KIN:
        _KIN.append(G1Kinematics(str(URDF)))
    kin = _KIN[0]
    m = np.load(motion_npz, allow_pickle=True)
    n_m = len(m["frames"])
    if qpos is None:
        S = soles(kin, m["g1_root_pos"], m["g1_root_quat_xyzw"], m["g1_dof"], list(m["g1_joint_names"].astype(str)))
        mi = np.arange(n_m)
    else:
        Q = np.asarray(qpos, np.float64)
        S = soles(kin, Q[:, :3], Q[:, [4, 5, 6, 3]], Q[:, 7:], [str(x) for x in qpos_names])
        mi = np.clip(np.arange(len(Q)) if motion_idx is None else np.asarray(motion_idx, int), 0, n_m - 1)
    sl = slice(int(t0), len(mi) if t1 is None else min(int(t1), len(mi)))
    S, mi = {s: P[sl] for s, P in S.items()}, mi[sl]
    lab = {g: np.asarray(v, bool)[mi] for g, v in json.loads(str(m["contacts"])).items()} if "contacts" in m.files else {}
    fr = recon
    if isinstance(recon, (str, Path)):
        z = np.load(recon, allow_pickle=True)
        fr = np.asarray(z["frames"], int) if "frames" in z.files else None
    a = argparse.Namespace(h_thr=h_thr, v_thr=v_thr)
    return score(S, np.load(terrain_npz), m["frames"].astype(int)[mi], lab, np.load(contact_npz), float(m["fps"]), a,
                 dict(meta or {}, recon_frames=fr))


def check_ref(mf, a):
    mf = Path(mf)
    clip, run_dir = mf.name[: -len("_motion.npz")], mf.parent.parent
    tf, cf = mf.with_name(f"{clip}_terrain_g1.npz"), Path(a.contacts) / f"{base_clip(clip)}__0.contact.npz"
    if not tf.exists() or not cf.exists():
        return None
    meta = dict(run=f"{run_dir.name}/{mf.parent.name}", kind="ref", clip=clip, category=category(run_dir, clip))
    return score_motion(mf, tf, cf, recon_frames(clip, run_dir, a), h_thr=a.h_thr, v_thr=a.v_thr, meta=meta)


def check_sim(tp, a):
    p = Path(tp)
    refdir, run_dir, tracker = p.parents[6], p.parents[8], p.parents[4].name
    setting = p.parents[7].name[len("run_"):]
    tag, clip = refdir.name.split("__", 1)
    view = run_dir / "view" / "view"
    mf, tf = run_dir / tag / f"{clip}_motion.npz", view / "terrain" / f"{refdir.name}.npz"
    cf = Path(a.contacts) / f"{base_clip(clip)}__0.contact.npz"
    bf = refdir / "stoop_board.json"
    if not (mf.exists() and tf.exists() and cf.exists() and bf.exists()):
        return None
    board = [r for j in json.loads(bf.read_text())["jobs"] if j["label"] == tracker for r in j["rows"]]
    fell, fall_t = (bool(board[0]["fell"]), board[0]["fall_t_s"]) if board else (None, None)
    z = np.load(p)
    Q = z["robot_qpos"].astype(np.float64)
    k_end = len(Q) if not fell or fall_t is None else min(len(Q), int(fall_t * 50))
    names = json.loads((view / "manifest.json").read_text())["qpos_names"][7:]
    meta = dict(run=f"{run_dir.name}/sim_{setting}/{tracker}/{tag}", kind="sim", clip=clip,
                category=category(run_dir, clip), fell=fell, fall_t_s=fall_t, sim_frames=int(k_end))
    return score_motion(mf, tf, cf, recon_frames(clip, run_dir, a), qpos=Q, qpos_names=names, motion_idx=z["motion_t"],
                        t1=k_end, h_thr=a.h_thr, v_thr=a.v_thr, meta=meta)


def check_port(tp, a):
    """A policy rollout <run>/port_<tag>[_limB ...]/traj/<clip>_motion.npz (qpos, qpos_names; sample k = reference
    frame k, MuJoCo, VideoMimic's terrain policy) on the reference's own terrain_g1 (no view shift): the first --port_skip
    s skipped (the port respawns the root 0.1 m up), cut at results.json's hard_fall_time_s (the gate rule's
    fall_time_s is reported)."""
    p = Path(tp)
    port, run_dir = p.parents[1], p.parents[2]
    clip = p.name[: -len("_motion.npz")]
    tag = port.name[len("port_"):]
    while tag and not (run_dir / tag / f"{clip}_motion.npz").exists():  # port_<tag>_<suffix> -> <tag>
        tag = tag.rsplit("_", 1)[0] if "_" in tag else ""
    if not tag:
        return None
    mf, tf = run_dir / tag / f"{clip}_motion.npz", run_dir / tag / f"{clip}_terrain_g1.npz"
    cf = Path(a.contacts) / f"{base_clip(clip)}__0.contact.npz"
    if not (tf.exists() and cf.exists()):
        return None
    rf = port / "results.json"
    info = next((r for r in json.loads(rf.read_text()).get("clips", []) if r.get("clip") in (p.stem, clip)), {}) if rf.exists() else {}
    if "error" in info or str(info.get("ref", "")).endswith(".h5"):  # VideoMimic's own reference lives in its world
        return None
    hard, gate = info.get("hard_fall_time_s"), info.get("fall_time_s")
    z = np.load(p)
    t0, t1 = int(round(a.port_skip * 50)), len(z["qpos"]) if hard is None else min(len(z["qpos"]), int(hard * 50))
    if t1 - t0 < 10:
        return None
    meta = dict(run=f"{run_dir.name}/{port.name}", kind="port", clip=clip, category=category(run_dir, clip),
                fell=hard is not None, fall_t_s=hard, gate_fall_t_s=gate, sim_frames=int(t1 - t0))
    return score_motion(mf, tf, cf, recon_frames(clip, run_dir, a), qpos=z["qpos"], qpos_names=list(z["qpos_names"])[7:],
                        t0=t0, t1=t1, h_thr=a.h_thr, v_thr=a.v_thr, meta=meta)


def summarise(rows):
    out = {}
    for r in rows:
        for key in (r["run"], f"{r['run']}|{r['category']}"):
            b = out.setdefault(key, dict(clips=0, frames=0, n_st=0, n_cst=0, on=[], off=[], pen=[], spd=[], fell=0, drift=[],
                                         **{k: np.zeros(3) for k in PAIRS}))
            b["clips"] += 1
            b["frames"] += r["frames"]
            b["fell"] += int(bool(r.get("fell")))
            b["pen"].append(r["sole_penetration_mm_p95"])
            for s in ("left", "right"):
                d = r[s]
                for k in PAIRS:
                    if k in d:
                        b[k] += [d[k]["tp"], d[k]["fp"], d[k]["fn"]]
                b["n_st"] += d["n_static"]
                b["n_cst"] += d["n_contact_on_static"]
                b["drift"] += d.get("stance_drift_cm", [])
                if d["speed_on_static_m_s"]:
                    b["spd"].append(d["speed_on_static_m_s"][0])
                for nm, lst in (("onset_lag_ms", "on"), ("offset_lag_ms", "off")):
                    if d[nm] is not None:
                        b[lst].append(d[nm])
    med = lambda x: round(float(np.median(x)), 3) if x else None  # noqa: E731
    summ = {}
    for key, b in sorted(out.items()):
        e = dict(clips=b["clips"], frames=b["frames"])
        if b["fell"]:
            e["fell"] = b["fell"]
        for k in PAIRS:
            tp, fp, fn = b[k]
            if tp + fp + fn:
                e[k] = dict(P=round(tp / max(tp + fp, 1), 3), R=round(tp / max(tp + fn, 1), 3),
                            F1=round(2 * tp / max(2 * tp + fp + fn, 1), 3))
        e.update(contact_on_static=round(b["n_cst"] / max(b["n_st"], 1), 3), speed_on_static_m_s_median=med(b["spd"]),
                 onset_lag_ms_median=med(b["on"]), offset_lag_ms_median=med(b["off"]),
                 sole_penetration_mm_p95_median=med(b["pen"]))
        if b["drift"]:  # stance drift per head-static run (cm; stance_drift)
            dr = np.asarray(b["drift"])
            e.update(stance_runs=int(len(dr)), stance_drift_cm_median=round(float(np.median(dr)), 2),
                     stance_drift_cm_p90=round(float(np.quantile(dr, 0.9)), 2), stance_drift_gt5cm=round(float((dr > 5).mean()), 3),
                     stance_drift_gt10cm=round(float((dr > 10).mean()), 3))
        summ[key] = e
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motions", default="", help="glob(s), comma separated, of <run>/<ref>/<clip>_motion.npz")
    ap.add_argument("--sims", default="", help="glob(s), comma separated, of tracker rollout trajectory npz")
    ap.add_argument("--ports", default="", help="glob(s), comma separated, of <run>/port_<tag>*/traj/<clip>_motion.npz")
    ap.add_argument("--port_skip", type=float, default=0.5, help="s skipped at a port rollout's start (respawn drop)")
    ap.add_argument("--contacts", required=True, help="dir of <clip>__0.contact.npz contact-head files")
    ap.add_argument("--recon_dirs", default="", help="comma list of recon dirs (<dir>/<clip>.npz with frames)")
    ap.add_argument("--h_thr", type=float, default=0.02)
    ap.add_argument("--v_thr", type=float, default=0.2)
    ap.add_argument("--min_run", type=int, default=10, help="frames (50 fps) of a head-static run for the stance drift")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows, skipped = [], 0
    for fn, globs in ((check_ref, a.motions), (check_sim, a.sims), (check_port, a.ports)):
        for g in [x for x in globs.split(",") if x]:
            for f in sorted(glob.glob(g)):
                r = fn(f, a)
                if r is None:
                    skipped += 1
                    continue
                rows.append(r)
                print(json.dumps({k: r[k] for k in ("run", "clip", "frames", "sole_penetration_mm_p95")}),
                      {s: (r[s]["planted_vs_static"]["F1"], r[s]["contact_vs_any"]["F1"], r[s]["onset_lag_ms"]) for s in ("left", "right")},
                      flush=True)
    summ = summarise(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(summary=summ, skipped=skipped, clips=rows), indent=1) + "\n")
    print(json.dumps({k: v for k, v in summ.items() if "|" not in k}, indent=1))
    print("skipped", skipped)


if __name__ == "__main__":
    main()
