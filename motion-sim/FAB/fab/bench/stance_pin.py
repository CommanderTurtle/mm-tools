"""Stance pinning: one post-solve step on a G1 reference so its stance feet stay still on the terrain and it passes a strict
MuJoCo validity gate (bench/g1_contact_check.strict_gate: foot slide, root step, foot / body penetration).
  stance     stance runs (bench/g1_contact_check.stance_runs) on the contact labels (hysteresis 0.6 / 0.4, gaps
             < 0.1 s filled, runs >= 0.2 s) or on ground-truth kinematic labels; per run, the core = the frames where heel and toe are both static
             (pin 3; the run itself when it has none). solver_stance = the cores (runs >= 3 frames): the frames where
             the whole foot is fixed. Heel-strike / toe-off frames of a run keep their heel / toe point still but are
             not in solver_stance (the gate measures the footprint centre, which moves in a roll).
  target     per core: the ankle-roll pose with the gate's sole point (footprint centre at its lowest, ankle-roll
             frame) at the core's median xy, the foot flat on the local terrain plane (horizontal across an edge), the
             lowest foot capsule 2 mm above the surface (the package hfield and the gate's 2 cm slot grid, the higher).
  solve      the G1 29-DoF MJCF (the gate's model) in torch, from the solver's solution: pin the cores exactly
             (then a legs-only polish), roll points on the edge frames, an all-geom terrain non-penetration hinge,
             Model 15 ranges as hard bounds and velocity limits, the solver's self-collision pairs, legs first (the root
             costs 100x a leg joint), smooth residuals (blends the run edges), root step <= 4.5 cm / frame.
    python -m bench.stance_pin --run <run dir> --tag solve --out_tag p --recon <recon dir> \
        [--contacts_dir <dir of <clip>__0.contact.npz> | --contacts_kin_dir <dir of <clip>.npz>] [--only a,b]
fab-retarget (scripts/fab_retarget.py) runs it with the released configuration (fab/retarget/release_config.PIN).
-> <run>/<out_tag>/<clip>_motion.npz (same keys; contacts = json {lfoot, rfoot} = solver_stance on the 50 Hz grid,
   + pin_roll (T, 2) the roll frames, pin_report json) + links to <clip>_terrain_g1.npz / _scene_g1.obj.
On CUDA the main and polish loops run as CUDA graphs (fab/retarget/cuda_graph.py; masks as their precomputed nonzero indices,
the constant tensors built once) with a penetration prefilter: points whose start gap bound is >= PIN_PEN_KEEP_M (0.5 m) in
every frame enter the hinge as exact zeros, and the clips are pinned again without the prefilter if one comes within
PEN_SAFE_M. Both are bitwise the plain loops; PIN_GRAPH=0 runs the plain loops (PIN_PEN_KEEP_M=0: no prefilter).
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from fab.retarget import g1_files
from fab.retarget.cuda_graph import graph_steps

GATE_XML = g1_files.gate_scene()
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
CFG = dict(iters=600, lr=0.01, w_pin=3e4, w_rot=300.0, w_roll=3e3, w_leg=1.0, w_waist=10.0, w_arm=10.0, w_root=100.0,
           w_rrot=30.0, w_smooth=3e3, w_jump=1e5, jump=0.045, w_pen=3e4, m_body=0.003, m_foot=0.001, w_vel=10.0,
           w_self=1000.0, lift=0.002, polish=300, roll_max=4, flat_deg=12.0, ramp=1.0,
           polish_mode="pin", w_ps=100.0, lr_root=1.0, drift_cap=0.0, max_tilt=180.0, rot_cap_deg=0.0,
           retry_mm=0.0, retry_iters=2.0, min_support=0.0, edge_shift_m=0.08, edge_drop=0.02,
           w_hspeed=0.0, hspeed_margin=0.3, foot_k=5, cap_eps=0.0, slot_tri=0, clip_norm=0)
# clip_norm (opt-in): every loss term is each clip's own mean (pins over its pinned frames, the rest over its
# frames / frame pairs / triples) summed over the batch's clips, instead of one mean over the whole batch. The clips' variables
# never interact, so each clip then follows the same Adam trajectory as when solved alone: a clip's pin does not depend on
# which clips it is batched with (with one clip both settings are the same objective)


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def axis_angle(ax, ang):
    """(..., 3) unit axes, (...) angles -> (..., 3, 3)."""
    K = torch.zeros(ang.shape + (3, 3), device=ang.device, dtype=ang.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -ax[..., 2], ax[..., 1], -ax[..., 0]
    K = K - K.transpose(-1, -2)
    s, c = torch.sin(ang)[..., None, None], torch.cos(ang)[..., None, None]
    return torch.eye(3, device=ang.device, dtype=ang.dtype) + s * K + (1 - c) * K @ K


def so3_exp(v):
    a = v.norm(dim=-1).clamp_min(1e-9)
    return axis_angle(v / a[..., None], a)


class MjFK:
    """Forward kinematics of a MuJoCo model's body tree (free root + hinges), torch, batched over frames; matches
    mj_kinematics (the gate's model) to float precision."""

    def __init__(self, xml, names, dev):
        import mujoco

        m = mujoco.MjModel.from_xml_path(str(xml))
        self.m, self.dev = m, dev
        self.names = [m.body(i).name for i in range(m.nbody)]
        self.parent = [int(m.body_parentid[i]) for i in range(m.nbody)]
        self.pos = torch.tensor(m.body_pos, dtype=torch.float32, device=dev)
        self.rot = torch.tensor(np.stack([quat_mat(q) for q in m.body_quat]), dtype=torch.float32, device=dev)
        col = {n: i for i, n in enumerate(names)}
        self.jnt = {}  # body -> (axis, anchor, column in q)
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                self.jnt[int(m.jnt_bodyid[j])] = (torch.tensor(m.jnt_axis[j], dtype=torch.float32, device=dev),
                                                  torch.tensor(m.jnt_pos[j], dtype=torch.float32, device=dev), col[m.joint(j).name])
        self.free = int(m.jnt_bodyid[next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)])

    def __call__(self, R, t, q):
        P, Q = [None] * len(self.names), [None] * len(self.names)
        T = len(t)
        P[0] = torch.zeros(T, 3, device=self.dev)
        Q[0] = torch.eye(3, device=self.dev).expand(T, 3, 3)
        for b in range(1, len(self.names)):
            if b == self.free:
                P[b], Q[b] = t, R
                continue
            p = self.parent[b]
            Rb = Q[p] @ self.rot[b]
            xb = P[p] + Q[p] @ self.pos[b]
            if b in self.jnt:
                ax, anc, k = self.jnt[b]
                Rj = axis_angle(ax.expand(T, 3), q[:, k])
                xb = xb + Rb @ anc - (Rb @ Rj) @ anc
                Rb = Rb @ Rj
            P[b], Q[b] = xb, Rb
        return P, Q


def collision_points(fk, k=5, k_foot=None, eps=0.0):
    """Every robot collision geom of the model as body-frame sample points + radii: capsules k points on the axis (the
    foot capsules k_foot: the sole footprint; eps > 0: at least so many that an edge between two spheres reaches at most eps
    into the capsule, spacing 2 sqrt(2 r eps): 5 points left 1.5 cm on the 25 cm knee capsule), spheres their centre.
    -> body ids (n,), points (n, 3), radii (n,), is_foot (n,)."""
    import mujoco

    m = fk.m
    B, Pt, Rd, Ft = [], [], [], []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0 or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        R = quat_mat(m.geom_quat[g])
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            kk = k_foot if k_foot and fk.names[int(m.geom_bodyid[g])] in FEET else k
            if eps > 0:
                r_, hl = float(m.geom_size[g][0]), float(m.geom_size[g][1])
                kk = max(kk, int(np.ceil(2 * hl / (2 * np.sqrt(2 * r_ * eps)))) + 1)
            s = np.linspace(-m.geom_size[g][1], m.geom_size[g][1], kk)
            pts = m.geom_pos[g][None] + s[:, None] * R[:, 2][None]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            pts = m.geom_pos[g][None]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            pts = m.geom_pos[g][None] + np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * m.geom_size[g] @ R.T
        else:
            continue
        r = float(m.geom_size[g][0]) if int(m.geom_type[g]) in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_SPHERE)) else 0.0
        b = int(m.geom_bodyid[g])
        for p in pts:
            B.append(b), Pt.append(p), Rd.append(r), Ft.append(fk.names[b] in FEET)
    return np.array(B), np.array(Pt, np.float32), np.array(Rd, np.float32), np.array(Ft)


def sole_geometry(fk):
    """In each ankle-roll frame: the gate's sole point (kin.G1Kin._sole_point: the lowest collision points of the foot's
    geoms in the standing pose, those within 1 cm of the lowest, averaged), the heel / toe points (the rearmost /
    foremost of them) and the foot capsules' bottom sample points (+ radii)."""
    import mujoco

    m = fk.m
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    d.qpos[2] = 1.0
    mujoco.mj_kinematics(m, d)
    out = {}
    for f in FEET:
        b = m.body(f).id
        Rb, pb = d.xmat[b].reshape(3, 3), d.xpos[b]
        pts = []
        for g in np.flatnonzero(m.geom_bodyid == b):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            gt, sz = m.geom_type[g], m.geom_size[g]
            down = sz[0] if gt == mujoco.mjtGeom.mjGEOM_SPHERE else (sz[2] if gt == mujoco.mjtGeom.mjGEOM_BOX else sz[0])
            pts.append(d.geom_xpos[g] - np.array([0.0, 0.0, down]))
        pts = np.array(pts)
        low = pts[np.abs(pts[:, 2] - pts[:, 2].min()) < 0.01]
        sole = Rb.T @ (low.mean(0) - pb)
        # the capsule ends along the foot (the heel / toe ends of the long capsules), their lowest points
        ends = []
        for g in np.flatnonzero(m.geom_bodyid == b):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            ax = d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g][1]
            for e in (d.geom_xpos[g] + ax, d.geom_xpos[g] - ax):
                ends.append(Rb.T @ (e - np.array([0, 0, m.geom_size[g][0]]) - pb))
        ends = np.array(ends)
        heel, toe = ends[ends[:, 0].argmin()], ends[ends[:, 0].argmax()]
        heel[1:] = sole[1:]
        toe[1:] = sole[1:]
        out[f] = dict(sole=sole.astype(np.float32), heel=heel.astype(np.float32), toe=toe.astype(np.float32))
    return out


def _fused_ok(xy, tri):
    """Surface.req's fused kernel applies: a CUDA float32 query, bilinear slot grid, Triton available (PIN_REQ_FUSED=0: the
    PyTorch expression; both give the same bits)."""
    if tri or not xy.is_cuda or xy.dtype != torch.float32 or os.environ.get("PIN_REQ_FUSED", "1") == "0":
        return False
    from fab.retarget.req_kernel import AVAILABLE

    return AVAILABLE


class Surface:
    """Terrain height = max(bilinear of the package hfield, bilinear of the gate's slot grid) at world xy (torch), for a
    batch of clips: grids padded to one shape, every query carries its clip index (cid). tri: the slot grid as MuJoCo's
    hfield surface (each cell two triangles split along the (x0, y0) -> (x1, y1) diagonal, mj_rayHfield), not bilinear."""

    def __init__(self, clips, dev, tri=False):
        """clips: [(package (H, x0, y0, res), slot (z, X0, Y0, RES, c, Rm, shift))] per clip."""
        t_ = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev)  # noqa: E731
        self.n, self.tri = len(clips), tri
        self.g = []
        for k in range(2):
            Hs = [cl[k][0] for cl in clips]
            ny, nx = max(h.shape[0] for h in Hs), max(h.shape[1] for h in Hs)
            G = np.zeros((len(Hs), ny, nx), np.float32)
            for i, h in enumerate(Hs):
                G[i, : h.shape[0], : h.shape[1]] = h
            par = dict(H=t_(G), x0=t_([cl[k][1] for cl in clips]), y0=t_([cl[k][2] for cl in clips]), res=t_([cl[k][3] for cl in clips]),
                       nx=t_([h.shape[1] for h in Hs]), ny=t_([h.shape[0] for h in Hs]))
            if k == 1:
                par.update(c=t_([cl[1][4] for cl in clips]), Rm=t_([cl[1][5] for cl in clips]), sh=t_([cl[1][6] for cl in clips]))
            self.g.append(par)

    def _grid(self, g, p, cid, outside=None, tri=False):
        """Bilinear (tri: MuJoCo's two-triangle) lookup of grid g at points p (..., 2) in that grid's own coordinates;
        outside the grid: the edge (clamped), or `outside` when given (the gate's slot: no terrain there, only its floor
        plane)."""
        nx, ny = g["nx"][cid], g["ny"][cid]
        u0, v0 = (p[..., 0] - g["x0"][cid]) / g["res"][cid], (p[..., 1] - g["y0"][cid]) / g["res"][cid]
        u = torch.minimum(u0.clamp_min(0), nx - 1.001)
        v = torch.minimum(v0.clamp_min(0), ny - 1.001)
        i, j = v.floor().long(), u.floor().long()
        a, b = v - i, u - j
        H = g["H"]
        if tri:  # a (y) <= b (x): triangle (i, j), (i, j+1), (i+1, j+1); else (i, j), (i+1, j+1), (i+1, j)
            h00, h01, h10, h11 = H[cid, i, j], H[cid, i, j + 1], H[cid, i + 1, j], H[cid, i + 1, j + 1]
            h = torch.where(a <= b, h00 + b * (h01 - h00) + a * (h11 - h01), h00 + a * (h10 - h00) + b * (h11 - h10))
        else:
            h = (1 - a) * (1 - b) * H[cid, i, j] + (1 - a) * b * H[cid, i, j + 1] + a * (1 - b) * H[cid, i + 1, j] + a * b * H[cid, i + 1, j + 1]
        if outside is not None:
            h = torch.where((u0 >= 0) & (u0 <= nx - 1) & (v0 >= 0) & (v0 <= ny - 1), h, torch.full_like(h, outside))
        return h

    def _slot(self, xy, cid):
        g = self.g[1]
        return torch.einsum("...ij,...j->...i", g["Rm"][cid], xy - g["c"][cid]) + g["sh"][cid]

    def __call__(self, xy, cid):
        cid = torch.as_tensor(cid, device=xy.device).expand(xy.shape[:-1])
        return torch.maximum(self._grid(self.g[0], xy, cid), self._grid(self.g[1], self._slot(xy, cid), cid, outside=-1e3, tri=self.tri))

    def req(self, xy, r, cid, const=None):
        """The lowest centre height of a sphere of radius r at xy (..., 2) that clears the surface: max over offsets
        within its footprint of h(xy + d) + sqrt(r^2 - |d|^2) (a capsule next to a riser collides sideways). The slot
        transform is applied to the centres once and to the offsets as a rotation (no per-offset gathers)."""
        if const is not None:  # the xy-independent tensors of this call site, computed once (req_const): the same values
            r, cid, d, ds, rs = const
        else:
            r, cid, d, ds, rs = self.req_const(r, cid, xy.shape[:-1], xy.dtype, xy.device)
        if _fused_ok(xy, self.tri):  # one fused kernel: the same values and gradient (fab.retarget.req_kernel)
            from fab.retarget.req_kernel import req_fused

            return req_fused(xy, self._slot(xy, cid), (r, cid, d, ds, rs), self.g[0], self.g[1])
        h0 = self._grid(self.g[0], xy[..., None, :] + d, cid[..., None])
        h1 = self._grid(self.g[1], self._slot(xy, cid)[..., None, :] + ds, cid[..., None], outside=-1e3, tri=self.tri)
        return (torch.maximum(h0, h1) + rs).max(-1).values

    def req_const(self, r, cid, shape, dtype, device):
        """req's tensors that do not depend on xy (radii, clip ids, the offsets and their slot rotation, the
        sphere-cap heights), as req computes them."""
        if getattr(self, "_req_c", None) is None or self._req_c[0].device != device:
            rho = torch.tensor([0.0] + [0.5] * 8 + [0.8] * 8 + [1.0] * 8, device=device)
            th = torch.tensor([0.0] + list(np.linspace(0, 2 * np.pi, 8, endpoint=False)) * 3, device=device, dtype=torch.float32)
            off = torch.stack([rho * torch.cos(th), rho * torch.sin(th)], -1)  # (25, 2) on the unit disk
            self._req_c = (rho, off, torch.sqrt((1 - rho ** 2).clamp_min(0)))
        rho, off, sq_ = self._req_c
        r = torch.as_tensor(r, device=device, dtype=dtype).expand(shape)
        cid = torch.as_tensor(cid, device=device).expand(shape)
        d = r[..., None, None] * off  # (..., 25, 2)
        ds = torch.einsum("...ij,...kj->...ki", self.g[1]["Rm"][cid], d)
        return r, cid, d, ds, r[..., None] * sq_

    def bound_setup(self, r_max):
        """Max-filtered copies of both grids (half-width ceil(r_max / res) + 2 cells): at a point's (clamped) cell
        they bound every height req can read for a sphere of radius <= r_max there (bilinear <= its corner cells)."""
        self._hb = []
        for g in self.g:
            k = int(np.ceil(r_max / float(g["res"].min()))) + 2
            self._hb.append(torch.nn.functional.max_pool2d(g["H"][None], 2 * k + 1, 1, k)[0])

    def _cell(self, g, p, cid):
        nx, ny = g["nx"][cid], g["ny"][cid]
        u = torch.minimum(((p[..., 0] - g["x0"][cid]) / g["res"][cid]).clamp_min(0), nx - 1.001)
        v = torch.minimum(((p[..., 1] - g["y0"][cid]) / g["res"][cid]).clamp_min(0), ny - 1.001)
        return v.floor().long(), u.floor().long()

    def hmax(self, xy, cid):
        """An upper bound of the terrain height req can read around xy (bound_setup first)."""
        i0, j0 = self._cell(self.g[0], xy, cid)
        i1, j1 = self._cell(self.g[1], self._slot(xy, cid), cid)
        return torch.maximum(self._hb[0][cid, i0, j0], self._hb[1][cid, i1, j1])

    def req_cols(self, xy, slot_xy, const):
        """req on a column subset: xy (T, m, 2), its slot coordinates (taken from _slot on all columns) and the
        req_const tensors of those columns; the per-element ops of req."""
        r, cid, d, ds, rs = const
        if _fused_ok(xy, self.tri):
            from fab.retarget.req_kernel import req_fused

            return req_fused(xy, slot_xy, const, self.g[0], self.g[1])
        h0 = self._grid(self.g[0], xy[..., None, :] + d, cid[..., None])
        h1 = self._grid(self.g[1], slot_xy[..., None, :] + ds, cid[..., None], outside=-1e3, tri=self.tri)
        return (torch.maximum(h0, h1) + rs).max(-1).values


def surfaces(tg, root_xy):
    """The package terrain (terrain_g1 grid) and the gate's view of it: terrain_hfield_2cm (nearest resample,
    bench.scene_r2s_simpkg.hfield_2cm) put on the gate's 14 x 4 m slot (nearest, outside = the lowest), in the clip's own
    frame (the slot is a yaw + shift of it)."""
    H = np.asarray(tg["height"], np.float64)
    res, x0, y0 = float(tg["res"]), float(tg["x0"]), float(tg["y0"])
    ny, nx = H.shape
    r2 = 0.02 if res > 0.03 else res
    n2y, n2x = int(np.floor((ny - 1) * res / r2)) + 1, int(np.floor((nx - 1) * res / r2)) + 1
    iy = np.clip(np.round(np.arange(n2y) * r2 / res).astype(int), 0, ny - 1)
    ix = np.clip(np.round(np.arange(n2x) * r2 / res).astype(int), 0, nx - 1)
    H2 = H[np.ix_(iy, ix)]
    X0, X1, Y0, Y1, RES = -2.5, 11.5, -2.0, 2.0, 0.02
    xy = np.asarray(root_xy, np.float64)
    c = xy.mean(0)
    _, V = np.linalg.eigh((xy - c).T @ (xy - c))
    ax = V[:, -1]
    if (xy[-1] - xy[0]) @ ax < 0:
        ax = -ax
    th = float(np.arctan2(ax[1], ax[0]))
    Rm = np.array([[np.cos(-th), -np.sin(-th)], [np.sin(-th), np.cos(-th)]])
    u = (xy - c) @ Rm.T
    shift = np.array([-u[0, 0], -u[:, 1].mean()])
    xs, ys = np.arange(X0, X1 + RES / 2, RES), np.arange(Y0, Y1 + RES / 2, RES)
    XX, YY = np.meshgrid(xs, ys)
    P = np.stack([XX.ravel() - shift[0], YY.ravel() - shift[1]], 1) @ Rm + c
    jx, jy = np.round((P[:, 0] - x0) / r2).astype(int), np.round((P[:, 1] - y0) / r2).astype(int)
    inside = (jx >= 0) & (jx < H2.shape[1]) & (jy >= 0) & (jy < H2.shape[0])
    z = np.full(len(P), np.nan)
    z[inside] = H2[jy[inside], jx[inside]]
    f0 = float(np.nanmin(z))
    z = np.maximum(np.where(np.isnan(z), f0, z).reshape(XX.shape), f0)
    z = np.round((z - f0) * 1000.0) / 1000.0 + f0
    return (H, x0, y0, res), (z, X0, Y0, RES, c, Rm, shift)


def longest_still(xy, cap, R=None, rot_cap=0.0):
    """The longest window [i, j) of xy (n, 2) whose points stay within cap of the window's median and, with R (n, 3, 3)
    and rot_cap (rad) > 0, whose foot orientation stays within rot_cap of the window's first frame (a pivot is not a
    stance)."""
    n = len(xy)

    def ok(i, j):
        if cap > 0 and np.linalg.norm(xy[i:j] - np.median(xy[i:j], 0), axis=1).max() > cap:
            return False
        if R is not None and rot_cap > 0:
            cosang = (np.einsum("ji,tjk->tik", R[i], R[i:j]).trace(axis1=1, axis2=2) - 1) / 2
            if np.arccos(np.clip(cosang, -1, 1)).max() > rot_cap:
                return False
        return True

    if n == 0 or ok(0, n):
        return 0, n
    best = (0, 1)
    for i in range(n):
        j = i + max(1, best[1] - best[0])
        while j <= n and ok(i, j):
            best = (i, j) if j - i > best[1] - best[0] else best
            j += 1
    return best


def stance_cores(st, fps=50.0, roll_max=4, sole_xy=None, cap=0.0, sole_R=None, rot_cap=0.0):
    """stance runs -> per foot: core (T,) bool (pin 3 frames of each run, the whole run when it has none; runs
    >= 3 frames), roll (T,) 1 heel / 2 toe on the run's other frames next to the core (at most roll_max on each side:
    heel strike / toe off are brief; the head's longer stance edges are left to the solve), run id per core frame."""
    out = {}
    for side in ("left", "right"):
        s, pin, run = st[side]["stance"], st[side]["pin"], st[side]["run"]
        T = len(s)
        core, roll, rid = np.zeros(T, bool), np.zeros(T, np.uint8), np.full(T, -1, int)
        for r in np.unique(run[run >= 0]):
            idx = np.flatnonzero(run == r)
            both = idx[pin[idx] == 3]
            if len(both):
                a, b = both.min(), both.max() + 1  # the core is contiguous: first to last both-static frame
            else:
                a, b = idx.min(), idx.max() + 1
            if sole_xy is not None and (cap > 0 or rot_cap > 0):  # keep the input's stillest part (drift / pivot)
                i, j = longest_still(sole_xy[side][a:b], cap, None if sole_R is None else sole_R[side][a:b], rot_cap)
                a, b = a + i, a + j
            if b - a < 3:
                continue
            core[a:b], rid[a:b] = True, r
            roll[idx[(idx < a) & (idx >= a - roll_max)]] = 1  # heel strike: the heel is the static point
            roll[idx[(idx >= b) & (idx < b + roll_max)]] = 2  # toe off
        out[side] = dict(core=core, roll=roll, run=rid)
    return out


def flat_rot(R_med, n):
    """A foot frame with z = the terrain normal n and x = the median foot heading projected on that plane."""
    x = R_med[:, 0] - (R_med[:, 0] @ n) * n
    x = x / max(np.linalg.norm(x), 1e-9)
    return np.stack([x, np.cross(n, x), n], 1)


def support_shift(surf, ci, xy, Rf, fp, frad, sole, c, t_):
    """The share of the sole's footprint (a 7 x 3 grid over the foot capsules' extent) that is not over a drop deeper than
    edge_drop once the foot sits on its highest point (a tread's nose, not TSDF bumps, with edge_drop ~5 cm); when it is
    below min_support, the smallest xy shift (along / across the foot, up to edge_shift_m) that brings it to min_support
    (else the best). -> (xy, support before, after or None)."""
    grid = np.array([[x, y, 0.0] for x in np.linspace(-0.054, 0.132, 7) for y in (-0.025, 0.0, 0.025)]) - [sole[0], sole[1], 0.0]
    gw = grid @ Rf.T
    fwd = Rf[:2, 0] / max(np.linalg.norm(Rf[:2, 0]), 1e-9)
    lat = np.array([-fwd[1], fwd[0]])
    r = np.arange(-c["edge_shift_m"], c["edge_shift_m"] + 1e-6, 0.01)
    D = np.array([[0.0, 0.0]] + [[a, b] for a in r for b in r[::2] if (a, b) != (0.0, 0.0)])
    cand = xy[None] + D[:, :1] * fwd + D[:, 1:] * lat  # (K, 2)
    rq = surf.req(t_(cand[:, None] + fp[None, :, :2]), t_(frad)[None].expand(len(cand), -1), ci).cpu().numpy()
    zc = (rq - fp[None, :, 2]).max(1)  # the sole point's height per candidate (resting on its highest point)
    h = surf(t_(cand[:, None] + gw[None, :, :2]), ci).cpu().numpy()
    sup = (h >= zc[:, None] + gw[None, :, 2] - c["edge_drop"]).mean(1)  # not hanging over a drop deeper than edge_drop
    if sup[0] >= c["min_support"]:
        return xy, float(sup[0]), None
    dist = np.hypot(D[:, 0], D[:, 1])
    ok = sup >= c["min_support"]
    k = int(np.argmin(np.where(ok, dist, np.inf))) if ok.any() else int(np.argmax(sup - 1e-3 * dist))
    return cand[k], float(sup[0]), float(sup[k])


def pin_clip(mo, tg, stance, dev="cuda", cfg=None):
    """One clip (pin_batch of one)."""
    return pin_batch([(mo, tg, stance)], dev, cfg)[0]


PEN_KEEP_M = float(os.environ.get("PIN_PEN_KEEP_M", "0.5"))  # penetration prefilter (graph path): keep points with a start gap below this
PEN_SAFE_M = 0.0031  # ... exact while every excluded point's gap is >= its margin (3 mm): a lower bound of it is tracked at every
# main-loop iteration inside the graph (the point's height above the max-filtered terrain, minus its radius) and the exact gap
# at the final pose; either below this -> the clip set again without the prefilter


def pin_batch(items, dev="cuda", cfg=None):
    """pin_batch with the penetration prefilter (PIN_PEN_KEEP_M > 0); the clips are pinned again without it when an excluded
    point came closer than PEN_SAFE_M."""
    out = _pin_batch(items, dev, cfg, prefilter=PEN_KEEP_M > 0)
    if out is None:
        out = _pin_batch(items, dev, cfg, prefilter=False)
        for o in out:
            o[5]["pen_prefilter"] = "fallback: an excluded point's gap bound fell below PEN_SAFE_M"
    return out


def _pin_batch(items, dev="cuda", cfg=None, prefilter=True):
    """items: [(mo, tg, stance)]: a reference (motion npz dict), its terrain_g1 dict and its stance runs on
    its 50 Hz grid; solved together (frames concatenated, time terms masked at the clip boundaries).
    -> per clip (new root pos, root quat xyzw, dof, solver_stance (T, 2), roll (T, 2), report)."""
    from scipy.spatial.transform import Rotation

    from fab.retarget.g1_model15 import RANGES, VELOCITY
    from bench.videomimic.scene_retarget import SELF_ARM, SELF_BODY, SELF_HAND, SELF_LEG, self_capsules

    c = dict(CFG, **(cfg or {}))
    names = [str(n) for n in items[0][0]["g1_joint_names"]]
    assert all([str(n) for n in mo["g1_joint_names"]] == names for mo, _, _ in items)
    fk = MjFK(GATE_XML, names, dev)
    t_ = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=dev)  # noqa: E731
    Ts = [len(mo["g1_dof"]) for mo, _, _ in items]
    off = np.r_[0, np.cumsum(Ts)]
    T = int(off[-1])
    cid_np = np.concatenate([np.full(n, i) for i, n in enumerate(Ts)])
    cid = torch.as_tensor(cid_np, device=dev)
    q0 = t_(np.concatenate([mo["g1_dof"] for mo, _, _ in items]))
    R0 = t_(np.concatenate([Rotation.from_quat(np.asarray(mo["g1_root_quat_xyzw"], np.float64)).as_matrix() for mo, _, _ in items]))
    p0 = t_(np.concatenate([mo["g1_root_pos"] for mo, _, _ in items]))
    rng = RANGES
    lo, hi = t_([rng[n][0] for n in names]), t_([rng[n][1] for n in names])
    vlim = VELOCITY
    vmax = t_([vlim[n] for n in names])
    surf = Surface([surfaces(tg, np.asarray(mo["g1_root_pos"])[:, :2]) for mo, tg, _ in items], dev, tri=bool(c["slot_tri"]))
    geo = sole_geometry(fk)
    fb = {f: fk.names.index(f) for f in FEET}
    same1 = t_(cid_np[1:] == cid_np[:-1]).bool()  # frame pairs / triples inside one clip
    # root speed cap from the human (w_hspeed > 0): the G1 root may not outrun scale x the recon pelvis speed + a margin
    hcap = t_(np.concatenate([np.asarray(mo.get("_hspeed", np.full(len(mo["g1_dof"]), np.inf)), np.float64) * float(mo["scale"])
                              for mo, _, _ in items])[1:] + c["hspeed_margin"])
    same2 = t_((cid_np[2:] == cid_np[1:-1]) & (cid_np[1:-1] == cid_np[:-2])).bool()
    with torch.no_grad():
        P0, Q0 = fk(R0, p0, q0.clamp(lo + 1e-4, hi - 1e-4))
        sxy = {s_: (P0[fb[f]] + (Q0[fb[f]] @ t_(geo[f]["sole"])[:, None])[..., 0])[:, :2].cpu().numpy() for f, s_ in zip(FEET, ("left", "right"))}
        sR = {s_: Q0[fb[f]].cpu().numpy().astype(np.float64) for f, s_ in zip(FEET, ("left", "right"))}
    cores = [stance_cores(st, roll_max=int(c["roll_max"]), sole_xy={k: v[off[i]: off[i + 1]] for k, v in sxy.items()}, cap=c["drift_cap"],
                          sole_R={k: v[off[i]: off[i + 1]] for k, v in sR.items()}, rot_cap=np.radians(c["rot_cap_deg"]))
             for i, (_, _, st) in enumerate(items)]
    tgt_p, tgt_R = {f: torch.zeros(T, 3, device=dev) for f in FEET}, {f: torch.eye(3, device=dev).repeat(T, 1, 1) for f in FEET}
    roll_p = {f: torch.zeros(T, 3, device=dev) for f in FEET}
    mask, rollm, roll_pt = {}, {}, {}
    Bc, Pc, Rc, Fc = collision_points(fk, k_foot=int(c["foot_k"]), eps=float(c["cap_eps"]))
    reps = [{"runs": {}} for _ in items]
    for f, side in zip(FEET, ("left", "right")):
        core_all = np.concatenate([cr[side]["core"] for cr in cores])
        roll_all = np.concatenate([cr[side]["roll"] for cr in cores])
        mask[f], rollm[f] = t_(core_all).bool(), t_(roll_all > 0).bool()
        roll_pt[f] = torch.where(t_(roll_all == 1)[:, None].bool(), t_(geo[f]["heel"]), t_(geo[f]["toe"]))
        fpts = Pc[(Bc == fb[f])]
        frad = Rc[(Bc == fb[f])]
        for ci, (cr_, (_, _, st)) in enumerate(zip(cores, items)):
            cr = cr_[side]
            n_runs = 0
            for r in np.unique(cr["run"][cr["run"] >= 0]):
                idx = np.flatnonzero(cr["run"] == r) + off[ci]
                Rm_ = Q0[fb[f]][idx].cpu().numpy()
                sole_w = (P0[fb[f]][idx] + (Q0[fb[f]][idx] @ t_(geo[f]["sole"])[:, None])[..., 0]).cpu().numpy()
                xy = np.median(sole_w[:, :2], 0)
                # the run's median foot orientation; flattened onto the local terrain plane when it is within flat_deg
                # of it (a large tilt is the solver's answer to Model 15's ankle range, e.g. the forefoot on a tread)
                Rmed = Rotation.from_matrix(Rm_).mean().as_matrix()
                Rh = flat_rot(Rmed, np.array([0, 0, 1.0]))
                fp = (fpts - geo[f]["sole"]) @ Rh.T
                hz = surf(t_(xy[None] + fp[:, :2]), ci).cpu().numpy()
                A_ = np.c_[fp[:, :2], np.ones(len(fp))]
                coef, *_ = np.linalg.lstsq(A_, hz, rcond=None)
                n = np.array([0, 0, 1.0])
                if np.abs(A_ @ coef - hz).max() < 0.01 and np.hypot(*coef[:2]) < np.tan(np.radians(20)):
                    n = np.array([-coef[0], -coef[1], 1.0]) / np.linalg.norm([-coef[0], -coef[1], 1.0])
                tilt = float(np.degrees(np.arccos(np.clip(Rmed[:, 2] @ n, -1, 1))))
                if tilt > c["max_tilt"]:  # not standing on its sole (a foot on its top / side in floor poses): not pinned
                    run_all = np.flatnonzero(st[side]["run"] == r)
                    cr["core"][idx - off[ci]], cr["roll"][run_all] = False, 0
                    mask[f][idx], rollm[f][run_all + off[ci]] = False, False
                    reps[ci].setdefault("unpinned_tilt_deg", []).append(round(tilt, 1))
                    continue
                Rf = flat_rot(Rmed, n) if tilt < c["flat_deg"] else Rmed
                fp = (fpts - geo[f]["sole"]) @ Rf.T
                if c["min_support"] > 0:  # sole over an edge (a tread's nose): slide the foot onto the support it rests on
                    xy, sup0, sup1 = support_shift(surf, ci, xy, Rf, fp, frad, geo[f]["sole"], c, t_)
                    if sup1 is not None:
                        reps[ci].setdefault("edge_shift", []).append([round(sup0, 2), round(sup1, 2)])
                if c.get("rest_plain", 0):  # each foot point on the heightfield directly under it (no footprint max)
                    rq = (surf._grid(surf.g[0], t_(xy[None] + fp[:, :2]), torch.full((len(fp),), ci, device=dev)) + t_(frad)).cpu().numpy()
                else:
                    rq = surf.req(t_(xy[None] + fp[:, :2]), t_(frad), ci).cpu().numpy()
                zc = float(np.max(rq - fp[:, 2])) + c["lift"]  # the sole point's height: the lowest capsule 2 mm clear
                reps[ci].setdefault("tilt_deg", []).append(round(tilt, 1))
                po = np.r_[xy, zc] - Rf @ geo[f]["sole"]
                tgt_p[f][idx], tgt_R[f][idx] = t_(po), t_(Rf)
                # roll frames of the same stance run: the heel (strike) / toe (off) point at its place in the core pose
                run_all = np.flatnonzero(st[side]["run"] == r)
                for k in run_all[cr["roll"][run_all] > 0]:
                    pt = geo[f]["heel"] if cr["roll"][k] == 1 else geo[f]["toe"]
                    roll_p[f][k + off[ci]] = t_(po + Rf @ pt)
                n_runs += 1
            reps[ci]["runs"][side] = n_runs
    gnd = None
    if c.get("w_ground", 0) > 0:  # --ground_aware (fab/retarget/release_config.py ground_aware_pin): on the frames the solve labels
        # a foot (lfoot / rfoot) that are not that foot's core or roll frames, the foot's lowest collision point is held within lift of
        # the terrain under it (the package heightfield, not the max-filtered surface; vertical only: the hold never slides a foot to a
        # lower cell); on the no-flight frames (anyfoot), the lower foot
        assert not c["clip_norm"], "w_ground with clip_norm"

        def lab_(g):
            return np.concatenate([np.r_[np.asarray(json.loads(str(mo["contacts"])).get(g, []), bool), np.zeros(len(mo["g1_dof"]), bool)][: len(mo["g1_dof"])]
                                   for mo, _, _ in items])

        busy = np.zeros(T, bool)
        gm = {}
        for f, g in zip(FEET, ("lfoot", "rfoot")):
            free = ~(mask[f].cpu().numpy() | rollm[f].cpu().numpy())
            gm[f] = lab_(g) & free
            busy |= ~free
        ga_ = lab_("anyfoot") & ~busy
        gnd = dict(m={f: t_(gm[f]) for f in FEET}, any=t_(ga_), n=max(1, int(sum(gm[f].sum() for f in FEET) + ga_.sum())),
                   cols={f: torch.as_tensor(np.flatnonzero(Bc == fb[f]), device=dev) for f in FEET})
        for ci in range(len(items)):
            sl_ = slice(off[ci], off[ci + 1])
            reps[ci]["ground_frames"] = {"lfoot": int(gm[FEET[0]][sl_].sum()), "rfoot": int(gm[FEET[1]][sl_].sum()), "anyfoot": int(ga_[sl_].sum())}
    zq = torch.logit(((q0 - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)).clone().requires_grad_(True)
    dp = torch.zeros(T, 3, device=dev, requires_grad=True)
    dr = torch.zeros(T, 3, device=dev, requires_grad=True)
    wj = t_([c["w_leg"] if any(k in n for k in ("hip", "knee", "ankle")) else c["w_waist"] if "waist" in n else c["w_arm"] for n in names])
    leg = t_([any(k in n for k in ("hip", "knee", "ankle")) for n in names]).bool()
    Bt, Pt_, Rt_, Ft_ = torch.as_tensor(Bc, device=dev), t_(Pc), t_(Rc), torch.as_tensor(Fc, device=dev)
    marg = torch.where(Ft_, c["m_foot"], c["m_body"])
    pairs = [(self_capsules(SELF_ARM), self_capsules(SELF_BODY)), (self_capsules(SELF_LEG["left"]), self_capsules(SELF_LEG["right"])),
             (self_capsules(SELF_HAND["left"]), self_capsules(SELF_HAND["right"]))]
    bi = {n: i for i, n in enumerate(fk.names)}

    caps_c = {}  # id(capsule list) -> ([(body index, points tensor)], radii): no host -> device copy per call

    def caps(P, Q, cl):
        if id(cl) not in caps_c:
            caps_c[id(cl)] = ([(bi[b], t_(p)) for b, p, _ in cl], torch.cat([torch.full((len(p),), r_, device=dev) for _, p, r_ in cl]))
        pts_, rr_ = caps_c[id(cl)]
        X = torch.cat([P[b_][:, None] + torch.einsum("tij,pj->tpi", Q[b_], p_) for b_, p_ in pts_], 1)
        return X, rr_

    with torch.no_grad():  # self pairs: only those deeper than in the input are penalised (the input already passed the solve's own self-collision term)
        self0 = []
        for a_, b_ in pairs:
            Xa, ra = caps(P0, Q0, a_)
            Xb, rb = caps(P0, Q0, b_)
            self0.append((torch.cdist(Xa, Xb) - ra[None, :, None] - rb[None, None]).clamp(max=0.01))

    def forward(zq, dp, dr):
        q = lo + (hi - lo) * torch.sigmoid(zq)
        R = R0 @ so3_exp(dr)
        t = p0 + dp
        P, Q = fk(R, t, q)
        return q, R, t, P, Q

    gidx = {}  # graph path, set before the loops: the boolean masks as their nonzero indices (x[mask] gathers exactly these)

    def d2(x):  # second differences inside each clip
        return (x[2:] - 2 * x[1:-1] + x[:-2])[gidx.get("same2", same2)]

    cid1, cid2 = cid[1:][same1], cid[2:][same2]

    def cm(x, ids):  # clip_norm: the sum over clips of each clip's mean of x (one value per element, ids its clip)
        s_ = torch.zeros(len(items), device=dev, dtype=x.dtype).index_add_(0, ids, x)
        return (s_ / torch.bincount(ids, minlength=len(items)).clamp(min=1).to(x.dtype)).sum()

    def losses_clip(q, t, P, Q, pin_only=False):
        e = {}
        pid = torch.cat([cid[mask[f]] for f in FEET])
        e["pin"] = cm(torch.cat([((P[fb[f]] - tgt_p[f]) ** 2).sum(-1)[mask[f]] for f in FEET]), pid)
        e["rot"] = cm(torch.cat([((Q[fb[f]] - tgt_R[f]) ** 2).sum((-1, -2))[mask[f]] for f in FEET]), pid)
        if pin_only:
            return e
        e["roll"] = cm(torch.cat([((P[fb[f]] + (Q[fb[f]] @ roll_pt[f][..., None])[..., 0] - roll_p[f]) ** 2).sum(-1)[rollm[f]] for f in FEET]),
                       torch.cat([cid[rollm[f]] for f in FEET]))
        e["prior"] = cm((wj * (q - q0) ** 2).sum(-1), cid) + c["w_root"] * cm((dp ** 2).sum(-1), cid) + c["w_rrot"] * cm((dr ** 2).sum(-1), cid)
        e["smooth"] = cm((d2(torch.cat([q - q0, dp, dr], 1)) ** 2).sum(-1), cid2)
        e["jump"] = cm(torch.relu((t[1:] - t[:-1]).norm(dim=-1) - c["jump"])[same1] ** 2, cid1)
        if c["w_hspeed"] > 0:
            hm = same1 & torch.isfinite(hcap)
            e["hspeed"] = cm(torch.relu((t[1:] - t[:-1]).norm(dim=-1) * 50.0 - hcap)[hm] ** 2, cid[1:][hm])
        e["vel"] = cm((torch.relu((q[1:] - q[:-1]).abs() * 50.0 - vmax)[same1] ** 2).sum(-1), cid1)
        W = torch.stack([P[b] for b in range(len(P))], 1)[:, Bt] + torch.einsum("tnij,nj->tni", torch.stack(Q, 1)[:, Bt], Pt_)
        gap = W[..., 2] - surf.req(W[..., :2], Rt_, cid[:, None], const=gidx.get("pen_c"))
        e["pen"] = cm((torch.relu(marg - gap) ** 2).sum(-1), cid)
        e["self"] = 0.0
        for (a_, b_), s0 in zip(pairs, self0):
            Xa, ra = caps(P, Q, a_)
            Xb, rb = caps(P, Q, b_)
            dd = torch.cdist(Xa, Xb) - ra[None, :, None] - rb[None, None]
            e["self"] = e["self"] + cm((torch.relu(s0.clamp(max=0.0) - dd) ** 2).sum((-1, -2)), cid)
        return e

    def losses(q, t, P, Q, pin_only=False):
        if c["clip_norm"]:
            return losses_clip(q, t, P, Q, pin_only)
        e = {}
        mk, rk = gidx.get("mask", mask), gidx.get("rollm", rollm)
        s1k, hk = gidx.get("same1", same1), gidx.get("hm", None)
        npin = gidx["npin"] if "npin" in gidx else max(1, sum(int(mask[f].sum()) for f in FEET))
        e["pin"] = sum((((P[fb[f]] - tgt_p[f]) ** 2).sum(-1)[mk[f]]).sum() for f in FEET) / npin
        e["rot"] = sum((((Q[fb[f]] - tgt_R[f]) ** 2).sum((-1, -2))[mk[f]]).sum() for f in FEET) / npin
        if pin_only:
            return e
        rl = [((P[fb[f]] + (Q[fb[f]] @ roll_pt[f][..., None])[..., 0] - roll_p[f]) ** 2).sum(-1)[rk[f]] for f in FEET]
        e["roll"] = sum(x.sum() for x in rl) / max(1, sum(len(x) for x in rl))
        e["prior"] = (wj * (q - q0) ** 2).sum(-1).mean() + c["w_root"] * (dp ** 2).sum(-1).mean() + c["w_rrot"] * (dr ** 2).sum(-1).mean()
        e["smooth"] = (d2(torch.cat([q - q0, dp, dr], 1)) ** 2).sum(-1).mean()
        e["jump"] = (torch.relu((t[1:] - t[:-1]).norm(dim=-1) - c["jump"])[s1k] ** 2).mean()
        if c["w_hspeed"] > 0:
            e["hspeed"] = (torch.relu((t[1:] - t[:-1]).norm(dim=-1) * 50.0 - hcap)[same1 & torch.isfinite(hcap) if hk is None else hk] ** 2).mean()
        e["vel"] = (torch.relu((q[1:] - q[:-1]).abs() * 50.0 - vmax)[s1k] ** 2).sum(-1).mean()
        W = torch.stack([P[b] for b in range(len(P))], 1)[:, Bt] + torch.einsum("tnij,nj->tni", torch.stack(Q, 1)[:, Bt], Pt_)
        if "pen_S" in gidx:  # prefilter: req on the kept columns only; the excluded ones are exact zeros (checked: inactive)
            S_ = gidx["pen_S"]
            WS = W[:, S_]
            gS = WS[..., 2] - surf.req_cols(WS[..., :2], surf._slot(W[..., :2], gidx["pen_c"][1])[:, S_], gidx["pen_cS"])
            e["pen"] = torch.zeros_like(W[..., 2]).index_copy(1, S_, torch.relu(gidx["margS"] - gS) ** 2).sum(-1).mean()
            if "run_min" in gidx:  # the excluded points' gap lower bound at this iteration (no grad; a static device buffer)
                with torch.no_grad():
                    Wx = W.detach()[:, gidx["pen_Sx"]]
                    lb = Wx[..., 2] - surf.hmax(Wx[..., :2], gidx["cidSx"]) - gidx["rSx"]
                    gidx["run_min"].copy_(torch.minimum(gidx["run_min"], lb.min()))
        else:
            gap = W[..., 2] - surf.req(W[..., :2], Rt_, cid[:, None], const=gidx.get("pen_c"))
            e["pen"] = (torch.relu(marg - gap) ** 2).sum(-1).mean()
        if gnd is not None:
            hh = {}
            for f in FEET:
                Wf = W[:, gnd["cols"][f]]
                hh[f] = (Wf[..., 2] - Rt_[gnd["cols"][f]] - surf._grid(surf.g[0], Wf[..., :2].detach(), cid[:, None].expand(Wf.shape[:2]))).min(-1).values
            e["ground"] = (sum((torch.relu(hh[f] - c["lift"]) ** 2 * gnd["m"][f]).sum() for f in FEET)
                           + (torch.relu(torch.minimum(hh[FEET[0]], hh[FEET[1]]) - c["lift"]) ** 2 * gnd["any"]).sum()) / gnd["n"]
        e["self"] = 0.0
        for (a_, b_), s0 in zip(pairs, self0):
            Xa, ra = caps(P, Q, a_)
            Xb, rb = caps(P, Q, b_)
            dd = torch.cdist(Xa, Xb) - ra[None, :, None] - rb[None, None]
            e["self"] = e["self"] + (torch.relu(s0.clamp(max=0.0) - dd) ** 2).sum((-1, -2)).mean()  # no deeper than the input
        return e

    W_ = dict(pin=c["w_pin"], rot=c["w_rot"], roll=c["w_roll"], prior=1.0, smooth=c["w_smooth"], jump=c["w_jump"], vel=c["w_vel"],
              pen=c["w_pen"], self=c["w_self"], hspeed=c["w_hspeed"])
    if gnd is not None:
        W_["ground"] = c["w_ground"]
    lrs = [c["lr"], c["lr"] * c["lr_root"], c["lr"] * c["lr_root"]]
    opt = torch.optim.Adam([{"params": [zq]}, {"params": [dp]}, {"params": [dr]}], lr=c["lr"])
    t0 = time.time()
    graph = os.environ.get("PIN_GRAPH", "1") != "0" and str(dev).startswith("cuda") and float(c["ramp"]) == 1.0 and not c["clip_norm"]
    pen_min = [float("inf")]
    if graph:  # static index tensors (after the setup edits the masks), then graph replays + eager Adam steps (fab.retarget.cuda_graph)
        gidx.update(mask={f: mask[f].nonzero()[:, 0] for f in FEET}, rollm={f: rollm[f].nonzero()[:, 0] for f in FEET},
                    same1=same1.nonzero()[:, 0], same2=same2.nonzero()[:, 0], npin=max(1, sum(int(mask[f].sum()) for f in FEET)),
                    hm=(same1 & torch.isfinite(hcap)).nonzero()[:, 0],
                    pen_c=surf.req_const(Rt_, cid[:, None], (T, len(Bc)), torch.float32, torch.device(dev)))

        def gap_full(zq_, dp_, dr_):  # every collision point's gap at these variables (the original pen term's gap)
            with torch.no_grad():
                _, _, _, P_, Q_ = forward(zq_, dp_, dr_)
                W_ = torch.stack([P_[b] for b in range(len(P_))], 1)[:, Bt] + torch.einsum("tnij,nj->tni", torch.stack(Q_, 1)[:, Bt], Pt_)
                return W_[..., 2] - surf.req(W_[..., :2], Rt_, cid[:, None], const=gidx["pen_c"])

        if prefilter:  # pen prefilter: columns (points) whose start gap BOUND (the tracked quantity) is >= PEN_KEEP_M in every
            # frame are left out of req
            surf.bound_setup(float(Rt_.max()))
            with torch.no_grad():
                _, _, _, P_, Q_ = forward(zq, dp, dr)
                Wb_ = torch.stack([P_[b] for b in range(len(P_))], 1)[:, Bt] + torch.einsum("tnij,nj->tni", torch.stack(Q_, 1)[:, Bt], Pt_)
                lb0 = Wb_[..., 2] - surf.hmax(Wb_[..., :2], cid[:, None].expand(T, len(Bc))) - Rt_
            keep = Ft_.bool() | (lb0.min(0).values < PEN_KEEP_M)
            if not bool(keep.all()):
                S_, Sx = keep.nonzero()[:, 0], (~keep).nonzero()[:, 0]
                gidx.update(pen_S=S_, pen_Sx=Sx, margS=marg[S_], pen_cS=tuple(x[:, S_] for x in gidx["pen_c"]),
                            cidSx=cid[:, None].expand(T, len(Sx)).contiguous(), rSx=Rt_[Sx],
                            run_min=torch.full((), float("inf"), device=dev))

        def set_lr(it):
            for g, l0 in zip(opt.param_groups, lrs):
                g["lr"] = l0 * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * it / c["iters"])))

        def loss_main():
            q, R, t, P, Q = forward(zq, dp, dr)
            e = losses(q, t, P, Q)
            ramp = c["ramp"] ** (0 / max(1, c["iters"] - 1) - 1.0)  # c["ramp"] == 1: 1.0 at every iteration
            return sum(W_[k] * v * (ramp if k in ("pin", "rot", "roll") else 1.0) for k, v in e.items())

        graph_steps([zq, dp, dr], opt, loss_main, c["iters"], pre=set_lr)
        if "run_min" in gidx:
            pen_min[0] = float(gidx.pop("run_min"))  # main loop done: the final eval below does not track it
    for it in range(0 if graph else c["iters"]):
        for g, l0 in zip(opt.param_groups, lrs):
            g["lr"] = l0 * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * it / c["iters"])))
        q, R, t, P, Q = forward(zq, dp, dr)
        e = losses(q, t, P, Q)
        ramp = c["ramp"] ** (it / max(1, c["iters"] - 1) - 1.0)  # the pin weights grow to their value (ramp > 1)
        loss = sum(W_[k] * v * (ramp if k in ("pin", "rot", "roll") else 1.0) for k, v in e.items())
        opt.zero_grad()
        loss.backward()
        opt.step()
    # polish, the legs only: "pin" the pin alone (exact; can snap per frame), "pin_smooth" + the polish's own change kept
    # smooth in time, "smooth" every term
    zl = zq.detach().clone().requires_grad_(True)
    opt2 = torch.optim.Adam([zl], lr=0.003)
    dp_, dr_ = dp.detach(), dr.detach()
    with torch.no_grad():
        q1 = lo + (hi - lo) * torch.sigmoid(zq)
    def loss_polish():
        zz = torch.where(leg, zl, zq.detach())
        q, R, t, P, Q = forward(zz, dp_, dr_)
        if c["polish_mode"] == "smooth":
            e = losses(q, t, P, Q)
            loss = sum(W_[k] * v for k, v in e.items())
        else:
            e = losses(q, t, P, Q, pin_only=True)
            loss = 1e3 * e["pin"] + 10 * e["rot"]
            if c["polish_mode"] == "pin_smooth":
                loss = loss + c["w_ps"] * (cm((d2(q - q1) ** 2).sum(-1), cid2) if c["clip_norm"] else (d2(q - q1) ** 2).sum(-1).mean())
        return loss

    if graph:
        graph_steps([zl], opt2, loss_polish, c["polish"])
    for it in range(0 if graph else c["polish"]):
        zz = torch.where(leg, zl, zq.detach())
        q, R, t, P, Q = forward(zz, dp_, dr_)
        if c["polish_mode"] == "smooth":
            e = losses(q, t, P, Q)
            loss = sum(W_[k] * v for k, v in e.items())
        else:
            e = losses(q, t, P, Q, pin_only=True)
            loss = 1e3 * e["pin"] + 10 * e["rot"]
            if c["polish_mode"] == "pin_smooth":
                loss = loss + c["w_ps"] * (cm((d2(q - q1) ** 2).sum(-1), cid2) if c["clip_norm"] else (d2(q - q1) ** 2).sum(-1).mean())
        opt2.zero_grad()
        loss.backward()
        opt2.step()
    if "pen_Sx" in gidx:  # the prefilter is exact only while every excluded point stays inactive: else redo without it
        pen_end = float(gap_full(torch.where(leg, zl, zq).detach(), dp_, dr_)[:, gidx["pen_Sx"]].min())
        if min(pen_min[0], pen_end) < PEN_SAFE_M:
            return None
    with torch.no_grad():
        zz = torch.where(leg, zl, zq)
        q, R, t, P, Q = forward(zz, dp_, dr_)
        e = {k: float(v) for k, v in losses(q, t, P, Q).items()}
        perr = {f: (P[fb[f]] - tgt_p[f]).norm(dim=-1) * mask[f] for f in FEET}
    dt = time.time() - t0
    quat = Rotation.from_matrix(R.cpu().numpy().astype(np.float64)).as_quat()
    tq, qq, dpn = t.cpu().numpy().astype(np.float64), q.cpu().numpy().astype(np.float64), dp_.norm(dim=-1).cpu().numpy()
    dq = (q - q0).abs().cpu().numpy()
    legn = leg.cpu().numpy()
    out = []
    for ci in range(len(items)):
        sl = slice(off[ci], off[ci + 1])
        rep = reps[ci]
        rep.update(terms_batch=e, pin_err_max_mm={f: round(1000 * float(perr[f][sl].max()), 2) for f in FEET}, solve_s=round(dt, 1),
                   batch=len(items), root_shift_max_cm=round(float(dpn[sl].max()) * 100, 2),
                   pen_prefilter={"kept": int(len(gidx["pen_S"])), "of": int(len(Bc)), "keep_m": PEN_KEEP_M,
                                  "min_excluded_gap_bound_m": round(pen_min[0], 4), "min_excluded_gap_end_m": round(pen_end, 4)}
                   if "pen_Sx" in gidx else "off",
                   dq_leg_max_rad=round(float(dq[sl][:, legn].max()), 3), dq_other_max_rad=round(float(dq[sl][:, ~legn].max()), 3), cfg=c)
        st = np.stack([cores[ci]["left"]["core"], cores[ci]["right"]["core"]], 1)
        rl = np.stack([cores[ci]["left"]["roll"], cores[ci]["right"]["roll"]], 1)
        out.append((tq[sl], quat[sl], qq[sl], st, rl, rep))
    return out


def gate_ds(mo, tg, stance, **kw):
    """strict_gate (bench/g1_contact_check) on a reference in the motion format: a temporary clip dir with
    refs/gate_ref.npz (joint_pos in the gate model's qpos order) and terrain/terrain_hfield_2cm.npz (the package's 2 cm
    resample of terrain_g1; the slot's floor shift makes the package dz irrelevant)."""
    import tempfile

    import mujoco

    from bench.g1_contact_check import strict_gate

    m = mujoco.MjModel.from_xml_path(str(GATE_XML))
    names = [str(n) for n in mo["g1_joint_names"]]
    T = len(mo["g1_dof"])
    jp = np.zeros((T, m.nq))
    jp[:, :3] = mo["g1_root_pos"]
    jp[:, 3:7] = np.asarray(mo["g1_root_quat_xyzw"])[:, [3, 0, 1, 2]]
    for k, n in enumerate(names):
        jp[:, m.jnt_qposadr[m.joint(n).id]] = np.asarray(mo["g1_dof"])[:, k]
    H = np.asarray(tg["height"], np.float64)
    res = float(tg["res"])
    ny, nx = H.shape
    r2 = 0.02 if res > 0.03 else res
    iy = np.clip(np.round(np.arange(int(np.floor((ny - 1) * res / r2)) + 1) * r2 / res).astype(int), 0, ny - 1)
    ix = np.clip(np.round(np.arange(int(np.floor((nx - 1) * res / r2)) + 1) * r2 / res).astype(int), 0, nx - 1)
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "refs").mkdir()
        (Path(td) / "terrain").mkdir()
        np.savez(Path(td) / "refs/gate_ref.npz", joint_pos=jp)
        np.savez(Path(td) / "terrain/terrain_hfield_2cm.npz", x0=tg["x0"], y0=tg["y0"], res=np.float64(r2),
                 height=H[np.ix_(iy, ix)].astype(np.float32))
        return strict_gate(td, stance=np.asarray(stance, bool), **kw)


def contact_file(a, clip, recon_npz, mo=None):
    """The clip's contact-head file, or ground-truth kinematic labels in its format (static as a probability); a clip
    without a head file falls back to the labels its solve used (the motion's contacts, per recon frame)."""
    fr = np.asarray(np.load(recon_npz, allow_pickle=True)["frames"], int)
    if a.contacts_kin_dir:  # static (< 10 cm and < 0.6 m/s) AND any (< 4 cm): a slow foot hovering 5-9 cm over
        # the floor (MOYO) is not pinned onto it
        kz = np.load(Path(a.contacts_kin_dir) / f"{clip}.npz", allow_pickle=True)
        return {"frame": fr, "video_static_prob": (np.asarray(kz["static"], bool) & np.asarray(kz["any"], bool)).astype(np.float64)}
    base = clip
    for suf in ("_w06", "_w20", "_full"):
        base = base[: -len(suf)] if base.endswith(suf) else base
    cf = Path(a.contacts_dir) / f"{base}__0.contact.npz"
    if cf.exists() or mo is None:
        return dict(np.load(cf))
    lab, rows = json.loads(str(mo["contacts"])), np.asarray(mo["frames"], int)
    rr = np.unique(rows)
    P = np.zeros((len(rr), 4))
    for g, cols in (("lfoot", [0, 1]), ("rfoot", [2, 3])):
        v = np.r_[np.asarray(lab.get(g, []), float), np.zeros(len(rows))][: len(rows)]
        P[:, cols] = np.array([v[rows == r].mean() for r in rr])[:, None]
    return {"frame": fr[np.clip(rr, 0, len(fr) - 1)], "video_static_prob": P, "fallback": "solve labels"}


class _NpzLike(dict):
    @property
    def files(self):
        return list(self.keys())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--tag", default="solve")
    p.add_argument("--out_tag", default="p")
    p.add_argument("--recon", required=True)
    p.add_argument("--contacts_dir", default="", help="dir of <clip>__0.contact.npz contact-head files (empty: the solve's labels)")
    p.add_argument("--contacts_kin_dir", default="")
    p.add_argument("--only", default="", help="comma list or @file")
    p.add_argument("--cfg", default="")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--gate", action="store_true", help="also score the input and the output with strict_gate (into pin_report)")
    p.add_argument("--batch_frames", type=int, default=4000, help="clips solved together up to this many frames")
    p.add_argument("--terrain_clean", default="", help="a clean terrain (<dir>/<clip>/terrain_hfield_2cm_clean.npz, "
                   "package frame): pin on it, - the package dz; written as <clip>_terrain_clean_g1.npz / .obj")
    p.add_argument("--pkg", default="", help="the packages' clips dir (<pkg>/<clip>/package.json dz_m), with --terrain_clean")
    p.add_argument("--reverse", action="store_true", help="walk the clips last to first (a second worker meets the first in the middle)")
    p.add_argument("--terrain_file", default="terrain_hfield_2cm_clean.npz", help="the file name under <terrain_clean>/<clip>/")
    a = p.parse_args()
    from bench.g1_contact_check import stance_runs

    run = Path(a.run)
    out = run / a.out_tag
    out.mkdir(parents=True, exist_ok=True)
    only = (Path(a.only[1:]).read_text().split() if a.only.startswith("@") else a.only.split(",")) if a.only else []
    files = sorted(f for f in (run / a.tag).glob("*_motion.npz") if not only or f.name[: -len("_motion.npz")] in only)
    todo = [f for f in files[a.shard:: a.num_shards] if a.force or not (out / f.name).exists()]
    todo = todo[::-1] if a.reverse else todo
    cfg = json.loads(a.cfg or "{}")
    batch = []

    def flush(batch):
        try:
            res = pin_batch([(mo, tg, st) for _, mo, tg, st in batch], cfg=cfg)
            c_ = dict(CFG, **cfg)
            redo = [k for k, r in enumerate(res) if c_["retry_mm"] > 0 and max(r[5]["pin_err_max_mm"].values()) > c_["retry_mm"]]
            if redo:  # convergence retry (one constant rule): the clips whose pin is still off get retry_iters x the iterations
                cfg2 = dict(cfg, iters=int(c_["iters"] * c_["retry_iters"]), polish=int(c_["polish"] * c_["retry_iters"]))
                res2 = pin_batch([(batch[k][1], batch[k][2], batch[k][3]) for k in redo], cfg=cfg2)
                for k, r2 in zip(redo, res2):
                    r2[5]["retry"] = {"pin_err_max_mm_first": res[k][5]["pin_err_max_mm"]}
                    res[k] = r2
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            for f, *_ in batch:
                print(f.name[: -len("_motion.npz")], "FAIL", repr(e)[:300], flush=True)
            return
        for (f, mo, tg, _), (pos, quat, dof, ss, rl, rep) in zip(batch, res):
            clip = f.name[: -len("_motion.npz")]
            if a.gate:
                import inspect

                from bench.g1_contact_check import strict_gate

                pm = dict(mo, g1_root_pos=pos, g1_root_quat_xyzw=quat, g1_dof=dof)
                rep["gate_before"], rep["gate_after"] = gate_ds(mo, tg, ss), gate_ds(pm, tg, ss)
                if "fix_mirror" in inspect.signature(strict_gate).parameters:  # the heightfield rows put in MJCF order
                    rep["gate_before_fixed"], rep["gate_after_fixed"] = gate_ds(mo, tg, ss, fix_mirror=True), gate_ds(pm, tg, ss, fix_mirror=True)
            if "_clean" in tg:
                rep["terrain"] = {"source": "clean terrain (terrain_hfield_2cm_clean) - package dz", "file": tg["_clean"], "dz_m": tg["_dz"]}
            lab = json.loads(str(mo["contacts"]))
            lab.update(lfoot=ss[:, 0].astype(int).tolist(), rfoot=ss[:, 1].astype(int).tolist())
            mo = dict(mo, g1_root_pos=pos, g1_root_quat_xyzw=quat, g1_dof=dof, contacts=json.dumps(lab), pin_roll=rl,
                      pin_report=json.dumps(rep, default=float), pin_source=f"{a.tag} + stance_pin")
            mo.pop("_hspeed", None)
            of = out / f.name
            np.savez(str(of)[:-4] + ".tmp.npz", **mo)
            os.replace(str(of)[:-4] + ".tmp.npz", of)
            for suf in ("_terrain_g1.npz", "_terrain_g1.obj", "_scene_g1.obj"):
                src, dst = f.with_name(f"{clip}{suf}"), out / f"{clip}{suf}"
                if src.exists() and not dst.exists():
                    dst.symlink_to(src.resolve())
            if "_clean" in tg:  # the terrain it was pinned on, in the run frame (audit, port mesh)
                from bench.scene_r2s_simpkg import terrain_mesh

                tc = {k: tg[k] for k in ("x0", "y0", "res", "height")}
                np.savez(out / f"{clip}_terrain_clean_g1.npz", **tc)
                terrain_mesh(tc).export(out / f"{clip}_terrain_clean_g1.obj")
            g = {k: {m_: round(v[m_], 4) for m_ in ("slide", "feet_pen", "body_pen", "root_step")} for k, v in rep.items() if k.startswith("gate_")}
            print(clip, "ok", json.dumps({k: rep.get(k) for k in ("runs", "pin_err_max_mm", "root_shift_max_cm", "dq_leg_max_rad", "solve_s", "batch", "retry")}),
                  json.dumps(g), flush=True)

    for f in todo:
        clip = f.name[: -len("_motion.npz")]
        if not a.force and (out / f.name).exists():  # pinned meanwhile by another process
            continue
        try:
            mo = dict(np.load(f, allow_pickle=True))
            tg = dict(np.load(f.with_name(f"{clip}_terrain_g1.npz"), allow_pickle=True))
            if a.terrain_clean:
                cf = Path(a.terrain_clean) / clip / a.terrain_file
                cz = np.load(cf)
                dz = float(json.loads((Path(a.pkg) / clip / "package.json").read_text())["dz_m"])
                tg = {"x0": float(cz["x0"]), "y0": float(cz["y0"]), "res": float(cz["res"]),
                      "height": np.asarray(cz["height"], np.float64) - dz, "_clean": str(cf), "_dz": dz}
            rnpz = Path(a.recon) / f"{clip}.npz"
            st = stance_runs(_NpzLike(contact_file(a, clip, rnpz, mo)), rows=np.asarray(mo["frames"], int), recon=str(rnpz), fps=float(mo.get("fps", 50.0)))
            rz = np.load(rnpz, allow_pickle=True)
            hv = np.r_[0.0, np.linalg.norm(np.diff(np.asarray(rz["joints"], np.float64)[:, 0], axis=0), axis=1)] * float(rz["fps"])
            hv = np.maximum.reduce([np.roll(hv, k) for k in (-1, 0, 1)])  # the larger of the neighbouring recon frames
            mo["_hspeed"] = hv[np.clip(np.asarray(mo["frames"], int), 0, len(hv) - 1)]
        except Exception as e:  # noqa: BLE001
            print(clip, "FAIL", repr(e)[:300], flush=True)
            continue
        batch.append((f, mo, tg, st))
        if sum(len(x[1]["g1_dof"]) for x in batch) >= a.batch_frames:
            flush(batch)
            batch = []
    if batch:
        flush(batch)


if __name__ == "__main__":
    main()
