"""Scene-contact quality of a G1 motion against its own scene (GT-free; a reference before tracking or a tracker
rollout), as a heightfield (bench/videomimic/recon.py *_hfield.npz, or a mesh rasterised by mesh_heightfield):
  pen      body penetration: per frame the deepest point of any G1 collision geom below the terrain (spheres / capsules:
           lowest surface point; meshes: vertices) -> frames deeper than 2 cm (%), mean / p95 depth (mm)
  contact  sole gap: while a foot is labelled in contact (the motion's own labels, else: the sole within 3 cm of the
           terrain and slower than 0.3 m/s), |lowest sole point - terrain| (mm)
  skate    horizontal speed of a foot (centre of its sole points) in contact (m/s)
  float    frames where the human (labels) has a foot contact but the robot's lowest sole point is > 5 cm above (%)
  jitter   mean |d2 q / dt2| (rad/s^2) on a common 30 Hz grid (linear resampling)
Robot: the 29-dof G1 MJCF of the retargeter ($FAB_HOME/g1/g1_29dof.xml, scripts/download_g1.sh); 23-dof motions get wrists 0.

    python -m bench.videomimic.scene_metrics --motion <name>_motion.npz --hfield <name>_hfield.npz [--scale s]
"""

import argparse
import json

import numpy as np

from fab.retarget import g1_files

MJCF = str(g1_files.mjcf())
SOLES = {"lfoot": [f"left_sole_{i}" for i in range(1, 5)] + ["left_toe"],
         "rfoot": [f"right_sole_{i}" for i in range(1, 5)] + ["right_toe"]}


def mesh_heightfield(mesh_path, xy, z_ref, res=0.04, margin=1.5, above=0.1):
    """A mesh (e.g. VideoMimic's background_mesh.obj) as a heightfield over the path xy (T, 2): per cell the highest
    surface below the nearest path sample's z_ref (pelvis) + above (bench/videomimic/recon.heightfield)."""
    import open3d as o3d

    from bench.videomimic.recon import heightfield

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    return heightfield(mesh, np.asarray(xy), np.asarray(z_ref), res, margin, above)


class HF:
    """Bilinear heightfield lookup. Multi-level terrains (`levels` (ny, nx, L), -inf
    padded) answer per query height z: each cell's highest level <= z + 0.3 m (the surface a point at z stands on);
    without z, or without levels, the single `height` grid."""

    def __init__(self, hf, scale=1.0):
        self.x0, self.y0, self.res = float(hf["x0"]) * scale, float(hf["y0"]) * scale, float(hf["res"]) * scale
        self.H = np.asarray(hf["height"], np.float64) * scale
        L = np.asarray(hf["levels"], np.float64) if "levels" in getattr(hf, "files", hf) else None
        self.L = L * scale if L is not None and L.ndim == 3 else None  # simready `levels` are 1-D (stair levels)

    def __call__(self, xy, z=None):
        ny, nx = self.H.shape
        u = np.clip((xy[..., 0] - self.x0) / self.res, 0, nx - 1.001)
        v = np.clip((xy[..., 1] - self.y0) / self.res, 0, ny - 1.001)
        i, j = np.floor(v).astype(int), np.floor(u).astype(int)
        a, b = v - i, u - j
        if self.L is not None and z is not None:
            zq = np.asarray(z, np.float64)[..., None] + 0.3
            pick = lambda ii, jj: np.where(self.L[ii, jj] <= zq, self.L[ii, jj], -np.inf).max(-1)  # noqa: E731
            c = [pick(i, j), pick(i, j + 1), pick(i + 1, j), pick(i + 1, j + 1)]
            c = [np.where(np.isfinite(x), x, -10.0) for x in c]
            return (1 - a) * (1 - b) * c[0] + (1 - a) * b * c[1] + a * (1 - b) * c[2] + a * b * c[3]
        H = self.H
        return (1 - a) * (1 - b) * H[i, j] + (1 - a) * b * H[i, j + 1] + a * (1 - b) * H[i + 1, j] + a * b * H[i + 1, j + 1]


class Robot:
    def __init__(self, mjcf=MJCF):
        import mujoco

        self.m = mujoco.MjModel.from_xml_path(mjcf)
        self.d = mujoco.MjData(self.m)
        self.geoms = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] != 0 and
                      (self.m.geom_contype[g] or self.m.geom_conaffinity[g])]
        if not self.geoms:  # visual-only models: every non-world geom
            self.geoms = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] != 0]
        self.mesh_v = {}
        for g in self.geoms:
            if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = self.m.geom_dataid[g]
                v = self.m.mesh_vert[self.m.mesh_vertadr[mid]: self.m.mesh_vertadr[mid] + self.m.mesh_vertnum[mid]]
                self.mesh_v[g] = v[:: max(1, len(v) // 64)].astype(np.float64)
        self.sole = {k: [self.m.body(n).id for n in v] for k, v in SOLES.items()}
        self.sole_r = {k: float(self.m.geom_size[np.flatnonzero(self.m.geom_bodyid == b[0])[0]][0]) for k, b in self.sole.items()}

    def points(self):
        """World points per geom approximating its lowest surface (current mj_data)."""
        import mujoco

        out = []
        for g in self.geoms:
            p, R, t = self.d.geom_xpos[g], self.d.geom_xmat[g].reshape(3, 3), self.m.geom_type[g]
            s = self.m.geom_size[g]
            if t == mujoco.mjtGeom.mjGEOM_SPHERE:
                out.append(p[None] - [0, 0, s[0]])
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                ax = R[:, 2] * s[1]
                out.append(np.stack([p + ax, p - ax, p]) - [0, 0, s[0]])
            elif t == mujoco.mjtGeom.mjGEOM_MESH:
                out.append(self.mesh_v[g] @ R.T + p)
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                c = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s
                out.append(c @ R.T + p)
            else:
                out.append(p[None] - [0, 0, self.m.geom_rbound[g]])
        return np.concatenate(out)


def metrics(motion, terr, contacts=None, robot=None, fps=None):
    """motion: dict with g1_dof (T, n), g1_joint_names, g1_root_pos (T, 3), g1_root_quat_xyzw (T, 4) [, fps];
    terr: HF (same world); contacts: {lfoot / rfoot: (T,) bool} of the human (at the motion's rate) or None."""
    import mujoco

    robot = robot or Robot()
    m, d = robot.m, robot.d
    jn = [str(j) for j in motion["g1_joint_names"]]
    adr = [m.joint(n).qposadr[0] for n in jn]
    q, pos, quat = (np.asarray(motion[k], np.float64) for k in ("g1_dof", "g1_root_pos", "g1_root_quat_xyzw"))
    fps = float(fps or motion.get("fps", 50.0))
    T = len(q)
    pen = np.zeros(T)
    sole = {k: np.zeros((T, 3)) for k in SOLES}
    for t in range(T):
        d.qpos[:] = 0
        d.qpos[:3], d.qpos[3:7] = pos[t], quat[t][[3, 0, 1, 2]]
        d.qpos[adr] = q[t]
        mujoco.mj_kinematics(m, d)
        P = robot.points()
        pen[t] = max(0.0, float((terr(P[:, :2], P[:, 2]) - P[:, 2]).max()))
        for k, bs in robot.sole.items():
            sp = d.xpos[bs] - [0, 0, robot.sole_r[k]]
            gap = sp[:, 2] - terr(sp[:, :2], sp[:, 2])
            sole[k][t] = [*sp[:, :2].mean(0), gap.min()]  # foot centre (no heel / toe switching), lowest gap
    out = {"frames": T, "pen_frames_2cm_pct": float((pen > 0.02).mean() * 100), "pen_mean_mm": float(pen.mean() * 1000),
           "pen_p95_mm": float(np.percentile(pen, 95) * 1000)}
    gaps, skates, floats = [], [], []
    for k in SOLES:
        g = sole[k][:, 2]
        v = np.r_[0.0, np.linalg.norm(np.diff(sole[k][:, :2], axis=0), axis=-1) * fps]
        c = np.asarray(contacts[k], bool)[:T] if contacts is not None and k in contacts else (np.abs(g) < 0.03) & (v < 0.3)
        if len(c) < T:
            c = np.r_[c, np.zeros(T - len(c), bool)]
        gaps.append(np.abs(g[c]))
        skates.append(v[c])
        if contacts is not None:
            floats.append(g[c] > 0.05)
    out["contact_gap_mm"] = float(np.concatenate(gaps).mean() * 1000) if sum(map(len, gaps)) else float("nan")
    out["skate_m_s"] = float(np.concatenate(skates).mean()) if sum(map(len, skates)) else float("nan")
    out["float_pct"] = float(np.concatenate(floats).mean() * 100) if floats and sum(map(len, floats)) else float("nan")
    out["contact_frames"] = int(sum(map(len, gaps)))
    # label-free skate: foot speed while its lowest sole point is within 2 cm of the terrain
    near = [np.r_[0.0, np.linalg.norm(np.diff(sole[k][:, :2], axis=0), axis=-1) * fps][np.abs(sole[k][:, 2]) < 0.02]
            for k in SOLES]
    out["skate_near_m_s"] = float(np.concatenate(near).mean()) if sum(map(len, near)) else float("nan")
    # jitter on a common 30 Hz grid (finite differences of noisy motions grow with the frame rate)
    tq = np.arange(0, (T - 1) / fps + 1e-9, 1 / 30.0) * fps
    i0 = np.clip(np.floor(tq).astype(int), 0, T - 1)
    a = (tq - i0)[:, None]
    q30 = (1 - a) * q[i0] + a * q[np.clip(i0 + 1, 0, T - 1)]
    out["jitter_rad_s2"] = float(np.abs(np.diff(q30, 2, axis=0)).mean() * 30.0 ** 2) if len(q30) > 2 else float("nan")
    nw = [i for i, n in enumerate(jn) if "wrist" not in n]  # the 23 joints VideoMimic's G1 has (wrists fixed)
    out["jitter23_rad_s2"] = (float(np.abs(np.diff(q30[:, nw], 2, axis=0)).mean() * 30.0 ** 2) if len(q30) > 2
                              else float("nan"))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion", required=True)
    p.add_argument("--hfield", default="")
    p.add_argument("--mesh", default="")
    p.add_argument("--scale", type=float, default=1.0, help="multiply the heightfield (x, y, z) by this (G1 world)")
    args = p.parse_args()
    z = dict(np.load(args.motion, allow_pickle=True))
    if args.hfield:
        terr = HF(np.load(args.hfield), args.scale)
    else:
        terr = HF(mesh_heightfield(args.mesh, z["g1_root_pos"][:, :2], z["g1_root_pos"][:, 2]))
    print(json.dumps(metrics(z, terr), indent=1))


if __name__ == "__main__":
    main()
