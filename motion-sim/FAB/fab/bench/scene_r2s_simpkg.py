"""Simulation scene package per clip: a reconstructed scene ready for a simulator (optionally textured, e.g. to train
perception-based trackers). One directory per clip, in the G1 world of the clip's references (recon world x s)
shifted up by dz so the lowest terrain sits 1 cm above z = 0 (simulators that add their own ground plane at 0):

  scene.xml            MJCF scene file (meant to be attached to a robot world through a frame): one
                       static body "terrain" = the collision terrain (hfield; multi-level terrains: box geoms) with
                       an environment collision class (contype 2, conaffinity 1), the visual mesh(es)
                       (contype 0; textured when a UV-mapped mesh is given), and a camera "input_cam" at the input
                       camera's pose (static cameras: that pose; moving: frame 0, the trajectory in camera.npz)
  terrain/terrain_hfield.npz   the collision heightfield (x0, y0, res, height (ny, nx), row = y [, levels, boxes])
  terrain/terrain_collision.obj  the same as a triangle mesh (Isaac load_obj / trimesh terrains)
  visual/              the visual mesh(es) (+ textures)
  refs/<tag>.npz       a whole-body-tracking motion file (fps, joint_pos (T, 36) = root xyz + quat wxyz + 29 joints,
                       joint_vel (T, 35) = MuJoCo qvel: root linear velocity, angular velocity (root frame), 29 joints, body_pos_w / body_quat_w /
                       body_lin_vel_w / body_ang_vel_w (T, B, .) of the MJCF's B bodies from its forward kinematics,
                       joint_names, body_names)
  refs/<tag>_motion.npz  the retargeter's reference (g1_dof, g1_root_pos, g1_root_quat_xyzw, contacts, frames), shifted
  refs/sim_<tag>[_motion].npz  the terrain policy's MuJoCo rollout of <tag> where it succeeded: a physically valid
                       reference next to the kinematic one (package.json refs: kinematic / physically_valid / success)
  contacts.npz         the contact heads (static / any, heel / toe, hands / knees) on the references' 50 Hz grid
  camera.npz           T_world_cam (N, 4, 4) OpenCV convention, mj_pos / mj_quat_wxyz (MuJoCo camera), K, width /
                       height (K's working resolution), fps, frames (image indices), static
  package.json         provenance, s, dz, the simulator config (scene file, camera sensor, motion files), tracking rows

    python -m bench.scene_r2s_simpkg build --run <run dir> --tags p --recon <recon dir> \
        --clip <clip> --out <dir> [--visual <mesh> --visual_world recon|g1]
    python -m bench.scene_r2s_simpkg check --pkg <dir> [--images clips.json:<clip>] --out <png>
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from fab.retarget import g1_files

MJCF = os.environ.get("G1_WBT_MJCF", str(g1_files.mjcf()))  # the 29-DoF model of the motion files' body states
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])



def hfield_2cm(pk, mode="edge", thr=0.05):
    """terrain/terrain_hfield_2cm.npz: the collision heightfield on a 2 cm grid for consumers that need <= 3 cm; same frame
    and keys. mode edge (edge-aware resample): bilinear where the native 2x2 block spans < thr x s (s = the package scale),
    nearest elsewhere, so slopes stay slopes and risers / walls stay crisp; nearest: 1-2 cell terraces on every slope."""
    hf = dict(np.load(pk / "terrain" / "terrain_hfield.npz"))
    res = float(hf["res"])
    H = np.asarray(hf["height"], np.float64)
    ny, nx = H.shape
    r2 = 0.02 if res > 0.03 else res
    n2y, n2x = int(np.floor((ny - 1) * res / r2)) + 1, int(np.floor((nx - 1) * res / r2)) + 1
    fy, fx = np.arange(n2y) * r2 / res, np.arange(n2x) * r2 / res
    iy = np.clip(np.round(fy).astype(int), 0, ny - 1)
    ix = np.clip(np.round(fx).astype(int), 0, nx - 1)
    H2 = H[np.ix_(iy, ix)]
    if mode == "edge" and r2 != res and ny > 1 and nx > 1:
        s = float(json.loads((pk / "package.json").read_text()).get("scale_s", 1.0))
        y0, x0 = np.clip(np.floor(fy).astype(int), 0, ny - 2), np.clip(np.floor(fx).astype(int), 0, nx - 2)
        ay, ax = np.clip(fy - y0, 0, 1)[:, None], np.clip(fx - x0, 0, 1)[None]
        a, b = H[np.ix_(y0, x0)], H[np.ix_(y0, x0 + 1)]
        c, d = H[np.ix_(y0 + 1, x0)], H[np.ix_(y0 + 1, x0 + 1)]
        bil = (1 - ay) * (1 - ax) * a + (1 - ay) * ax * b + ay * (1 - ax) * c + ay * ax * d
        span = np.maximum.reduce([a, b, c, d]) - np.minimum.reduce([a, b, c, d])
        H2 = np.where(span < thr * s, bil, H2)
    out = {"x0": hf["x0"], "y0": hf["y0"], "res": np.float64(r2), "height": H2.astype(np.float32)}
    if "valid" in hf:
        out["valid"] = np.asarray(hf["valid"])[np.ix_(iy, ix)]
    np.savez(pk / "terrain" / "terrain_hfield_2cm.npz", **out)

def quat_mul(a, b):  # wxyz
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def so3_derivative(q, dt):
    """convert_data_format_mj.MotionLoader._so3_derivative: world angular velocity from wxyz quaternions."""
    qc = q * np.array([1, -1, -1, -1])
    r = quat_mul(q[2:], qc[:-2])
    r = np.where(r[:, :1] < 0, -r, r)
    w = np.clip(r[:, 0], -1, 1)
    ang = 2 * np.arccos(w)
    s = np.sqrt(np.clip(1 - w * w, 0, None))
    axis = np.where(s[:, None] > 1e-8, r[:, 1:] / np.maximum(s[:, None], 1e-12), 0.0)
    om = axis * ang[:, None] / (2 * dt)
    return np.concatenate([om[:1], om, om[-1:]])


def wbt_motion(mo, dz, fps=50.0, local_omega=True):
    """Our reference -> the whole-body-tracking motion npz (body states from the MJCF's forward kinematics; layout above)."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    names = [str(j) for j in mo["g1_joint_names"]]
    jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)][1:]
    col = [names.index(j) for j in jn]
    pos = np.asarray(mo["g1_root_pos"], np.float64) + [0, 0, dz]
    quat = np.asarray(mo["g1_root_quat_xyzw"], np.float64)[:, [3, 0, 1, 2]]
    for k in range(1, len(quat)):  # sign continuity
        if (quat[k] * quat[k - 1]).sum() < 0:
            quat[k] = -quat[k]
    q = np.asarray(mo["g1_dof"], np.float64)[:, col]
    dt = 1.0 / fps
    om = so3_derivative(quat, dt)
    if local_omega:  # MuJoCo's free joint takes the angular velocity in the body frame; a world-frame omega there makes
        # every body velocity wrong (stance feet at ~1-2 m/s): rotated into the root frame
        from scipy.spatial.transform import Rotation

        om = np.einsum("tji,tj->ti", Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix(), om)
    vel = np.concatenate([np.gradient(pos, dt, axis=0), om, np.gradient(q, dt, axis=0)], 1)
    T = len(q)
    out = {k: [] for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
    v6 = np.zeros((m.nbody, 6))
    for t in range(T):
        d.qpos[:] = np.r_[pos[t], quat[t], q[t]]
        d.qvel[:] = vel[t]
        mujoco.mj_forward(m, d)
        for b in range(m.nbody):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v6[b], 0)
        out["joint_pos"].append(d.qpos.copy())
        out["joint_vel"].append(d.qvel.copy())
        out["body_pos_w"].append(d.xpos.copy())
        out["body_quat_w"].append(d.xquat.copy())
        out["body_lin_vel_w"].append(v6[:, 3:6].copy())
        out["body_ang_vel_w"].append(v6[:, 0:3].copy())
    out = {k: np.stack(v) for k, v in out.items()}
    out["fps"] = np.array([int(round(fps))])
    out["joint_names"] = np.array(jn)
    out["body_names"] = np.array([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)])
    return out


def terrain_mesh(tg):
    """The collision terrain as triangles: the heightfield grid, plus the boxes of a multi-level terrain."""
    import trimesh

    H = np.asarray(tg["height"], np.float64)
    ny, nx = H.shape
    x0, y0, res = float(tg["x0"]), float(tg["y0"]), float(tg["res"])
    X, Y = np.meshgrid(x0 + res * np.arange(nx), y0 + res * np.arange(ny))
    v = np.stack([X.ravel(), Y.ravel(), H.ravel()], 1)
    i = np.arange(ny * nx).reshape(ny, nx)
    a, b, c, e = i[:-1, :-1].ravel(), i[:-1, 1:].ravel(), i[1:, :-1].ravel(), i[1:, 1:].ravel()
    parts = [trimesh.Trimesh(v, np.concatenate([np.stack([a, b, e], 1), np.stack([a, e, c], 1)]), process=False)]
    for bx in np.asarray(tg.get("boxes", np.zeros((0, 6))), np.float64):
        box = trimesh.creation.box(extents=2 * np.maximum(bx[3:], 1e-3))
        box.apply_translation(bx[:3])
        parts.append(box)
    return trimesh.util.concatenate(parts)


def visual_meshes(path, s, dz, out_dir):
    """Visual mesh(es) -> OBJ (+ PNG textures) in the package world; one (obj, texture or None) per material."""
    import trimesh

    sc = trimesh.load(path, process=False)
    geoms = list(sc.geometry.values()) if isinstance(sc, trimesh.Scene) else [sc]
    out = []
    for k, g in enumerate(geoms):
        g = g.copy()
        g.vertices = np.asarray(g.vertices, np.float64) * s + [0, 0, dz]
        tex = None
        vis = getattr(g, "visual", None)
        if isinstance(vis, trimesh.visual.TextureVisuals) and vis.uv is not None and getattr(vis.material, "image", None) is not None:
            tex = out_dir / f"visual_{k}.png"
            vis.material.image.save(tex)
        f = out_dir / f"visual_{k}.obj"
        g.export(f, include_texture=False) if tex is None else g.export(f)
        for extra in out_dir.glob("material*"):  # trimesh writes its own MTL / texture copies: the PNG above is used
            extra.unlink()
        out.append((f, tex))
    return out


def scene_xml(tg, vis, cam, out):
    """The MJCF scene file (the terrain body, its visual meshes and the input camera)."""
    import xml.etree.ElementTree as ET

    H = np.asarray(tg["height"], np.float64)
    ny, nx = H.shape
    res, x0, y0 = float(tg["res"]), float(tg["x0"]), float(tg["y0"])
    zmin, zmax = float(H.min()), float(H.max())
    sz = max(zmax - zmin, 1e-3)
    root = ET.Element("mujoco", model=out.parent.name)
    ET.SubElement(root, "compiler", angle="radian", meshdir=".", texturedir=".")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "hfield", name="terrain_hf", nrow=str(ny), ncol=str(nx),
                  size=f"{(nx - 1) * res / 2:.6f} {(ny - 1) * res / 2:.6f} {sz:.6f} 1.0",
                  elevation=" ".join(f"{x:.5f}" for x in ((H[::-1] - zmin) / sz).ravel()))  # MJCF: first row = +y
    ET.SubElement(asset, "material", name="terrain_mat", rgba="0.55 0.55 0.52 1")
    wb = ET.SubElement(root, "worldbody")
    body = ET.SubElement(wb, "body", name="terrain")
    phys = dict(contype="2", conaffinity="1", friction="0.7 0.005 0.001", solimp="0.99 0.99 0.01 0.5 2", solref="0.001 1")
    ET.SubElement(body, "geom", name="terrain_hf", type="hfield", hfield="terrain_hf", material="terrain_mat",
                  pos=f"{x0 + (nx - 1) * res / 2:.6f} {y0 + (ny - 1) * res / 2:.6f} {zmin:.6f}",
                  rgba="0.55 0.55 0.52 0" if vis else "0.55 0.55 0.52 1", **phys)
    for k, bx in enumerate(np.asarray(tg.get("boxes", np.zeros((0, 6))), np.float64)):
        ET.SubElement(body, "geom", name=f"terrain_box_{k}", type="box", pos=" ".join(f"{x:.5f}" for x in bx[:3]),
                      size=" ".join(f"{x:.5f}" for x in np.maximum(bx[3:], 1e-3)), material="terrain_mat",
                      rgba="0.55 0.55 0.52 0" if vis else "0.55 0.55 0.52 1", **phys)
    for k, (f, tex, *xf) in enumerate(vis):
        # xf: a shared mesh placed by the geom (scans: mesh scale s, geom pos / quat = the window's alignment)
        ET.SubElement(asset, "mesh", name=f"visual_{k}", file=os.path.relpath(f, out.parent),
                      **({"scale": " ".join([f"{xf[0]['scale']:.6f}"] * 3)} if xf else {}))
        if tex is not None:
            ET.SubElement(asset, "texture", name=f"visual_tex_{k}", type="2d", file=os.path.relpath(tex, out.parent))
            ET.SubElement(asset, "material", name=f"visual_mat_{k}", texture=f"visual_tex_{k}", specular="0.0", shininess="0.0")
        else:
            ET.SubElement(asset, "material", name=f"visual_mat_{k}", rgba="0.78 0.76 0.72 1", specular="0.1")
        ET.SubElement(body, "geom", name=f"visual_{k}", type="mesh", mesh=f"visual_{k}", material=f"visual_mat_{k}",
                      contype="0", conaffinity="0", group="2", mass="0",
                      **({"pos": " ".join(f"{x:.6f}" for x in xf[0]["pos"]), "quat": " ".join(f"{x:.8f}" for x in xf[0]["quat"])}
                         if xf else {}))
    ET.SubElement(wb, "camera", name="input_cam", mode="fixed", pos=" ".join(f"{x:.5f}" for x in cam["pos"]),
                  quat=" ".join(f"{x:.6f}" for x in cam["quat"]), fovy=f"{cam['fovy']:.4f}")
    ET.indent(root)
    ET.ElementTree(root).write(out)


def build(a):
    tags = a.tags.split(",")
    run = Path(a.run)
    pk = Path(a.out) / a.clip
    for sub in ("terrain", "visual", "refs"):
        (pk / sub).mkdir(parents=True, exist_ok=True)
    mo0 = dict(np.load(run / tags[0] / f"{a.clip}_motion.npz", allow_pickle=True))
    s = float(mo0["scale"])
    tg = dict(np.load(run / tags[0] / f"{a.clip}_terrain_g1.npz", allow_pickle=True))
    lows = [float(np.asarray(tg["height"]).min())]
    if "boxes" in tg:
        B = np.asarray(tg["boxes"], np.float64)
        lows.append(float((B[:, 2] - B[:, 5]).min()))
    dz = 0.01 - min(lows)
    tg["height"] = np.asarray(tg["height"], np.float64) + dz
    if "levels" in tg:
        tg["levels"] = np.asarray(tg["levels"], np.float64) + dz
    if "boxes" in tg:
        tg["boxes"] = np.asarray(tg["boxes"], np.float64) + [0, 0, dz, 0, 0, 0]
    np.savez(pk / "terrain/terrain_hfield.npz", **{k: v for k, v in tg.items()})
    terrain_mesh(tg).export(pk / "terrain/terrain_collision.obj")
    # visual: the textured mesh if given, else the reference's scene mesh (G1 world, untextured)
    for f in (pk / "visual").glob("*"):
        f.unlink()
    if a.visual:
        vm = Path(a.visual).parent / "meta.json"  # parts' collision stamp: refuse a visual baked for another terrain
        if vm.exists() and "collision" in json.loads(vm.read_text()):
            import hashlib

            hfile = json.loads(vm.read_text())["collision"].get("hfield_file", "terrain/terrain_hfield.npz")
            # a visual stamped on the clean 2 cm file: checked against that file (--stamp_hfield)
            hf_ = np.load(a.stamp_hfield if a.stamp_hfield and hfile != "terrain/terrain_hfield.npz" else pk / hfile)
            data = np.asarray(hf_["height"], np.float32).tobytes() + np.array([hf_["x0"], hf_["y0"], hf_["res"]], np.float64).tobytes()
            if hashlib.sha256(data).hexdigest() != json.loads(vm.read_text())["collision"]["hfield_sha256_data"]:
                raise SystemExit(f"{a.clip}: {a.visual} was baked for another collision terrain (hfield sha256 mismatch): re-bake")
        vis = visual_meshes(a.visual, s if a.visual_world == "recon" else 1.0, dz, pk / "visual")
    elif a.scan_visual:
        # a shared scan visual in scan coordinates: G1 world = s R (x - origin) + dz
        from scipy.spatial.transform import Rotation

        rc = np.load(Path(a.recon) / f"{a.clip}.npz", allow_pickle=True)
        R, mu = np.asarray(rc["R_align"], np.float64), np.asarray(rc["origin"], np.float64)
        q = Rotation.from_matrix(R).as_quat()
        sv = Path(a.scan_visual)  # <scan>/scene_visual.obj + _albedo.png, or <scan>/visual.obj + visual.png
        fo, ft = ((sv / "scene_visual.obj", sv / "scene_visual_albedo.png") if (sv / "scene_visual.obj").exists()
                  else (sv / "visual.obj", sv / "visual.png"))
        vis = [(fo, ft, {"scale": s, "pos": np.array([0, 0, dz]) - s * R @ mu, "quat": q[[3, 0, 1, 2]]})]
    elif a.scene_vertex_colors and (Path(a.recon) / f"{a.clip}_scene.ply").exists():
        # a coloured scan / mesh (no video): its vertex colours baked to a per-face texture
        import trimesh

        raise NotImplementedError("textured scene visuals are not part of this release")

        tm = trimesh.load(Path(a.recon) / f"{a.clip}_scene.ply", process=False, force="mesh")
        SV = np.asarray(tm.vertices, np.float64) * s + [0, 0, dz]
        vc = np.asarray(getattr(tm.visual, "vertex_colors", np.full((len(SV), 4), 150)))[:, :3].astype(np.uint8)
        vis = [textured_obj(SV, np.asarray(tm.faces, np.int64), vc, pk / "visual", "visual_0")] if len(tm.faces) >= 4 else []
    elif a.color_images and len(getattr(__import__("trimesh").load(Path(a.recon) / f"{a.clip}_scene.ply", process=False,
                                                                     force="mesh"), "faces", [])) >= 4:
        # the recon's TSDF coloured from the video (bench/scene_r2s_combined.colorize), per-face texture; an empty scene
        # (ViPE depth failure) falls through to the reference's own scene mesh below
        import trimesh

        raise NotImplementedError("textured scene visuals are not part of this release")

        rc = np.load(Path(a.recon) / f"{a.clip}.npz", allow_pickle=True)
        tm = trimesh.load(Path(a.recon) / f"{a.clip}_scene.ply", process=False, force="mesh")
        T_ = np.asarray(rc["T_world_cam"], np.float64).copy()
        T_[:, :3, 3] = T_[:, :3, 3] * s + [0, 0, dz]
        SV = np.asarray(tm.vertices, np.float64) * s + [0, 0, dz]
        Cc, _ = colorize(SV, np.asarray(tm.faces, np.int64), T_, np.asarray(rc["K"], np.float64).reshape(-1, 3, 3)[0],
                         np.asarray(rc["frames"], int), load_images(a.color_images),
                         np.asarray(rc["verts"], np.float32) * s + np.array([0, 0, dz], np.float32))
        vis = [textured_obj(SV, np.asarray(tm.faces, np.int64), Cc, pk / "visual", "visual_0")]
    else:
        src = run / tags[0] / f"{a.clip}_scene_g1.obj"
        vis = visual_meshes(str(src), 1.0, dz, pk / "visual") if src.exists() else []
    # camera
    rec = np.load(Path(a.recon) / f"{a.clip}.npz", allow_pickle=True)
    Twc = np.asarray(rec["T_world_cam"], np.float64).copy()
    Twc[:, :3, 3] = Twc[:, :3, 3] * s + [0, 0, dz]
    K = np.asarray(rec["K"], np.float64).reshape(-1, 3, 3)[0]
    import mujoco

    mjq = np.zeros((len(Twc), 4))
    for t in range(len(Twc)):
        mujoco.mju_mat2Quat(mjq[t], (Twc[t, :3, :3] @ CV_TO_MJ).reshape(-1))
    fovy = float(np.degrees(2 * np.arctan(K[1, 2] / K[1, 1])))
    dpos = np.linalg.norm(Twc[:, :3, 3] - np.median(Twc[:, :3, 3], 0), axis=1)
    dang = np.degrees(2 * np.arccos(np.clip(np.abs((mjq * mjq[len(mjq) // 2]).sum(1)), 0, 1)))
    static = bool(np.percentile(dpos, 95) < 0.02 and np.percentile(dang, 95) < 1.0)
    c0 = len(Twc) // 2 if static else 0
    np.savez(pk / "camera.npz", T_world_cam=Twc, mj_pos=Twc[:, :3, 3], mj_quat_wxyz=mjq, K=K,
             width=int(round(2 * K[0, 2])), height=int(round(2 * K[1, 2])), fovy_deg=fovy, fps=float(rec["fps"]),
             frames=np.asarray(rec["frames"], int), static=static)
    scene_xml(tg, vis, {"pos": Twc[c0, :3, 3], "quat": mjq[c0], "fovy": fovy}, pk / "scene.xml")
    # references
    rows = []
    for f in glob_rows(run):
        rows += [r for r in json.load(open(f)) if r["clip"] == a.clip]
    refs = {}
    for tag in tags:
        mf = run / tag / f"{a.clip}_motion.npz"
        if not mf.exists():
            continue
        mo = dict(np.load(mf, allow_pickle=True))
        np.savez(pk / "refs" / f"{tag}.npz", **wbt_motion(mo, dz, float(mo.get("fps", 50.0))))
        mo["g1_root_pos"] = np.asarray(mo["g1_root_pos"], np.float64) + [0, 0, dz]
        np.savez(pk / "refs" / f"{tag}_motion.npz", **mo)
        refs[tag] = {"wbt": f"refs/{tag}.npz", "ours": f"refs/{tag}_motion.npz",
                     "tracking": [{k: r.get(k) for k in ("tracker", "setting", "fell", "fall_t_s", "link_err_med_m",
                                                         "vm_term_s", "pen_frames_2cm_pct", "float_pct", "bone_rec_deg")}
                                  for r in rows if r["tag"] == tag],
                     "kinematic": True, "physically_valid": False}
        # the terrain policy's MuJoCo rollout of this reference: a physically valid reference where it succeeded
        tf, rf = run / f"port_{tag}" / "traj" / f"{a.clip}_motion.npz", run / f"port_{tag}" / "results.json"
        if tf.exists() and rf.exists():
            res = next((r for r in json.load(open(rf))["clips"] if "error" not in r and Path(r["ref"]).name == mf.name), None)
            ok = bool(res and res["success"] and res.get("success_hard", True))
            if ok or a.all_rollouts:
                tz = np.load(tf)
                qn = [str(x) for x in tz["qpos_names"]]
                qp = np.asarray(tz["qpos"], np.float64)
                ro = {"fps": 50.0, "g1_dof": qp[:, 7:], "g1_joint_names": np.array(qn[7:]), "g1_root_pos": qp[:, :3],
                      "g1_root_quat_xyzw": qp[:, [4, 5, 6, 3]], "scale": mo["scale"], "frames": mo["frames"],
                      "contacts": mo["contacts"]}
                np.savez(pk / "refs" / f"sim_{tag}.npz", **wbt_motion(ro, dz, 50.0))
                ro["g1_root_pos"] = ro["g1_root_pos"] + [0, 0, dz]
                np.savez(pk / "refs" / f"sim_{tag}_motion.npz", **ro)
                refs[f"sim_{tag}"] = {"wbt": f"refs/sim_{tag}.npz", "ours": f"refs/sim_{tag}_motion.npz",
                                      "kinematic": False, "physically_valid": ok, "rollout_of": tag,
                                      "tracker": "VideoMimic terrain policy, MuJoCo port (Model 15 plant)",
                                      "success": ok, "link_err_med_m": float(np.median(tz["link_err"])),
                                      "plant": res.get("plant") if res else None}
    # the per-frame contact schedule on the references' 50 Hz grid (the contact heads; reward / termination signal)
    cpath = ""
    base = a.clip
    for suf in ("_w06", "_w20", "_full"):
        base = base[: -len(suf)] if base.endswith(suf) else base
    cf = Path(a.contacts) / f"{base}__0.contact.npz"
    pinned = "pin_report" in mo0  # a pinned reference: its contacts are exactly the pinned stance cores (bench/stance_pin.py)
    if cf.exists() and refs:
        cz = np.load(cf, allow_pickle=True)
        img = np.asarray(rec["frames"], int)[np.clip(np.asarray(mo0["frames"], int), 0, len(rec["frames"]) - 1)]
        rows_c = {int(f): i for i, f in enumerate(np.asarray(cz["frame"], int))}
        kk = np.array([rows_c.get(int(f), -1) for f in img])
        okc = kk >= 0
        lab0 = json.loads(str(mo0["contacts"]))  # the solver's own stance mask (what the reference was solved with)
        T0 = len(img)
        out_c = {"solver_stance": np.stack([np.r_[np.asarray(lab0.get(g, []), bool), np.zeros(T0, bool)][:T0]
                                            for g in ("lfoot", "rfoot")], 1),
                 "fps": 50.0, "image_frame": img, "valid": okc, "channels": cz["channels"],
                 "hk_channels": cz["hk_channels"] if "hk_channels" in cz.files else np.array([]),
                 "heuristic_labels": mo0["contacts"],
                 "solver_stance_source": a.stance_source or ("pinned stance cores (stance runs on the video head)" if pinned else "video head")}
        for k in cz.files:
            if k.endswith("_prob"):
                v = np.zeros((len(kk),) + cz[k].shape[1:], np.float32)
                v[okc] = cz[k][kk[okc]]
                out_c[k] = v
                out_c[k[: -len("_prob")]] = v >= float(cz["thr"]) if "thr" in cz.files else v >= 0.5
        if a.contacts_kin and Path(a.contacts_kin).exists():  # ground-truth kinematic channels kept as reference fields
            kz = np.load(a.contacts_kin, allow_pickle=True)
            ri = np.clip(np.asarray(mo0["frames"], int), 0, len(kz["static"]) - 1)
            out_c.update(gt_static=np.asarray(kz["static"])[ri], gt_any=np.asarray(kz["any"])[ri], gt_height_m=np.asarray(kz["h"])[ri],
                         gt_speed_m_s=np.asarray(kz["s"])[ri], gt_channels=kz["channels"], contacts_source=str(kz["contacts_source"]))
        np.savez(pk / "contacts.npz", **out_c)
        cpath = "contacts.npz"
    elif a.contacts_kin and Path(a.contacts_kin).exists() and refs:  # ground-truth kinematic labels (no video, no heads)
        kz = np.load(a.contacts_kin, allow_pickle=True)
        T0 = len(mo0["frames"])
        ri = np.clip(np.asarray(mo0["frames"], int), 0, len(kz["static"]) - 1)  # 50 Hz grid -> the window's frames
        lab0 = json.loads(str(mo0["contacts"]))
        out_c = {"solver_stance": np.stack([np.r_[np.asarray(lab0.get(g, []), bool), np.zeros(T0, bool)][:T0]
                                            for g in ("lfoot", "rfoot")], 1),
                 "fps": 50.0, "valid": np.ones(T0, bool), "channels": kz["channels"],
                 "gt_static": np.asarray(kz["static"])[ri], "gt_any": np.asarray(kz["any"])[ri],
                 "gt_height_m": np.asarray(kz["h"])[ri], "gt_speed_m_s": np.asarray(kz["s"])[ri],
                 "contacts_source": str(kz["contacts_source"]),
                 "solver_stance_source": a.stance_source or ("pinned stance cores (stance runs on the kinematic labels, static AND any)" if pinned else "kinematic labels")}
        np.savez(pk / "contacts.npz", **out_c)
        cpath = "contacts.npz"
    elif refs:  # no head file: the stance mask alone (the solve's labels; pinned when pin_report)
        lab0 = json.loads(str(mo0["contacts"]))
        T0 = len(mo0["frames"])
        np.savez(pk / "contacts.npz", solver_stance=np.stack([np.r_[np.asarray(lab0.get(g, []), bool), np.zeros(T0, bool)][:T0]
                                                               for g in ("lfoot", "rfoot")], 1),
                 fps=50.0, valid=np.ones(T0, bool), heuristic_labels=mo0["contacts"],
                 solver_stance_source=a.stance_source or ("pinned stance cores (stance runs on the solve's own labels: no head file)" if pinned
                 else "the solve's own labels (no head file)"))
        cpath = "contacts.npz"
    W, Hh = int(round(2 * K[0, 2])), int(round(2 * K[1, 2]))
    meta = {"clip": a.clip, "run": str(run), "recon": str(a.recon), "scale_s": s, "dz_m": dz,
            "world": "G1 world (recon world x s) + (0, 0, dz); z up; the lowest terrain 1 cm above a ground plane at z = 0",
            "terrain": {"cells": list(np.asarray(tg["height"]).shape), "res_m": float(tg["res"]),
                        "boxes": int(len(tg.get("boxes", []))), "levels": "levels" in tg},
            "visual": [os.path.relpath(f, pk) for f, *_ in vis], "textured": any(t is not None for _, t, *_ in vis),
            **({"visual_shared": {"scan_visual": str(a.scan_visual), "placement": "mesh scale s, geom pos / quat = s R_align (x - origin) + dz"}}
               if a.scan_visual else {}),
            "camera": {"static": static, "fovy_deg": fovy, "width": W, "height": Hh,
                       "position": Twc[c0, :3, 3].round(5).tolist(), "orientation_wxyz": mjq[c0].round(6).tolist()},
            "refs": refs,
            "contact_schedule": {"file": cpath, "source": str(cf) if cpath else None,
                                 "what": "contact heads on the references' 50 Hz grid: {image,video}_{static,any}"
                                         "[_prob] (T, 4) = l_heel, l_toe, r_heel, r_toe (+ _hk_: hands, knees), valid, "
                                         "image_frame; heuristic_labels = the recon's labels the retargeter used"},
            "simulator": {
                "scene_file": str((pk / "scene.xml").resolve()),
                "ground": "a plane at z = 0, under the scene",
                "camera_sensor": {"name": "input_cam", "mount": "world", "position": Twc[c0, :3, 3].round(5).tolist(),
                                  "orientation_wxyz": mjq[c0].round(6).tolist(), "width": W, "height": Hh,
                                  "vertical_fov_deg": round(fovy, 4)},
                "motion_files": {t: str((pk / v["wbt"]).resolve()) for t, v in refs.items()}}}
    (pk / "package.json").write_text(json.dumps(meta, indent=1))
    print(pk, json.dumps({k: meta[k] for k in ("scale_s", "dz_m", "terrain", "visual", "camera")}), flush=True)


def glob_rows(run):
    import glob

    return glob.glob(str(run / "score_*_rows.json"))


def check(a):
    """Load the package as a simulator would (robot + a ground plane + the scene file attached through a frame),
    verify the hfield against terrain_hfield.npz by vertical rays, and render the input camera at sample frames."""
    import mujoco

    from bench.videomimic.scene_metrics import HF

    pk = Path(a.pkg)
    meta = json.loads((pk / "package.json").read_text())
    world = mujoco.MjSpec.from_file(a.robot)
    world.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], contype=2, conaffinity=1)
    world.worldbody.add_light(pos=[0, 0, 30], dir=[0, 0, -1])
    fr = world.worldbody.add_frame()
    world.attach(mujoco.MjSpec.from_file(str(pk / "scene.xml")), frame=fr, prefix="r2s_")
    m = world.compile()
    d = mujoco.MjData(m)
    d.qpos[0] = 1e3  # the robot out of the way of the rays
    mujoco.mj_forward(m, d)
    tg = dict(np.load(pk / "terrain/terrain_hfield.npz"))
    terr = HF(tg)
    rng = np.random.default_rng(0)
    H = np.asarray(tg["height"])
    ny, nx = H.shape
    xs = float(tg["x0"]) + rng.uniform(0.1, nx - 1.1, 300) * float(tg["res"])
    ys = float(tg["y0"]) + rng.uniform(0.1, ny - 1.1, 300) * float(tg["res"])
    gid = np.zeros(1, np.int32)
    geom_group = np.array([0, 0, 0, 0, 0, 0], np.uint8)
    geom_group[0] = 1  # rays only hit group-0 geoms (the collision terrain; visual meshes are group 2)
    err = []
    for x, y in zip(xs, ys):
        dist = mujoco.mj_ray(m, d, np.array([x, y, 50.0]), np.array([0, 0, -1.0]), geom_group, 1, -1, gid)
        if dist >= 0:
            err.append(abs((50.0 - dist) - float(terr(np.array([[x, y]]))[0])))
    err = np.array(err)
    rep = {"compiled": True, "nbody": m.nbody, "ngeom": m.ngeom, "nhfield": m.nhfield,
           "hfield_vs_npz_mm": {"n": int(len(err)), "median": float(np.median(err) * 1000), "max": float(err.max() * 1000)}}
    print(json.dumps(rep), flush=True)
    if a.out:
        import cv2

        cam = np.load(pk / "camera.npz")
        tag = a.tag or next(iter(meta["refs"]))
        ref = np.load(pk / meta["refs"][tag]["wbt"])
        jn = [str(j) for j in ref["joint_names"]]
        adr = [m.joint(n).qposadr[0] for n in jn]
        W, Hh = int(cam["width"]), int(cam["height"])
        sc = a.height / Hh
        rnd = mujoco.Renderer(m, a.height, int(round(W * sc)) // 2 * 2)
        cid = m.camera("r2s_input_cam").id
        mo = np.load(pk / meta["refs"][tag]["ours"])
        idx = np.asarray(mo["frames"], int)
        imgs = None
        if a.images:
            raise NotImplementedError("camera-view renders are not part of this release")

            imgs = load_images(a.images)
        tiles = []
        for k in np.linspace(0, len(ref["joint_pos"]) - 1, a.n).astype(int):
            d.qpos[:7] = ref["joint_pos"][k, :7]
            d.qpos[adr] = ref["joint_pos"][k, 7:]
            r = min(int(idx[k]), len(cam["mj_pos"]) - 1)
            m.cam_pos[cid], m.cam_quat[cid] = cam["mj_pos"][r], cam["mj_quat_wxyz"][r]
            mujoco.mj_forward(m, d)
            rnd.update_scene(d, camera=cid)
            sim = rnd.render().copy()
            if imgs is not None:
                fi = int(cam["frames"][r])
                src = video_frames(imgs, [fi]).get(fi)
                src = cv2.resize(src, (sim.shape[1], sim.shape[0])) if src is not None else np.zeros_like(sim)
                tiles.append(np.concatenate([src, sim, (0.5 * src + 0.5 * sim).astype(np.uint8)], 1))
            else:
                tiles.append(sim)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(a.out, cv2.cvtColor(np.concatenate(tiles, 0), cv2.COLOR_RGB2BGR))
        print("wrote", a.out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["build", "check"])
    p.add_argument("--run", default="")
    p.add_argument("--tags", default="p")
    p.add_argument("--recon", default="")
    p.add_argument("--clip", default="")
    p.add_argument("--out", default="")
    p.add_argument("--visual", default="", help="UV-textured visual mesh (obj / glb / ply)")
    p.add_argument("--visual_world", default="recon", choices=["recon", "g1"])
    p.add_argument("--stamp_hfield", default="", help="the file a visual's collision stamp names, when it is not terrain_hfield.npz")
    p.add_argument("--contacts", default="", help="dir of <clip>__0.contact.npz contact-head files")
    p.add_argument("--all_rollouts", action="store_true", help="also store failed rollouts (flagged physically_valid false)")
    p.add_argument("--color_images", default="", help="images (clips.json:<clip>): colour the TSDF visual from the video")
    p.add_argument("--scene_vertex_colors", action="store_true", help="visual = the scene mesh with its vertex colours")
    p.add_argument("--scan_visual", default="", help="a shared scan visual dir, placed by the "
                   "recon's R_align / origin")
    p.add_argument("--contacts_kin", default="", help="ground-truth kinematic contact npz (static / any / h / s) for contacts.npz")
    p.add_argument("--stance_source", default="", help="contacts.npz solver_stance_source text (the head / kinematic channels stay "
                   "as reference fields)")
    p.add_argument("--pkg", default="")
    p.add_argument("--robot", default=MJCF, help="check: the robot MJCF the scene is attached to")
    p.add_argument("--tag", default="")
    p.add_argument("--images", default="")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--height", type=int, default=360)
    a = p.parse_args()
    build(a) if a.cmd == "build" else check(a)


if __name__ == "__main__":
    main()
