"""MuJoCo replay of a real-to-sim clip (track.py output, or DexYCB GT for reference) -> <clip>/sim_<tag>/: panel videos,
the composed side-by-side video and metrics.json.

World = support-plane frame (z up, origin under the object's start position), gravity -z. Replays:
  kin     kinematic: object mesh at the tracked pose + the deforming MANO hand mesh per frame (camera + side view);
  phys    physics: free object (CoACD convex parts, density --density), hand = 16 convex MANO parts (skinning-weight
          argmax, rigid per frame by Kabsch), each a free body welded to a mocap target that follows the hand
          (gravity-compensated, hand-hand and hand-table collisions off); reports whether the object follows the
          reference (tracked) trajectory; --hold_until_lift 1 keeps the object anchored until it starts to move in
          the video (tests whether the grasp carries it, not whether the approach knocks it over); --controller 1
          closes the loop on the simulated object: once the hand has reached it, every finger segment within 2 cm gets
          its target moved onto the object's surface (+2 mm squeeze, <= 15 mm, low-passed), touching ones hold;
          --ctrl_flex 1 closes the fingers in joint space instead: per finger near the object and not touching it, the
          flexion about its MCP / PIP / DIP joint centres (from the estimated mesh) grows until a segment touches or
          --flex_max (forward kinematics of the part targets; the palms keep only the penetration relief);
  robot   dex-retargeting position retargeting (fingertips + middle joints, free wrist) to a robot hand (dex-urdf),
          replayed kinematically with the tracked object (link poses from pinocchio FK -> mocap bodies).
Panels (--grid): input | hand + object overlay | kinematic (side) | physics (side); physics on the image and the robot are
written as separate panel videos (phys_cam.mp4, robot.mp4 with --robot).

    MUJOCO_GL=egl python scripts/real2sim/sim.py --clip $R2S/mustard --hand fah --object recon --contact 1
    MUJOCO_GL=egl python scripts/real2sim/sim.py --clip <clip> --hand gt --object gt --tag gt   # DexYCB reference
"""
import argparse
import glob
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

FAH_HOME = os.environ.get("FAH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "fah"))
URDF_DIR = os.environ.get("DEX_URDF_DIR", os.path.join(FAH_HOME, "dex-urdf", "robots", "hands"))  # --robot only
SKIN = (0.93, 0.72, 0.6, 1)


def quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, np.float64).ravel())
    return q


def slerp(q0, q1, a):
    d = q0 @ q1
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)


def kabsch(A, B):
    """R, t with R A + t ~ B (rows = points)."""
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ma).T @ (B - mb))
    R = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
    return R, mb - R @ ma


def rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def fmt(a):
    return " ".join("%.6g" % v for v in np.ravel(a))


def look_at(eye, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    return np.r_[x, np.cross(x, f)]


# ---------------------------------------------------------------- inputs
def interp_hands(V, J, valid):
    """Fill missing frames between detections linearly; hold the last one; leading frames stay invalid."""
    idx = np.nonzero(valid)[0]
    V, J, valid = V.copy(), J.copy(), valid.copy()
    if not len(idx):
        return V, J, valid
    for i in range(idx[0], len(V)):
        if valid[i]:
            continue
        nxt = idx[idx > i]
        prv = idx[idx < i][-1]
        if len(nxt):
            a = (i - prv) / (nxt[0] - prv)
            V[i], J[i] = (1 - a) * V[prv] + a * V[nxt[0]], (1 - a) * J[prv] + a * J[nxt[0]]
        else:
            V[i], J[i] = V[prv], J[prv]
        valid[i] = True
    return V, J, valid


def load_source(clip_dir, hand, obj, track="track.npz", two_hands=False, gravity="plane"):
    c = json.load(open(os.path.join(clip_dir, "clip.json")))
    tr = np.load(os.path.join(clip_dir, track))
    gt = np.load(os.path.join(clip_dir, "gt.npz")) if os.path.exists(os.path.join(clip_dir, "gt.npz")) else None
    K = tr["K"]
    if hand == "gt":
        hs = [(gt["hand_verts"], gt["hand_joints"], gt["hand_annotated"], bool(gt["is_right"]))]
    else:
        hs = [(tr["hand_verts"], tr["hand_joints"], tr["hand_valid"], bool(tr["hand_side"]))]
        if two_hands and "hand2_verts" in tr.files:  # the other hand too (two-hand / handover clips)
            hs.append((tr["hand2_verts"], tr["hand2_joints"], tr["hand2_valid"], bool(tr["hand2_side"])))
    hs = [interp_hands(np.asarray(V_, np.float64), np.asarray(J_, np.float64), np.asarray(v_, bool)) + (s_,) for V_, J_, v_, s_ in hs]
    V, J, valid, side = hs[0]
    if obj == "gt":
        mesh = trimesh.load(c["gt_mesh"], force="mesh")
        T = gt["T_cam_obj"]
        Twc = gt["T_world_cam"]
        n_, d_ = Twc[2, :3], Twc[2, 3]
        plane = np.r_[n_, d_] if d_ > 0 else -np.r_[n_, d_]
    else:
        rd = str(tr["recon"]) if "recon" in tr.files else "recon"
        mesh = trimesh.load(os.path.join(clip_dir, rd, "mesh_metric.glb" if track == "track.npz" else track[:-4] + "_mesh.glb"), force="mesh")
        T, plane = tr["T_cam_obj"], tr["plane"]
    frames = sorted(glob.glob(os.path.join(clip_dir, "frames", "*.jpg")))
    T, motion = np.asarray(T, np.float64), np.tile(np.eye(4), (len(frames), 1, 1))
    if os.path.exists(os.path.join(clip_dir, "cams.npz")):  # moving camera (HOT3D): everything in the frame-0 camera
        Twc = np.load(os.path.join(clip_dir, "cams.npz"))["T_world_cam"]
        motion = np.linalg.inv(Twc[0]) @ Twc  # camera t -> camera 0
        T = motion @ T
        mv = lambda X: np.einsum("nij,nvj->nvi", motion[:, :3, :3], X) + motion[:, None, :3, 3]
        hs = [(mv(V_), mv(J_), v_, s_) for V_, J_, v_, s_ in hs]
        V, J = hs[0][0], hs[0][1]
        f0 = int(tr["frame0"])
        n0 = motion[f0, :3, :3] @ plane[:3]
        plane = np.r_[n0, plane[3] - n0 @ motion[f0, :3, 3]]
        if gravity == "headset":  # up = the headset world's +z (HOT3D: gravity -z) in the frame-0 camera; the support plane
            up = Twc[0][:3, :3].T @ np.array([0, 0, 1.0])  # keeps its offset if it is horizontal, else it runs under the object
            if plane[:3] @ up > np.cos(np.radians(20)):
                plane = np.r_[up, plane[3] * (plane[:3] @ up)]
            else:
                X0 = np.asarray(mesh.vertices) @ T[0][:3, :3].T + T[0][:3, 3]
                plane = np.r_[up, -(X0 @ up).min()]
    return dict(K=K, fps=c["fps"], frames=frames, n=len(frames), mesh=mesh, T_cam_obj=T, V=V, J=J,
                valid=valid, side=side, plane=plane, object=c.get("object", "object"), source=c["source"], motion=motion,
                hands=[dict(V=V_, J=J_, valid=v_, side=s_) for V_, J_, v_, s_ in hs])


def world_frame(src):
    """cam -> world: z = plane normal (towards the camera side), origin = object start position on the plane."""
    n, d = src["plane"][:3], src["plane"][3]
    c0 = src["T_cam_obj"][0][:3, 3]
    o = c0 - (n @ c0 + d) * n
    x = np.array([1.0, 0, 0]) - n[0] * n
    x /= np.linalg.norm(x)
    R_cw = np.stack([x, np.cross(n, x), n], 1)  # world axes in camera coordinates
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R_cw.T, -R_cw.T @ o
    return T


# ---------------------------------------------------------------- assets
def export_object(mesh, out, seed=0):
    """Textured OBJ (+ texture png) for the visual mesh and CoACD convex parts for collisions."""
    import coacd

    os.makedirs(out, exist_ok=True)
    tex = None
    try:
        text, texture = trimesh.exchange.obj.export_obj(mesh, include_texture=True, return_texture=True, mtl_name="obj.mtl")
        for k, v in texture.items():
            open(os.path.join(out, k), "wb").write(v)
            if k.lower().endswith((".png", ".jpg", ".jpeg")):
                tex = os.path.join(out, k)
    except Exception as e:
        print("texture export failed:", repr(e))
        text = trimesh.exchange.obj.export_obj(mesh, include_texture=False)
    open(os.path.join(out, "obj.obj"), "w").write(text)
    cm = coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    parts = coacd.run_coacd(cm, threshold=0.04, max_convex_hull=24, seed=seed)
    hulls = []
    for k, (v, f) in enumerate(parts):
        p = os.path.join(out, f"hull_{k}.obj")
        trimesh.Trimesh(v, f).export(p)
        hulls.append(p)
    return os.path.join(out, "obj.obj"), tex, hulls


def mano_parts(side):
    from bench.common import ManoLayer

    ml = ManoLayer(os.environ.get("MANO_DIR", os.path.join(FAH_HOME, "mano")))
    W = ml.layers[side].lbs_weights.detach().cpu().numpy()
    return W, ml.faces[side]


def urdf_visuals(urdf):
    root = ET.parse(urdf).getroot()
    out = {}
    for link in root.findall("link"):
        vs = []
        for v in link.findall("visual"):
            m = v.find("geometry/mesh")
            if m is None:
                continue
            o = v.find("origin")
            xyz = np.array([float(a) for a in o.get("xyz", "0 0 0").split()]) if o is not None else np.zeros(3)
            rpy = np.array([float(a) for a in o.get("rpy", "0 0 0").split()]) if o is not None else np.zeros(3)
            sc = np.array([float(a) for a in m.get("scale", "1 1 1").split()])
            col = v.find("material/color")
            rgba = [float(a) for a in col.get("rgba").split()] if col is not None else [0.25, 0.25, 0.28, 1]
            vs.append((os.path.join(os.path.dirname(urdf), m.get("filename")), xyz, quat(rpy_mat(*rpy)), sc, rgba))
        if vs:
            out[link.get("name")] = vs
    return out


# ---------------------------------------------------------------- scene
def camera_xml(name, K, H, W, T_wc):
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    xy = np.r_[R[:, 0], -R[:, 1]]  # OpenCV (x right, y down, z forward) -> MuJoCo (x right, y up, -z forward)
    return (f'<camera name="{name}" pos="{fmt(t)}" xyaxes="{fmt(xy)}" resolution="{W} {H}" sensorsize="{W * 1e-5:.6g} {H * 1e-5:.6g}" '
            f'focalpixel="{K[0, 0]:.4f} {K[1, 1]:.4f}" principalpixel="{(W - 1) / 2 - K[0, 2]:.4f} {(H - 1) / 2 - K[1, 2]:.4f}"/>')  # offsets checked by projection


def skin_xml(skin, bones, rgba, name="hand"):
    """MANO mesh as a MuJoCo skin: bound to the 16 part bodies of one hand (global part indices from skin["offset"]) with the
    MANO skinning weights (bind pose = the reference frame), so the rendered hand deforms smoothly with the parts."""
    W, o = skin["W"], skin["offset"]
    out = [f'<skin name="{name}" rgba="{fmt(rgba)}" vertex="{fmt(skin["V0"])}" face="{fmt(skin["faces"])}">']
    for k in range(16):
        v = np.nonzero(W[:, k] > 0.01)[0]
        out.append(f'<bone body="{bones}{o + k}" bindpos="{fmt(skin["cen"][k])}" bindquat="1 0 0 0" vertid="{fmt(v)}" '
                   f'vertweight="{fmt(W[v, k])}"/>')
    return "".join(out) + "</skin>"


PHYS = dict(weld="0.01 1", friction="1.2 0.02 0.0002", solimp="0.95 0.99 0.001", margin=0.0)
PARENT = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]  # MANO kinematic tree (16 parts)
FINGERS = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15)]  # index, middle, little, ring, thumb: MCP / PIP / DIP parts
FLEX_GAIN = (1.0, 1.0, 0.8)  # per-joint limit as a share of --flex_max (MCP, PIP, DIP)


def joint_pivots(V, W):
    """MANO joint centres from the mesh: per part the mean of the vertices skinned to both it and its parent (the joint's
    ring of vertices); part 0 = the palm's centroid."""
    piv = np.zeros((16, 3))
    piv[0] = V[W.argmax(1) == 0].mean(0)
    for k in range(1, 16):
        piv[k] = V[(W[:, k] > 0.2) & (W[:, PARENT[k]] > 0.2)].mean(0)
    return piv


def flex_axes(piv, right):
    """Per finger the flexion axis: rotating its bones about it by a positive angle curls them towards the palm (fingers:
    the palmar normal of the palm plane through the index / little MCP; thumb: towards the index / middle DIP joints,
    i.e. opposition)."""
    n = np.cross(piv[1] - piv[0], piv[7] - piv[0])
    n = n / np.linalg.norm(n) * (1.0 if right else -1.0)
    ax = []
    for f, (a, b, _) in enumerate(FINGERS):
        bone = piv[b] - piv[a]
        nf = n
        if f == 4:
            nf = 0.5 * (piv[3] + piv[6]) - piv[a]
            nf = nf - bone * (nf @ bone) / (bone @ bone)
            nf /= np.linalg.norm(nf)
        u = np.cross(bone, nf)
        ax.append(u / np.linalg.norm(u))
    return ax


def axis_rot(u, th):
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def flex_pose(p, piv, axes, theta, off=0):
    """Part poses (NP, 7) of one hand (parts off .. off + 15) with each finger's MCP / PIP / DIP closed by theta[f, j] (rad)
    in joint space: forward kinematics about the joint centres, distal parts carried along."""
    p = p.copy()
    for f, chain in enumerate(FINGERS):
        if np.abs(theta[f]).max() < 1e-6:
            continue
        P = piv.copy()
        for j, k in enumerate(chain):
            G = axis_rot(axes[f], theta[f, j])
            c = P[k].copy()
            for k2 in chain[j:]:
                g = off + k2
                Rk = np.zeros(9)
                mujoco.mju_quat2Mat(Rk, p[g, 3:])
                p[g, :3] = G @ (p[g, :3] - c) + c
                p[g, 3:] = quat(G @ Rk.reshape(3, 3))
                P[k2] = G @ (P[k2] - c) + c
    return p


def build_xml(src, T_wc, assets, mode, hand=None, robot=None, side_cam=None, density=300.0, center=(0, 0, 0.1)):
    H, W = cv2.imread(src["frames"][0]).shape[:2]
    obj_obj, tex, hulls = assets
    a = ['<asset>', '<texture name="grid" type="2d" builtin="checker" rgb1=".82 .82 .80" rgb2=".70 .70 .68" width="512" height="512"/>',
         '<material name="grid" texture="grid" texrepeat="12 12" reflectance="0"/>',
         f'<mesh name="obj_vis" file="{obj_obj}" inertia="shell"/>']
    if tex:
        a += [f'<texture name="obj_tex" type="2d" file="{tex}"/>', '<material name="obj_mat" texture="obj_tex" specular="0.2"/>']
    else:
        a += ['<material name="obj_mat" rgba=".85 .65 .2 1"/>']
    a += [f'<mesh name="hull{k}" file="{h}"/>' for k, h in enumerate(hulls)]
    b = ['<worldbody>', f'<light pos="{fmt(np.add(center, [0.1, 0.2, 1.2]))}" dir="-0.08 -0.16 -1" cutoff="55" exponent="2" '
         'diffuse=".6 .6 .6" specular=".1 .1 .1" castshadow="true"/>',
         '<geom name="floor" type="plane" size="3 3 .01" material="grid" group="2" condim="3" friction="1 0.005 0.0001"/>',
         camera_xml("input", src["K"], H, W, T_wc)]
    if side_cam is not None:
        b.append(f'<camera name="side" pos="{fmt(side_cam[0])}" xyaxes="{fmt(side_cam[1])}" fovy="45"/>')
    vis = '<geom name="obj_vis" type="mesh" mesh="obj_vis" material="obj_mat" contype="0" conaffinity="0" mass="0" group="1"/>'
    eq, dfm = [], []
    if mode == "phys":
        b.append('<body name="obj" pos="0 0 0"><freejoint name="obj"/>' + vis + "".join(
            f'<geom type="mesh" mesh="hull{k}" density="{density}" group="3" condim="4" friction="1 0.02 0.0002" '
            f'contype="1" conaffinity="3" solimp="{PHYS["solimp"]}"/>' for k in range(len(hulls))) + '</body>')
        b.append('<body name="anchor" mocap="true"/>')
        eq.append('<weld name="anchor" body1="obj" body2="anchor" relpose="0 0 0 1 0 0 0" solref="0.02 1"/>')
    else:
        b.append('<body name="obj" mocap="true">' + vis + '</body>')
    if hand is not None:
        b += [f'<body name="hm{k}" mocap="true" pos="{fmt(c)}"/>' for k, c in enumerate(hand["cen"])]
    if mode == "kin":
        dfm += [skin_xml(sk, "hm", SKIN, f"hand{h}") for h, sk in enumerate(hand["skins"])]
    if mode == "phys":
        for k, P in enumerate(hand["parts"]):
            a.append(f'<mesh name="part{k}" vertex="{fmt(P)}"/>')
            b.append(f'<body name="hp{k}" pos="{fmt(hand["cen"][k])}" gravcomp="1"><freejoint/><geom type="mesh" mesh="part{k}" '
                     f'mass="0.2" rgba="{fmt(SKIN)}" condim="4" friction="{PHYS["friction"]}" contype="2" conaffinity="0" group="3" '
                     f'solimp="{PHYS["solimp"]}" margin="{PHYS["margin"]}"/></body>')
            eq.append(f'<weld body1="hp{k}" body2="hm{k}" relpose="0 0 0 1 0 0 0" solref="{PHYS["weld"]}"/>')
        dfm += [skin_xml(sk, "hp", SKIN, f"hand{h}") for h, sk in enumerate(hand["skins"])]
    if mode == "robot":
        names = {}
        for link, vs in robot["visuals"].items():
            g = []
            for f, xyz, q, sc, rgba in vs:
                key = (f, tuple(sc))
                if key not in names:
                    names[key] = f"rm{len(names)}"
                    a.append(f'<mesh name="{names[key]}" file="{f}" scale="{fmt(sc)}" inertia="shell"/>')
                g.append(f'<geom type="mesh" mesh="{names[key]}" pos="{fmt(xyz)}" quat="{fmt(q)}" rgba="{fmt(rgba)}" '
                         f'contype="0" conaffinity="0" mass="0" group="1"/>')
            b.append(f'<body name="rl_{link}" mocap="true">' + "".join(g) + '</body>')
        if hand is not None:  # the human hand as a faint ghost next to the robot
            dfm += [skin_xml(sk, "hm", (*SKIN[:3], 0.3), f"hand{h}") for h, sk in enumerate(hand["skins"])]
    a.append('</asset>')
    b.append('</worldbody>')
    xml = ('<mujoco model="real2sim"><compiler angle="radian" balanceinertia="true"/>'
           '<option timestep="0.002" iterations="60" ls_iterations="20" cone="elliptic" impratio="10" gravity="0 0 -9.81"/>'
           '<visual><map znear="0.01" zfar="20"/><quality shadowsize="8192" offsamples="8"/>'
           '<headlight ambient=".42 .42 .42" diffuse=".45 .45 .45"/>'
           f'<global offwidth="{max(W, 1280)}" offheight="{max(H, 960)}"/></visual><statistic extent="0.6" center="{fmt(center)}"/>'
           + "".join(a) + "".join(b) + (f'<deformable>{"".join(dfm)}</deformable>' if dfm else "")
           + (f'<equality>{"".join(eq)}</equality>' if eq else "") + '</mujoco>')
    return xml


class Renderer:
    def __init__(self, model, H, W):
        self.r = mujoco.Renderer(model, H, W)

    def render(self, data, cam, groups=(1, 2), depth=False):
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        for g in groups:
            opt.geomgroup[g] = 1
        self.r.update_scene(data, camera=cam, scene_option=opt)
        if depth:
            self.r.enable_depth_rendering()
            out = self.r.render()
            self.r.disable_depth_rendering()
            return out
        return self.r.render()[..., ::-1].copy()  # BGR

    def set_input_camera(self, data, T_wc):
        """Move the input camera (moving-camera clips): OpenCV camera -> world pose."""
        cid = mujoco.mj_name2id(self.r.model, mujoco.mjtObj.mjOBJ_CAMERA, "input")
        self.r.model.cam_pos[cid] = T_wc[:3, 3]
        self.r.model.cam_quat[cid] = quat(T_wc[:3, :3] @ np.diag([1.0, -1.0, -1.0]))
        mujoco.mj_camlight(self.r.model, data)

    def overlay(self, data, img, alpha=0.8):
        """Hand + object (floor hidden) rendered from the input camera, blended where something was drawn."""
        rgb = self.render(data, "input", groups=(1,))
        dep = self.render(data, "input", groups=(1,), depth=True)
        a = (dep < 0.98 * dep.max())[..., None] * alpha
        return (img * (1 - a) + rgb * a).astype(np.uint8)


def page_success(obj, Tw, ncon, pen, fps, lift=0.05, held_h=0.03, held_frac=0.8, held_min=0.5, end_s=0.3, max_pen=0.010, follow=0.05):
    """Success rule of the page examples (fixed 2026-10-28 before the candidates ran; the duration and penetration terms were set
    after one pilot replay of an earlier build on DexYCB mustard: DexYCB lifts last ~1 s, MuJoCo reports single-frame penetration
    spikes). The video lifts the object >= `lift` (else not applicable) and the simulation (1) lifts it >= `lift`, (2) holds it
    (>= `held_h` above its start and touching the hand) in >= `held_frac` of the frames where the video has it >= `held_h` up,
    and for >= `held_min` s, (3) still holds it in the last `end_s` (>= `held_h` up in every frame, touching in most), (4) the
    95th percentile over frames of the deepest hand-object penetration <= `max_pen`, (5) its centre stays within `follow` of
    the video's track (median over the held frames)."""
    up = obj[:, 2] - obj[0, 2]
    ref_up = (Tw[:, 2, 3] - Tw[0, 2, 3]) >= held_h
    held = (up >= held_h) & (ncon > 0)
    k = max(int(round(end_s * fps)), 1)
    err = np.linalg.norm(obj[:, :3] - Tw[:, :3, 3], axis=1)
    frac = float((held & ref_up).sum() / max(ref_up.sum(), 1))
    out = dict(applicable=bool((Tw[:, 2, 3] - Tw[0, 2, 3]).max() >= lift), lift_mm=round(float(up.max()) * 1e3, 1),
               held_frac=round(frac, 2), held_s=round(float(held.sum()) / fps, 2),
               end_held=bool((up[-k:] >= held_h).all() and (ncon[-k:] > 0).sum() * 2 > k),
               pen95_mm=round(float(np.percentile(pen, 95)) * 1e3, 1), max_pen_mm=round(float(pen.max()) * 1e3, 1),
               follow_mm=round(float(np.median(err[held])) * 1e3, 1) if held.any() else None)
    out["success"] = bool(out["applicable"] and up.max() >= lift and frac >= held_frac and held.sum() / fps >= held_min
                          and out["end_held"] and np.percentile(pen, 95) <= max_pen and out["follow_mm"] is not None
                          and out["follow_mm"] <= follow * 1e3)
    return out


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def write_mp4(path, frames, fps):
    import imageio

    w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=8, pixelformat="yuv420p")
    for f in frames:
        w.append_data(f[..., ::-1])
    w.close()


# ---------------------------------------------------------------- replays
def main(args):
    t_start = time.time()
    PHYS.update(weld=f"{args.weld_tc} 1", friction=f"{args.friction} 0.02 0.0002", margin=args.margin,
                solimp="0.9 0.95 0.003" if args.soft else PHYS["solimp"])
    src = load_source(args.clip, args.hand, args.object, args.track, two_hands=args.two_hands, gravity=args.gravity)
    out = os.path.join(args.clip, f"sim_{args.tag}")
    os.makedirs(out, exist_ok=True)
    n, fps = src["n"], src["fps"]
    H, W = cv2.imread(src["frames"][0]).shape[:2]
    T_cw = world_frame(src)  # camera -> world
    Tw = np.einsum("ij,njk->nik", T_cw, src["T_cam_obj"])  # object in the world per frame
    HS = []  # per hand: world vertices / joints, validity, MANO skinning weights, faces, part labels
    for hd in src["hands"]:
        Wsk_, faces_ = mano_parts(hd["side"])
        HS.append(dict(V=hd["V"] @ T_cw[:3, :3].T + T_cw[:3, 3], J=hd["J"] @ T_cw[:3, :3].T + T_cw[:3, 3], valid=hd["valid"],
                       W=Wsk_, faces=faces_, labels=Wsk_.argmax(1), side=hd["side"]))
    Vw, Jw, valid, labels, Wsk, faces = HS[0]["V"], HS[0]["J"], HS[0]["valid"], HS[0]["labels"], HS[0]["W"], HS[0]["faces"]
    if args.axis_scale:  # diagnostic: fixed per-axis scale of the object (object frame)
        src["mesh"].apply_transform(np.diag(np.r_[[float(v) for v in args.axis_scale.split(",")], 1]))
    ad = os.path.join(out, "assets")
    reuse = args.reuse_assets and os.path.exists(os.path.join(ad, "hull_0.obj"))
    if reuse:  # the convex parts (and textured mesh) of a previous run of this tag: CoACD is not deterministic
        hulls = sorted(glob.glob(os.path.join(ad, "hull_*.obj")), key=lambda f: int(f.rsplit("_", 1)[1][:-4]))
        tex = [os.path.join(ad, f) for f in os.listdir(ad) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        assets = (os.path.join(ad, "obj.obj"), tex[0] if tex else None, hulls)
    else:
        assets = export_object(src["mesh"], ad, args.coacd_seed)
    contact = None
    if args.contact:
        from scripts.real2sim.contact import refine

        for h, hs in enumerate(HS):  # each hand refined on its own; only the first may rescale a reconstructed object
            shape = h == 0 and args.object == "recon" and args.shape
            cache = os.path.join(args.clip, f"contact_{args.hand}_{args.object}{'_shape' if shape else ''}"
                                            f"{'' if args.track == 'track.npz' else '_' + args.track[:-4]}{'' if h == 0 else f'_h{h}'}.npz")
            if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(os.path.join(args.clip, args.track)):
                z = np.load(cache, allow_pickle=True)
                delta, axis_scale, info = z["delta"], z["axis_scale"], z["info"].item()
            else:
                delta, axis_scale, info = refine(hs["V"], hs["valid"], Tw, assets[2], hs["labels"], shape=shape)
                np.savez(cache, delta=delta, axis_scale=axis_scale, info=np.array(info, dtype=object))
            hs["V"], hs["J"] = hs["V"] + delta[:, None], hs["J"] + delta[:, None]
            contact = info if h == 0 else contact
            print("contact refinement, hand", h, info)
            if h == 0 and np.abs(axis_scale - 1).max() > 0.01:  # grasp-informed object scale: re-export mesh and convex parts
                src["mesh"].apply_transform(np.diag(np.r_[axis_scale, 1]))
                if not reuse:
                    assets = export_object(src["mesh"], ad, args.coacd_seed)
        Vw, Jw = HS[0]["V"], HS[0]["J"]
    # rigid MANO parts per hand: canonical = the median valid frame, per-frame poses by Kabsch; parked 5 m up when no hand.
    # Global part index = 16 * hand + MANO part.
    NP = 16 * len(HS)
    pose, cen, parts, skins = np.zeros((n, NP, 7)), np.zeros((NP, 3)), [], []
    for h, hs in enumerate(HS):
        vi = np.nonzero(hs["valid"])[0]
        ref = vi[len(vi) // 2]
        for k in range(16):
            P = hs["V"][ref][hs["labels"] == k]
            cen[16 * h + k] = P.mean(0)
            parts.append(P - P.mean(0))
        for i in range(n):
            for k in range(16):
                g = 16 * h + k
                R, t = kabsch(parts[g], hs["V"][i][hs["labels"] == k]) if hs["valid"][i] else (np.eye(3), cen[g] + [0, 0, 5.0 + h])
                pose[i, g, :3], pose[i, g, 3:] = t, quat(R)
        skins.append(dict(V0=hs["V"][ref], faces=hs["faces"], W=hs["W"], cen=cen[16 * h:16 * h + 16], offset=16 * h))
    hand = dict(cen=cen, parts=parts, skins=skins)
    valid_any = np.any([hs["valid"] for hs in HS], axis=0)
    # side camera: --side_angle around the vertical from the input camera, framing the object's path and the hand
    obj_c = Tw[:, :3, 3]
    pts = np.concatenate([obj_c] + [hs["J"][hs["valid"]][:, 0] for hs in HS])
    lo, hi = np.percentile(pts, 5, 0), np.percentile(pts, 95, 0)
    target = np.r_[np.median(obj_c[:, :2], 0), 0.4 * obj_c[:, 2].min() + 0.6 * obj_c[:, 2].max()]
    v = T_cw[:3, 3] - target  # towards the input camera, horizontal
    v[2] = 0
    ang = np.radians(args.side_angle)
    v = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]]) @ (v / np.linalg.norm(v))
    dist = max(args.side_dist, 1.8 * np.linalg.norm(hi - lo))
    eye = target + dist * (np.cos(np.radians(15)) * v + np.sin(np.radians(15)) * np.array([0, 0, 1]))
    side = (eye, look_at(eye, target))
    panels = {k: [] for k in ("input", "overlay", "kin", "phys", "phys_cam")}
    mocap = lambda m, name: m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)]

    def set_hand(m, d, p):
        for k in range(NP):
            j = mocap(m, f"hm{k}")
            d.mocap_pos[j], d.mocap_quat[j] = p[k, :3], p[k, 3:]

    # kinematic replay + overlay
    m = mujoco.MjModel.from_xml_string(build_xml(src, T_cw, assets, "kin", hand, side_cam=side, center=target))
    d = mujoco.MjData(m)
    rd = Renderer(m, H, W) if args.render else None  # --render 0: no GL context (CPU only)
    jo = mocap(m, "obj")
    for i in range(n if args.render else 0):
        img = cv2.imread(src["frames"][i])
        d.mocap_pos[jo], d.mocap_quat[jo] = Tw[i][:3, 3], quat(Tw[i][:3, :3])
        set_hand(m, d, pose[i])
        mujoco.mj_forward(m, d)
        panels["input"].append(img)
        rd.set_input_camera(d, T_cw @ src["motion"][i])
        panels["overlay"].append(rd.overlay(d, img))
        panels["kin"].append(rd.render(d, "side"))
    if rd is not None:
        rd.r.close()

    # physics replay
    m = mujoco.MjModel.from_xml_string(build_xml(src, T_cw, assets, "phys", hand, side_cam=side, density=args.density, center=target))
    d = mujoco.MjData(m)
    rd = Renderer(m, H, W) if args.render else None  # --render 0: no GL context (CPU only)
    sub = int(round(1.0 / (fps * m.opt.timestep)))
    hull_v = np.concatenate([trimesh.load(h).vertices for h in assets[2]])
    z0 = (hull_v @ Tw[0][:3, :3].T + Tw[0][:3, 3])[:, 2].min()
    z0 = z0 if z0 < 0.03 else 0.001  # resting on the support plane: put it exactly on it; already in the air (held): keep
    q0 = np.r_[Tw[0][:3, 3] - [0, 0, z0 - 0.001], quat(Tw[0][:3, :3])]
    d.qpos[:7] = q0
    ja = mocap(m, "anchor")
    d.mocap_pos[ja], d.mocap_quat[ja] = q0[:3], q0[3:]
    eq_anchor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "anchor")
    if not args.anchor:
        d.eq_active[eq_anchor] = 0
    pid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"hp{k}") for k in range(NP)]
    jadr = [m.jnt_qposadr[m.body_jntadr[b]] for b in pid]
    dadr = [m.jnt_dofadr[m.body_jntadr[b]] for b in pid]

    def teleport(p, ks=None):
        for k in (range(NP) if ks is None else ks):
            d.qpos[jadr[k]:jadr[k] + 7] = p[k]
            d.qvel[dadr[k]:dadr[k] + 6] = 0

    set_hand(m, d, pose[0])
    teleport(pose[0])
    mujoco.mj_forward(m, d)
    for _ in range(int(0.3 / m.opt.timestep)):  # settle on the plane
        mujoco.mj_step(m, d)
    settle_tilt = settle_shift = None
    if args.free_settle > 0:  # stable resting pose: the object settles under gravity (hand parked away), then is anchored there
        q_tr = d.qpos[:7].copy()
        park = pose[0].copy()
        park[:, 2] += 5.0
        set_hand(m, d, park)
        teleport(park)
        d.eq_active[eq_anchor] = 0
        for _ in range(int(args.free_settle / m.opt.timestep)):
            mujoco.mj_step(m, d)
        d.qvel[:6] = 0
        d.mocap_pos[ja], d.mocap_quat[ja] = d.qpos[:3], d.qpos[3:7]
        d.eq_active[eq_anchor] = int(bool(args.anchor))
        set_hand(m, d, pose[0])
        teleport(pose[0])
        mujoco.mj_forward(m, d)
        settle_tilt = float(np.degrees(2 * np.arccos(min(1.0, abs(float(q_tr[3:] @ d.qpos[3:7]))))))
        settle_shift = float(np.linalg.norm(d.qpos[:3] - q_tr[:3]))
    hand_geoms = set(np.nonzero(np.isin(m.geom_bodyid, pid))[0])
    obj_geoms = set(np.nonzero(m.geom_bodyid == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj"))[0])
    # surface sample for the hand-reached test (anchor release): unseeded in the reported study (one source of its run-to-run
    # variance besides the convex parts); --sample_seed >= 0 makes a replay repeatable in one environment
    Pobj = np.asarray(src["mesh"].sample(2000) if args.sample_seed < 0 else src["mesh"].sample(2000, seed=args.sample_seed))
    obj_sim, ncon, pen, released, reached = [], [], [], None, None
    moved = np.linalg.norm(Tw[:, :3, 3] - Tw[0, :3, 3], axis=1) > 0.01  # the object has started to move in the video
    part_geom = [int(np.nonzero(m.geom_bodyid == b)[0][0]) for b in pid]
    hull_geoms = [g for g in sorted(obj_geoms) if m.geom_contype[g]]
    shift, fromto, shifts = np.zeros((NP, 3)), np.zeros(6), []
    fingers = [16 * h + k for h in range(len(HS)) for k in range(1, 16)]  # finger segments; the palms follow the estimate
    ctrl_parts = list(range(NP)) if args.ctrl_pen > 0 else fingers  # --ctrl_pen: palms too, for the penetration relief only
    if args.ctrl_flex:  # fingers close in joint space (flex_pose); the Cartesian target shifts are left to the palms
        ctrl_parts = [16 * h for h in range(len(HS))] if args.ctrl_pen > 0 else []
    theta, flexes, t_touch = np.zeros((len(HS), 5, 3)), [], np.full((len(HS), 5, 3), np.nan)
    th_max = np.radians(args.flex_max) * np.array(FLEX_GAIN)
    piv_axes = lambda i: [(lambda pv: (pv, flex_axes(pv, hs["side"])))(joint_pivots(hs["V"][i], hs["W"])) for hs in HS]

    def flexed(i):
        p = pose[i]
        if not args.ctrl_flex or not theta.any():
            return p
        for h, (pv, ax) in enumerate(piv_axes(i)):
            if HS[h]["valid"][i]:
                p = flex_pose(p, pv, ax, theta[h], 16 * h)
        return p

    prev_flexed = pose[0]
    for i in range(n):
        if args.ctrl_flex and reached is not None and valid_any[i]:  # joint-space closing until contact or the joint limit
            step = np.radians(args.flex_step)
            for h in range(len(HS)):
                if not HS[h]["valid"][i]:
                    theta[h] *= 0.7
                    continue
                for f, chain in enumerate(FINGERS):
                    dist = np.array([min([mujoco.mj_geomDistance(m, d, part_geom[16 * h + k], g, args.ctrl_near, fromto)
                                          for g in hull_geoms] or [args.ctrl_near]) for k in chain])
                    if args.ctrl_pen > 0 and dist.min() < -args.ctrl_pen:  # too deep: open the whole finger
                        theta[h, f] = np.maximum(theta[h, f] - step, np.radians(-args.flex_open))
                        t_touch[h, f] = np.nan
                        continue
                    for j in range(3):  # a joint closes until a segment distal of it touches (MCP stops first, the tip wraps)
                        if dist[j:].min() <= args.flex_touch:  # touching: squeeze --flex_squeeze past first touch, then hold
                            if np.isnan(t_touch[h, f, j]):
                                t_touch[h, f, j] = theta[h, f, j]
                            theta[h, f, j] = min(theta[h, f, j] + step, t_touch[h, f, j] + np.radians(args.flex_squeeze), th_max[j])
                            continue
                        t_touch[h, f, j] = np.nan
                        if dist.min() < args.ctrl_near:  # finger near, not touching here: close
                            theta[h, f, j] = min(theta[h, f, j] + step, th_max[j])
                        else:  # nothing within reach: relax towards the estimate
                            theta[h, f, j] *= 0.7
        elif args.ctrl_flex:
            theta *= 0.7
        flexes.append(np.degrees(theta.max(-1)).copy())
        if args.controller and reached is not None and valid_any[i]:  # contact-seeking grasp controller (closed loop on the sim)
            for k in ctrl_parts:
                if not HS[k // 16]["valid"][i]:
                    shift[k] *= 0.7
                    continue
                best, ft = args.ctrl_near, None
                for g in hull_geoms:
                    dist = mujoco.mj_geomDistance(m, d, part_geom[k], g, args.ctrl_near, fromto)
                    if dist < best:
                        best, ft = dist, fromto.copy()
                u = None if ft is None else (ft[3:] - ft[:3]) / max(np.linalg.norm(ft[3:] - ft[:3]), 1e-9)  # moves the part to contact
                if ft is None:  # nothing within reach: relax towards the estimate
                    want = 0.7 * shift[k]
                elif args.ctrl_pen > 0 and best < -args.ctrl_pen:  # too deep inside: back out to --ctrl_pen
                    want = shift[k] + u * (-best - args.ctrl_pen)
                elif best > 0 and k % 16:  # finger near but not touching: move the target onto the surface (+ squeeze)
                    want = shift[k] + u * (best + args.ctrl_squeeze)
                else:  # touching: hold
                    want = shift[k]
                nw = np.linalg.norm(want)
                want = want * min(1.0, args.ctrl_max / max(nw, 1e-9))
                shift[k] = 0.5 * shift[k] + 0.5 * want
        elif args.controller:
            shift *= 0.7
        shifts.append(np.linalg.norm(shift, axis=1).max())
        cur = flexed(i)
        if i > 0:
            tele = [k for k in range(NP) if HS[k // 16]["valid"][i] and not HS[k // 16]["valid"][i - 1]]  # hands appearing
            if tele:
                set_hand(m, d, cur)
                teleport(cur, tele)
            for s_ in range(sub):
                a = (s_ + 1) / sub
                p = cur.copy()
                for k in range(NP):
                    if k not in tele:
                        p[k, :3] = (1 - a) * prev_flexed[k, :3] + a * cur[k, :3]
                        p[k, 3:] = slerp(prev_flexed[k, 3:], cur[k, 3:], a)
                p[:, :3] += shift
                set_hand(m, d, p)
                mujoco.mj_step(m, d)
        prev_flexed = cur
        c = [d.contact[j] for j in range(d.ncon) if (d.contact[j].geom1 in hand_geoms and d.contact[j].geom2 in obj_geoms)
             or (d.contact[j].geom2 in hand_geoms and d.contact[j].geom1 in obj_geoms)]
        if reached is None:
            Ro = np.zeros(9)
            mujoco.mju_quat2Mat(Ro, d.qpos[3:7])
            X = Pobj @ Ro.reshape(3, 3).T + d.qpos[:3]
            if c or any(hs["valid"][i] and np.min(np.linalg.norm(hs["V"][i][::4, None] - X[None, ::4], axis=-1)) < args.release_dist
                        for hs in HS):
                reached = i
        if released is None and d.eq_active[eq_anchor] and reached is not None and (not args.hold_until_lift or moved[i]):
            d.eq_active[eq_anchor] = 0  # the object rests where it was until the hand reaches it (--hold_until_lift: until it moves)
            released = i
        ncon.append(len(c))
        pen.append(-min([x.dist for x in c], default=0.0))
        obj_sim.append(d.qpos[:7].copy())
        if args.render:
            panels["phys"].append(rd.render(d, "side"))
            rd.set_input_camera(d, T_cw @ src["motion"][i])
            panels["phys_cam"].append(rd.overlay(d, cv2.imread(src["frames"][i])))
    if rd is not None:
        rd.r.close()
    obj_sim = np.array(obj_sim)
    perr = np.linalg.norm(obj_sim[:, :3] - Tw[:, :3, 3], axis=1)
    lift_ref = Tw[:, 2, 3] - Tw[0, 2, 3]
    lift_sim = obj_sim[:, 2] - obj_sim[0, 2]
    held = lift_ref > 0.03
    metrics = dict(hand=args.hand, object=args.object, frames=n, contact_refine=contact, phys=dict(PHYS),
                   mass_kg=round(float(m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj")]), 3),
                   anchor_released_frame=released if args.anchor else None, hand_reached_frame=reached,
                   hold_until_lift=bool(args.hold_until_lift), free_settle_s=args.free_settle,
                   settle_tilt_deg=None if settle_tilt is None else round(settle_tilt, 1),
                   settle_shift_mm=None if settle_shift is None else round(settle_shift * 1e3, 1),
                   obj_pos_err_mm_median=round(float(np.median(perr)) * 1e3, 1), obj_pos_err_mm_final=round(float(perr[-1]) * 1e3, 1),
                   lift_ref_mm=round(float(lift_ref[-1]) * 1e3, 1), lift_sim_mm=round(float(lift_sim[-1]) * 1e3, 1),
                   follow_frames=int(((perr < 0.03) & held).sum()), lifted_frames=int(held.sum()),
                   success=bool(lift_ref[-1] > 0.03 and lift_sim[-1] > 0.5 * lift_ref[-1] and perr[-1] < 0.05),
                   hand_obj_contacts_median=float(np.median(ncon)), max_penetration_mm=round(float(max(pen)) * 1e3, 1),
                   page=page_success(obj_sim, Tw, np.array(ncon), np.array(pen), fps),
                   controller=bool(args.controller), ctrl_shift_mm_max=round(float(max(shifts)) * 1e3, 1),
                   ctrl_shift_mm_median=round(float(np.median([x for x in shifts if x > 0] or [0])) * 1e3, 1))
    if args.ctrl_flex:  # per hand and finger (index, middle, little, ring, thumb): largest closing angle, frames closed > 5 deg
        fl = np.array(flexes)
        metrics.update(ctrl_flex=True, flex_deg_max=np.round(fl.max(0), 1).tolist(), flex_frames=(fl > 5).sum(0).tolist(),
                       fingers_closed=int((fl.max(0) > 5).sum()))
    # robot retarget (kinematic), human hand as a ghost
    if args.robot:
        import dex_retargeting
        from dex_retargeting.retargeting_config import RetargetingConfig

        RetargetingConfig.set_default_urdf_dir(URDF_DIR)
        cfg = RetargetingConfig.load_from_file(os.path.join(os.path.dirname(dex_retargeting.__file__), "configs", "offline",
                                                            f"{args.robot}_hand_{'right' if src['side'] else 'left'}.yml"))
        rt = cfg.build()
        robot = rt.optimizer.robot
        idx = np.asarray(rt.optimizer.target_link_human_indices)
        vis = urdf_visuals(os.path.join(URDF_DIR, cfg.urdf_path))
        m = mujoco.MjModel.from_xml_string(build_xml(src, T_cw, assets, "robot", hand, robot=dict(visuals=vis), side_cam=side,
                                                     center=target))
        d = mujoco.MjData(m)
        rd = Renderer(m, H, W)
        links = list(vis)
        lid = [robot.get_link_index(nm) for nm in links]
        mids = [mocap(m, f"rl_{nm}") for nm in links]
        tid = [robot.get_link_index(nm) for nm in cfg.target_link_names]
        jo = mocap(m, "obj")
        panels["robot"], tip_err, qs, first = [], [], [], True
        for i in range(n):
            d.mocap_pos[jo], d.mocap_quat[jo] = Tw[i][:3, 3], quat(Tw[i][:3, :3])
            set_hand(m, d, pose[i])
            if valid[i]:
                ref_v = Jw[i][idx]
                for _ in range(20 if first else 1):
                    q = rt.retarget(ref_v)
                first = False
                robot.compute_forward_kinematics(q)
                for l, k in zip(lid, mids):
                    P = robot.get_link_pose(l)
                    d.mocap_pos[k], d.mocap_quat[k] = P[:3, 3], quat(P[:3, :3])
                tip_err.append(np.linalg.norm(np.array([robot.get_link_pose(t)[:3, 3] for t in tid]) - ref_v, axis=1).mean())
                qs.append(q)
            else:
                for k in mids:
                    d.mocap_pos[k] = [0, 0, 5.0]
            mujoco.mj_forward(m, d)
            panels["robot"].append(rd.render(d, "side"))
        rd.r.close()
        metrics.update(robot=args.robot, retarget_keypoint_err_mm=round(float(np.mean(tip_err)) * 1e3, 1))
        np.save(os.path.join(out, f"qpos_{args.robot}.npy"), np.array(qs))
    hs, os_ = ("FAH" if args.hand == "fah" else "GT"), ("reconstructed" if args.object == "recon" else "GT")
    names = dict(input="input video", overlay=f"{hs} hand + {os_} object", kin="MuJoCo kinematic replay",
                 phys="MuJoCo physics: free object, hand = 16 convex parts", phys_cam="MuJoCo physics, camera view",
                 robot=f"retarget to {args.robot.upper()} hand (kinematic)")
    if not args.render:
        panels = {}
    stills = [int(v) for v in args.stills.split(",") if v]
    seq = list(range(0, n, args.still_stride)) if args.still_stride else []
    for i in stills + seq:  # unlabelled frames for figures: stills/<panel>_<frame>.png (+ the sequence in stills/seq/)
        sd = os.path.join(out, "stills", "" if i in stills else "seq")
        os.makedirs(sd, exist_ok=True)
        for k in ("input", "overlay", "phys", "phys_cam", "kin"):
            if k in panels and i < len(panels[k]):
                cv2.imwrite(os.path.join(sd, f"{k}_{i:06d}.png"), panels[k][i])
    for k, v in panels.items():
        write_mp4(os.path.join(out, f"{k}.mp4"), [label(f.copy(), names[k]) for f in v], args.out_fps or fps)
    order = [k for k in args.grid.split(",") if k in panels]
    cols = 2 if len(order) <= 4 else 3
    grid = []
    for i in range(n if order else 0):
        tiles = [label(panels[k][i].copy(), names[k]) for k in order]
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        grid.append(np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], 1) for r in range(len(tiles) // cols)], 0))
    if grid:
        write_mp4(os.path.join(out, "side_by_side.mp4"), grid, args.out_fps or fps)
        pf = int(np.argmax(lift_ref > 0.7 * lift_ref.max())) if lift_ref.max() > 0.03 else n // 2
        cv2.imwrite(os.path.join(out, "poster.jpg"), grid[pf], [cv2.IMWRITE_JPEG_QUALITY, 88])
    metrics["sim_s"] = round(time.time() - t_start, 1)
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    np.savez(os.path.join(out, "sim_traj.npz"), obj_sim=obj_sim, obj_ref=Tw, T_world_cam=T_cw, ncon=np.array(ncon), pen=np.array(pen))
    print(out, json.dumps(metrics))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True)
    p.add_argument("--hand", default="fah", choices=["fah", "gt"])
    p.add_argument("--track", default="track.npz", help="track.py output in the clip")
    p.add_argument("--object", default="recon", choices=["recon", "gt"])
    p.add_argument("--robot", default="", help="dex-retargeting offline config prefix (leap, allegro, shadow, inspire; '' = off; "
                   "dropped from the demo 2026-10-27, user)")
    p.add_argument("--grid", default="input,overlay,kin,phys", help="panels of side_by_side.mp4 (+ phys_cam, robot); 2 columns "
                   "up to 4 panels, else 3")
    p.add_argument("--density", type=float, default=300.0, help="object density kg/m^3")
    p.add_argument("--side_angle", type=float, default=75.0)
    p.add_argument("--side_dist", type=float, default=0.75)
    p.add_argument("--out_fps", type=float, default=15.0, help="playback fps of the videos (DexYCB clips are 2.4 s at 30 fps)")
    p.add_argument("--anchor", type=int, default=1, help="hold the object at its start pose until the hand reaches it")
    p.add_argument("--release_dist", type=float, default=0.01, help="m: anchor release when a hand vertex is this close")
    p.add_argument("--contact", type=int, default=0, help="hand-object contact refinement (contact.py) before the replays")
    p.add_argument("--shape", type=int, default=1, help="with --contact and a reconstructed object: grasp-informed per-axis object scale")
    p.add_argument("--axis_scale", default="", help="diagnostic: sx,sy,sz object scale before everything")
    p.add_argument("--weld_tc", type=float, default=0.01, help="hand part -> target weld time constant (s)")
    p.add_argument("--friction", type=float, default=2.0, help="hand-object sliding friction (defaults = best of a 6-setting sweep on "
                   "the 3 DexYCB demo clips: 8 / 9 replays hold vs 4 / 9 with 1.2, hard contacts, no margin)")
    p.add_argument("--margin", type=float, default=0.003, help="hand geom contact margin (m)")
    p.add_argument("--soft", type=int, default=1, help="softer contacts (solimp 0.9 0.95 3 mm)")
    p.add_argument("--coacd_seed", type=int, default=0, help="CoACD seed (repeats: the convex parts differ slightly)")
    p.add_argument("--sample_seed", type=int, default=-1, help="seed of the object surface sample of the hand-reached test "
                   "(-1: unseeded, as in the reported study)")
    p.add_argument("--render", type=int, default=1, help="0: physics metrics only (repeats)")
    p.add_argument("--reuse_assets", type=int, default=0, help="reuse sim_<tag>/assets of a previous run (same convex parts: "
                   "re-render a metrics-only run without re-running CoACD)")
    p.add_argument("--free_settle", type=float, default=0.0, help="s: let the object settle freely under gravity (hand parked "
                   "away) into a stable resting pose before the replay; it is then anchored there as usual")
    p.add_argument("--two_hands", type=int, default=0, help="also simulate the other hand (track.py --two_hands 1)")
    p.add_argument("--gravity", default="plane", choices=["plane", "headset"], help="up direction: the support plane fitted on the "
                   "depth (track.py), or for moving cameras the headset world's +z (HOT3D)")
    p.add_argument("--hold_until_lift", type=int, default=0, help="keep the object anchored until it starts to move in the video "
                   "(tests whether the grasp carries it, not whether the approach knocks it over)")
    p.add_argument("--controller", type=int, default=0, help="contact-seeking grasp controller: after the hand reaches the "
                   "object, finger segments within --ctrl_near of it get their targets moved onto its surface (closed loop)")
    p.add_argument("--ctrl_near", type=float, default=0.02)
    p.add_argument("--ctrl_squeeze", type=float, default=0.002)
    p.add_argument("--ctrl_max", type=float, default=0.015, help="m: largest target shift per finger segment")
    p.add_argument("--ctrl_pen", type=float, default=0.0, help="m: > 0: any hand part deeper than this inside the object gets its target "
                   "backed out to it (penetration relief, palms included)")
    p.add_argument("--ctrl_flex", type=int, default=0, help="fingers close in joint space instead of the Cartesian target shifts: "
                   "per finger, while a segment is within --ctrl_near of the object and none touches it, its MCP / PIP / DIP flexion "
                   "grows by --flex_step deg per frame up to --flex_max (opens when deeper than --ctrl_pen, relaxes when far)")
    p.add_argument("--flex_step", type=float, default=4.0)
    p.add_argument("--flex_max", type=float, default=30.0, help="deg: largest added flexion per finger (MCP / PIP; DIP x 0.8)")
    p.add_argument("--flex_open", type=float, default=10.0, help="deg: largest opening below the estimate (penetration relief)")
    p.add_argument("--flex_touch", type=float, default=0.003, help="m: a segment this close counts as touching (= the contact margin)")
    p.add_argument("--flex_squeeze", type=float, default=0.0, help="deg: a touching finger keeps closing this much past first touch "
                   "(grip force through the weld)")
    p.add_argument("--stills", default="", help="frames to save unlabelled (stills/<panel>_<frame>.png), e.g. 110")
    p.add_argument("--still_stride", type=int, default=0, help="also every Nth frame into stills/seq/")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    args.tag = args.tag or f"{args.hand}_{args.object}" + ("_contact" if args.contact else "") + ("_settle" if args.free_settle > 0 else "") + ("_ctrl" if args.controller else "") + ("_flex" if args.ctrl_flex else "") + ("_hold" if args.hold_until_lift else "")
    main(args)
