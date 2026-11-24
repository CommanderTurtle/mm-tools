"""Unlabelled kinematic stills of one frame of a real-to-sim clip for figures: the reconstructed object at its tracked pose and
FAH's hand meshes of that frame as estimated (no contact refinement, no controller, no physics step) -> <clip>/still_<tag>/:
input_<f>.png, overlay_<f>.png (hands + object over the input frame, as sim.py's overlay), kin_side_a<angle>_<f>.png (MuJoCo
side view on a plain light background, cropped to the hands and object, input resolution) and still.json (provenance).
The side views use sim.py's world (support plane up) and --side_angles around the vertical from the frame's camera.

    MUJOCO_GL=egl python scripts/real2sim/still.py --clip $R2S/h3_milk --frame 110 --two_hands 1
"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

from scripts.real2sim.sim import SKIN, Renderer, camera_xml, fmt, load_source, look_at, mano_parts, quat, world_frame  # noqa: E402

BG = 0.96


def export_visual(mesh, out):
    """Textured OBJ (+ texture png) of the object, no collision parts."""
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
    return os.path.join(out, "obj.obj"), tex


def scene_xml(obj_file, tex, T_obj, hands, K, H, W, T_wc, sides, fovy, size):
    a = ['<asset>', f'<texture type="skybox" builtin="flat" rgb1="{BG} {BG} {BG}" rgb2="{BG} {BG} {BG}" width="16" height="16"/>',
         f'<mesh name="obj" file="{obj_file}" inertia="shell"/>']
    a += [f'<texture name="tex" type="2d" file="{tex}"/>', '<material name="obj" texture="tex" specular="0.2"/>'] if tex else \
        ['<material name="obj" rgba=".85 .65 .2 1"/>']
    a += [f'<mesh name="hand{h}" vertex="{fmt(V)}" face="{fmt(F)}" inertia="shell"/>' for h, (V, F) in enumerate(hands)]
    b = ['<worldbody>', '<light directional="true" dir="-0.3 0.4 -1" diffuse=".5 .5 .5" specular=".1 .1 .1" castshadow="false"/>',
         camera_xml("input", K, H, W, T_wc)]
    b += [f'<camera name="side{k}" pos="{fmt(e)}" xyaxes="{fmt(x)}" fovy="{fovy}"/>' for k, (e, x) in enumerate(sides)]
    b.append(f'<body name="obj" pos="{fmt(T_obj[:3, 3])}" quat="{fmt(quat(T_obj[:3, :3]))}"><geom type="mesh" mesh="obj" '
             'material="obj" group="1" contype="0" conaffinity="0"/></body>')
    b += [f'<geom type="mesh" mesh="hand{h}" rgba="{fmt(SKIN)}" group="1" contype="0" conaffinity="0"/>' for h in range(len(hands))]
    return ('<mujoco model="still"><visual><map znear="0.01" zfar="20"/><quality offsamples="8"/>'
            '<headlight ambient=".45 .45 .45" diffuse=".4 .4 .4"/>'
            f'<global offwidth="{max(W, size)}" offheight="{max(H, size)}"/></visual>'
            + "".join(a) + '</asset>' + "".join(b) + '</worldbody></mujoco>')


def crop(rgb, dep, out, margin=0.06):
    """Square crop around everything drawn (+ margin), resized to out x out."""
    ys, xs = np.nonzero(dep < 0.98 * dep.max())
    c = np.array([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2])
    r = 0.5 * max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * margin)
    x0, y0 = int(round(c[0] - r)), int(round(c[1] - r))
    s = int(round(2 * r))
    pad = max(0, -x0, -y0, x0 + s - rgb.shape[1], y0 + s - rgb.shape[0])
    bg = np.full(rgb.shape[2], int(round(BG * 255)), np.uint8)
    img = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg.tolist())
    return cv2.resize(img[y0 + pad:y0 + pad + s, x0 + pad:x0 + pad + s], (out, out), interpolation=cv2.INTER_AREA)


def main(args):
    src = load_source(args.clip, "fah", "recon", args.track, two_hands=args.two_hands)
    out = os.path.join(args.clip, f"still_{args.tag}")
    os.makedirs(out, exist_ok=True)
    i = args.frame
    H, W = cv2.imread(src["frames"][i]).shape[:2]
    T_cw = world_frame(src)
    T_obj = T_cw @ src["T_cam_obj"][i]
    hands = []
    for hd in src["hands"]:
        if not hd["valid"][i]:
            continue
        hands.append((hd["V"][i] @ T_cw[:3, :3].T + T_cw[:3, 3], mano_parts(hd["side"])[1]))
    P = np.concatenate([np.asarray(src["mesh"].vertices) @ T_obj[:3, :3].T + T_obj[:3, 3]] + [V for V, _ in hands])
    target = (P.min(0) + P.max(0)) / 2
    r = np.linalg.norm(P - target, axis=1).max()
    dist = 1.1 * r / np.sin(np.radians(args.fovy / 2))
    T_wc = T_cw @ src["motion"][i]
    v = T_wc[:3, 3] - target  # towards the frame's camera, horizontal
    v[2] = 0
    v /= np.linalg.norm(v)
    angles = [float(a) for a in args.side_angles.split(",")]
    sides = []
    for ang in np.radians(angles):
        u = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]]) @ v
        eye = target + dist * (np.cos(np.radians(args.elev)) * u + np.sin(np.radians(args.elev)) * np.array([0, 0, 1]))
        sides.append((eye, look_at(eye, target)))
    obj_file, tex = export_visual(src["mesh"], os.path.join(out, "assets"))
    m = mujoco.MjModel.from_xml_string(scene_xml(obj_file, tex, T_obj, hands, src["K"], H, W, T_wc, sides, args.fovy, args.render_size))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    img = cv2.imread(src["frames"][i])
    rd = Renderer(m, H, W)
    cv2.imwrite(os.path.join(out, f"input_{i:06d}.png"), img)
    cv2.imwrite(os.path.join(out, f"overlay_{i:06d}.png"), rd.overlay(d, img))
    rd.r.close()
    rd = Renderer(m, args.render_size, args.render_size)
    for k, ang in enumerate(angles):
        rgb, dep = rd.render(d, f"side{k}", groups=(1,)), rd.render(d, f"side{k}", groups=(1,), depth=True)
        cv2.imwrite(os.path.join(out, f"kin_side_a{ang:g}_{i:06d}.png"), crop(rgb, dep, args.size or W))
    rd.r.close()
    hp = pickle.load(open(os.path.join(args.clip, "hands.pkl"), "rb"))
    info = dict(clip=args.clip, frame=i, track=args.track, hands_tree=hp.get("tree"), method_args=hp.get("method_args"),
                hands_drawn=len(hands), hands_detected_in_frame=len(hp["frames"][i]), side_angles=angles, elev=args.elev,
                fovy=args.fovy, K=np.asarray(src["K"]).tolist(), contact_refinement=False, physics=False)
    json.dump(info, open(os.path.join(out, "still.json"), "w"), indent=1, default=str)
    print(out, len(hands), "hands")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--track", default="track.npz")
    p.add_argument("--two_hands", type=int, default=0)
    p.add_argument("--side_angles", default="45,75,105")
    p.add_argument("--elev", type=float, default=15.0)
    p.add_argument("--fovy", type=float, default=30.0)
    p.add_argument("--render_size", type=int, default=1408)
    p.add_argument("--size", type=int, default=0, help="output px of the side views (0: input width)")
    p.add_argument("--tag", default="kin")
    main(p.parse_args())
