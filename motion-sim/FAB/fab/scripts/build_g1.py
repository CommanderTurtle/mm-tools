"""Build FAB's Unitree G1 (29 DoF) description from Unitree's own g1_description (unitree_ros, robots/g1_description;
BSD-3-Clause, Copyright Unitree Robotics), as models/fab_assets.py installs it under $FAB_HOME/g1:

    g1_29dof.urdf        Unitree's g1_29dof.urdf with the Model 15 effort / velocity limits of 14 joints
                         (fab/retarget/g1_model15.py) and FAB's frames (below) as fixed links (1 g / 10 g marker inertials,
                         Unitree's convention for its own fixed links), each ankle point listed before its ankle pitch joint
                         and the sole points front to back (ball spheres, toe, heel spheres); one IMU frame (the torso's:
                         Unitree's imu_in_pelvis is left out)
    g1_29dof.xml         Unitree's g1_29dof.xml (bodies, joints, inertials, meshes, collision geoms) with
                         - a body for each fixed URDF link that carries a mesh: pelvis_contour_link and <side>_rubber_hand
                           (their meshes move into them, the hand mesh as a visual and a collision geom) and
                           waist_support_link (a copy of the waist-support meshes; the torso keeps its own);
                         - FAB's frames as child bodies, each with two non-colliding marker spheres (a 5 mm visual one and
                           the frame's contact radius: 5 mm on the sole and toe points, 1 mm on the hand points, 1 cm on the
                           ankle point);
                         - the meshes as OBJ files: each STL's distinct vertices in their order of first appearance, at 8
                           decimal places (metres), and its triangles in file order;
                         Unitree's actuators, sensors and demo floor are dropped
    unitree_meshes/      the meshes that model uses (OBJ) and the URDF's STL files
    g1_gate_robot.xml    the collision model of the stance pin's physics checks: Unitree's g1_29dof_rev_1_0.xml (bodies,
                         joints, inertials) with its geoms replaced by FAB's collision primitives (CAPSULES below)
    scene_g1_gate.xml    FAB's MuJoCo scene of those checks: g1_gate_robot.xml on a floor plane, timestep 2 ms, Newton
                         solver, 50 iterations

FAB's frames (per side, in the parent link's coordinates), all placed from Unitree's geometry:
    <side>_sole_1 .. _4  in <side>_ankle_roll_link: the centres of the four foot contact spheres of Unitree's MJCF
                         (heel left / right, ball left / right; radius 5 mm)
    <side>_toe           in <side>_ankle_roll_link: 2 cm in front of the midpoint of the two front (ball) contact spheres,
                         at their height (radius 5 mm)
    <side>_ankle         in <side>_knee_link: on the link's z axis, 0.28 m below the knee joint (2 cm above the ankle pitch
                         joint)
    <side>_thumb         in <side>_rubber_hand (Unitree's rubber-hand frame): (0.08, 0, 0.05), the thumb side of the hand
    <side>_pinky         in <side>_rubber_hand: (0.12, 0, -0.04), the little-finger side (radius 1 mm)
The skeleton correspondence (fab/retarget/g1.py) uses <side>_ankle, <side>_toe and Unitree's rubber-hand frame.

    python scripts/build_g1.py --src <unitree_ros>/robots/g1_description --out <dir>
"""
import argparse
import copy
import re
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SIDES = ("left", "right")
# Model 15 (Unitree's g1_29dof_mode_15 values): joint -> (effort N m, velocity rad / s)
M15 = {"left_hip_pitch_joint": ("139", "20"), "left_hip_roll_joint": ("139", "20"), "left_ankle_pitch_joint": ("35", "30"),
       "left_ankle_roll_joint": ("35", "30"), "right_hip_pitch_joint": ("139", "20"), "right_hip_roll_joint": ("139", "20"),
       "right_ankle_pitch_joint": ("35", "30"), "right_ankle_roll_joint": ("35", "30"), "waist_roll_joint": ("35", "30"),
       "waist_pitch_joint": ("35", "30"), "left_wrist_pitch_joint": ("10", "27"), "left_wrist_yaw_joint": ("10", "27"),
       "right_wrist_pitch_joint": ("10", "27"), "right_wrist_yaw_joint": ("10", "27")}
SOLE_R = 0.005  # the contact spheres' radius (Unitree's)
POINT_R = 0.001  # the thumb / pinky markers: points
ANKLE_R = 0.01  # the ankle point's marker
VIS_R = 0.005  # the visual marker sphere of every frame
TOE_AHEAD = 0.02  # the toe point in front of the ball contact spheres (m)
ANKLE_BELOW_KNEE = 0.28  # the ankle point below the knee joint (m)
THUMB, PINKY = (0.08, 0.0, 0.05), (0.12, 0.0, -0.04)  # in the rubber-hand frame (m)
# FAB's collision primitives of the stance pin's gate model, per body in Unitree's body order: (body, type, size, fromto or
# pos). Pelvis and torso / head: spheres and capsules around the shells; legs: one thigh and one shin capsule; feet: seven
# 1 cm sole capsules along the foot (8 mm for the right foot's second and sixth); arms: upper-arm, forearm and hand capsules.
_FOOT = [("0.1 -0.026 -0.025 0.05 -0.027 -0.025", 0.01), ("-0.044 -0.018 -0.025 0.123 -0.018 -0.025", 0.01),
         ("-0.052 -0.01 -0.025 0.13 -0.01 -0.025", 0.01), ("-0.054 0 -0.025 0.132 0 -0.025", 0.01),
         ("-0.052 0.01 -0.025 0.13 0.01 -0.025", 0.01), ("-0.044 0.018 -0.025 0.123 0.018 -0.025", 0.01),
         ("0.1 0.026 -0.025 0.05 0.026 -0.025", 0.01)]
_FOOT_R = [("0.1 -0.026 -0.025 0.05 -0.026 -0.025", 0.01), ("-0.044 -0.018 -0.025 0.123 -0.018 -0.025", 0.008)] + _FOOT[2:5] + \
          [("-0.044 0.018 -0.025 0.123 0.018 -0.025", 0.008), _FOOT[6]]
CAPSULES = [("pelvis", "sphere", "0.07", "0 0 -0.08")]
for _s, _ft in (("left", _FOOT), ("right", _FOOT_R)):
    CAPSULES += [(f"{_s}_hip_roll_link", "capsule", "0.05", "0.02 0 0 0.02 0 -0.2"),
                 (f"{_s}_knee_link", "capsule", "0.04", "0.02 0 0 0.02 0 -0.25")]
    CAPSULES += [(f"{_s}_ankle_roll_link", "capsule", f"{r:g}", ft) for ft, r in _ft]
CAPSULES += [("torso_link", "capsule", "0.073", "0.005 -0.032 .22 0.005 0.032 .22"),
             ("torso_link", "capsule", "0.07", "0.005 -0.028 .13 0.005 0.028 .13"),
             ("torso_link", "capsule", "0.065", "0.005 -0.02 .06 0.005 0.02 .06"),
             ("torso_link", "capsule", "0.068", "0.01 0 .41 0.01 0 .42")]
for _s in SIDES:
    CAPSULES += [(f"{_s}_shoulder_yaw_link", "capsule", "0.035", "0 0 -0.08 0 0 0.05"),
                 (f"{_s}_elbow_link", "capsule", "0.035", "-0.01 0 -0.01 0.12 0 -0.01"),
                 (f"{_s}_wrist_yaw_link", "capsule", "0.05", "0.05 0 0 0.1 0 0")]


def stl_triangles(path):
    """Binary STL -> (n, 3, 3) float32 triangle vertices."""
    b = Path(path).read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    assert 84 + 50 * n == len(b), f"{path}: not a binary STL"
    a = np.frombuffer(b[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n)
    return a["v"]


def stl_to_obj(src, dst):
    """The STL's distinct vertices in their order of first appearance at 8 decimal places, and its triangles in file order."""
    V = stl_triangles(src).reshape(-1, 3)
    _, first, inv = np.unique(V, axis=0, return_index=True, return_inverse=True)
    rank = np.empty(len(first), int)
    rank[np.argsort(first)] = np.arange(len(first))
    U, F = V[np.sort(first)], rank[inv.reshape(-1)].reshape(-1, 3) + 1
    with open(dst, "w") as f:
        f.writelines("v %.8f %.8f %.8f\n" % tuple(float(x) for x in u) for u in U)
        f.writelines("f %d %d %d\n" % tuple(t) for t in F)


def fmt(p):
    return " ".join(f"{float(x):.6g}" for x in p)


def frames(mj):
    """{side: {name: (parent link, xyz, marker sphere radius or None)}} from Unitree's MJCF (the foot spheres)."""
    out = {}
    for s in SIDES:
        foot = mj.find(f".//body[@name='{s}_ankle_roll_link']")
        sph = [g for g in foot.findall("geom") if g.get("type", "sphere") == "sphere" and g.get("mesh") is None]  # (MJCF default type)
        assert len(sph) == 4, (s, len(sph))
        sole = {f"{s}_sole_{k + 1}": (f"{s}_ankle_roll_link", [float(x) for x in g.get("pos").split()], SOLE_R) for k, g in enumerate(sph)}
        assert all(abs(float(g.get("size").split()[0]) - SOLE_R) < 1e-9 for g in sph)
        front = sorted(sole, key=lambda n: -sole[n][1][0])[:2]  # the two spheres with the largest x (Unitree's 3 and 4)
        assert front == [f"{s}_sole_3", f"{s}_sole_4"], front
        toe = np.mean([sole[n][1] for n in front], 0) + [TOE_AHEAD, 0.0, 0.0]
        f = {n: sole[n] for n in front}  # front to back: the ball spheres, the toe, the heel spheres
        f[f"{s}_toe"] = (f"{s}_ankle_roll_link", toe.tolist(), SOLE_R)
        f.update({n: sole[n] for n in sole if n not in front})
        f[f"{s}_ankle"] = (f"{s}_knee_link", [0.0, 0.0, -ANKLE_BELOW_KNEE], ANKLE_R)
        f[f"{s}_thumb"] = (f"{s}_rubber_hand", list(THUMB), POINT_R)
        f[f"{s}_pinky"] = (f"{s}_rubber_hand", list(PINKY), POINT_R)
        out[s] = f
    return out


def build_urdf(src, fr, out):
    s = (src / "g1_29dof.urdf").read_text()
    for name, (effort, vel) in M15.items():
        m = re.search(r'(<joint name="%s"[^>]*>.*?<limit[^>]*/>)' % re.escape(name), s, re.S)
        assert m, name
        blk = re.sub(r'effort="[^"]*"', f'effort="{effort}"', m.group(1))
        blk = re.sub(r'velocity="[^"]*"', f'velocity="{vel}"', blk)
        s = s[:m.start()] + blk + s[m.end():]
    s = s.replace('filename="meshes/', 'filename="unitree_meshes/')
    for pat in (r'\s*<link name="imu_in_pelvis">.*?</link>', r'\s*<joint name="imu_in_pelvis_joint".*?</joint>'):  # one IMU frame
        s, n = re.subn(pat, "", s, flags=re.S)
        assert n == 1, pat

    def block(name, parent, xyz):
        mass, inertia = ("0.01", "1e-6") if name.endswith("_ankle") else ("0.001", "1e-7")
        return [f'  <link name="{name}">', '    <inertial>', '      <origin xyz="0 0 0" rpy="0 0 0"/>', f'      <mass value="{mass}"/>',
                f'      <inertia ixx="{inertia}" ixy="0" ixz="0" iyy="{inertia}" iyz="0" izz="{inertia}"/>', '    </inertial>',
                '  </link>', f'  <joint name="{name}_joint" type="fixed">', f'    <origin xyz="{fmt(xyz)}" rpy="0 0 0"/>',
                f'    <parent link="{parent}"/>', f'    <child link="{name}"/>', "  </joint>"]

    add = ["", "  <!-- FAB's frames (scripts/build_g1.py) -->"]
    for side in SIDES:
        for name, (parent, xyz, _) in fr[side].items():
            if name.endswith("_ankle"):  # listed before the ankle pitch joint (the knee link's first child)
                i = s.index(f'<joint name="{side}_ankle_pitch_joint"')
                s = s[:i] + "\n".join(block(name, parent, xyz)).lstrip() + "\n  " + s[i:]
            else:
                add += block(name, parent, xyz)
    i = s.rindex("</robot>")
    (out / "g1_29dof.urdf").write_text(s[:i] + "\n".join(add) + "\n" + s[i:])


def _first_child(parent, el):
    kids = [i for i, e in enumerate(list(parent)) if e.tag == "body"]
    parent.insert(kids[0] if kids else len(parent), el)


def build_mjcf(src, mj, fr, out):
    root = mj
    root.find("compiler").set("meshdir", "unitree_meshes")
    for tag in ("actuator", "sensor"):  # Unitree's motors and sensors (FAB replays kinematics)
        for e in root.findall(tag):
            root.remove(e)
    wbs = root.findall("worldbody")
    for wb in wbs[1:]:  # Unitree's demo floor and light
        root.remove(wb)
    # bodies for the fixed URDF links that carry a mesh
    pel = root.find(".//body[@name='pelvis']")
    pc = ET.Element("body", {"name": "pelvis_contour_link"})
    for g in [g for g in pel.findall("geom") if g.get("mesh") == "pelvis_contour_link"]:
        pel.remove(g)
        pc.append(g)
    assert len(pc) == 2
    _first_child(pel, pc)
    tor = root.find(".//body[@name='torso_link']")
    ws = [g for g in tor.findall("geom") if g.get("mesh") == "waist_support_link"]
    assert len(ws) == 2
    wb = ET.Element("body", {"name": "waist_support_link", "pos": ws[0].get("pos")})
    for g in ws:
        g2 = copy.deepcopy(g)
        for k in ("pos", "quat"):
            g2.attrib.pop(k, None)
        wb.append(g2)
    _first_child(tor, wb)
    for side in SIDES:
        wy = root.find(f".//body[@name='{side}_wrist_yaw_link']")
        hand = next(g for g in wy.findall("geom") if g.get("mesh") == f"{side}_rubber_hand")
        hp = [float(x) for x in hand.get("pos").split()]
        wy.remove(hand)
        del hand.attrib["pos"]
        hb = ET.SubElement(wy, "body", {"name": f"{side}_rubber_hand", "pos": fmt(hp)})
        hb.append(hand)
        col = copy.deepcopy(hand)  # the hand mesh as a collision geom too
        for k in ("contype", "conaffinity", "group"):
            col.attrib.pop(k, None)
        hb.append(col)
    for side in SIDES:
        for name, (parent, xyz, r) in fr[side].items():
            pb = root.find(f".//body[@name='{parent}']")
            b = ET.SubElement(pb, "body", {"name": name, "pos": fmt(xyz)})
            for rr in ((ANKLE_R if r == ANKLE_R else VIS_R), r):  # the visual marker, then the frame's contact radius
                ET.SubElement(b, "geom", {"type": "sphere", "size": f"{rr:g}", "contype": "0", "conaffinity": "0", "group": "4",
                                          "density": "0", "rgba": "1 0.4 0 1"})
    md = out / "unitree_meshes"
    md.mkdir(parents=True, exist_ok=True)
    used = sorted({m.get("file") for m in root.iter("mesh")})
    for m in root.iter("mesh"):
        m.set("file", Path(m.get("file")).stem + ".obj")
    for f in used:
        shutil.copyfile(src / "meshes" / f, md / f)  # the STL too (the URDF names them)
        stl_to_obj(src / "meshes" / f, md / (Path(f).stem + ".obj"))
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(out / "g1_29dof.xml", encoding="unicode")
    return used


def build_gate(src, out):
    """g1_gate_robot.xml: Unitree's g1_29dof_rev_1_0.xml without its geoms, actuators, sensors and assets, with CAPSULES."""
    t = ET.parse(src / "g1_29dof_rev_1_0.xml")
    r = t.getroot()
    for e in list(r):
        if e.tag in ("actuator", "sensor", "contact", "keyframe", "asset"):
            r.remove(e)
    for wb in r.findall("worldbody")[1:]:
        r.remove(wb)
    comp = r.find("compiler")
    comp.attrib.pop("meshdir", None)
    for b in r.iter("body"):
        for g in b.findall("geom"):
            b.remove(g)
    bodies = {b.get("name"): b for b in r.iter("body")}
    last = {}
    for body, typ, size, where in CAPSULES:
        b = bodies[body]
        a = {"type": typ, "size": size, ("pos" if typ == "sphere" else "fromto"): where, "contype": "1", "conaffinity": "1",
             "group": "3", "rgba": ".2 .6 .2 .3"}
        g = ET.Element("geom", a)
        kids = [i for i, e in enumerate(list(b)) if e.tag == "body"]
        at = last[body] + 1 if body in last else (kids[0] if kids else len(b))  # a body's geoms before its child bodies
        b.insert(at, g)
        last[body] = at
    ET.indent(t, space="  ")
    t.write(out / "g1_gate_robot.xml", encoding="unicode")


SCENE = """<mujoco model="g1 gate scene (FAB)">
  <!-- FAB's scene for the stance pin's physics checks (scripts/build_g1.py): the gate model on a floor plane -->
  <include file="g1_gate_robot.xml"/>
  <option timestep="0.002" iterations="50" solver="Newton"/>
  <asset/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.01"/>
  </worldbody>
</mujoco>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="unitree_ros/robots/g1_description")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mj = ET.parse(src / "g1_29dof.xml").getroot()
    fr = frames(mj)
    build_urdf(src, fr, out)
    used = build_mjcf(src, mj, fr, out)
    build_gate(src, out)
    (out / "scene_g1_gate.xml").write_text(SCENE)
    for side in SIDES:
        for name, (parent, xyz, _) in fr[side].items():
            print(f"{name:14s} in {parent:22s} {fmt(xyz)}")
    print(f"built g1_29dof.urdf, g1_29dof.xml ({len(used)} meshes), g1_gate_robot.xml, scene_g1_gate.xml in {out}")


if __name__ == "__main__":
    main()
