"""The Unitree G1 robot description files the retargeter loads, under $FAB_HOME/g1 (fab.paths.g1_dir();
models/fab_assets.py installs them: Unitree's g1_description, built into FAB's files by scripts/build_g1.py):

    g1_29dof.urdf          kinematics, joint limits, link inertials, FAB's frames (the retargeting solver, the G1 head)
    g1_29dof.xml           the 29-DoF MuJoCo model with FAB's frames (support points, scene metrics, the simulation package)
    g1_mjlab.xml           mjlab's G1 model with collision capsules (the solver's self-collision term)
    scene_g1_gate.xml      FAB's scene of the stance pin's physics checks (g1_gate_robot.xml: FAB's collision primitives on
                           Unitree's rev 1.0 model, on a floor plane)
"""
from fab import paths


def urdf():
    return paths.g1_dir() / "g1_29dof.urdf"


def mjcf():
    return paths.g1_dir() / "g1_29dof.xml"


def harness_mjcf():
    return paths.g1_dir() / "g1_mjlab.xml"


def gate_scene():
    return paths.g1_dir() / "scene_g1_gate.xml"
