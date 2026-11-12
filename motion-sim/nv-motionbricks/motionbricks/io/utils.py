# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Find bundled assets, and load the SOMA skeletal mesh or the Unitree G1 Newton model."""

import re
from pathlib import Path

import motionbricks.io.usd as usd_utils


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# The skeleton prim inside the bundled SOMA character file.
SOMA_SKELETON_PRIM_PATH = "/OUTPUT/c_skeleton_grp/Hips"
_SOMA_FINGER_JOINT = re.compile(r"(Left|Right)Hand(Thumb|Index|Middle|Ring|Pinky)")


def get_package_root() -> Path:
    """Return the filesystem path to the package root."""
    return _PACKAGE_ROOT


def get_assets_dir() -> Path:
    """Return the assets directory path."""
    return get_package_root() / 'assets'


def get_asset_file(*relative_parts: str) -> Path:
    """Return a path to a specific file under assets/.

    Args:
        *relative_parts: Path parts under assets/, such as "human" and the file name.
    """
    return get_assets_dir().joinpath(*relative_parts)


def get_soma_human_model():
    """Return the SOMA mesh with its skeleton in the reference pose.

    The mesh holds the file's RelaxedHands override pose and a Fingers mask.
    A renderer can select them to draw relaxed fingers.
    """

    mesh = usd_utils.load_skeletal_mesh_with_skeleton_from_usd(
        get_asset_file('human', 'soma_base_skel_minimal_no_root.usd'),
        '/OUTPUT/c_geometry_grp',
        SOMA_SKELETON_PRIM_PATH,
    )
    skeleton = mesh.skeleton
    fingers = [name for name in skeleton.joint_names if _SOMA_FINGER_JOINT.match(name)]
    mesh.override_pose_masks['Fingers'] = skeleton.joint_mask(fingers)
    return mesh


def get_unitree_g1_model(*, device=None, load_visuals: bool = True):
    """Load the bundled Unitree G1 robot as a Newton model.

    Args:
        device: Device for new arrays.
        load_visuals: Load the visual meshes. False skips them.
    """
    import newton

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_mjcf(
        str(get_asset_file("robots", "unitree_g1", "g1_29dof_rev_1_0.xml")),
        parse_meshes=load_visuals,
        parse_visuals=load_visuals,
        parse_sites=False,
        collapse_fixed_joints=False,
        enable_self_collisions=False,
    )
    return builder.finalize(device=device)
