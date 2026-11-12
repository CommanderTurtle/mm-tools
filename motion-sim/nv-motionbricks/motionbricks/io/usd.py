# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Load skeletons and skinned meshes from USD.

The USD rest pose becomes the skeleton's reference pose.
"""

import logging

import numpy as np
import warp as wp
from pxr import Tf, Usd, UsdGeom, UsdSkel

from motionbricks.animation.mesh import SkeletalMesh, SkinnedMesh
from motionbricks.animation.skeleton import Skeleton

logger = logging.getLogger(__name__)

_EPSILON = 1e-6

###############################################################################
# USD Stage Metadata Helpers
###############################################################################


def _resolve_time_code(stage):
    """Return the time code for evaluating transforms and attributes.

    A stage with a time range uses its start time. Exporters often animate
    xformOps without a meaningful default value, and bind pose loading expects
    the first frame.
    """
    start = stage.GetStartTimeCode()
    if start < stage.GetEndTimeCode():
        return Usd.TimeCode(start)
    return Usd.TimeCode.Default()


def _open_stage(usd_file_path):
    """Open a USD stage. Every failure raises ValueError."""
    try:
        stage = Usd.Stage.Open(str(usd_file_path))
    except Tf.ErrorException as exc:
        raise ValueError(f"Could not open USD stage: {usd_file_path}.") from exc
    if stage is None:
        raise ValueError(f"Could not open USD stage: {usd_file_path}.")
    return stage


def get_stage_meters_per_unit(stage):
    """Return the stage's meters per unit, such as 0.01 for centimeters.

    Args:
        stage: Open USD stage.
    """
    return UsdGeom.GetStageMetersPerUnit(stage)


def get_stage_up_axis(stage):
    """Return the stage's up axis, 'Y' or 'Z'.

    Args:
        stage: Open USD stage.
    """
    up_axis = UsdGeom.GetStageUpAxis(stage)
    return "Y" if up_axis == UsdGeom.Tokens.y else "Z"


###############################################################################
# Matrix Decomposition Helpers
###############################################################################


def decompose_matrix(mat, translation_scale=1.0):
    """Split a row major 4x4 matrix into a position, a rotation, and a scale.

    Returns (wp.vec3, wp.quat, np.ndarray).

    Args:
        mat: Row major 4x4 matrix, as USD stores it.
        translation_scale: Factor applied to the position, such as 0.01 to convert
            centimeters to meters.
    """
    # USD matrices are row major, so the translation is in row 3.
    pos = mat[3, :3] * translation_scale

    # The upper 3x3 block holds rotation and scale.
    rot_matrix = mat[:3, :3].copy()

    # Each column's length is the scale along that axis.
    scale = np.array(
        [
            np.linalg.norm(rot_matrix[:, 0]),
            np.linalg.norm(rot_matrix[:, 1]),
            np.linalg.norm(rot_matrix[:, 2]),
        ]
    )

    for i in range(3):
        if scale[i] > _EPSILON:
            rot_matrix[:, i] /= scale[i]

    # Transpose to the layout Warp expects before converting to a quaternion.
    rot = wp.quat_from_matrix(wp.mat33(rot_matrix.T.flatten()))

    return wp.vec3(pos[0], pos[1], pos[2]), rot, scale


def get_prim_world_scale(prim, time_code):
    """Return a prim's world scale [sx, sy, sz] from its world transform.

    Args:
        prim: USD prim.
        time_code: Time at which to evaluate the transform.
    """
    xform_cache = UsdGeom.XformCache(time_code)
    mat = np.array(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float32)

    # Each column's length is the scale along that axis.
    scale = np.array(
        [
            np.linalg.norm(mat[:3, 0]),
            np.linalg.norm(mat[:3, 1]),
            np.linalg.norm(mat[:3, 2]),
        ]
    )

    return scale


def get_prim_meters_scale(stage, prim):
    """Return the factor that converts a prim's data to meters.

    It combines the stage's meters per unit with the prim's world scale. The
    scale is assumed uniform, so only its X component is used.

    Args:
        stage: Stage that holds the prim.
        prim: Prim whose data to convert.
    """
    meters_per_unit = get_stage_meters_per_unit(stage)
    prim_scale = get_prim_world_scale(prim, _resolve_time_code(stage))

    combined_scale = meters_per_unit * prim_scale[0]

    return combined_scale


###############################################################################
# USD Loading Functions
###############################################################################


def get_bind_transform_as_warp_transform(bind_matrix, scale=1.0):
    """Convert a row major 4x4 bind matrix to a transform, scaling its translation.

    Args:
        bind_matrix: Row major 4x4 bind matrix.
        scale: Factor applied to the translation.
    """
    # USD multiplies row vectors, so its matrix is the transpose of Warp's.
    transform = wp.transform_from_matrix(wp.transpose(wp.mat44(*bind_matrix.flatten())))
    return wp.transform(transform.p * scale, transform.q)


def _load_skeleton(stage, skeleton_prim_path):
    skeleton_prim = stage.GetPrimAtPath(skeleton_prim_path)
    if not skeleton_prim.IsValid():
        raise ValueError(f"Skeleton prim not found at path {skeleton_prim_path}.")

    usd_skeleton = UsdSkel.Skeleton(skeleton_prim)
    joint_paths = usd_skeleton.GetJointsAttr().Get()
    rest_transforms = usd_skeleton.GetRestTransformsAttr().Get()
    if not joint_paths or not rest_transforms:
        raise ValueError(
            f"Skeleton {skeleton_prim_path} has no joints or rest transforms."
        )
    if len(joint_paths) != len(rest_transforms):
        raise ValueError(
            f"Skeleton {skeleton_prim_path} has {len(joint_paths)} joints but "
            f"{len(rest_transforms)} rest transforms."
        )

    translation_scale = get_prim_meters_scale(stage, skeleton_prim)

    joint_path_strings = [str(path) for path in joint_paths]
    path_to_index = {path: index for index, path in enumerate(joint_path_strings)}
    joint_names = [path.rsplit("/", 1)[-1] for path in joint_path_strings]
    if len(set(joint_names)) != len(joint_names):
        raise ValueError(f"Skeleton {skeleton_prim_path} has duplicate joint names.")

    parent_indices = []
    for path in joint_path_strings:
        parent_path = path.rsplit("/", 1)[0] if "/" in path else None
        parent_indices.append(-1 if parent_path is None else path_to_index[parent_path])

    local_transforms = []
    for matrix in rest_transforms:
        position, rotation, _ = decompose_matrix(
            np.asarray(matrix, dtype=np.float32),
            translation_scale=translation_scale,
        )
        local_transforms.append(wp.transform(position, rotation))

    skeleton = Skeleton(joint_names, parent_indices, local_transforms)
    if get_stage_up_axis(stage) == "Y":
        skeleton.up_axis = wp.vec3(0.0, 1.0, 0.0)
        skeleton.forward_axis = wp.vec3(0.0, 0.0, 1.0)
    return skeleton


def _get_value_or_first_sample(attribute, time_code):
    """Return the attribute's value at time_code, or else its first time sample."""
    value = attribute.Get(time_code)
    return attribute.Get(Usd.TimeCode.EarliestTime()) if value is None else value


def _load_override_poses(stage, skeleton_prim, skeleton, translation_scale):
    """Return each SkelAnimation under the skeleton prim as local joint transforms for every joint, by prim name.

    Joints missing from the animation use their reference transforms.
    If the animation only has time samples, use its first sample.
    """
    time_code = _resolve_time_code(stage)
    rest = skeleton.reference_local_transforms.numpy()
    poses = {}
    for prim in skeleton_prim.GetChildren():
        if not prim.IsA(UsdSkel.Animation):
            continue
        animation = UsdSkel.Animation(prim)
        joint_paths = animation.GetJointsAttr().Get() or []
        translations = _get_value_or_first_sample(animation.GetTranslationsAttr(), time_code) or []
        rotations = _get_value_or_first_sample(animation.GetRotationsAttr(), time_code) or []
        if not len(joint_paths) == len(translations) == len(rotations):
            raise ValueError(f"SkelAnimation {prim.GetPath()} needs a translation and a rotation per joint.")

        pose = rest.copy()
        for path, translation, rotation in zip(joint_paths, translations, rotations):
            name = str(path).rsplit("/", 1)[-1]
            joint = skeleton.joint_index(name)
            if joint < 0:
                raise ValueError(f"SkelAnimation {prim.GetPath()} names unknown joint {name}.")
            pose[joint, :3] = np.asarray(translation) * translation_scale
            pose[joint, 3:6] = rotation.GetImaginary()
            pose[joint, 6] = rotation.GetReal()
        poses[prim.GetName()] = wp.array(pose, dtype=wp.transform, device=skeleton.reference_local_transforms.device)
    return poses


def load_skeleton_from_usd(usd_file_path, skeleton_prim_path):
    """Load a Skeleton with its reference pose from USD.

    A stage that fails to open, or a missing skeleton, raises ValueError.

    Args:
        usd_file_path: USD file to open.
        skeleton_prim_path: Path of the UsdSkel.Skeleton prim.
    """

    stage = _open_stage(usd_file_path)
    return _load_skeleton(stage, skeleton_prim_path)


def _load_skinned_meshes(stage, mesh_prim, skeleton, skeleton_world_transform, scale):
    """Load every skinned mesh at or below mesh_prim, with its points in skeleton space.

    Meshes without a skinning binding are skipped with a warning. A mesh without a joints
    relationship uses the skeleton's joints. Quads are split into two triangles, and
    larger polygons are skipped. A binding without joint indices or weights raises
    ValueError.

    Args:
        stage: Stage that holds the meshes.
        mesh_prim: Mesh prim, or a prim whose descendants hold meshes.
        skeleton: Skeleton whose joint indices the meshes use.
        skeleton_world_transform: World transform of the skeleton. It moves the mesh
            points into skeleton space.
        scale: Factor that converts the mesh's data to meters.
    """
    time_code = _resolve_time_code(stage)
    # XformCache handles every xformOp, reset, and pivot in the world transform.
    xform_cache = UsdGeom.XformCache(time_code)
    inv_skel_rot = wp.quat_inverse(skeleton_world_transform.q)
    meshes = []
    for prim in Usd.PrimRange(mesh_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        binding_api = UsdSkel.BindingAPI(prim)
        if not binding_api:
            logger.warning("Mesh %s has no skinning binding and is skipped.", prim.GetPath())
            continue
        joint_indices = binding_api.GetJointIndicesAttr().Get()
        joint_weights = binding_api.GetJointWeightsAttr().Get()
        if joint_indices is None or joint_weights is None:
            raise ValueError(f"Mesh {prim.GetPath()} needs jointIndices and jointWeights.")

        # The geom bind transform moves mesh points into skeleton space, where the bind
        # transforms are. An authored geomBindTransform, USD's standard mapping, wins.
        geom_bind_value = binding_api.GetGeomBindTransformAttr().Get(time_code)
        if geom_bind_value is not None:
            position, rotation, _ = decompose_matrix(
                np.array(geom_bind_value, dtype=np.float32).reshape(4, 4), translation_scale=scale,
            )
            geom_bind = wp.transform(position, rotation)
        else:
            # inv(skeleton_world) * mesh_world, keeping position and rotation. Scale is
            # applied to the points separately.
            mesh_position, mesh_rotation, _ = decompose_matrix(
                np.array(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float32), translation_scale=scale,
            )
            # Subtract the skeleton position, then rotate into skeleton space.
            geom_bind = wp.transform(
                wp.quat_rotate(inv_skel_rot, mesh_position - skeleton_world_transform.p),
                wp.mul(inv_skel_rot, mesh_rotation),
            )

        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32) * scale
        for i in range(len(points)):
            points[i] = wp.transform_point(geom_bind, wp.vec3(points[i]))
        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        faces = []
        face_id = 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            if count == 3:
                faces.append(indices[face_id : face_id + 3])
            elif count == 4:
                faces.append(indices[face_id : face_id + 3])
                faces.append(indices[[face_id, face_id + 2, face_id + 3]])
            face_id += count

        # Joint indices refer to the mesh's own joint list. Map each one to the skeleton
        # by name. A joint the skeleton lacks gets -1.
        joints = binding_api.GetJointsAttr().Get()
        names = skeleton.joint_names if joints is None else [str(joint).rsplit("/", 1)[-1] for joint in joints]
        skeleton_joints = np.array([skeleton.joint_index(name) for name in names], dtype=int)
        meshes.append(SkinnedMesh(
            points,
            np.array(faces, dtype=np.int32).flatten(),
            skeleton_joints[np.array(joint_indices, dtype=np.int32)],
            np.array(joint_weights, dtype=np.float32),
        ))
    return meshes


def load_skeletal_mesh_with_skeleton_from_usd(
    usd_file_path,
    mesh_prim_path,
    skeleton_prim_path,
):
    """Load a skeletal mesh and construct its Skeleton from the same USD stage.

    Every SkelAnimation under the skeleton prim becomes an override pose named
    after its prim. Failures raise ValueError.

    Args:
        usd_file_path: USD file to open.
        mesh_prim_path: Path of the mesh prim, or of a prim whose descendants hold the
            meshes.
        skeleton_prim_path: Path of the UsdSkel.Skeleton prim.
    """

    stage = _open_stage(usd_file_path)
    skeleton = _load_skeleton(stage, skeleton_prim_path)
    mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
    if not mesh_prim.IsValid():
        raise ValueError(f"Mesh prim not found at path {mesh_prim_path}.")

    # The mesh and skeleton prims can be scaled differently, so each uses its own scale.
    skeleton_prim = stage.GetPrimAtPath(skeleton_prim_path)
    skeleton_scale = get_prim_meters_scale(stage, skeleton_prim)
    skeleton_position, skeleton_rotation, _ = decompose_matrix(
        np.array(
            UsdGeom.XformCache(_resolve_time_code(stage)).GetLocalToWorldTransform(skeleton_prim),
            dtype=np.float32,
        ),
        translation_scale=skeleton_scale,
    )
    skinned_meshes = _load_skinned_meshes(
        stage,
        mesh_prim,
        skeleton,
        wp.transform(skeleton_position, skeleton_rotation),
        get_prim_meters_scale(stage, mesh_prim),
    )
    if not skinned_meshes:
        raise ValueError(f"No skinned meshes found under {mesh_prim_path}.")

    # The skeleton was built from the same joints attribute, so the binds come in joint
    # order. They use the skeleton prim's scale, not the mesh prim's.
    bind_matrices = UsdSkel.Skeleton(skeleton_prim).GetBindTransformsAttr().Get()
    if bind_matrices is None or len(bind_matrices) != skeleton.num_joints:
        raise ValueError(f"Skeleton {skeleton_prim_path} needs one bind transform per joint.")
    inverse_bind_transforms = [
        wp.transform_inverse(get_bind_transform_as_warp_transform(
            np.array(matrix, dtype=np.float32).reshape(4, 4), scale=skeleton_scale,
        ))
        for matrix in bind_matrices
    ]
    return SkeletalMesh(
        skinned_meshes,
        skeleton,
        inverse_bind_transforms,
        override_poses=_load_override_poses(stage, skeleton_prim, skeleton, skeleton_scale),
    )
