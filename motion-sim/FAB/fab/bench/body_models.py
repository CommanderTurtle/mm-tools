"""Minimal SMPL body model (torch) for evaluation, plus a chumpy-free loader for the original MPI .pkl files.

The official SMPL pickles store some arrays as chumpy objects; chumpy does not import on current numpy, so the
unpickler maps every chumpy class to a stub that keeps the wrapped array.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scipy.sparse
import torch


class _ChumpyStub:
    def __init__(self, *args, **kwargs):
        self.x = None

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"x": state})

    def __array__(self, dtype=None, copy=None):
        arr = np.asarray(self.__dict__.get("x", self.__dict__.get("a")))
        return arr.astype(dtype) if dtype is not None else arr


class _NoChumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _ChumpyStub
        return super().find_class(module, name)


def load_smpl_pkl(path: str | Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        raw = _NoChumpyUnpickler(f, encoding="latin1").load()

    def dense(value):
        if scipy.sparse.issparse(value):
            return np.asarray(value.todense())
        return np.asarray(value)

    return {
        "v_template": dense(raw["v_template"]).astype(np.float32),
        "shapedirs": dense(raw["shapedirs"]).astype(np.float32),
        "posedirs": dense(raw["posedirs"]).astype(np.float32),
        "J_regressor": dense(raw["J_regressor"]).astype(np.float32),
        "weights": dense(raw["weights"]).astype(np.float32),
        "kintree_table": dense(raw["kintree_table"]).astype(np.int64),
        "faces": dense(raw["f"]).astype(np.int64),
    }


def _rodrigues(axis_angle: torch.Tensor) -> torch.Tensor:
    """[..., 3] axis-angle -> [..., 3, 3] rotation matrices."""
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = axis_angle / angle
    x, y, z = axis.unbind(-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1).reshape(*axis.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand_as(skew)
    sin, cos = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
    return eye + sin * skew + (1 - cos) * (skew @ skew)


class SMPL(torch.nn.Module):
    """SMPL forward kinematics + linear blend skinning (vertices and the 24 SMPL joints)."""

    def __init__(self, model_path: str | Path, num_betas: int = 10):
        super().__init__()
        data = load_smpl_pkl(model_path)
        self.num_betas = num_betas
        self.register_buffer("v_template", torch.from_numpy(data["v_template"]))
        self.register_buffer("shapedirs", torch.from_numpy(data["shapedirs"][:, :, :num_betas]))
        self.register_buffer("posedirs", torch.from_numpy(data["posedirs"].reshape(-1, data["posedirs"].shape[-1]).T))
        self.register_buffer("J_regressor", torch.from_numpy(data["J_regressor"]))
        self.register_buffer("weights", torch.from_numpy(data["weights"]))
        self.register_buffer("faces", torch.from_numpy(data["faces"]))
        parents = data["kintree_table"][0].copy()
        parents[0] = -1
        self.parents = parents.tolist()

    def forward(self, global_orient: torch.Tensor, body_pose: torch.Tensor, betas: torch.Tensor,
                transl: torch.Tensor | None = None, rotmats: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """global_orient [B, 3], body_pose [B, 69], betas [B, >=num_betas], transl [B, 3] -> (verts [B, 6890, 3], joints [B, 24, 3]).
        rotmats [B, 24, 3, 3]: joint rotations given directly (the axis-angle inputs are then ignored)."""
        batch = global_orient.shape[0]
        betas = betas[:, : self.num_betas]
        v_shaped = self.v_template + torch.einsum("bl,vcl->bvc", betas, self.shapedirs)
        joints = torch.einsum("jv,bvc->bjc", self.J_regressor, v_shaped)

        pose = torch.cat([global_orient, body_pose], dim=1).reshape(batch, 24, 3)
        rot = _rodrigues(pose) if rotmats is None else rotmats
        eye = torch.eye(3, dtype=rot.dtype, device=rot.device)
        pose_feature = (rot[:, 1:] - eye).reshape(batch, -1)
        v_posed = v_shaped + (pose_feature @ self.posedirs).reshape(batch, -1, 3)

        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, self.parents[1:]]
        transforms = [self._rt(rot[:, 0], rel_joints[:, 0])]
        for i in range(1, 24):
            transforms.append(transforms[self.parents[i]] @ self._rt(rot[:, i], rel_joints[:, i]))
        transforms = torch.stack(transforms, dim=1)  # [B, 24, 4, 4]
        posed_joints = transforms[:, :, :3, 3].clone()

        # remove the rest-pose joint location so the transforms act on rest-pose vertices
        joints_h = torch.cat([joints, torch.zeros_like(joints[..., :1])], dim=-1)[..., None]
        transforms = torch.cat([transforms[..., :3], transforms[..., 3:] - transforms @ joints_h], -1)  # autograd-safe
        skin = torch.einsum("vj,bjrc->bvrc", self.weights, transforms)
        v_h = torch.cat([v_posed, torch.ones_like(v_posed[..., :1])], dim=-1)
        verts = (skin @ v_h[..., None])[..., :3, 0]

        if transl is not None:
            verts = verts + transl[:, None]
            posed_joints = posed_joints + transl[:, None]
        return verts, posed_joints

    @staticmethod
    def _rt(rot: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(rot.shape[0], 4, 4, dtype=rot.dtype, device=rot.device)
        out[:, :3, :3] = rot
        out[:, :3, 3] = t
        out[:, 3, 3] = 1
        return out


def load_smplx_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Official SMPL-X .npz (e.g. SMPLX_NEUTRAL_2020.npz: 300 shape + 100 expression components)."""
    raw = np.load(path, allow_pickle=True)

    def dense(value):
        value = value.item() if value.dtype == object and value.shape == () else value
        return np.asarray(value.todense()) if scipy.sparse.issparse(value) else np.asarray(value)

    return {
        "v_template": dense(raw["v_template"]).astype(np.float32),
        "shapedirs": dense(raw["shapedirs"]).astype(np.float32),
        "posedirs": dense(raw["posedirs"]).astype(np.float32),
        "J_regressor": dense(raw["J_regressor"]).astype(np.float32),
        "weights": dense(raw["weights"]).astype(np.float32),
        "kintree_table": dense(raw["kintree_table"]).astype(np.int64),
        "faces": dense(raw["f"]).astype(np.int64),
    }


class SMPLX(SMPL):
    """SMPL-X forward kinematics + LBS (55 joints: body 22, jaw, eyes, 2 x 15 hand joints), axis-angle hands.

    pose layout (165): global (3) | body (63) | jaw (3) | left eye (3) | right eye (3) | left hand (45) | right hand (45).
    Expressions are not modelled (zero), which only affects face vertices."""

    def __init__(self, model_path: str | Path, num_betas: int = 300):
        torch.nn.Module.__init__(self)
        data = load_smplx_npz(model_path)
        n_shape = min(num_betas, data["shapedirs"].shape[-1] - 100) if data["shapedirs"].shape[-1] > 300 else num_betas
        self.num_betas = n_shape
        self.register_buffer("v_template", torch.from_numpy(data["v_template"]))
        self.register_buffer("shapedirs", torch.from_numpy(data["shapedirs"][:, :, :n_shape]))
        self.register_buffer("posedirs", torch.from_numpy(data["posedirs"].reshape(-1, data["posedirs"].shape[-1]).T))
        self.register_buffer("J_regressor", torch.from_numpy(data["J_regressor"]))
        self.register_buffer("weights", torch.from_numpy(data["weights"]))
        self.register_buffer("faces", torch.from_numpy(data["faces"]))
        parents = data["kintree_table"][0].copy()
        parents[0] = -1
        self.parents = parents.tolist()
        self.num_joints = len(self.parents)

    def forward_smplx(self, global_orient, body_pose, betas, transl=None, left_hand_pose=None, right_hand_pose=None,
                      jaw_pose=None):
        batch = global_orient.shape[0]
        zeros3 = global_orient.new_zeros(batch, 3)
        zeros45 = global_orient.new_zeros(batch, 45)
        pose = torch.cat([global_orient, body_pose, jaw_pose if jaw_pose is not None else zeros3, zeros3, zeros3,
                          left_hand_pose if left_hand_pose is not None else zeros45,
                          right_hand_pose if right_hand_pose is not None else zeros45], dim=1)
        return self._forward_pose(pose, betas, transl)

    def _forward_pose(self, pose, betas, transl):
        batch = pose.shape[0]
        betas = betas[:, : self.num_betas]
        v_shaped = self.v_template + torch.einsum("bl,vcl->bvc", betas, self.shapedirs)
        joints = torch.einsum("jv,bvc->bjc", self.J_regressor, v_shaped)
        rot = _rodrigues(pose.reshape(batch, self.num_joints, 3))
        eye = torch.eye(3, dtype=rot.dtype, device=rot.device)
        v_posed = v_shaped + (((rot[:, 1:] - eye).reshape(batch, -1)) @ self.posedirs).reshape(batch, -1, 3)
        rel = joints.clone()
        rel[:, 1:] -= joints[:, self.parents[1:]]
        transforms = [self._rt(rot[:, 0], rel[:, 0])]
        for i in range(1, self.num_joints):
            transforms.append(transforms[self.parents[i]] @ self._rt(rot[:, i], rel[:, i]))
        transforms = torch.stack(transforms, dim=1)
        posed_joints = transforms[:, :, :3, 3].clone()
        joints_h = torch.cat([joints, torch.zeros_like(joints[..., :1])], dim=-1)[..., None]
        transforms = transforms.clone()
        transforms[:, :, :, 3:] -= transforms @ joints_h
        skin = torch.einsum("vj,bjrc->bvrc", self.weights, transforms)
        verts = (skin @ torch.cat([v_posed, torch.ones_like(v_posed[..., :1])], dim=-1)[..., None])[..., :3, 0]
        if transl is not None:
            verts, posed_joints = verts + transl[:, None], posed_joints + transl[:, None]
        return verts, posed_joints
