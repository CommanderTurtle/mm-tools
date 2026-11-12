"""
Generate camera trajectory .npz files for camera control inference.

Produces c2ws [N,4,4] and intrinsics [N,3,3] in the same format as
master_nomas_1.5_haoyi/hymm/utils/camera_generator.py.

Coordinate system: w2c internally, exported as c2w (np.linalg.inv).
  x-right, y-up (look_at convention with Y-up), z-forward-to-camera.

Usage:
    python hymm/models/camera/geometry/generate_npz_trajectories.py \
        --traj move_left --num_frames 129 --output_dir outputs/camera_trajs

    # Composite trajectory (comma-separated):
    python hymm/models/camera/geometry/generate_npz_trajectories.py \
        --traj pull_out,pan_left --num_frames 129

    # Generate ALL standard trajectories at once:
    python hymm/models/camera/geometry/generate_npz_trajectories.py \
        --all --num_frames 129 --output_dir outputs/camera_trajs
"""

import argparse
import copy
import math
import os
from typing import List

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


VALID_TRAJECTORIES = [
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "push_in",
    "pull_out",
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "orbit_left",
    "orbit_right",
    "circle",
    "spiral",
    "fixed",
]

LINEAR_TRAJECTORIES = [
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "move_left_forward",
    "move_right_forward",
    "move_back_left",
    "move_back_right",
    "push_in",
    "pull_out",
    "orbit_left",
    "orbit_right",
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "fixed",
]

DEFAULT_SCALE = 0.05
DEFAULT_CENTER_DEPTH = 1.0
DEFAULT_FOCAL_LENGTH = 2500.0
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 352
DEFAULT_ROTATION_THETA = 30.0


def look_at_matrix(camera_pos, target, z_forward_to_camera=False, invert_pos=True):
    forward = (target - camera_pos).float()
    forward = forward / torch.norm(forward)
    if z_forward_to_camera:
        forward = -forward

    up = torch.tensor([0.0, 1.0, 0.0], device=camera_pos.device)
    right = torch.cross(up, forward)
    right = right / torch.norm(right)
    up = torch.cross(forward, right)

    mat = torch.eye(4, device=camera_pos.device)
    mat[0, :3] = right
    mat[1, :3] = up
    mat[2, :3] = forward
    mat[:3, 3] = (-camera_pos) if invert_pos else camera_pos
    return mat


class _Cam:
    """Lightweight camera pose holder (w2c)."""

    def __init__(self, R=None, T=None, device="cpu"):
        self.device = torch.device(device)
        self.R = (R if R is not None else torch.eye(3)).to(self.device)
        self.T = (T if T is not None else torch.zeros(3, 1)).to(self.device)
        self._sync()

    def _sync(self):
        RT = torch.cat((self.R, self.T), dim=1)
        last = torch.tensor([[0, 0, 0, 1]], device=self.device, dtype=self.R.dtype)
        self.RT = torch.cat((RT, last), dim=0)

    def set_R(self, R):
        self.R = R.to(self.device)
        self._sync()

    def set_T(self, T):
        self.T = T.to(self.device)
        self._sync()

    def set_RT(self, RT):
        self.RT = RT.to(self.device)
        self.R = self.RT[:3, :3]
        self.T = self.RT[:3, 3:4]

    def clone(self):
        return copy.deepcopy(self)


def _move(cam, axis, delta, rotation, center_depth):
    cam = cam.clone()
    dev = cam.device
    pos = torch.zeros(3, device=dev)
    pos[axis] = delta

    T = cam.T.clone()
    T += pos.unsqueeze(-1)
    camera_pos = T.squeeze(-1)

    sign = 1.0
    look_at_pt = torch.tensor([0.0, 0.0, sign * center_depth], device=dev)
    if rotation == "trajectory_aligned":
        _la = look_at_pt + pos * 2
    elif rotation == "center_facing":
        _la = look_at_pt
    else:
        _la = look_at_pt + pos

    vm = look_at_matrix(camera_pos, _la, z_forward_to_camera=False, invert_pos=True)
    cam.set_RT(vm)
    return cam


def _pan(cam, axis, direction, theta_deg):
    cam = cam.clone()
    dev = cam.device
    theta = torch.deg2rad(torch.tensor(direction * theta_deg, device=dev))
    c, s = torch.cos(theta), torch.sin(theta)

    if axis == 0:
        rot = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], device=dev)
    elif axis == 1:
        rot = torch.tensor([[c, 0, -s], [0, 1, 0], [s, 0, c]], device=dev)
    else:
        rot = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], device=dev)

    cam.set_R(rot @ cam.R)
    return cam


def _orbit(cam, axis, direction, center_depth, theta_deg):
    cam = cam.clone()
    dev = cam.device
    radius = center_depth
    orbit_center = torch.tensor([0, 0, radius], device=dev, dtype=torch.float32)
    current_pos = cam.T.squeeze()
    relative_pos = current_pos - orbit_center

    theta = torch.deg2rad(torch.tensor(direction * theta_deg, device=dev))
    c, s = torch.cos(theta), torch.sin(theta)
    if axis == 1:
        rot = torch.tensor([[c, 0, -s], [0, 1, 0], [s, 0, c]], device=dev)
    else:
        raise ValueError("orbit only supports axis=1")

    new_rel = rot.T @ relative_pos
    new_pos = new_rel + orbit_center
    cam.set_T(-new_pos.unsqueeze(1))

    look_dir = orbit_center - new_pos
    look_dir = look_dir / torch.norm(look_dir)
    forward = look_dir
    up = torch.tensor([0, 1, 0], device=dev, dtype=torch.float32)
    right = torch.cross(up, forward)
    right = right / torch.norm(right)
    up = torch.cross(forward, right)
    new_R = torch.stack([right, up, forward], dim=1)
    cam.set_R(new_R.T)
    return cam


def _interp_segment(cam1, cam2, num_points, include_start, device="cpu"):
    R1, R2 = cam1.R.cpu().numpy(), cam2.R.cpu().numpy()
    T1, T2 = cam1.T.cpu().numpy(), cam2.T.cpu().numpy()
    slerp = Slerp([0.0, 1.0], R.from_matrix([R1, R2]))
    if include_start:
        ts = (
            [i / max(num_points - 1, 1) for i in range(num_points)]
            if num_points > 0
            else []
        )
    else:
        ts = (
            [i / num_points for i in range(1, num_points + 1)] if num_points > 0 else []
        )

    cams = []
    for t in ts:
        Ri = torch.from_numpy(slerp([t]).as_matrix()[0]).to(device)
        Ti = torch.from_numpy(T1 + t * (T2 - T1)).to(device)
        cams.append(_Cam(Ri, Ti, device=device))
    return cams


def _anchor_to_origin(cameras):
    inv = torch.inverse(cameras[0].RT)
    for cam in cameras:
        cam.set_RT(inv @ cam.RT)


def generate_trajectory_npz(
    traj: str,
    num_frames: int = 129,
    camera_rotation: str = "no_rotation",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    focal_length: float = DEFAULT_FOCAL_LENGTH,
    scale: float = DEFAULT_SCALE,
    center_depth: float = DEFAULT_CENTER_DEPTH,
    device: str = "cpu",
) -> dict:
    """Generate a camera trajectory and return c2ws + intrinsics as numpy arrays.

    Returns dict with 'c2ws' [N,4,4] float32 and 'intrinsics' [N,3,3] float32.
    """
    traj_list = [t.strip() for t in traj.split(",")]
    rotation_list = [camera_rotation] * len(traj_list)
    n_segments = len(traj_list)
    total_interp = num_frames - 1

    base = total_interp // n_segments
    rem = total_interp % n_segments
    segment_frames = [base] * n_segments
    for idx in range(rem):
        segment_frames[n_segments - 1 - idx] += 1

    dev = torch.device(device)
    keyframes = [_Cam(device=dev)]

    action_map = {
        "push_in": lambda c: _move(c, 2, scale, "no_rotation", center_depth),
        "pull_out": lambda c: _move(c, 2, -scale, "no_rotation", center_depth),
        "move_left": lambda c: _move(c, 0, -scale, rotation_list[0], center_depth),
        "move_right": lambda c: _move(c, 0, scale, rotation_list[0], center_depth),
        "move_up": lambda c: _move(c, 1, -scale, rotation_list[0], center_depth),
        "move_down": lambda c: _move(c, 1, scale, rotation_list[0], center_depth),
        "orbit_left": lambda c: _orbit(c, 1, 1, center_depth, DEFAULT_ROTATION_THETA),
        "orbit_right": lambda c: _orbit(c, 1, -1, center_depth, DEFAULT_ROTATION_THETA),
        "pan_left": lambda c: _pan(c, 1, -1, DEFAULT_ROTATION_THETA),
        "pan_right": lambda c: _pan(c, 1, 1, DEFAULT_ROTATION_THETA),
        "tilt_up": lambda c: _pan(c, 0, -1, DEFAULT_ROTATION_THETA),
        "tilt_down": lambda c: _pan(c, 0, 1, DEFAULT_ROTATION_THETA),
        "fixed": lambda c: c.clone(),
    }

    all_linear = all(t in LINEAR_TRAJECTORIES for t in traj_list)

    if all_linear:
        for t_name, rot in zip(traj_list, rotation_list):
            _anchor_to_origin(keyframes)
            next_cam = action_map[t_name](keyframes[-1])
            keyframes.append(next_cam)
            _anchor_to_origin(keyframes)

        interp = []
        for idx, (c1, c2) in enumerate(zip(keyframes[:-1], keyframes[1:])):
            interp.extend(
                _interp_segment(
                    c1, c2, segment_frames[idx], include_start=(idx == 0), device=dev
                )
            )
    else:
        if len(traj_list) != 1:
            raise ValueError("Non-linear trajectories only support a single traj value")
        t_name = traj_list[0]
        frames = total_interp + 1
        sign = 1.0
        look_at_pt = torch.tensor([0.0, 0.0, sign * center_depth], device=dev)
        interp = []

        if "circle" in t_name:
            radius_x, radius_y = 0.03 * scale, 0.02 * scale
            example_scale = 10.0
            for i in range(frames):
                theta = 2 * math.pi * i / (frames - 1)
                x = radius_x * (math.cos(theta) - 1) * (center_depth / example_scale)
                y = radius_y * math.sin(theta) * (center_depth / example_scale) * sign
                pos = torch.tensor([x, y, 0.0], device=dev)
                vm = look_at_matrix(
                    pos, look_at_pt, z_forward_to_camera=False, invert_pos=True
                )
                c = _Cam(device=dev)
                c.set_R(vm[:3, :3])
                interp.append(c)
        elif "spiral" in t_name:
            radius_x, radius_y = 0.3 * scale, 0.2 * scale
            for i in range(frames):
                theta = 2 * math.pi * i / (frames - 1)
                x = radius_x * (math.cos(theta) - 1) * (center_depth / 1.0)
                y = radius_y * math.sin(theta) * (center_depth / 1.0) * sign
                pos = torch.tensor([x, y, 0.0], device=dev)
                cr = camera_rotation
                if cr == "center_facing":
                    vm = look_at_matrix(
                        pos, look_at_pt, z_forward_to_camera=False, invert_pos=True
                    )
                elif cr == "trajectory_aligned":
                    vm = look_at_matrix(
                        pos,
                        look_at_pt + pos * 2,
                        z_forward_to_camera=False,
                        invert_pos=True,
                    )
                else:
                    vm = look_at_matrix(
                        pos,
                        look_at_pt + pos,
                        z_forward_to_camera=False,
                        invert_pos=True,
                    )
                c = _Cam(device=dev)
                c.set_RT(vm)
                interp.append(c)
        else:
            raise ValueError(f"Unknown non-linear trajectory: {t_name}")

    # Convert w2c -> c2w
    w2cs = torch.stack([c.RT for c in interp], dim=0)
    c2ws = np.linalg.inv(w2cs.cpu().numpy()).astype(np.float32)

    fx = focal_length / width
    fy = focal_length / height
    K = np.array([[fx, 0.0, 0.5], [0.0, fy, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    intrinsics = np.repeat(K[None], c2ws.shape[0], axis=0)

    return {"c2ws": c2ws, "extrinsics": c2ws, "intrinsics": intrinsics}


# Convenience list for batch generation
STANDARD_TRAJECTORIES = [
    ("move_left", "no_rotation"),
    ("move_right", "no_rotation"),
    ("move_up", "no_rotation"),
    ("move_down", "no_rotation"),
    ("push_in", "no_rotation"),
    ("pull_out", "no_rotation"),
    ("pan_left", "no_rotation"),
    ("pan_right", "no_rotation"),
    ("tilt_up", "no_rotation"),
    ("tilt_down", "no_rotation"),
    ("orbit_left", "no_rotation"),
    ("orbit_right", "no_rotation"),
    ("move_left,move_right", "no_rotation"),
    ("move_left,move_up,move_down,move_right", "no_rotation"),
    ("pull_out,pan_left", "no_rotation"),
    ("pull_out,pan_left,pull_out", "no_rotation"),
    ("pull_out,push_in", "no_rotation"),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate camera trajectory .npz files"
    )
    parser.add_argument(
        "--traj",
        type=str,
        default="move_left",
        help="Trajectory name (comma-separated for composite)",
    )
    parser.add_argument(
        "--camera_rotation",
        type=str,
        default="no_rotation",
        choices=["no_rotation", "center_facing", "trajectory_aligned"],
    )
    parser.add_argument("--num_frames", type=int, default=129)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--focal_length", type=float, default=DEFAULT_FOCAL_LENGTH)
    parser.add_argument("--output_dir", type=str, default="outputs/camera_trajs")
    parser.add_argument(
        "--all", action="store_true", help="Generate all standard trajectories"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.all:
        trajs = STANDARD_TRAJECTORIES
    else:
        trajs = [(args.traj, args.camera_rotation)]

    for traj_name, rot in trajs:
        result = generate_trajectory_npz(
            traj=traj_name,
            num_frames=args.num_frames,
            camera_rotation=rot,
            width=args.width,
            height=args.height,
            focal_length=args.focal_length,
        )
        fname = f"videox_camera_{traj_name.replace(',', '_')}.npz"
        out_path = os.path.join(args.output_dir, fname)
        np.savez(out_path, **result)
        print(
            f"  {fname}: c2ws={result['c2ws'].shape}, intrinsics={result['intrinsics'].shape}"
        )

    print(f"\nAll trajectories saved to {args.output_dir}")
