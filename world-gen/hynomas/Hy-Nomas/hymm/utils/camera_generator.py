import copy
import json
import os
from typing import List

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import argparse
import math

VALID_TRAJECTORIES = [
    "move_left",
    "move_right",
    "move_left_forward",
    "move_right_forward",
    "move_back_left",
    "move_back_right",
    "move_up",
    "move_down",
    "push_in",
    "pull_out",
    "orbit_left",
    "orbit_right",
    "pan_left",
    "pan_right",
    "circle",
    "spiral",
    "circle_counter",
    "spiral_counter",
    "fixed",
]
VALID_ROTATIONS = ["no_rotation", "center_facing", "trajectory_aligned"]
LINEAR_TRAJECTORIES = [
    "move_left",
    "move_right",
    "move_left_forward",
    "move_right_forward",
    "move_back_left",
    "move_back_right",
    "move_up",
    "move_down",
    "push_in",
    "pull_out",
    "orbit_left",
    "orbit_right",
    "pan_left",
    "pan_right",
    "fixed",
]

# -----------------------------------------------------------------------------
#  Utility functions
# -----------------------------------------------------------------------------


def look_at_matrix(camera_pos, target, z_forward_to_camera=True, invert_pos=True):
    """Creates a 4x4 look-at matrix, keeping the camera pointing towards a target."""
    forward = (target - camera_pos).float()
    forward = forward / torch.norm(forward)
    if z_forward_to_camera:
        forward = -forward

    up = torch.tensor(
        [0.0, 1.0, 0.0], device=camera_pos.device
    )  # assuming Y-up coordinate system
    right = torch.cross(up, forward)
    right = right / torch.norm(right)
    up = torch.cross(forward, right)

    look_at = torch.eye(4, device=camera_pos.device)
    look_at[0, :3] = right
    look_at[1, :3] = up
    look_at[2, :3] = forward
    look_at[:3, 3] = (-camera_pos) if invert_pos else camera_pos

    return look_at


# -----------------------------------------------------------------------------
#  Core classes
# -----------------------------------------------------------------------------


class CameraExtrinsics:
    """Lightweight holder for a camera pose (rotation + translation).

    The internal representation is *camera‑to‑world* (c2w).
    All tensors live on self.device.
    """

    def __init__(self, R: torch.Tensor | None = None, T: torch.Tensor | None = None):
        if R is None:
            R = torch.eye(3)
        if T is None:
            T = torch.zeros(3, 1)
        self.R: torch.Tensor = R
        self.T: torch.Tensor = T
        self.device: torch.device = self.R.device
        self.update_RT()

    # ----------------------- helpers -----------------------
    def update_RT(self):
        RT = torch.cat((self.R, self.T), dim=1)
        last = torch.tensor([[0, 0, 0, 1]], device=self.device, dtype=self.R.dtype)
        self.RT = torch.cat((RT, last), dim=0)

    def update_R_T(self):
        self.R = self.RT[:3, :3]
        self.T = self.RT[:3, 3:4]

    # ----------------------- setters -----------------------
    def set_R(self, R: torch.Tensor):
        self.R = R.to(self.device)
        self.update_RT()

    def set_T(self, T: torch.Tensor):
        self.T = T.to(self.device)
        self.update_RT()

    def set_RT(self, RT: torch.Tensor):
        self.RT = RT.to(self.device)
        self.update_R_T()

    # ----------------------- misc --------------------------
    def to(self, device: torch.device | str):
        self.device = torch.device(device)
        self.R = self.R.to(device)
        self.T = self.T.to(device)
        self.RT = self.RT.to(device)
        return self


class CameraGen:
    """Generate key‑frame and interpolated camera trajectories.

    The public method is `generate_cameras(camera_path, save_path)` which returns
    numpy arrays of keyframes and full interpolated frames.
    """

    def __init__(self, args: argparse.Namespace, scales: dict):
        self.args = args
        self.camera_type = args.camera_type
        if self.camera_type != "videox":
            raise ValueError(
                f"Only 'videox' camera_type is supported, got '{self.camera_type}'"
            )
        self.width = args.width
        self.height = args.height
        self.scales = scales

        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.focal_length = args.focal_length
        self.traj = self._parse_csv_arg(args.traj)
        if not self.traj:
            raise ValueError("traj cannot be empty")
        invalid_traj = [item for item in self.traj if item not in VALID_TRAJECTORIES]
        if invalid_traj:
            raise ValueError(
                f"Unsupported trajectory {invalid_traj}. "
                f"Valid values: {VALID_TRAJECTORIES}"
            )

        camera_rotation = self._parse_csv_arg(args.camera_rotation)
        if not camera_rotation:
            raise ValueError("camera_rotation cannot be empty")
        if len(camera_rotation) == 1 and len(self.traj) > 1:
            camera_rotation = camera_rotation * len(self.traj)
        if len(camera_rotation) != len(self.traj):
            raise ValueError("camera_rotation and traj must have the same length")
        invalid_rotation = [
            item for item in camera_rotation if item not in VALID_ROTATIONS
        ]
        if invalid_rotation:
            raise ValueError(
                f"Unsupported camera_rotation {invalid_rotation}. "
                f"Valid values: {VALID_ROTATIONS}"
            )
        self.camera_rotation = camera_rotation

        self.output_dir = args.output_dir
        self.num_frames = args.num_frames
        self.segment_frames = self._split_frames_evenly(self.num_frames, len(self.traj))
        self.total_frames = (
            args.num_frames - 1
        )  # num_frames‑1 as in the original script
        self.rotation_range_theta = 30.0  # degrees

        self.camera_types = ["videox"]
        self.camera_type_current = None
        self.clear()

    @staticmethod
    def _parse_csv_arg(value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @staticmethod
    def _split_frames_evenly(total_frames: int, num_segments: int) -> List[int]:
        if total_frames < 1:
            raise ValueError("num_frames must be >= 1")
        if num_segments < 1:
            raise ValueError("At least one trajectory segment is required")
        if total_frames < num_segments:
            raise ValueError(
                f"num_frames ({total_frames}) must be >= number of trajectory segments ({num_segments})"
            )

        base = total_frames // num_segments
        remainder = total_frames % num_segments
        segment_frames = [base] * num_segments
        for idx in range(remainder):
            segment_frames[num_segments - 1 - idx] += 1
        return segment_frames

    # ---------------- public API ----------------
    def clear(self):
        self.cameras: List[CameraExtrinsics] = [CameraExtrinsics().to(self.device)]
        self.cameras_interp: List[CameraExtrinsics] = []

    # ------------- camera transformations --------------

    def _move(
        self,
        axis: int,
        delta: float,
        rotation: str,
        center_depth: float = 1.0,
        camera: CameraExtrinsics | None = None,
    ):

        invert_pos = True
        z_forward_to_camera = False
        is_c2w = False
        sign = 1.0
        look_at = torch.tensor([0.0, 0.0, sign * center_depth]).to(self.device)

        if camera is None:
            camera = copy.deepcopy(self.cameras[-1])

        if axis == 0:
            pos = torch.tensor([delta, 0.0, 0.0], device=self.device)
        elif axis == 1:
            pos = torch.tensor([0.0, delta, 0.0], device=self.device)
        elif axis == 2:
            pos = torch.tensor([0.0, 0.0, delta], device=self.device)
        else:
            raise ValueError("Axis should be 0, 1 or 2")

        T = camera.T.clone()
        T += pos.unsqueeze(-1)

        camera_pos = T.squeeze(-1)
        if rotation == "trajectory_aligned":
            _look_at = look_at + pos * 2
        elif rotation == "center_facing":
            _look_at = look_at
        elif rotation == "no_rotation":
            _look_at = look_at + pos
        else:
            raise ValueError(
                "Camera rotation should be center_facing or trajectory_aligned"
            )
        view_matrix = look_at_matrix(
            camera_pos,
            _look_at,
            z_forward_to_camera=z_forward_to_camera,
            invert_pos=invert_pos,
        )

        # for c2w matrix, we need to invert the rotation
        if is_c2w:
            R = view_matrix[:3, :3]
            view_matrix[:3, :3] = R.T

        camera.set_RT(view_matrix)
        return camera

    def _pan(
        self,
        axis: int,
        direction: int,
        camera: CameraExtrinsics | None = None,
        theta: float = None,
    ):
        rotation_range_theta = self.rotation_range_theta if theta is None else theta

        if camera is None:
            camera = copy.deepcopy(self.cameras[-1])
        sign = direction
        theta = torch.deg2rad(
            torch.tensor(sign * rotation_range_theta, device=self.device)
        )

        if axis == 0:
            rot = torch.tensor(
                [
                    [1, 0, 0],
                    [0, torch.cos(theta), -torch.sin(theta)],
                    [0, torch.sin(theta), torch.cos(theta)],
                ],
                device=self.device,
            )
        elif axis == 1:
            rot = torch.tensor(
                [
                    [torch.cos(theta), 0, -torch.sin(theta)],
                    [0, 1, 0],
                    [torch.sin(theta), 0, torch.cos(theta)],
                ],
                device=self.device,
            )
        elif axis == 2:
            rot = torch.tensor(
                [
                    [torch.cos(theta), -torch.sin(theta), 0],
                    [torch.sin(theta), torch.cos(theta), 0],
                    [0, 0, 1],
                ],
                device=self.device,
            )

        camera.set_R(rot @ camera.R)

        # # For gen3c (w2c), we want to rotate the camera in world space, so we multiply from the right
        # # For c2w, we want to rotate the world in camera space, so we multiply from the left with transpose
        # if self.camera_type_current == "gen3c":
        #     camera.set_R(camera.R @ rot)  # w2c: R_new = R_old @ R_delta
        # else:
        #     camera.set_R(rot.T @ camera.R)  # c2w: R_new = R_delta.T @ R_old
        return camera

    def _fixed(self):
        return self.cameras[-1]

    def _orbit(
        self,
        axis: int,
        direction: int,
        camera: CameraExtrinsics | None = None,
        center_depth: float = 1.0,
        theta: float = 30,
    ):
        rotation_range_theta = self.rotation_range_theta if theta is None else theta
        invert_pos = True
        z_forward_to_camera = False

        if camera is None:
            camera = copy.deepcopy(self.cameras[-1])

        sign = 1.0
        radius = center_depth

        orbit_center = torch.tensor(
            [0, 0, sign * radius], device=self.device, dtype=torch.float32
        )
        current_pos = camera.T.squeeze()
        relative_pos = current_pos - orbit_center

        theta = torch.deg2rad(
            torch.tensor(direction * rotation_range_theta, device=self.device)
        )

        if axis == 0:
            rot = torch.tensor(
                [
                    [1, 0, 0],
                    [0, torch.cos(theta), -torch.sin(theta)],
                    [0, torch.sin(theta), torch.cos(theta)],
                ],
                device=self.device,
            )
        elif axis == 1:
            rot = torch.tensor(
                [
                    [torch.cos(theta), 0, -torch.sin(theta)],
                    [0, 1, 0],
                    [torch.sin(theta), 0, torch.cos(theta)],
                ],
                device=self.device,
            )
        elif axis == 2:
            rot = torch.tensor(
                [
                    [torch.cos(theta), -torch.sin(theta), 0],
                    [torch.sin(theta), torch.cos(theta), 0],
                    [0, 0, 1],
                ],
                device=self.device,
            )

        new_relative_pos = rot.T @ relative_pos  # For c2w
        new_pos = new_relative_pos + orbit_center

        if invert_pos:
            camera.set_T(-new_pos.unsqueeze(1))
        else:
            camera.set_T(new_pos.unsqueeze(1))

        look_dir = orbit_center - new_pos
        look_dir = look_dir / torch.norm(look_dir)

        if z_forward_to_camera:
            forward = -look_dir
        else:
            forward = look_dir

        up = torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32)
        right = torch.cross(up, forward)
        right = right / torch.norm(right)
        up = torch.cross(forward, right)

        new_R = torch.stack([right, up, forward], dim=1)
        if invert_pos:
            camera.set_R(new_R.T)
        else:
            camera.set_R(new_R)
        return camera

    # ------------- keyframe & interpolation --------------

    def generate_keyframe_camera(self, camera_type: str, rotation: str):
        if camera_type != "videox":
            raise ValueError(
                f"Only 'videox' camera_type is supported, got '{camera_type}'"
            )
        scale = self.scales[camera_type]["scale"]
        center_depth = self.scales[camera_type]["center_depth"]
        # x-right, y-down, z-forward, left-hand
        actions = {
            "push_in": lambda: self._move(
                axis=2, delta=scale, rotation="no_rotation", center_depth=center_depth
            ),
            "pull_out": lambda: self._move(
                axis=2, delta=-scale, rotation="no_rotation", center_depth=center_depth
            ),
            "move_left": lambda: self._move(
                axis=0, delta=-scale, rotation=rotation, center_depth=center_depth
            ),
            "move_right": lambda: self._move(
                axis=0, delta=scale, rotation=rotation, center_depth=center_depth
            ),
            "move_left_forward": lambda: self._move(
                axis=2,
                delta=-scale,
                rotation=rotation,
                center_depth=center_depth,
                camera=self._move(
                    axis=0, delta=-scale, rotation=rotation, center_depth=center_depth
                ),
            ),
            "move_right_forward": lambda: self._move(
                axis=2,
                delta=scale,
                rotation=rotation,
                center_depth=center_depth,
                camera=self._move(
                    axis=0, delta=scale, rotation=rotation, center_depth=center_depth
                ),
            ),
            "move_back_left": lambda: self._move(
                axis=2,
                delta=-scale,
                rotation=rotation,
                center_depth=center_depth,
                camera=self._move(
                    axis=0, delta=-scale, rotation=rotation, center_depth=center_depth
                ),
            ),
            "move_back_right": lambda: self._move(
                axis=2,
                delta=scale,
                rotation=rotation,
                center_depth=center_depth,
                camera=self._move(
                    axis=0, delta=scale, rotation=rotation, center_depth=center_depth
                ),
            ),
            "move_up": lambda: self._move(
                axis=1, delta=-scale, rotation=rotation, center_depth=center_depth
            ),
            "move_down": lambda: self._move(
                axis=1, delta=scale, rotation=rotation, center_depth=center_depth
            ),
            "orbit_left": lambda: self._orbit(
                axis=1, direction=1, center_depth=center_depth, theta=30
            ),
            "orbit_right": lambda: self._orbit(
                axis=1, direction=-1, center_depth=center_depth, theta=30
            ),
            "pan_left": lambda: self._pan(axis=1, direction=-1, theta=30),
            "pan_right": lambda: self._pan(axis=1, direction=1, theta=30),
            "fixed": lambda: self._fixed(),
        }

        if self.layout not in actions:
            raise ValueError(f"Unknown layout command '{self.layout}'")
        next_cam = actions[self.layout]()
        self.cameras.append(copy.deepcopy(next_cam))

    def interp_cameras(
        self, cam1: CameraExtrinsics, cam2: CameraExtrinsics
    ) -> List[CameraExtrinsics]:
        R1, R2 = cam1.R.cpu().numpy(), cam2.R.cpu().numpy()
        T1, T2 = cam1.T.cpu().numpy(), cam2.T.cpu().numpy()
        slerp = Slerp([0.0, 1.0], R.from_matrix([R1, R2]))
        cams: List[CameraExtrinsics] = []
        for i in range(1, self.total_frames):
            t = i / self.total_frames
            R_interp = torch.from_numpy(slerp([t]).as_matrix()[0]).to(self.device)
            T_interp = torch.from_numpy(T1 + t * (T2 - T1)).to(self.device)
            cams.append(CameraExtrinsics(R_interp, T_interp).to(self.device))
        return cams

    def generate_interp_cameras(self):
        self.cameras_interp.append(copy.deepcopy(self.cameras[0]))
        for c1, c2 in zip(self.cameras[:-1], self.cameras[1:]):
            self.cameras_interp.extend(self.interp_cameras(c1, c2))
            self.cameras_interp.append(copy.deepcopy(c2))

    def _interp_segment_cameras(
        self,
        cam1: CameraExtrinsics,
        cam2: CameraExtrinsics,
        num_points: int,
        include_start: bool,
    ) -> List[CameraExtrinsics]:
        if num_points <= 0:
            return []

        R1, R2 = cam1.R.cpu().numpy(), cam2.R.cpu().numpy()
        T1, T2 = cam1.T.cpu().numpy(), cam2.T.cpu().numpy()
        slerp = Slerp([0.0, 1.0], R.from_matrix([R1, R2]))

        if include_start:
            if num_points == 1:
                ts = [0.0]
            else:
                ts = [i / (num_points - 1) for i in range(num_points)]
        else:
            ts = [i / num_points for i in range(1, num_points + 1)]

        cams: List[CameraExtrinsics] = []
        for t in ts:
            R_interp = torch.from_numpy(slerp([t]).as_matrix()[0]).to(self.device)
            T_interp = torch.from_numpy(T1 + t * (T2 - T1)).to(self.device)
            cams.append(CameraExtrinsics(R_interp, T_interp).to(self.device))
        return cams

    def _generate_linear_interp_cameras(self):
        self.cameras_interp = []
        for idx, (c1, c2) in enumerate(zip(self.cameras[:-1], self.cameras[1:])):
            frames = self.segment_frames[idx]
            self.cameras_interp.extend(
                self._interp_segment_cameras(c1, c2, frames, include_start=idx == 0)
            )

    # ------------- alignment helpers --------------

    def transform_all_cam_to_current_cam(self, index: int = -1):
        """Left‑multiply every camera so the chosen camera becomes the origin."""
        inv = torch.inverse(self.cameras[index].RT)
        for cam in self.cameras:
            cam.set_RT(inv @ cam.RT)

    def _export_matrices(self, cam_list: List[CameraExtrinsics]):
        mats = [(cam.RT).unsqueeze(0) for cam in cam_list]
        return torch.cat(mats, dim=0).cpu().numpy().tolist()

    def _save_cameras_arrays(self):
        """Export VideoX camera trajectory as dense arrays.

        - c2ws: (n, 4, 4)
        - intrinsics: (n, 3, 3)
        """
        camera_type = self.camera_type_current
        width, height = self.width, self.height
        traj_str = "_".join(self.traj)
        fx = self.focal_length / width
        fy = self.focal_length / height
        cx = 0.5  # center x in normalized coordinates
        cy = 0.5  # center y in normalized coordinates

        # Internal trajectory is view matrices (w2c); export as c2w for downstream use.
        w2cs = torch.stack([camera.RT for camera in self.cameras_interp], dim=0)
        c2ws = torch.inverse(w2cs).cpu().numpy().astype(np.float32)
        intrinsic = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        intrinsics = np.repeat(intrinsic[None, :, :], c2ws.shape[0], axis=0)

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(
            self.output_dir, f"{camera_type}_camera_{traj_str}.npz"
        )
        np.savez(output_path, c2ws=c2ws, extrinsics=c2ws, intrinsics=intrinsics)

        print(
            f"Exported cameras to {output_path}, "
            f"c2ws shape={c2ws.shape}, intrinsics shape={intrinsics.shape}"
        )

    def _save_cameras(self, camera_type: str):
        if camera_type != "videox":
            raise ValueError(
                f"Only 'videox' camera_type is supported, got '{camera_type}'"
            )
        self._save_cameras_arrays()

    def _generate_cameras(self, camera_type: str):
        if all(layout in LINEAR_TRAJECTORIES for layout in self.traj):
            for layout, rotation in zip(self.traj, self.camera_rotation):
                self.layout = layout
                self.transform_all_cam_to_current_cam()
                self.generate_keyframe_camera(camera_type, rotation)
                # anchor all poses so that the first keyframe sits at origin
                self.transform_all_cam_to_current_cam(index=0)
            self._generate_linear_interp_cameras()
            return

        if len(self.traj) != 1:
            raise ValueError(
                "Non-linear trajectories (circle/spiral*) currently only support a single traj value"
            )

        for layout, rotation in zip(self.traj, self.camera_rotation):
            self.layout = layout
            if layout in LINEAR_TRAJECTORIES:
                self.transform_all_cam_to_current_cam()
                self.generate_keyframe_camera(camera_type, rotation)

                # anchor all poses so that the first keyframe sits at origin
                self.transform_all_cam_to_current_cam(index=0)
                self.generate_interp_cameras()
            else:
                if "circle" in layout:
                    self._create_circle_cameras()
                elif "spiral" in layout:
                    self._create_spiral_cameras(rotation)
                else:
                    raise ValueError(f"Unknown layout '{layout}'")

    def _create_circle_cameras(self):
        camera_type = self.camera_type_current
        layout = self.layout
        isclockwise = "counter" not in layout
        frames = self.total_frames + 1
        center_depth = self.scales[camera_type]["center_depth"]
        scale = self.scales[camera_type]["scale"]
        is_c2w = False

        radius_x = 0.03 * scale
        radius_y = 0.02 * scale
        radius_z = 0.00 * scale

        z_forward_to_camera = False
        invert_pos = True
        sign = 1.0
        look_at = torch.tensor([0.0, 0.0, sign * center_depth]).to(self.device)

        # Spiral motion key points
        spiral_positions = []
        initial_camera_pos = torch.tensor(
            [0, 0, 0], device=self.device
        )  # world_to_camera_matrix[:3, 3].clone()

        example_scale = 10.0
        theta_max = 2 * math.pi

        for i in range(frames):
            theta = theta_max * i / (frames - 1)  # angle for each point
            x = (
                radius_x
                * (math.cos(theta) - 1)
                * (1 if isclockwise else -1)
                * (center_depth / example_scale)
            )

            y = radius_y * math.sin(theta) * (center_depth / example_scale) * sign
            z = radius_z * math.sin(theta) * (center_depth / example_scale) * sign
            spiral_positions.append(torch.tensor([x, y, z], device=self.device))

        for pos in spiral_positions:
            view_matrix = look_at_matrix(
                initial_camera_pos + pos,
                look_at,
                z_forward_to_camera=z_forward_to_camera,
                invert_pos=invert_pos,
            )
            # for c2w matrix, we need to invert the rotation
            if is_c2w:
                R = view_matrix[:3, :3]
                view_matrix[:3, :3] = R.T

            camera = CameraExtrinsics()
            camera.set_R(view_matrix[:3, :3])
            self.cameras_interp.append(camera)

    def _create_spiral_cameras(self, camera_rotation: str):
        camera_type = self.camera_type_current
        layout = self.layout
        isclockwise = "counter" not in layout
        frames = self.total_frames + 1
        center_depth = self.scales[camera_type]["center_depth"]
        scale = self.scales[camera_type]["scale"]
        is_c2w = False

        radius_x = 0.3 * scale
        radius_y = 0.2 * scale
        radius_z = 0.0 * scale

        z_forward_to_camera = False
        invert_pos = True
        sign = 1.0
        look_at = torch.tensor([0.0, 0.0, sign * center_depth]).to(self.device)

        # Spiral motion key points
        spiral_positions = []
        initial_camera_pos = torch.tensor(
            [0, 0, 0], device=self.device
        )  # world_to_camera_matrix[:3, 3].clone()

        example_scale = 1.0
        theta_max = 2 * math.pi

        for i in range(frames):
            theta = theta_max * i / (frames - 1)  # angle for each point
            x = (
                radius_x
                * (math.cos(theta) - 1)
                * (1 if isclockwise else -1)
                * (center_depth / example_scale)
            )

            y = radius_y * math.sin(theta) * (center_depth / example_scale) * sign
            z = radius_z * math.sin(theta) * (center_depth / example_scale) * sign
            spiral_positions.append(torch.tensor([x, y, z], device=self.device))

        for pos in spiral_positions:
            if camera_rotation == "center_facing":
                view_matrix = look_at_matrix(
                    initial_camera_pos + pos,
                    look_at,
                    z_forward_to_camera=z_forward_to_camera,
                    invert_pos=invert_pos,
                )
            elif camera_rotation == "trajectory_aligned":
                view_matrix = look_at_matrix(
                    initial_camera_pos + pos,
                    look_at + pos * 2,
                    z_forward_to_camera=z_forward_to_camera,
                    invert_pos=invert_pos,
                )
            elif camera_rotation == "no_rotation":
                view_matrix = look_at_matrix(
                    initial_camera_pos + pos,
                    look_at + pos,
                    z_forward_to_camera=z_forward_to_camera,
                    invert_pos=invert_pos,
                )
            else:
                raise ValueError(
                    "Camera rotation should be center_facing, trajectory_aligned or no_rotation"
                )

            # for c2w matrix, we need to invert the rotation
            if is_c2w:
                R = view_matrix[:3, :3]
                view_matrix[:3, :3] = R.T

            camera = CameraExtrinsics()
            camera.set_RT(view_matrix)
            self.cameras_interp.append(camera)

    def generate_cameras(self):
        """High‑level helper wrapping everything: create, convert & save."""
        self.camera_type_current = self.camera_type
        self._generate_cameras(self.camera_type)
        self._save_cameras(self.camera_type)

        self.clear()

    # ------------- saving --------------


# -----------------------------------------------------------------------------
#  Script entry‑point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera_type",
        "-c",
        type=str,
        default="videox",
        help="camera type",
        choices=["videox"],
    )
    parser.add_argument("--width", type=int, default=640, help="width")
    parser.add_argument("--height", type=int, default=352, help="height")
    parser.add_argument("--num_frames", type=int, default=129, help="number of frames")
    parser.add_argument(
        "--focal_length", type=float, default=2500.0, help="focal length"
    )
    parser.add_argument(
        "--traj",
        type=str,
        default="move_left",
        help="trajectory, supports comma-separated values such as pull_out,pan_left",
    )
    parser.add_argument(
        "--camera_rotation",
        type=str,
        default="no_rotation",
        help="camera rotation; supports comma-separated values. A single value is broadcast to all traj segments",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/camera_trajs",
        help="output directory",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    scales = {
        "videox": {
            "scale": 0.05,
            "center_depth": 1,
        },
    }
    gen = CameraGen(args, scales)

    gen.generate_cameras()
