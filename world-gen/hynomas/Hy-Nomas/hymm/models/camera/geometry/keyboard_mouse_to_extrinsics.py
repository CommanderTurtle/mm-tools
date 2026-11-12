"""
Keyboard + Mouse → Camera Extrinsics Trajectory Generator

Converts game-style inputs (WSAD, arrow keys, mouse deltas) into camera
extrinsics trajectories (c2w 4×4 matrices) compatible with the Nomas camera
control pipeline.

Key design differences from Hunyuan-GameCraft:
  - Supports **simultaneous** key presses (e.g. W+A = forward-left diagonal)
  - Mouse delta maps directly to yaw/pitch rotation (FPS-style)
  - Arrow keys also map to rotation
  - Per-frame accumulation: every frame applies all active inputs

Coordinate system (aligned with camera_generator.py / videox c2w output):
  - +X  = right   (D key)
  - +Y  = down    (opposite of move_up)
  - +Z  = forward (W key)
  - Positive yaw   = look right  (right arrow / mouse_dx > 0)
  - Positive pitch = look up     (up arrow / mouse_dy < 0)

Output format (aligned with `generate_npz_trajectories.py`):
  .npz file containing:
    c2ws       : np.ndarray [T, 4, 4] float32
    extrinsics : np.ndarray [T, 4, 4] float32  (same as c2ws)
    intrinsics : np.ndarray [T, 3, 3] float32
  Filename pattern: `videox_camera_<traj_name>.npz`
"""

import os
import math
import json
import argparse
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import torch


DEFAULT_FPS = 24
DEFAULT_MOVE_SPEED = 2.8 / DEFAULT_FPS
DEFAULT_ROT_SPEED = 25.0 / DEFAULT_FPS


class CameraConfig(object):
    """Camera intrinsic and movement parameters.

    Movement defaults assume 24 FPS: 2.8 units/s and 25 degrees/s.
    ``move_speed`` and ``rot_speed`` remain per-frame increments; mouse
    sensitivity remains degrees per input delta unit.

    Defaults aligned with `generate_npz_trajectories.py`:
        focal_length=2500, width=640, height=352
        => fx = 2500/640 ≈ 3.90625
           fy = 2500/352 ≈ 7.10227
           cx = cy = 0.5  (normalized principal point)
    """

    DEFAULT_FOCAL_LENGTH = 2500.0
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 352

    def __init__(
        self,
        fx=None,
        fy=None,
        cx=0.5,
        cy=0.5,
        focal_length=DEFAULT_FOCAL_LENGTH,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        move_speed=DEFAULT_MOVE_SPEED,
        rot_speed=DEFAULT_ROT_SPEED,
        mouse_sensitivity=0.1,
    ):
        self.fx = fx if fx is not None else focal_length / width
        self.fy = fy if fy is not None else focal_length / height
        self.cx = cx
        self.cy = cy
        self.focal_length = focal_length
        self.width = width
        self.height = height
        self.move_speed = move_speed
        self.rot_speed = rot_speed
        self.mouse_sensitivity = mouse_sensitivity


class FrameInput(object):
    """Input state for a single frame.

    All fields are booleans for key-down state, or floats for mouse deltas.
    Multiple keys can be True simultaneously.
    """

    def __init__(
        self,
        w=False,
        s=False,
        a=False,
        d=False,
        up=False,
        down=False,
        left=False,
        right=False,
        mouse_dx=0.0,
        mouse_dy=0.0,
    ):
        self.w = w  # forward
        self.s = s  # backward
        self.a = a  # strafe left
        self.d = d  # strafe right
        self.up = up  # pitch up (look up)
        self.down = down  # pitch down (look down)
        self.left = left  # yaw left (look left)
        self.right = right  # yaw right (look right)
        self.mouse_dx = mouse_dx  # horizontal: positive = move right = yaw right
        self.mouse_dy = mouse_dy  # vertical:   positive = move down  = pitch down


def euler_to_rotation_matrix(
    pitch_deg: float, yaw_deg: float, roll_deg: float = 0.0
) -> np.ndarray:
    """Convert Euler angles (degrees) to a 3×3 rotation matrix (OpenCV c2w).

    OpenCV convention: +X right, +Y down, +Z forward.
    Ry(yaw) @ Rx(pitch) @ Rz(roll), where yaw rotates around Y-axis (left/right)
    and pitch rotates around X-axis (up/down).
    """
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    r = math.radians(roll_deg)

    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)

    Ry = np.array(
        [
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy],
        ]
    )
    Rx = np.array(
        [
            [1, 0, 0],
            [0, cp, -sp],
            [0, sp, cp],
        ]
    )
    Rz = np.array(
        [
            [cr, -sr, 0],
            [sr, cr, 0],
            [0, 0, 1],
        ]
    )

    return Ry @ Rx @ Rz


def get_forward_vec(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Camera forward direction in world space (+Z column of c2w rotation).

    OpenCV: camera looks along +Z.
    """
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    return np.array(
        [
            math.sin(y) * math.cos(p),
            -math.sin(p),
            math.cos(y) * math.cos(p),
        ]
    )


def get_right_vec(yaw_deg: float) -> np.ndarray:
    """Camera right direction in world space (+X column of c2w rotation).

    Right vector is always horizontal (no pitch dependency), matching FPS games.
    """
    y = math.radians(yaw_deg)
    return np.array(
        [
            math.cos(y),
            0.0,
            -math.sin(y),
        ]
    )


def generate_trajectory(
    frame_inputs: List[FrameInput],
    config: CameraConfig = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate camera extrinsics trajectory from per-frame inputs.

    Output length == len(frame_inputs).  Frame 0 is the identity pose (no
    input applied yet); frame i reflects the cumulative effect of inputs
    0..i-1.  This way duration N produces exactly N frames.

    Parameters
    ----------
    frame_inputs : list of FrameInput
        One entry per frame describing which keys/mouse are active.
    config : CameraConfig
        Camera parameters.

    Returns
    -------
    norm_extrinsics : torch.Tensor, shape [T, 4, 4]
        Camera-to-world (c2w) matrices for each frame.
    norm_intrinsics : torch.Tensor, shape [T, 3, 3]
        Camera intrinsic matrices for each frame (constant across frames).
    """
    if config is None:
        config = CameraConfig()

    position = np.array([0.0, 0.0, 0.0])
    pitch = 0.0  # degrees
    yaw = 0.0  # degrees
    roll = 0.0  # degrees

    c2w_list = []

    for idx, fi in enumerate(frame_inputs):
        if idx == 0:
            # Frame 0 is always the identity (origin) pose
            c2w_list.append(np.eye(4, dtype=np.float64))
            continue

        # --- Accumulate rotation ---
        # Positive yaw  = look right (+X), matching camera_generator's pan_right
        # Positive pitch = look up    (-Y), matching camera_generator's tilt_up
        if fi.up:
            pitch += config.rot_speed
        if fi.down:
            pitch -= config.rot_speed
        if fi.left:
            yaw -= config.rot_speed
        if fi.right:
            yaw += config.rot_speed

        yaw += fi.mouse_dx * config.mouse_sensitivity
        pitch -= fi.mouse_dy * config.mouse_sensitivity

        pitch = max(-89.0, min(89.0, pitch))

        # --- Accumulate translation ---
        forward = get_forward_vec(pitch, yaw)
        right = get_right_vec(yaw)

        move = np.zeros(3)
        if fi.w:
            move += forward * config.move_speed
        if fi.s:
            move -= forward * config.move_speed
        if fi.a:
            move -= right * config.move_speed
        if fi.d:
            move += right * config.move_speed

        position = position + move

        # --- Build c2w ---
        R = euler_to_rotation_matrix(pitch, yaw, roll)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R
        c2w[:3, 3] = position
        c2w_list.append(c2w.copy())

    T = len(c2w_list)

    # Build intrinsics (constant across frames)
    K = np.array(
        [
            [config.fx, 0, config.cx],
            [0, config.fy, config.cy],
            [0, 0, 1.0],
        ],
        dtype=np.float64,
    )

    norm_extrinsics = torch.tensor(
        np.stack(c2w_list, axis=0), dtype=torch.float32
    )  # [T, 4, 4]
    norm_intrinsics = (
        torch.tensor(K, dtype=torch.float32).unsqueeze(0).expand(T, -1, -1).contiguous()
    )  # [T, 3, 3]

    return norm_extrinsics, norm_intrinsics


# ──────────────────────────────────────────────────────────────────
# Convenience: build frame_inputs from a compact action-sequence spec
# ──────────────────────────────────────────────────────────────────


def parse_action_sequence(action_spec: List[Dict]) -> List[FrameInput]:
    """Build per-frame FrameInput list from a high-level action sequence.

    Each element in action_spec is a dict:
        {
            "keys": ["w", "a"],         # active keys this segment
            "mouse_dx": 0.0,            # per-frame mouse delta X
            "mouse_dy": 0.0,            # per-frame mouse delta Y
            "duration": 33              # number of frames
        }

    Keys can be any combination of: w, s, a, d, up, down, left, right.
    Multiple segments are concatenated to form the full trajectory.

    Example:
        [
            {"keys": ["w"],          "duration": 30},
            {"keys": ["w", "a"],     "duration": 20},
            {"keys": ["d"],          "mouse_dx": 2.0, "duration": 15},
        ]
    """
    valid_keys = {"w", "s", "a", "d", "up", "down", "left", "right"}
    frames = []

    for seg in action_spec:
        keys = set(seg.get("keys", []))
        unknown = keys - valid_keys
        if unknown:
            raise ValueError(f"Unknown keys: {unknown}. Valid: {valid_keys}")

        mdx = seg.get("mouse_dx", 0.0)
        mdy = seg.get("mouse_dy", 0.0)
        dur = seg.get("duration", 1)

        for _ in range(dur):
            frames.append(
                FrameInput(
                    w="w" in keys,
                    s="s" in keys,
                    a="a" in keys,
                    d="d" in keys,
                    up="up" in keys,
                    down="down" in keys,
                    left="left" in keys,
                    right="right" in keys,
                    mouse_dx=mdx,
                    mouse_dy=mdy,
                )
            )

    return frames


def action_sequence_to_extrinsics(
    action_spec: List[Dict],
    config: CameraConfig = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """End-to-end: action spec → (norm_extrinsics, norm_intrinsics)."""
    frames = parse_action_sequence(action_spec)
    return generate_trajectory(frames, config)


# ──────────────────────────────────────────────────────────────────
# Preset trajectories for common camera motions
# ──────────────────────────────────────────────────────────────────


def preset_forward(
    duration: int = 33, speed: float = DEFAULT_MOVE_SPEED
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Move forward for `duration` frames."""
    cfg = CameraConfig(move_speed=speed)
    spec = [{"keys": ["w"], "duration": duration}]
    return action_sequence_to_extrinsics(spec, cfg)


def preset_backward(
    duration: int = 33, speed: float = DEFAULT_MOVE_SPEED
) -> Tuple[torch.Tensor, torch.Tensor]:
    cfg = CameraConfig(move_speed=speed)
    spec = [{"keys": ["s"], "duration": duration}]
    return action_sequence_to_extrinsics(spec, cfg)


def preset_strafe_left(
    duration: int = 33, speed: float = DEFAULT_MOVE_SPEED
) -> Tuple[torch.Tensor, torch.Tensor]:
    cfg = CameraConfig(move_speed=speed)
    spec = [{"keys": ["a"], "duration": duration}]
    return action_sequence_to_extrinsics(spec, cfg)


def preset_strafe_right(
    duration: int = 33, speed: float = DEFAULT_MOVE_SPEED
) -> Tuple[torch.Tensor, torch.Tensor]:
    cfg = CameraConfig(move_speed=speed)
    spec = [{"keys": ["d"], "duration": duration}]
    return action_sequence_to_extrinsics(spec, cfg)


def preset_look_left(
    duration: int = 33, rot_speed: float = DEFAULT_ROT_SPEED
) -> Tuple[torch.Tensor, torch.Tensor]:
    cfg = CameraConfig(rot_speed=rot_speed)
    spec = [{"keys": ["left"], "duration": duration}]
    return action_sequence_to_extrinsics(spec, cfg)


def preset_look_right(
    duration: int = 33, rot_speed: float = DEFAULT_ROT_SPEED
) -> Tuple[torch.Tensor, torch.Tensor]:
    cfg = CameraConfig(rot_speed=rot_speed)
    spec = [{"keys": ["right"], "duration": duration}]
    return action_sequence_to_extrinsics(spec, cfg)


def preset_look_up(
    duration: int = 33, rot_speed: float = DEFAULT_ROT_SPEED
) -> Tuple[torch.Tensor, torch.Tensor]:
    cfg = CameraConfig(rot_speed=rot_speed)
    spec = [{"keys": ["up"], "duration": duration}]
    return action_sequence_to_extrinsics(spec, cfg)


def preset_look_down(
    duration: int = 33, rot_speed: float = DEFAULT_ROT_SPEED
) -> Tuple[torch.Tensor, torch.Tensor]:
    cfg = CameraConfig(rot_speed=rot_speed)
    spec = [{"keys": ["down"], "duration": duration}]
    return action_sequence_to_extrinsics(spec, cfg)


def preset_forward_left(
    duration: int = 33, speed: float = DEFAULT_MOVE_SPEED
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Diagonal: forward + strafe left."""
    cfg = CameraConfig(move_speed=speed)
    spec = [{"keys": ["w", "a"], "duration": duration}]
    return action_sequence_to_extrinsics(spec, cfg)


def preset_forward_look_right(
    duration: int = 33, speed: float = DEFAULT_MOVE_SPEED, rot_speed: float = DEFAULT_ROT_SPEED
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Move forward while turning right (arc motion)."""
    cfg = CameraConfig(move_speed=speed, rot_speed=rot_speed)
    spec = [{"keys": ["w", "right"], "duration": duration}]
    return action_sequence_to_extrinsics(spec, cfg)


def preset_mouse_pan(
    duration: int = 33,
    dx_per_frame: float = 2.0,
    dy_per_frame: float = 0.0,
    sensitivity: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simulate mouse movement for panning."""
    cfg = CameraConfig(mouse_sensitivity=sensitivity)
    spec = [
        {
            "keys": [],
            "mouse_dx": dx_per_frame,
            "mouse_dy": dy_per_frame,
            "duration": duration,
        }
    ]
    return action_sequence_to_extrinsics(spec, cfg)


# ──────────────────────────────────────────────────────────────────
# Unified "generator-style" interface (aligned with
# `hymm/models/camera/geometry/generate_trajectories.py::TRAJECTORY_GENERATORS`)
#
# Each generator takes `num_frames` (+ optional kwargs) and returns
#   (c2ws [N,4,4] float32, intrinsics [N,3,3] float32)
# so downstream tooling (e.g. `tools_tmp/build_camera_bench_keyboard.py`)
# can compose / re-base / save NPZ in exactly the same way it does for the
# analytic generators.
# ──────────────────────────────────────────────────────────────────


def _run_action_to_arrays(
    action_spec: List[Dict],
    config: Optional["CameraConfig"] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run an action spec → (c2ws[N,4,4] float32, intrinsics[N,3,3] float32)."""
    if config is None:
        config = CameraConfig()
    ext, intr = action_sequence_to_extrinsics(action_spec, config)
    c2ws = ext.detach().cpu().numpy().astype(np.float32)
    intrinsics = intr.detach().cpu().numpy().astype(np.float32)
    return c2ws, intrinsics


def _kb_static(num_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """No keys, no mouse: identity poses for every frame."""
    return _run_action_to_arrays([{"keys": [], "duration": num_frames}])


def _kb_w(num_frames: int, move_speed: float = DEFAULT_MOVE_SPEED) -> Tuple[np.ndarray, np.ndarray]:
    return _run_action_to_arrays(
        [{"keys": ["w"], "duration": num_frames}],
        CameraConfig(move_speed=move_speed),
    )


def _kb_s(num_frames: int, move_speed: float = DEFAULT_MOVE_SPEED) -> Tuple[np.ndarray, np.ndarray]:
    return _run_action_to_arrays(
        [{"keys": ["s"], "duration": num_frames}],
        CameraConfig(move_speed=move_speed),
    )


def _kb_a(num_frames: int, move_speed: float = DEFAULT_MOVE_SPEED) -> Tuple[np.ndarray, np.ndarray]:
    return _run_action_to_arrays(
        [{"keys": ["a"], "duration": num_frames}],
        CameraConfig(move_speed=move_speed),
    )


def _kb_d(num_frames: int, move_speed: float = DEFAULT_MOVE_SPEED) -> Tuple[np.ndarray, np.ndarray]:
    return _run_action_to_arrays(
        [{"keys": ["d"], "duration": num_frames}],
        CameraConfig(move_speed=move_speed),
    )


def _kb_arrow_left(
    num_frames: int, rot_speed: float = DEFAULT_ROT_SPEED
) -> Tuple[np.ndarray, np.ndarray]:
    return _run_action_to_arrays(
        [{"keys": ["left"], "duration": num_frames}],
        CameraConfig(rot_speed=rot_speed),
    )


def _kb_arrow_right(
    num_frames: int, rot_speed: float = DEFAULT_ROT_SPEED
) -> Tuple[np.ndarray, np.ndarray]:
    return _run_action_to_arrays(
        [{"keys": ["right"], "duration": num_frames}],
        CameraConfig(rot_speed=rot_speed),
    )


def _kb_arrow_up(
    num_frames: int, rot_speed: float = DEFAULT_ROT_SPEED
) -> Tuple[np.ndarray, np.ndarray]:
    return _run_action_to_arrays(
        [{"keys": ["up"], "duration": num_frames}],
        CameraConfig(rot_speed=rot_speed),
    )


def _kb_arrow_down(
    num_frames: int, rot_speed: float = DEFAULT_ROT_SPEED
) -> Tuple[np.ndarray, np.ndarray]:
    return _run_action_to_arrays(
        [{"keys": ["down"], "duration": num_frames}],
        CameraConfig(rot_speed=rot_speed),
    )


# Aliases accepted by the "combo" generator so callers can use the same
# names that appear as standalone entries in KEYBOARD_TRAJECTORY_GENERATORS
# (e.g. "arrow_right") instead of the raw flag names used by FrameInput
# (e.g. "right").
_COMBO_KEY_ALIASES: Dict[str, str] = {
    "arrow_left": "left",
    "arrow_right": "right",
    "arrow_up": "up",
    "arrow_down": "down",
}


def _kb_combo(
    num_frames: int,
    keys: Optional[List[str]] = None,
    mouse_dx: float = 0.0,
    mouse_dy: float = 0.0,
    move_speed: float = DEFAULT_MOVE_SPEED,
    rot_speed: float = DEFAULT_ROT_SPEED,
    mouse_sensitivity: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generic simultaneous-keys generator.

    Use this when you want >=2 keys held together for the whole segment, or
    constant per-frame mouse deltas. Examples:
      _kb_combo(N, keys=["w","a"])              # forward + strafe-left
      _kb_combo(N, keys=["w","right"])          # forward + look-right (arc)
      _kb_combo(N, keys=["w","arrow_right"])    # same, but using the alias
      _kb_combo(N, keys=["w"], mouse_dx=2.0)    # forward + mouse pan right
    """
    keys_list: List[str] = [_COMBO_KEY_ALIASES.get(k, k) for k in (keys or [])]
    return _run_action_to_arrays(
        [
            {
                "keys": keys_list,
                "mouse_dx": float(mouse_dx),
                "mouse_dy": float(mouse_dy),
                "duration": num_frames,
            }
        ],
        CameraConfig(
            move_speed=move_speed,
            rot_speed=rot_speed,
            mouse_sensitivity=mouse_sensitivity,
        ),
    )


def _kb_mouse_pan(
    num_frames: int,
    mouse_dx: float = 2.0,
    mouse_dy: float = 0.0,
    mouse_sensitivity: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure mouse-look (no keys); equivalent to preset_mouse_pan."""
    return _run_action_to_arrays(
        [
            {
                "keys": [],
                "mouse_dx": float(mouse_dx),
                "mouse_dy": float(mouse_dy),
                "duration": num_frames,
            }
        ],
        CameraConfig(mouse_sensitivity=mouse_sensitivity),
    )


# Mirrors the spirit of `TRAJECTORY_GENERATORS` in
# `hymm/models/camera/geometry/generate_trajectories.py`, but every entry
# is a "single-key (or simple combo) per-frame action" semantically tied
# to keyboard/mouse input. Composite trajectories are built on top of
# these atoms by tools that want sequential / simultaneous mixing.
#
# Naming conventions:
#   - WSAD movement keys are kept as their literal names ("w","s","a","d")
#     so the table is grep-friendly to game-input traces.
#   - Arrow-key rotation is prefixed with "arrow_" to avoid colliding with
#     the WSAD letters in user-facing logs.
#   - "combo" is the generic 1+ keys (+ optional mouse) generator.
KEYBOARD_TRAJECTORY_GENERATORS: Dict[str, callable] = {
    "static": _kb_static,
    "w": _kb_w,
    "s": _kb_s,
    "a": _kb_a,
    "d": _kb_d,
    "arrow_left": _kb_arrow_left,
    "arrow_right": _kb_arrow_right,
    "arrow_up": _kb_arrow_up,
    "arrow_down": _kb_arrow_down,
    "combo": _kb_combo,
    "mouse_pan": _kb_mouse_pan,
}


# ──────────────────────────────────────────────────────────────────
# Ground-truth per-frame key/mouse extraction
#
# Lets a downstream "overlay" tool draw the *exact* keys/mouse the
# trajectory was generated from (no need to inverse-discretise the c2w
# sequence).  Channel order matches the 8-key panel layout used by
# `tools_tmp/overlay_camera_keys_on_video.py`:
#
#     idx : 0  1  2  3  4  5  6  7
#     key : W  A  S  D  ↑  ↓  ←  →
# ──────────────────────────────────────────────────────────────────

GT_KEY_CHANNEL_ORDER: Tuple[str, ...] = (
    "w",
    "a",
    "s",
    "d",
    "up",
    "down",
    "left",
    "right",
)


def frame_inputs_to_gt_arrays(
    frame_inputs: List[FrameInput],
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame ground-truth flatten of `frame_inputs`.

    Returns
    -------
    gt_keys  : np.ndarray [N, 8] uint8
        Channel order: W, A, S, D, ↑, ↓, ←, →  (== GT_KEY_CHANNEL_ORDER).
    gt_mouse : np.ndarray [N, 2] float32
        (mouse_dx, mouse_dy) raw deltas (no sensitivity multiplied).
    """
    n = len(frame_inputs)
    gt_keys = np.zeros((n, 8), dtype=np.uint8)
    gt_mouse = np.zeros((n, 2), dtype=np.float32)
    for i, fi in enumerate(frame_inputs):
        gt_keys[i, 0] = int(bool(fi.w))
        gt_keys[i, 1] = int(bool(fi.a))
        gt_keys[i, 2] = int(bool(fi.s))
        gt_keys[i, 3] = int(bool(fi.d))
        gt_keys[i, 4] = int(bool(fi.up))
        gt_keys[i, 5] = int(bool(fi.down))
        gt_keys[i, 6] = int(bool(fi.left))
        gt_keys[i, 7] = int(bool(fi.right))
        gt_mouse[i, 0] = float(fi.mouse_dx)
        gt_mouse[i, 1] = float(fi.mouse_dy)
    return gt_keys, gt_mouse


def action_spec_to_gt_arrays(
    action_spec: List[Dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """End-to-end: action spec → (gt_keys[N,8] uint8, gt_mouse[N,2] float32).

    The caller is responsible for ensuring `sum(seg.duration) == num_frames`.
    """
    frames = parse_action_sequence(action_spec)
    return frame_inputs_to_gt_arrays(frames)


# ──────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────


def visualize_trajectory(
    c2w: torch.Tensor,
    save_path: str = "trajectory.png",
    title: str = "Camera Trajectory",
):
    """Plot camera trajectory with a 3-D perspective view and a top-down view.

    Left subplot  : 3-D view with an oblique angle so the cube structure is
                    visible.  Axes are remapped so that forward (W) points into
                    the screen-up direction, matching user intuition.
    Right subplot : top-down (bird's-eye) 2-D view.  Screen-up = forward (W),
                    screen-right = right (D).

    Parameters
    ----------
    c2w : torch.Tensor [T, 4, 4]
    save_path : str
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for projection='3d')

    c2w_np = c2w.numpy()
    T = c2w_np.shape[0]
    positions = c2w_np[:, :3, 3]  # [T, 3]  world (x, y, z)

    # Remap OpenCV world coords to user-friendly 3-D frame:
    #   OpenCV: +X=right, +Y=down, +Z=forward
    #   A (left) moves -X, D (right) moves +X  →  plot_X = world_X
    #   W (forward) moves +Z                   →  plot_Y = world_Z
    #   Up moves -Y (OpenCV Y-down)            →  plot_Z = -world_Y
    plot_x = positions[:, 0]
    plot_y = positions[:, 2]
    plot_z = -positions[:, 1]

    fig = plt.figure(figsize=(16, 7))

    # ─── Left: 3-D perspective view ───────────────────────────────
    ax3d = fig.add_subplot(121, projection="3d")

    # Color-gradient path
    for i in range(T - 1):
        ratio = i / max(T - 1, 1)
        c = plt.cm.viridis(ratio)
        ax3d.plot(
            plot_x[i : i + 2],
            plot_y[i : i + 2],
            plot_z[i : i + 2],
            color=c,
            linewidth=2.5,
        )

    ax3d.scatter(
        plot_x[0],
        plot_y[0],
        plot_z[0],
        c="green",
        s=90,
        marker="o",
        label="start",
        edgecolors="black",
        linewidths=0.6,
        zorder=5,
    )
    ax3d.scatter(
        plot_x[-1],
        plot_y[-1],
        plot_z[-1],
        c="red",
        s=90,
        marker="^",
        label="end",
        edgecolors="black",
        linewidths=0.6,
        zorder=5,
    )

    # Gaze arrows every N-th frame
    step = max(1, T // 12)
    all_range = max(
        plot_x.max() - plot_x.min(),
        plot_y.max() - plot_y.min(),
        plot_z.max() - plot_z.min(),
        0.01,
    )
    arrow_len = all_range * 0.10
    for i in range(0, T, step):
        fwd_w = c2w_np[i, :3, 2]  # forward in world = +Z col (OpenCV)
        # remap to plot coords: plot_X=X, plot_Y=Z, plot_Z=-Y
        dx, dy, dz = fwd_w[0], fwd_w[2], -fwd_w[1]
        norm = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
        dx, dy, dz = dx / norm * arrow_len, dy / norm * arrow_len, dz / norm * arrow_len
        ratio = i / max(T - 1, 1)
        ax3d.quiver(
            plot_x[i],
            plot_y[i],
            plot_z[i],
            dx,
            dy,
            dz,
            color=plt.cm.Oranges(0.45 + 0.55 * ratio),
            arrow_length_ratio=0.25,
            linewidth=1.5,
        )

    # Oblique viewing angle: elev=25 gives a nice tilt, azim=-135 so
    # forward (plot_Y+) points roughly "into the screen / upper-left".
    ax3d.view_init(elev=25, azim=-50)

    ax3d.set_xlabel("Right (D) \u2192", fontsize=10, labelpad=8)
    ax3d.set_ylabel("Forward (W) \u2192", fontsize=10, labelpad=8)
    ax3d.set_zlabel("Up \u2192", fontsize=10, labelpad=8)
    ax3d.set_title(title + "  (3-D)", fontsize=12)
    ax3d.legend(loc="upper left", fontsize=9)

    # Equal aspect via cubic bounding box
    mid_x = (plot_x.max() + plot_x.min()) / 2
    mid_y = (plot_y.max() + plot_y.min()) / 2
    mid_z = (plot_z.max() + plot_z.min()) / 2
    half = max(all_range / 2, 0.02) * 1.15
    ax3d.set_xlim(mid_x - half, mid_x + half)
    ax3d.set_ylim(mid_y - half, mid_y + half)
    ax3d.set_zlim(mid_z - half, mid_z + half)
    ax3d.grid(True, alpha=0.25)

    # ─── Right: top-down 2-D view ─────────────────────────────────
    ax2d = fig.add_subplot(122)

    screen_x = positions[:, 0]  # right (D) = +world_X  (OpenCV)
    screen_y = positions[:, 2]  # forward (W) = +world_Z (OpenCV)

    for i in range(T - 1):
        ratio = i / max(T - 1, 1)
        c = plt.cm.viridis(ratio)
        ax2d.plot(screen_x[i : i + 2], screen_y[i : i + 2], color=c, linewidth=2.5)

    ax2d.scatter(
        screen_x[0],
        screen_y[0],
        c="green",
        s=100,
        marker="o",
        edgecolors="black",
        linewidths=0.8,
        zorder=4,
        label="start",
    )
    ax2d.scatter(
        screen_x[-1],
        screen_y[-1],
        c="red",
        s=100,
        marker="^",
        edgecolors="black",
        linewidths=0.8,
        zorder=4,
        label="end",
    )

    # Gaze arrows
    arrow_2d = (
        max(
            screen_x.max() - screen_x.min(),
            screen_y.max() - screen_y.min(),
            0.01,
        )
        * 0.08
    )
    for i in range(0, T, step):
        fwd_w = c2w_np[i, :3, 2]  # +Z col (OpenCV forward)
        dx = fwd_w[0]  # screen_x = world_X
        dy = fwd_w[2]  # screen_y = world_Z
        norm = math.sqrt(dx * dx + dy * dy) + 1e-12
        dx, dy = dx / norm * arrow_2d, dy / norm * arrow_2d
        ratio = i / max(T - 1, 1)
        ax2d.annotate(
            "",
            xy=(screen_x[i] + dx, screen_y[i] + dy),
            xytext=(screen_x[i], screen_y[i]),
            arrowprops=dict(
                arrowstyle="->", color=plt.cm.Oranges(0.45 + 0.55 * ratio), lw=1.8
            ),
        )

    ax2d.set_xlabel("Right (D) \u2192", fontsize=11)
    ax2d.set_ylabel("Forward (W) \u2192", fontsize=11)
    ax2d.set_title(title + "  (top-down)", fontsize=12)
    ax2d.legend(loc="best", fontsize=9)
    ax2d.set_aspect("equal")
    ax2d.grid(True, alpha=0.3)

    pad_x = max((screen_x.max() - screen_x.min()) * 0.15, 0.01)
    pad_y = max((screen_y.max() - screen_y.min()) * 0.15, 0.01)
    ax2d.set_xlim(screen_x.min() - pad_x, screen_x.max() + pad_x)
    ax2d.set_ylim(screen_y.min() - pad_y, screen_y.max() + pad_y)

    # WSAD labels at edges
    ax2d.annotate(
        "W",
        xy=(0.5, 0.97),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=10,
        color="gray",
        fontweight="bold",
    )
    ax2d.annotate(
        "S",
        xy=(0.5, 0.03),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=10,
        color="gray",
        fontweight="bold",
    )
    ax2d.annotate(
        "A",
        xy=(0.03, 0.5),
        xycoords="axes fraction",
        ha="left",
        va="center",
        fontsize=10,
        color="gray",
        fontweight="bold",
        rotation=90,
    )
    ax2d.annotate(
        "D",
        xy=(0.97, 0.5),
        xycoords="axes fraction",
        ha="right",
        va="center",
        fontsize=10,
        color="gray",
        fontweight="bold",
        rotation=90,
    )

    plt.tight_layout()
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Trajectory saved to {save_path}")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate camera extrinsics from keyboard/mouse inputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Forward for 33 frames
  python keyboard_mouse_to_extrinsics.py --preset forward --duration 33

  # Forward + strafe left simultaneously
  python keyboard_mouse_to_extrinsics.py --preset forward_left --duration 33

  # From JSON action spec
  python keyboard_mouse_to_extrinsics.py --action_json '[
      {"keys": ["w"], "duration": 20},
      {"keys": ["w", "d"], "duration": 15},
      {"keys": ["right"], "mouse_dx": 1.5, "duration": 10}
  ]'

  # Save and visualize
  python keyboard_mouse_to_extrinsics.py --preset forward_look_right \\
      --save_path traj.pt --viz_path traj.png
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--preset",
        type=str,
        choices=[
            "forward",
            "backward",
            "strafe_left",
            "strafe_right",
            "look_left",
            "look_right",
            "look_up",
            "look_down",
            "forward_left",
            "forward_look_right",
            "mouse_pan",
        ],
        help="Use a preset trajectory",
    )
    group.add_argument(
        "--action_json",
        type=str,
        help="JSON string or path to JSON file with action sequence",
    )

    parser.add_argument("--duration", type=int, default=129)
    parser.add_argument(
        "--move_speed", type=float, default=DEFAULT_MOVE_SPEED,
        help="Units per frame; default 2.8 units/s at 24 FPS",
    )
    parser.add_argument(
        "--rot_speed", type=float, default=DEFAULT_ROT_SPEED,
        help="Degrees per frame; default 25 degrees/s at 24 FPS",
    )
    parser.add_argument("--mouse_sensitivity", type=float, default=0.1)
    parser.add_argument(
        "--save_path",
        type=str,
        default="outputs/keyboard/trajectory.pt",
        help="Save (extrinsics, intrinsics) to .pt file",
    )
    parser.add_argument(
        "--viz_path",
        type=str,
        default="outputs/keyboard/trajectory.png",
        help="Save trajectory visualization to image",
    )

    args = parser.parse_args()

    config = CameraConfig(
        move_speed=args.move_speed,
        rot_speed=args.rot_speed,
        mouse_sensitivity=args.mouse_sensitivity,
    )

    """
    [{"keys": ["w"], "duration": 32}, {"keys": ["s"], "duration": 32}, {"keys": ["a"], "duration": 32}, {"keys": ["d"], "duration": 33}]
    """

    if args.preset:
        preset_funcs = {
            "forward": preset_forward,
            "backward": preset_backward,
            "strafe_left": preset_strafe_left,
            "strafe_right": preset_strafe_right,
            "look_left": preset_look_left,
            "look_right": preset_look_right,
            "look_up": preset_look_up,
            "look_down": preset_look_down,
            "forward_left": preset_forward_left,
            "forward_look_right": preset_forward_look_right,
            "mouse_pan": preset_mouse_pan,
        }
        fn = preset_funcs[args.preset]
        if args.preset in (
            "forward",
            "backward",
            "strafe_left",
            "strafe_right",
            "forward_left",
        ):
            ext, intr = fn(duration=args.duration, speed=args.move_speed)
        elif args.preset in ("look_left", "look_right", "look_up", "look_down"):
            ext, intr = fn(duration=args.duration, rot_speed=args.rot_speed)
        elif args.preset == "forward_look_right":
            ext, intr = fn(
                duration=args.duration, speed=args.move_speed, rot_speed=args.rot_speed
            )
        elif args.preset == "mouse_pan":
            ext, intr = fn(duration=args.duration, sensitivity=args.mouse_sensitivity)
    else:
        action_str = args.action_json
        # Try reading as file first
        try:
            with open(action_str, "r") as f:
                action_spec = json.load(f)
        except (FileNotFoundError, OSError):
            action_spec = json.loads(action_str)
        ext, intr = action_sequence_to_extrinsics(action_spec, config)

    print(f"Generated trajectory: {ext.shape[0]} frames")
    print(f"  norm_extrinsics: {ext.shape}  (c2w)")
    print(f"  norm_intrinsics: {intr.shape}")

    # Print first and last frame
    print(f"\n  Frame 0 (c2w):\n{ext[0]}")
    print(f"\n  Frame {ext.shape[0] - 1} (c2w):\n{ext[-1]}")

    # Translation stats
    positions = ext[:, :3, 3].numpy()
    total_dist = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
    print(f"\n  Total distance traveled: {total_dist:.6f}")
    print(f"  Final position: {positions[-1]}")

    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        save_dict = {"norm_extrinsics": ext, "norm_intrinsics": intr}
        if args.preset:
            save_dict["action_json"] = [
                {"keys": [args.preset], "duration": args.duration}
            ]
        else:
            save_dict["action_json"] = action_spec
        torch.save(save_dict, args.save_path)
        print(f"\n  Saved to {args.save_path}")

    if args.viz_path:
        visualize_trajectory(ext, args.viz_path)


if __name__ == "__main__":
    main()
