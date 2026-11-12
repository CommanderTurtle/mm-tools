"""
Generate various camera trajectories for testing and evaluation.

Coordinate system (OpenCV convention):
    +X = right
    +Y = down
    +Z = forward (camera looking direction)
"""

import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R


def make_intrinsic(fx=969.7, fy=969.7, cx=960.0, cy=540.0):
    """Create a 3x3 intrinsic matrix."""
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def make_c2w(rotation_matrix, translation):
    """Create a 4x4 camera-to-world matrix."""
    c2w = np.eye(4)
    c2w[:3, :3] = rotation_matrix
    c2w[:3, 3] = translation
    return c2w.tolist()


def look_at(camera_pos, target_pos, up=None):
    """
    Compute rotation matrix for camera looking at target.

    OpenCV convention: +X=right, +Y=down, +Z=forward
    """
    if up is None:
        up = np.array([0.0, -1.0, 0.0])  # Y-down world up

    forward = target_pos - camera_pos
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        # forward is parallel to up, choose different up
        up = np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    # Rotation matrix: columns are [right, down, forward]
    rotation = np.stack([right, down, forward], axis=1)
    return rotation


def generate_forward_trajectory(num_frames=33, step_size=0.08):
    """
    Generate a straight forward trajectory (camera moves along +Z axis).

    相机沿着 Z 轴向前移动。
    """
    poses = {}
    K = make_intrinsic()

    for i in range(num_frames):
        rotation = np.eye(3)
        translation = np.array([0.0, 0.0, i * step_size])
        poses[str(i)] = {"extrinsic": make_c2w(rotation, translation), "K": K}

    return poses


def generate_circular_trajectory(num_frames=33, radius=1.0, height=0.0):
    """
    Generate a circular trajectory around the origin.

    相机绕原点做圆周运动，始终朝向中心。
    OpenCV: +X=right, +Y=down, +Z=forward
    """
    poses = {}
    K = make_intrinsic()

    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames

        # Camera position on circle (XZ plane, Y is height/down)
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        translation = np.array([x, height, z])

        # Camera looks at origin
        target = np.array([0.0, height, 0.0])
        rotation = look_at(translation, target)

        poses[str(i)] = {"extrinsic": make_c2w(rotation, translation), "K": K}

    return poses


def generate_spiral_trajectory(
    num_frames=33, radius=1.0, height_range=1.0, num_loops=1
):
    """
    Generate a spiral trajectory going downward while circling.

    相机螺旋下降（+Y=down），同时绕原点旋转。
    """
    poses = {}
    K = make_intrinsic()

    for i in range(num_frames):
        t = i / (num_frames - 1)
        angle = 2 * np.pi * num_loops * t

        # Position: spiral downward (+Y = down in OpenCV)
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        y = height_range * t  # going down
        translation = np.array([x, y, z])

        # Camera looks at center axis
        target = np.array([0.0, y, 0.0])
        rotation = look_at(translation, target)

        poses[str(i)] = {"extrinsic": make_c2w(rotation, translation), "K": K}

    return poses


def generate_pan_left_trajectory(num_frames=33, total_angle=45.0, step_size=0.0):
    """
    Generate a pan left trajectory (camera rotates left around Y axis).

    相机原地向左旋转（yaw left）。
    OpenCV (+X=right, +Y=down, +Z=forward): c2w 的第三列就是相机 forward
    在世界系下的方向。`R_y(a) @ (0,0,1) = (sin a, 0, cos a)`，所以
    `a > 0` 会让 forward 偏向世界 +X（右） = pan_right；要 pan_left
    必须用负角度。
    """
    poses = {}
    K = make_intrinsic()

    for i in range(num_frames):
        angle = np.radians(total_angle * i / (num_frames - 1))
        rotation = R.from_euler("y", -angle).as_matrix()
        translation = np.array([0.0, 0.0, i * step_size])

        poses[str(i)] = {"extrinsic": make_c2w(rotation, translation), "K": K}

    return poses


def generate_pan_right_trajectory(num_frames=33, total_angle=45.0, step_size=0.0):
    """
    Generate a pan right trajectory (camera rotates right around Y axis).

    相机原地向右旋转（yaw right）。
    OpenCV: `R_y(+a) @ (0,0,1) = (sin a, 0, cos a)`，正角度让相机 forward
    偏向世界 +X（右） = pan_right。
    """
    poses = {}
    K = make_intrinsic()

    for i in range(num_frames):
        angle = np.radians(total_angle * i / (num_frames - 1))
        rotation = R.from_euler("y", angle).as_matrix()
        translation = np.array([0.0, 0.0, i * step_size])

        poses[str(i)] = {"extrinsic": make_c2w(rotation, translation), "K": K}

    return poses


def generate_tilt_up_trajectory(num_frames=33, total_angle=30.0):
    """
    Generate a tilt up trajectory (camera rotates up around X axis).

    相机原地向上抬头（pitch up）。
    OpenCV: `R_x(+a) @ (0,0,1) = (0, -sin a, cos a)`，正角度让 forward 的
    Y 分量为负；OpenCV 中 +Y=down，所以 -Y=up，即相机看向上 = tilt_up。
    """
    poses = {}
    K = make_intrinsic()

    for i in range(num_frames):
        angle = np.radians(total_angle * i / (num_frames - 1))
        rotation = R.from_euler("x", angle).as_matrix()
        translation = np.array([0.0, 0.0, 0.0])

        poses[str(i)] = {"extrinsic": make_c2w(rotation, translation), "K": K}

    return poses


def generate_tilt_down_trajectory(num_frames=33, total_angle=30.0):
    """
    Generate a tilt down trajectory (camera rotates down around X axis).

    相机原地向下低头（pitch down）。
    OpenCV: `R_x(-a) @ (0,0,1) = (0, sin a, cos a)`，负角度让 forward 的
    Y 分量为正（+Y=down），即相机看向下 = tilt_down。
    """
    poses = {}
    K = make_intrinsic()

    for i in range(num_frames):
        angle = np.radians(total_angle * i / (num_frames - 1))
        rotation = R.from_euler("x", -angle).as_matrix()
        translation = np.array([0.0, 0.0, 0.0])

        poses[str(i)] = {"extrinsic": make_c2w(rotation, translation), "K": K}

    return poses


def generate_dolly_zoom_trajectory(num_frames=33, start_z=0.0, end_z=2.0):
    """
    Generate a dolly (push in) trajectory.

    相机沿着 Z 轴推进（dolly in）。
    """
    poses = {}
    K = make_intrinsic()

    for i in range(num_frames):
        t = i / (num_frames - 1)
        z = start_z + t * (end_z - start_z)
        rotation = np.eye(3)
        translation = np.array([0.0, 0.0, z])

        poses[str(i)] = {"extrinsic": make_c2w(rotation, translation), "K": K}

    return poses


def generate_crane_up_trajectory(num_frames=33, height_change=1.0):
    """
    Generate a crane up trajectory (camera moves upward in world).

    相机垂直向上移动（crane shot）。
    OpenCV: +Y=down, so moving up = negative Y
    """
    poses = {}
    K = make_intrinsic()

    for i in range(num_frames):
        t = i / (num_frames - 1)
        y = -t * height_change  # negative Y = up in OpenCV
        rotation = np.eye(3)
        translation = np.array([0.0, y, 0.0])

        poses[str(i)] = {"extrinsic": make_c2w(rotation, translation), "K": K}

    return poses


def generate_crane_down_trajectory(num_frames=33, height_change=1.0):
    """
    Generate a crane down trajectory (camera moves downward in world).

    相机垂直向下移动。
    OpenCV: +Y=down
    """
    poses = {}
    K = make_intrinsic()

    for i in range(num_frames):
        t = i / (num_frames - 1)
        y = t * height_change  # positive Y = down in OpenCV
        rotation = np.eye(3)
        translation = np.array([0.0, y, 0.0])

        poses[str(i)] = {"extrinsic": make_c2w(rotation, translation), "K": K}

    return poses


def generate_truck_left_trajectory(num_frames=33, distance=1.0):
    """
    Generate a truck left trajectory (camera moves left).

    相机向左平移（truck left）。
    OpenCV: +X=right, so left = negative X
    """
    poses = {}
    K = make_intrinsic()

    for i in range(num_frames):
        t = i / (num_frames - 1)
        x = -t * distance
        rotation = np.eye(3)
        translation = np.array([x, 0.0, 0.0])

        poses[str(i)] = {"extrinsic": make_c2w(rotation, translation), "K": K}

    return poses


def generate_truck_right_trajectory(num_frames=33, distance=1.0):
    """
    Generate a truck right trajectory (camera moves right).

    相机向右平移（truck right）。
    OpenCV: +X=right
    """
    poses = {}
    K = make_intrinsic()

    for i in range(num_frames):
        t = i / (num_frames - 1)
        x = t * distance
        rotation = np.eye(3)
        translation = np.array([x, 0.0, 0.0])

        poses[str(i)] = {"extrinsic": make_c2w(rotation, translation), "K": K}

    return poses


def generate_arc_trajectory(num_frames=33, radius=2.0, arc_angle=90.0):
    """
    Generate an arc trajectory (camera moves in an arc while looking at origin).

    相机沿弧线移动，始终看向原点。
    """
    poses = {}
    K = make_intrinsic()

    start_angle = -np.radians(arc_angle / 2)
    end_angle = np.radians(arc_angle / 2)

    for i in range(num_frames):
        t = i / (num_frames - 1)
        angle = start_angle + t * (end_angle - start_angle)

        # Camera position on arc (XZ plane)
        x = radius * np.sin(angle)
        z = radius * np.cos(angle)
        translation = np.array([x, 0.0, z])

        # Camera looks at origin
        target = np.array([0.0, 0.0, 0.0])
        rotation = look_at(translation, target)

        poses[str(i)] = {"extrinsic": make_c2w(rotation, translation), "K": K}

    return poses


def generate_zigzag_trajectory(num_frames=33, amplitude=0.5, forward_speed=0.1):
    """
    Generate a zigzag trajectory (camera moves forward while oscillating left-right).

    相机前进的同时左右摆动。
    """
    poses = {}
    K = make_intrinsic()

    for i in range(num_frames):
        t = i / (num_frames - 1)
        x = amplitude * np.sin(4 * np.pi * t)  # oscillate left-right
        z = i * forward_speed
        rotation = np.eye(3)
        translation = np.array([x, 0.0, z])

        poses[str(i)] = {"extrinsic": make_c2w(rotation, translation), "K": K}

    return poses


def generate_random_walk_trajectory(
    num_frames=33, step_size=0.05, rot_std=2.0, seed=42
):
    """
    Generate a random walk trajectory.

    相机随机游走。
    """
    np.random.seed(seed)
    poses = {}
    K = make_intrinsic()

    # Start at origin
    position = np.array([0.0, 0.0, 0.0])
    rotation = np.eye(3)

    for i in range(num_frames):
        poses[str(i)] = {"extrinsic": make_c2w(rotation, position.copy()), "K": K}

        # Random step in forward direction (camera local Z)
        forward = rotation[:, 2]
        position += forward * step_size

        # Small random translation
        position += np.random.randn(3) * step_size * 0.3

        # Small random rotation
        euler_delta = np.random.randn(3) * np.radians(rot_std)
        delta_rot = R.from_euler("xyz", euler_delta).as_matrix()
        rotation = rotation @ delta_rot

    return poses


# Registry of all trajectory generators
TRAJECTORY_GENERATORS = {
    "forward": generate_forward_trajectory,
    "circular": generate_circular_trajectory,
    "spiral": generate_spiral_trajectory,
    "pan_left": generate_pan_left_trajectory,
    "pan_right": generate_pan_right_trajectory,
    "tilt_up": generate_tilt_up_trajectory,
    "tilt_down": generate_tilt_down_trajectory,
    "dolly": generate_dolly_zoom_trajectory,
    "crane_up": generate_crane_up_trajectory,
    "crane_down": generate_crane_down_trajectory,
    "truck_left": generate_truck_left_trajectory,
    "truck_right": generate_truck_right_trajectory,
    "arc": generate_arc_trajectory,
    "zigzag": generate_zigzag_trajectory,
    "random_walk": generate_random_walk_trajectory,
}


def generate_all_trajectories(output_dir, num_frames=33):
    """
    Generate all trajectory types and save to output directory.

    Args:
        output_dir: directory to save trajectory JSON files
        num_frames: number of frames per trajectory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(TRAJECTORY_GENERATORS)} trajectory types...")

    for name, generator in TRAJECTORY_GENERATORS.items():
        poses = generator(num_frames=num_frames)
        output_path = output_dir / f"{name}.json"

        with open(output_path, "w") as f:
            json.dump(poses, f, indent=4)

        print(f"  Generated: {output_path.name} ({len(poses)} frames)")

    print(f"\nAll trajectories saved to {output_dir}")


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "trajectories"
    generate_all_trajectories(output_dir, num_frames=33)
