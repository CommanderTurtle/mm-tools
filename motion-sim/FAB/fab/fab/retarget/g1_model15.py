"""Unitree G1 29-DoF, Model 15 (mode_machine 15): per joint (URDF / MuJoCo serial order) the range (rad), the default
standing pose (rad) and the velocity limit (rad/s).

Ranges and velocity limits are those of Unitree's g1_29dof_mode_15.urdf (unitree_ros, robots/g1_description; BSD-3-Clause,
Copyright Unitree Robotics); they equal the joint limits of the G1 URDF that scripts/download_g1.sh installs. The default
pose is the G1's standard standing pose (hips -0.312, knees 0.669, ankles -0.363; arms slightly bent).
"""

# (name, lower, upper, default, velocity_limit)
JOINTS = [
    ("left_hip_pitch_joint", -2.5307, 2.8798, -0.312, 20.0),
    ("left_hip_roll_joint", -0.5236, 2.9671, 0.0, 20.0),
    ("left_hip_yaw_joint", -2.7576, 2.7576, 0.0, 32.0),
    ("left_knee_joint", -0.087267, 2.8798, 0.669, 20.0),
    ("left_ankle_pitch_joint", -0.87267, 0.5236, -0.363, 30.0),
    ("left_ankle_roll_joint", -0.2618, 0.2618, 0.0, 30.0),
    ("right_hip_pitch_joint", -2.5307, 2.8798, -0.312, 20.0),
    ("right_hip_roll_joint", -2.9671, 0.5236, 0.0, 20.0),
    ("right_hip_yaw_joint", -2.7576, 2.7576, 0.0, 32.0),
    ("right_knee_joint", -0.087267, 2.8798, 0.669, 20.0),
    ("right_ankle_pitch_joint", -0.87267, 0.5236, -0.363, 30.0),
    ("right_ankle_roll_joint", -0.2618, 0.2618, 0.0, 30.0),
    ("waist_yaw_joint", -2.618, 2.618, 0.0, 32.0),
    ("waist_roll_joint", -0.52, 0.52, 0.0, 30.0),
    ("waist_pitch_joint", -0.52, 0.52, 0.0, 30.0),
    ("left_shoulder_pitch_joint", -3.0892, 2.6704, 0.2, 37.0),
    ("left_shoulder_roll_joint", -1.5882, 2.2515, 0.2, 37.0),
    ("left_shoulder_yaw_joint", -2.618, 2.618, 0.0, 37.0),
    ("left_elbow_joint", -1.0472, 2.0944, 0.6, 37.0),
    ("left_wrist_roll_joint", -1.972222054, 1.972222054, 0.0, 37.0),
    ("left_wrist_pitch_joint", -1.61443, 1.61443, 0.0, 27.0),
    ("left_wrist_yaw_joint", -1.61443, 1.61443, 0.0, 27.0),
    ("right_shoulder_pitch_joint", -3.0892, 2.6704, 0.2, 37.0),
    ("right_shoulder_roll_joint", -2.2515, 1.5882, -0.2, 37.0),
    ("right_shoulder_yaw_joint", -2.618, 2.618, 0.0, 37.0),
    ("right_elbow_joint", -1.0472, 2.0944, 0.6, 37.0),
    ("right_wrist_roll_joint", -1.972222054, 1.972222054, 0.0, 37.0),
    ("right_wrist_pitch_joint", -1.61443, 1.61443, 0.0, 27.0),
    ("right_wrist_yaw_joint", -1.61443, 1.61443, 0.0, 27.0),
]
NAMES = [j[0] for j in JOINTS]
RANGES = {j[0]: (j[1], j[2]) for j in JOINTS}
DEFAULT = {j[0]: j[3] for j in JOINTS}
VELOCITY = {j[0]: j[4] for j in JOINTS}
assert len(JOINTS) == 29
