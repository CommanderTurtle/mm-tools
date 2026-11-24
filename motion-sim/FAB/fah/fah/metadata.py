# Hand keypoint conventions.
#
# The model predicts the 70 MHR keypoints (see sam_3d_body/metadata/mhr70.py); in
# hand mode only the right-hand block (indices 21..41) is non-zero. Per finger the
# MHR order is [tip, first, second, third joint] (tip -> knuckle), wrist last.
# We re-order to the common 21-joint OpenPose / FreiHAND / MANO-21 layout:
#   0 wrist, 1-4 thumb (CMC/MCP -> tip), 5-8 index (MCP -> tip), 9-12 middle,
#   13-16 ring, 17-20 pinky.

MHR70_RIGHT_HAND_START = 21

MHR_HAND21_TO_OPENPOSE = [
    20,
    3, 2, 1, 0,
    7, 6, 5, 4,
    11, 10, 9, 8,
    15, 14, 13, 12,
    19, 18, 17, 16,
]  # fmt: skip

MHR70_TO_OPENPOSE_HAND = [MHR70_RIGHT_HAND_START + i for i in MHR_HAND21_TO_OPENPOSE]

HAND_KEYPOINT_NAMES = [
    "wrist",
    "thumb1", "thumb2", "thumb3", "thumb4",
    "index1", "index2", "index3", "index4",
    "middle1", "middle2", "middle3", "middle4",
    "ring1", "ring2", "ring3", "ring4",
    "pinky1", "pinky2", "pinky3", "pinky4",
]  # fmt: skip

HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]  # fmt: skip

# BGR colors per finger (thumb, index, middle, ring, pinky)
FINGER_COLORS = [
    (0, 0, 255),
    (0, 200, 255),
    (0, 255, 0),
    (255, 128, 0),
    (255, 0, 255),
]

# MHR skeleton joints (127-joint rig)
MHR_RIGHT_WRIST_TWIST_JOINT = 41
MHR_RIGHT_WRIST_JOINT = 42
