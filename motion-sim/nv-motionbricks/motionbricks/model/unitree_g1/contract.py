# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The Unitree G1 model's contract, ONNX paths, and foot contact points.

The contract comes from the model's manifest and describes a batch size of one.
"""

from pathlib import Path

from ..contract import load_model_manifest


UNITREE_G1_N1_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2] / "assets/onnx/unitree_g1/unitree_g1_n1.json"
)
UNITREE_G1_N1_CONTRACT, UNITREE_G1_N1_FP32_ONNX_PATH, UNITREE_G1_N1_FP16_ONNX_PATH = load_model_manifest(
    UNITREE_G1_N1_MANIFEST_PATH,
)
UNITREE_G1_N1_FOOT_CONTACT_BODIES = ("left_ankle_roll_link",) * 2 + ("right_ankle_roll_link",) * 2
UNITREE_G1_N1_FOOT_CONTACT_OFFSETS = ((0.0, 0.0, 0.0), (0.14, 0.0, -0.035)) * 2
