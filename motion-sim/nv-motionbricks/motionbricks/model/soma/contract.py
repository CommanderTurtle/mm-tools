# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The SOMA model's contract, ONNX paths, foot contact joints, and heading joints.

The contract comes from the model's manifest and describes a batch size of one.
"""

from pathlib import Path

from ..contract import load_model_manifest

SOMA_N1_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "assets/onnx/soma/soma_n1.json"
SOMA_N1_CONTRACT, SOMA_N1_FP32_ONNX_PATH, SOMA_N1_FP16_ONNX_PATH = load_model_manifest(SOMA_N1_MANIFEST_PATH)
SOMA_N1_FOOT_CONTACT_JOINTS = ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")
# The right and left leg joints whose positions give a pose's heading, as in training.
SOMA_N1_HEADING_JOINTS = ("RightLeg", "LeftLeg")
