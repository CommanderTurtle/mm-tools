# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Discover subpackages under the installed name while keeping the source layout."""

from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        packages=[
            "motionbricks_onnx",
            *(
                f"motionbricks_onnx.{name}"
                for name in find_packages(exclude=["tests", "tests.*"])
            ),
        ]
    )
