#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Training code for every FastFrames stage (see docs/training.md).

Each stage module exposes ``train(config)`` and is launched with
``torchrun ... scripts/train.py --config configs/<stage>.yaml``.
"""
