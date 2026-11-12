# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#

# pyre-unsafe

from .shapenet_core import ShapeNetCore


__all__ = [k for k in globals().keys() if not k.startswith("_")]
