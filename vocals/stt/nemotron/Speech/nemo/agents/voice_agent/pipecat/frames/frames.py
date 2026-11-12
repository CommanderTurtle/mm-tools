# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.


from dataclasses import dataclass
import numpy as np
from pipecat.frames.frames import DataFrame


@dataclass
class DiarResultFrame(DataFrame):
    """Diarization frame."""

    diar_result: np.ndarray | int
    stream_id: str = "default"
