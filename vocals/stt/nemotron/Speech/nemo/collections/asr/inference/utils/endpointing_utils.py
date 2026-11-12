# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.


def millisecond_to_frames(millisecond: int, ms_per_timestep: int) -> int:
    """
    Convert milliseconds to frames
    Args:
        millisecond (int): milliseconds to convert
        ms_per_timestep (int): milliseconds per timestep
    Returns:
        int: number of frames
    """
    return (millisecond + ms_per_timestep - 1) // ms_per_timestep


def get_custom_stop_history_eou(
    stop_history_eou: int | None, default_stop_history_eou: int, ms_per_timestep: int
) -> int:
    """
    Get the custom stop history of EOU
    Args:
        stop_history_eou (int): stop history of EOU
        default_stop_history_eou (int): default stop history of EOU
        ms_per_timestep (int): milliseconds per timestep
    Returns:
        int: custom stop history of EOU
    """
    if stop_history_eou is None:
        return default_stop_history_eou
    if stop_history_eou > 0:
        return millisecond_to_frames(stop_history_eou, ms_per_timestep)
    return 0 if stop_history_eou == 0 else -1
