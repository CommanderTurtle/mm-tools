# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import librosa
import torch


def read_audio(audio_file: str, target_sr: int, mono: bool = True) -> torch.Tensor:
    """
    Read audio file and return samples with target sampling rate
    Args:
        audio_file: (str) audio file path
        target_sr: (int) target sampling rate
        mono: (bool) whether to convert to mono
    Returns:
        (torch.Tensor) audio samples
    """
    return torch.tensor(librosa.load(audio_file, sr=target_sr, mono=mono)[0]).float()
