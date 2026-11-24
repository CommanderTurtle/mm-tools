# Copyright 2026 Inworld AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Audio I/O and high-quality resampling shared by the package."""

import os
import secrets

import numpy as np
import soundfile as sf
import torch
import torchaudio

SR = 48000

KAISER_BEST = {
    "resampling_method": "sinc_interp_kaiser",
    "lowpass_filter_width": 64,
    "rolloff": 0.9475937167399596,
    "beta": 14.769656459379492,
}


def resample(wav: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    """kaiser_best resampling; identity when the rates match."""
    if orig_freq == new_freq:
        return wav
    return torchaudio.functional.resample(wav, orig_freq, new_freq,
                                          **KAISER_BEST)


def fit_length(wav: torch.Tensor, n: int) -> torch.Tensor:
    """Trims the last axis to ``n`` samples or pads it with trailing zeros."""
    if wav.shape[-1] >= n:
        return wav[..., :n]
    return torch.nn.functional.pad(wav, (0, n - wav.shape[-1]))


def to_mono(audio) -> torch.Tensor:
    """Converts [T], [C, T] or [T, C] audio to a float32 tensor [T] in [-1, 1].

    Integer PCM (int16/int32/uint8) is scaled by its dtype range; floats pass
    through. Up to eight channels are averaged.

    Raises:
        ValueError: The dtype has no PCM scale (int64), or both dimensions
            exceed eight, so the channel axis cannot be told from the time axis.
    """
    x = torch.as_tensor(
        np.asarray(audio) if not isinstance(audio, torch.Tensor) else audio)
    if x.dtype == torch.int64:
        raise ValueError(
            "int64 samples have no PCM scale: pass float, int16, int32 or"
            " uint8.")
    if not torch.is_floating_point(x):
        info = torch.iinfo(x.dtype)
        x = x.float()
        if info.min < 0:
            x = x / -float(info.min)
        else:
            x = (x - (info.max + 1) / 2) / ((info.max + 1) / 2)
    x = x.detach().float()
    if x.ndim == 2:
        if min(x.shape) > 8:
            raise ValueError(
                f"Expected at most eight channels, got shape {tuple(x.shape)}.")
        x = (x.mean(0)
             if x.shape[0] <= 8 and x.shape[0] < x.shape[1] else x.mean(1))
    elif x.ndim != 1:
        raise ValueError(
            f"Expected 1-D or 2-D audio, got shape {tuple(x.shape)}.")
    return x


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Reads a soundfile-readable file as (mono float32 tensor [T], rate)."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(x.mean(1)), int(sr)


def create_staging_file(destination: str, prefix: str = ".partial-") -> str:
    """Creates an empty hidden file beside ``destination`` to write it under.

    The name keeps the destination's extension, from which soundfile takes the
    format. The file has the permissions of any new file (0o666 less the
    umask, applied by the system as it creates the file), which renaming it
    into place keeps; ``tempfile.mkstemp`` would give 0o600.

    Raises:
        FileExistsError: Every random name tried was taken.
    """
    folder = os.path.dirname(os.path.abspath(destination))
    suffix = os.path.splitext(destination)[1]
    for _ in range(100):
        path = os.path.join(folder, f"{prefix}{secrets.token_hex(4)}{suffix}")
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
        except FileExistsError:
            continue
        return path
    raise FileExistsError(f"No free staging name beside {destination}")


def save_audio(path: str,
               wav: np.ndarray | torch.Tensor,
               sample_rate: int,
               subtype: str = "PCM_16") -> None:
    """Writes a mono waveform.

    Integer subtypes cannot hold samples beyond full scale, so a waveform
    peaking above 1.0 is scaled down to peak at 1.0 instead of being clipped
    (isolated speech overshoots the input peak on roughly a quarter of loud
    inputs). Float subtypes are written unchanged. The file is written under a
    temporary name and renamed into place, so an interrupted write never leaves
    a partial file that a resumed run would take as done; it gets the
    permissions of any new file.
    """
    y = torch.as_tensor(wav).detach().float().cpu()
    if subtype.startswith("PCM") and y.numel():
        peak = float(y.abs().max())
        if peak > 1.0:
            y = y / peak
    tmp = create_staging_file(path)
    try:
        sf.write(tmp, y.numpy(), sample_rate, subtype=subtype)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
