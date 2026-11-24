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
"""Lists the demo's example audio: the audio files of one folder."""

import dataclasses
import pathlib
import re

import numpy as np
import soundfile as sf

DEFAULT_DIRECTORY = pathlib.Path(__file__).resolve().parent / "examples"
AUDIO_FORMATS = {
    ".wav": frozenset(("WAV", "WAVEX", "RF64")),
    ".flac": frozenset(("FLAC",)),
    ".mp3": frozenset(("MP3",)),
    ".ogg": frozenset(("OGG",)),
}


@dataclasses.dataclass(frozen=True)
class Examples:
    """The audio files and the label each one is shown with."""

    rows: list[list[str]]
    labels: list[str]


def _validate_audio(path):
    try:
        with sf.SoundFile(path) as audio:
            if audio.format not in AUDIO_FORMATS[path.suffix.lower()]:
                raise ValueError(
                    "Example audio format does not match its filename: "
                    f"{path.name}")
            if audio.frames < 1 or audio.channels < 1 or audio.samplerate < 1:
                raise ValueError(
                    f"Example audio is empty or invalid: {path.name}")
            frames = 0
            for block in audio.blocks(blocksize=65536, dtype="float32"):
                if not np.isfinite(block).all():
                    raise ValueError(
                        "Example audio contains non-finite samples: "
                        f"{path.name}")
                frames += len(block)
            if frames == 0:
                raise ValueError(
                    f"Example audio has no decodable frames: {path.name}")
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Cannot decode example audio: {path.name}") from error


def _name_order(path):
    return [(0, int(part), "") if part.isdigit() else (1, 0, part)
            for part in re.split(r"(\d+)", path.name)]


def load_examples(directory: str | pathlib.Path) -> Examples:
    """Returns the audio files of a folder and their labels, in name order.

    Numbers in names sort by value (``sample_2`` before ``sample_10``). A label
    is the file name without its extension. Other files and subfolders are
    left out.

    Args:
        directory: The folder of example audio.

    Returns:
        Gradio rows and display labels. Only these individual files may be
        passed to Gradio's allowed_paths, never the directory.

    Raises:
        ValueError: The folder is missing or holds no audio, or an audio file
            is a link or is not the audio its name says.
    """
    root = pathlib.Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f"No examples folder at {root}.")
    examples, labels = [], []
    for path in sorted(root.iterdir(), key=_name_order):
        if path.suffix.lower() not in AUDIO_FORMATS:
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"Example must be a regular, non-symlink file: {path.name}")
        _validate_audio(path)
        examples.append([str(path)])
        labels.append(path.stem)
    if not examples:
        raise ValueError(f"No example audio in {root}.")
    return Examples(examples, labels)
