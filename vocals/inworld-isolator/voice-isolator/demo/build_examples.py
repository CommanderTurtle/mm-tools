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
"""Rebuilds the two generated demo examples from their checked-in sources.

    python demo/build_examples.py [--out DIRECTORY]

Each one is a LibriSpeech utterance mixed with noise generated here. The other
examples are finished recordings and are not touched.
"""

import argparse
import hashlib
import pathlib

import numpy as np
import soundfile as sf
import torch

from inworld import voice_isolation

EXAMPLES = pathlib.Path(__file__).resolve().parent / "examples"
SAMPLE_RATE = 48000
RECIPES = (
    {
        "file":
            "sample_14.wav",
        "source":
            "sources/2277-149896-0032.flac",
        "source_sha256":
            "56b640a079f27f9473478a20f068b8bd8af3afa7030d7a07cbbc7b3f19211689",
        "noise":
            "hiss",
        "seed":
            7,
        "snr_db":
            6,
    },
    {
        "file":
            "sample_15.wav",
        "source":
            "sources/2035-147960-0000.flac",
        "source_sha256":
            "0ee6731694bcc55349cf4f70123aaff81a7779c782c4a3ec4579b243b1552748",
        "noise":
            "hum_taps",
        "seed":
            11,
        "snr_db":
            9,
    },
)


def _noise(kind: str, size: int, rate: int, seed: int) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    if kind == "hiss":
        return rng.standard_normal(size)
    if kind != "hum_taps":
        raise ValueError(f"Unknown generated noise: {kind}")
    time = np.arange(size) / rate
    noise = sum(
        np.sin(2 * np.pi * frequency * time) / harmonic
        for harmonic, frequency in enumerate((50, 100, 150), 1))
    for start in (2.0, 4.5, 7.0):
        begin = int(start * rate)
        length = min(int(0.2 * rate), size - begin)
        if length > 0:
            noise[begin:begin + length] += (8 * rng.standard_normal(length) *
                                            np.exp(-np.arange(length) /
                                                   (0.025 * rate)))
    return noise


def build_examples(destination: pathlib.Path) -> None:
    """Writes the generated examples, checking their sources first.

    Args:
        destination: Directory receiving the WAVs.

    Raises:
        ValueError: A source recording differs from its recorded checksum.
    """
    for recipe in RECIPES:
        source = EXAMPLES / recipe["source"]
        if hashlib.sha256(
                source.read_bytes()).hexdigest() != recipe["source_sha256"]:
            raise ValueError(f"Source SHA-256 mismatch: {recipe['source']}")
    destination.mkdir(parents=True, exist_ok=True)
    for recipe in RECIPES:
        speech, rate = sf.read(EXAMPLES / recipe["source"], dtype="float32")
        speech = voice_isolation.resample(torch.from_numpy(speech), rate,
                                          SAMPLE_RATE).numpy()
        speech = speech.astype(np.float64)
        speech *= 0.08 / np.sqrt(np.mean(speech**2))
        noise = _noise(recipe["noise"], len(speech), SAMPLE_RATE,
                       recipe["seed"])
        noise *= 0.08 / (10**(recipe["snr_db"] / 20) *
                         np.sqrt(np.mean(noise**2)))
        mixture = speech + noise
        mixture *= min(1.0, 0.95 / np.max(np.abs(mixture)))
        sf.write(destination / recipe["file"],
                 mixture,
                 SAMPLE_RATE,
                 subtype="PCM_16")


def main() -> None:
    """Rebuilds the generated examples at the requested output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=EXAMPLES)
    args = parser.parse_args()
    build_examples(args.out)


if __name__ == "__main__":
    main()
