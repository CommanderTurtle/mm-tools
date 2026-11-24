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
"""Named model registry: release names, Hub locations and checksums."""

import hashlib
import os

MODELS = {
    "voice-isolator-v1": {
        "repo_id":
            "inworld/voice-isolator",
        "filename":
            "denoiser.safetensors",
        "sha256":
            "647335a6aeb4dc706b895532c209d51e0e84a4849d17b339f00d4392a050ae53",
        "revision":
            "9b0285b3fe8403aa8e2f1dbb2165a80a4294c970",
        "stage":
            1,
        "description":
            ("Voice Isolator v1 for recorded audio, 48 kHz: offline only,"
             " keeps the foreground voice, 8 s chunks overlapping by 2 s."),
    },
    "voice-isolator-v1-realtime": {
        "repo_id":
            "inworld/voice-isolator",
        "filename":
            "denoiser_realtime.safetensors",
        "sha256":
            "76598f05523c0669e76562a8b2f31c9d395fca316e80c907ee11775e7497ae0f",
        "revision":
            "9b0285b3fe8403aa8e2f1dbb2165a80a4294c970",
        "stage":
            1,
        "description":
            ("Voice Isolator v1 for live audio, 48 kHz: streaming only, keeps"
             " the foreground voice, look-ahead 0, 20 or 40 ms."),
    },
}

# the weights the published v0 board rows were rendered from (their render
# stamps record these checksums), not part of the registry: the benchmark's
# digest-bound helper (bench/scripts/common.py) passes voice-isolator-v0 as its
# default, so these names stay on exactly those files until the next full
# re-score. v1 ships the same files today, but these are not aliases of v1 and
# never take its checksums: a later v1 is another file and fails theirs instead
# of standing in for the v0 rows.
PINNED = {
    "voice-isolator-v0": {
        "repo_id":
            "inworld/voice-isolator",
        "filename":
            "denoiser.safetensors",
        "sha256":
            "647335a6aeb4dc706b895532c209d51e0e84a4849d17b339f00d4392a050ae53",
        "revision":
            "9b0285b3fe8403aa8e2f1dbb2165a80a4294c970",
        "stage":
            1,
        "description":
            "The weights of the published v0 offline board row.",
    },
    "voice-isolator-v0-realtime": {
        "repo_id":
            "inworld/voice-isolator",
        "filename":
            "denoiser_realtime.safetensors",
        "sha256":
            "76598f05523c0669e76562a8b2f31c9d395fca316e80c907ee11775e7497ae0f",
        "revision":
            "9b0285b3fe8403aa8e2f1dbb2165a80a4294c970",
        "stage":
            1,
        "description":
            "The weights of the published v0 realtime board row.",
    },
}

HF_SCHEME = "hf://"
# the Hub repository of the released models names them all, and the inference
# mode picks its checkpoint
REPO_ID = "inworld/voice-isolator"
MODE_MODELS = {
    "offline": "voice-isolator-v1",
    "streaming": "voice-isolator-v1-realtime"
}


def parse_hf(spec: str) -> tuple[str, str, str | None]:
    """Splits ``hf://<org>/<repo>/<file>[@<revision>]`` into its parts.

    Returns:
        (repo_id, filename, revision or None).
    """
    body = spec[len(HF_SCHEME):]
    body, _, revision = body.partition("@")
    parts = body.split("/")
    if len(parts) < 3 or not all(parts):
        raise ValueError(
            f"'{spec}' must look like hf://<org>/<repo>/<file>[@<revision>].")
    return "/".join(parts[:2]), "/".join(parts[2:]), revision or None


def sha256_file(path: str) -> str:
    """Returns the sha256 of a file, read in 16 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(name_or_path: str, mode: str = "offline") -> str:
    """Returns a local file for a released model, an ``hf://`` spec or a path.

    The repository id, ``inworld/voice-isolator``, names the released models
    and ``mode`` picks its checkpoint: ``voice-isolator-v1`` offline and
    ``voice-isolator-v1-realtime`` for streaming. A registry name picks one
    directly, whatever the mode. Both come first: they download the pinned
    file from the Hub and verify its sha256; a name whose file is not released
    yet is refused, and a Hub failure says to load the file by its path.
    ``hf://<org>/<repo>/<file>[@<revision>]`` downloads any Hub file (no
    checksum: pin a revision instead). Anything else that exists on disk is
    returned as is; a local file that shares a registry name is loaded by its
    path (``./voice-isolator-v1``).
    """
    if name_or_path == REPO_ID:
        if mode not in MODE_MODELS:
            raise ValueError(
                f"mode must be one of {tuple(MODE_MODELS)}, got {mode!r}.")
        name_or_path = MODE_MODELS[mode]
    entry = MODELS.get(name_or_path) or PINNED.get(name_or_path)
    if entry is not None:
        if not entry["sha256"]:
            raise ValueError(
                f"'{name_or_path}' is not released yet; load a local checkpoint"
                " by its path instead.")
        # pylint: disable-next=import-outside-toplevel
        import huggingface_hub

        try:
            path = huggingface_hub.hf_hub_download(
                repo_id=entry["repo_id"],
                filename=entry["filename"],
                revision=entry["revision"],
            )
        except OSError as error:
            raise RuntimeError(
                f"'{name_or_path}' could not be fetched from the Hub "
                f"({entry['repo_id']}/{entry['filename']}): the weights are "
                "not published there yet, or this machine has no access or no "
                "network. Load the checkpoint by its path instead.") from error
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise RuntimeError(
                f"Checksum mismatch for {name_or_path}: expected "
                f"{entry['sha256']}, got {digest}.")
        return path
    if name_or_path.startswith(HF_SCHEME):
        # pylint: disable-next=import-outside-toplevel
        import huggingface_hub

        repo_id, filename, revision = parse_hf(name_or_path)
        return huggingface_hub.hf_hub_download(repo_id=repo_id,
                                               filename=filename,
                                               revision=revision)
    if os.path.exists(name_or_path):
        return name_or_path
    raise ValueError(
        f"'{name_or_path}' is neither a local file, an hf:// spec, nor a known "
        f"model ({REPO_ID}, {', '.join(sorted(MODELS))}).")
