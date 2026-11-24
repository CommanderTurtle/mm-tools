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
"""`voice-isolate-doctor`: whether this machine is ready to isolate voices.

The default level reads only this machine: versions, the device inference
would use and the weights already in the Hugging Face cache. ``--full`` also
fetches both released models (or the given checkpoints) and runs half a second
of silence through each. Every line carries a level, OK, INFO, WARN or FAIL,
and the report ends with one summary line; the exit status is 1 when any line
failed. ``--json`` prints the same report as one JSON object.

Running this module imports the package, so it cannot diagnose an install that
failed; it reports the versions an install produced.
"""

import argparse
import dataclasses
import importlib.util
import json
import platform
import sys

import numpy as np
import torch
import torchaudio

from inworld import voice_isolation
from inworld.voice_isolation import isolator
from inworld.voice_isolation import registry

OK, INFO, WARN, FAIL = "OK", "INFO", "WARN", "FAIL"
# what the realtime model is run with: the README's 20 ms of look-ahead
STREAM_LOOKAHEAD_FRAMES = 2
# short windows make half a second of audio cross several crossfades, so the
# stream's two-lane path runs too
STREAM_WINDOW_FRAMES = 8
STREAM_CROSSFADE_FRAMES = 4


@dataclasses.dataclass
class Check:
    """One line of the report."""
    level: str
    name: str
    message: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the doctor's options."""
    p = argparse.ArgumentParser(
        prog="voice-isolate-doctor",
        description=("Checks whether this machine is ready to run the Inworld"
                     " Voice Isolator."),
    )
    p.add_argument(
        "--full",
        action="store_true",
        help=("Also fetch both models (downloading them on first use) and run"
              " half a second of silence through each."),
    )
    p.add_argument(
        "--offline-model",
        default=registry.REPO_ID,
        help=("With --full: the offline checkpoint, as a registry name, an"
              " hf:// location or a local file (default: the released"
              " voice-isolator-v1)."),
    )
    p.add_argument(
        "--online-model",
        default=registry.REPO_ID,
        help=("With --full: the streaming checkpoint, as a registry name, an"
              " hf:// location or a local file (default: the released"
              " voice-isolator-v1-realtime)."),
    )
    p.add_argument(
        "--device",
        default=None,
        help="cpu, mps or cuda[:N] (default: the one inference would pick).",
    )
    p.add_argument("--json",
                   action="store_true",
                   help="Print the report as one JSON object.")
    return p.parse_args(argv)


def _environment() -> list[Check]:
    return [
        Check(INFO, "package",
              f"inworld-voice-isolation {voice_isolation.__version__}"),
        Check(INFO, "python",
              f"Python {platform.python_version()} on {platform.platform()}"),
        Check(
            INFO, "torch",
            f"torch {torch.__version__}, torchaudio {torchaudio.__version__}"),
    ]


def _device_check(device: torch.device, chosen: bool) -> Check:
    """Reports the device; warns on an Apple-silicon Mac left on the CPU."""
    old_torch = torch.torch_version.TorchVersion(
        torch.__version__) < isolator.MPS_MIN_TORCH
    if (not chosen and device.type == "cpu" and sys.platform == "darwin" and
            platform.machine() == "arm64" and old_torch):
        return Check(
            WARN, "device",
            f"cpu: upgrade torch to {isolator.MPS_MIN_TORCH} or later to use"
            " the Metal GPU")
    return Check(INFO, "device", f"inference runs on {device}")


def _kernels_check() -> Check:
    if importlib.util.find_spec("mamba_ssm") is not None:
        return Check(INFO, "kernels", "optional CUDA kernels installed")
    return Check(INFO, "kernels",
                 "optional CUDA kernels not installed (CUDA only, not needed)")


def _cached_path(entry: dict) -> str | None:
    """The cached file of a registry entry, or None when it is not cached."""
    # pylint: disable-next=import-outside-toplevel
    import huggingface_hub

    found = huggingface_hub.try_to_load_from_cache(entry["repo_id"],
                                                   entry["filename"],
                                                   revision=entry["revision"])
    return found if isinstance(found, str) else None


def _cache_checks() -> tuple[list[Check], bool]:
    """Checks each released model's cached file; also says whether all are."""
    checks, all_cached = [], True
    for name, entry in registry.MODELS.items():
        if not entry["sha256"]:
            checks.append(Check(INFO, name, "not released yet"))
            continue
        path = _cached_path(entry)
        if path is None:
            all_cached = False
            checks.append(
                Check(INFO, name, "not downloaded yet; run with --full"))
        elif registry.sha256_file(path) != entry["sha256"]:
            checks.append(
                Check(FAIL, name,
                      f"checksum mismatch: delete {path} and run again"))
        else:
            checks.append(Check(OK, name, f"cached at {path}"))
    return checks, all_cached


def _hub_failure(name: str, spec: str, error: Exception) -> str:
    """Says why ``registry.resolve`` could not fetch a model from the Hub.

    For a released model ``registry.resolve`` wraps the Hub's error in a
    RuntimeError; for an ``hf://`` location the Hub's own error arrives as it
    was raised. Both are told apart here by the Hub error underneath.
    """
    # pylint: disable-next=import-outside-toplevel
    from huggingface_hub import errors as hub_errors

    if str(error).startswith("Checksum mismatch"):
        entry = registry.MODELS.get(name) or {}
        path = _cached_path(entry) if entry else None
        return f"checksum mismatch: delete {path or 'the cached file'} and run again"
    cause = error.__cause__ if isinstance(error, RuntimeError) else error
    if isinstance(cause, hub_errors.HfHubHTTPError):
        if spec.startswith(registry.HF_SCHEME):
            return ("not reachable: check the location and your access to it,"
                    " or pass a local .safetensors file")
        return (f"{registry.REPO_ID} is not reachable: pass --offline-model and"
                " --online-model with local .safetensors files")
    # what the Hub client raises when it has no connection and no cached copy
    if isinstance(cause, hub_errors.LocalEntryNotFoundError):
        return ("cannot reach huggingface.co: check the network, or pass a"
                " local file")
    return f"could not fetch {spec}: {cause or error}"


def _run_model(path: str, mode: str, device: torch.device) -> None:
    """Loads a checkpoint and runs half a second of silence through it."""
    silence = np.zeros(voice_isolation.SR // 2, dtype=np.float32)
    if mode == "offline":
        model = isolator.VoiceIsolator.load(path, device=device, mode=mode)
        model.isolate(silence, voice_isolation.SR)
        return
    model = isolator.VoiceIsolator.load(
        path,
        device=device,
        mode=mode,
        lookahead_frames=STREAM_LOOKAHEAD_FRAMES)
    session = model.stream(window_frames=STREAM_WINDOW_FRAMES,
                           crossfade_frames=STREAM_CROSSFADE_FRAMES)
    session.push(silence)
    session.flush()


def _model_check(spec: str, mode: str, device: torch.device) -> Check:
    """Fetches one model and runs it; reports it under its release name."""
    name = registry.MODE_MODELS[mode] if spec == registry.REPO_ID else spec
    try:
        path = registry.resolve(spec, mode)
    # RuntimeError: a released model; OSError: the Hub's own error, raised
    # as it is for an hf:// location
    except (RuntimeError, OSError) as error:
        return Check(FAIL, name, _hub_failure(name, spec, error))
    except ValueError as error:
        return Check(FAIL, name, str(error))
    try:
        _run_model(path, mode, device)
    # any failure to load or run is reported as this model's FAIL line
    except Exception as error:  # pylint: disable=broad-exception-caught
        hint = "" if device.type == "cpu" else ": try --device cpu"
        return Check(FAIL, name, f"failed on {device}: {error}{hint}")
    return Check(OK, name, f"loads and runs on {device}")


def _summary(checks: list[Check], full: bool, all_cached: bool,
             names: tuple[str, str], device: torch.device) -> str:
    failed = sum(check.level == FAIL for check in checks)
    if failed:
        noun = "problem" if failed == 1 else "problems"
        return f"{failed} {noun}: see the FAIL lines above"
    if full:
        return (f"All set: {names[0]} and {names[1]} on {device}"
                f" (torch {torch.__version__})")
    if not all_cached:
        return "Ready to install weights: run voice-isolate-doctor --full"
    return "Ready: run voice-isolate-doctor --full to run both models"


def diagnose(args: argparse.Namespace) -> tuple[list[Check], str]:
    """Runs the checks the options ask for; returns them and the summary."""
    device = (torch.device(args.device)
              if args.device else isolator.default_device())
    checks = _environment()
    checks.append(_device_check(device, chosen=args.device is not None))
    checks.append(_kernels_check())
    cache, all_cached = _cache_checks()
    checks.extend(cache)
    names = []
    for spec, mode in ((args.offline_model, "offline"), (args.online_model,
                                                         "streaming")):
        names.append(registry.MODE_MODELS[mode] if spec ==
                     registry.REPO_ID else spec)
        if args.full:
            checks.append(_model_check(spec, mode, device))
    return checks, _summary(checks, args.full, all_cached, tuple(names), device)


def main(argv: list[str] | None = None) -> int:
    """Prints the report and returns 1 when any check failed, else 0."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    checks, summary = diagnose(args)
    failed = any(check.level == FAIL for check in checks)
    if args.json:
        print(
            json.dumps(
                {
                    "checks": [dataclasses.asdict(c) for c in checks],
                    "summary": summary,
                    "ok": not failed,
                },
                indent=2))
    else:
        for check in checks:
            print(f"{check.level:<4}  {check.name}: {check.message}")
        print(summary)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
