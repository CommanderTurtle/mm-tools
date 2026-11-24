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
"""`voice-isolate`: isolate one file or a whole directory.

A file input always writes its output (an existing output folder receives
``<stem>.wav``). Offline also writes ``<output-stem>_background.wav`` as the
input minus the isolated voice. Both are float WAVs at the same sample rate
and level. Directory inputs resume only when every requested output exists.
"""

import argparse
import os
import stat
import string
import sys
import tempfile
import time
import unicodedata

import torch

from inworld.voice_isolation import audio as audio_lib
from inworld.voice_isolation import isolator
from inworld.voice_isolation import registry

_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".opus")
# the probe for a case-insensitive folder swaps ASCII letters only, which every
# such file system folds alike
_SWAP_CASE = str.maketrans(string.ascii_lowercase + string.ascii_uppercase,
                           string.ascii_uppercase + string.ascii_lowercase)
# on macOS the CPU path runs about 4x faster on two threads than on PyTorch's
# default (one per core): the network is many small ops, and the threads spend
# their time synchronising rather than computing
_MACOS_CPU_THREADS = 2


def _limit_macos_cpu_threads(device):
    """Caps PyTorch's CPU threads on macOS unless the user chose a count."""
    if (sys.platform == "darwin" and device.type == "cpu" and
            "OMP_NUM_THREADS" not in os.environ and
            torch.get_num_threads() > _MACOS_CPU_THREADS):
        torch.set_num_threads(_MACOS_CPU_THREADS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses the audio paths and model options for the command line."""
    p = argparse.ArgumentParser(
        prog="voice-isolate",
        description="Inworld Voice Isolator: 48 kHz full-band voice isolation.",
    )
    p.add_argument("input", help="Audio file, or a directory of audio files.")
    p.add_argument("output",
                   help="Output file, or directory for a directory input.")
    background = p.add_mutually_exclusive_group()
    background.add_argument(
        "--background",
        help=("Offline, single-file input: background WAV path. The default"
              " is <output-stem>_background.wav. Both outputs are float WAVs"
              " whose sum is the mono input at the output sample rate."),
    )
    background.add_argument(
        "--no-background",
        action="store_true",
        help=
        "Offline: save only the isolated voice. Streaming saves only voice.",
    )
    p.add_argument(
        "--model",
        default=registry.REPO_ID,
        help=("The Hub repository inworld/voice-isolator (the default: --mode"
              " picks its checkpoint), a registry name (voice-isolator-v1, or"
              " voice-isolator-v1-realtime, which streams),"
              " hf://<org>/<repo>/<file>[@<revision>], or a local checkpoint"
              " (.safetensors / .pt)."),
    )
    p.add_argument(
        "--mode",
        default="offline",
        choices=isolator.MODES,
        help=("offline: every frame sees its whole chunk (best quality, the"
              " default; voice-isolator-v1). streaming: causal, as a live"
              " stream runs (voice-isolator-v1-realtime)."),
    )
    p.add_argument(
        "--lookahead",
        type=int,
        default=0,
        help=("Streaming mode: frames of look-ahead (10 ms each for the"
              " released model), one of the checkpoint's trained positions."),
    )
    p.add_argument(
        "--block-frames",
        type=int,
        default=4,
        help="Streaming mode: maximum frames processed together; tunes"
        " throughput without changing the stream's latency.",
    )
    p.add_argument(
        "--window-frames",
        type=int,
        default=None,
        help="Streaming mode: the time attention never looks back across a"
        " window of this many frames (a second lane crossfades in before"
        " each boundary, --crossfade-frames). Default: 600 frames (6 s);"
        " 0 turns the windows off.",
    )
    p.add_argument(
        "--crossfade-frames",
        type=int,
        default=None,
        help="Streaming mode: frames over which the two lanes blend before"
        " a window boundary. Default: 200 frames (2 s).",
    )
    p.add_argument(
        "--output-sr",
        type=int,
        default=48000,
        help="Output sample rate (default 48000; 0 = keep each input's rate).",
    )
    p.add_argument(
        "--device",
        default=None,
        help=("cuda | mps | cpu (default: cuda if available, else mps on an"
              " Apple-silicon Mac with PyTorch 2.6 or later, else cpu)."),
    )
    p.add_argument(
        "--precision",
        default="auto",
        choices=("auto", "fp32", "fp16", "bf16"),
        help=("Network precision: auto runs bf16 on a CUDA GPU, the precision"
              " the model was trained in, and float32 elsewhere (CPU, mps);"
              " the others force one."),
    )
    p.add_argument(
        "--offline-passes",
        type=int,
        default=None,
        help=("Offline mode: passes of the model over its own output (default:"
              " the checkpoint's own count, 1 for the released files); each"
              " pass costs one offline render."),
    )
    p.add_argument(
        "--chunk-seconds",
        type=float,
        default=None,
        help=("Offline mode: seconds per render chunk (default: the"
              " checkpoint's, 8 s for voice-isolator-v1); streaming ignores"
              " it."),
    )
    p.add_argument(
        "--overlap-seconds",
        type=float,
        default=None,
        help=("Offline mode: seconds two chunks overlap and crossfade"
              " (default: the checkpoint's, 2 s)."),
    )
    p.add_argument(
        "--shard",
        type=int,
        default=0,
        help="With a directory input: this worker's shard.",
    )
    p.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="With a directory input: total workers.",
    )
    p.add_argument(
        "--float32",
        action="store_true",
        help=("With voice-only output, write float32 WAV instead of PCM_16."
              " Offline voice/background pairs always use float32 to preserve"
              " their original levels; PCM_16 voice-only output scales down"
              " peaks above full scale."),
    )
    return p.parse_args(argv)


def _outputs(inp,
             out,
             shard,
             num_shards,
             include_background=False,
             background=None):
    """Returns (input, voice, optional background) jobs and whether to resume.

    Refuses outputs that would collide or overwrite an input.
    """
    if os.path.isdir(inp):
        if background:
            sys.exit("--background needs a single input file.")
        if os.path.lexists(out) and not os.path.isdir(out):
            sys.exit(f"A directory input needs an output directory, and {out}"
                     " is not one.")
        if _inside(out, inp):
            sys.exit(
                "The output directory must be outside the input directory.")
        files = _audio_files(inp)
        pairs = [(
            f,
            os.path.join(out,
                         os.path.splitext(os.path.relpath(f, inp))[0] + ".wav"),
        ) for f in files]
        resume = True
    else:
        if os.path.isdir(out):
            out = os.path.join(
                out,
                os.path.splitext(os.path.basename(inp))[0] + ".wav")
        pairs, resume = [(inp, out)], False
    jobs = [(src, dst,
             (background or os.path.splitext(dst)[0] + "_background.wav")
             if include_background else None) for src, dst in pairs]
    if include_background and any(not path.lower().endswith(".wav")
                                  for _, voice, noise in jobs
                                  for path in (voice, noise)):
        sys.exit("Voice/background outputs must be WAV files; use"
                 " --no-background for a voice-only output in another format.")
    _check_destinations(jobs)  # all shards share the same collision check
    if resume:
        try:
            os.makedirs(out, exist_ok=True)
        except OSError as error:
            sys.exit(f"Cannot create the output directory {out}:"
                     f" {error.strerror}.")
    return jobs[shard::num_shards] if resume else jobs, resume


def _audio_files(folder):
    """Lists the audio files below a folder, sorted, skipping hidden names.

    The folder's path is taken literally (brackets, ``*`` and ``?`` in it are
    not patterns); symlinked folders below it are followed.
    """
    found = []
    for root, folders, names in os.walk(folder, followlinks=True):
        folders[:] = [name for name in folders if not name.startswith(".")]
        found += [
            os.path.join(root, name)
            for name in names
            if not name.startswith(".") and name.lower().endswith(_EXTS)
        ]
    return sorted(path for path in found if os.path.isfile(path))


def _inside(path, folder):
    """Whether a path, existing or not, is an existing folder or lies in it.

    The folders on the path are compared by identity, so ``folder`` spelled
    another way counts: through a symlink, or in other case where the file
    system ignores case.
    """
    current = os.path.realpath(path)
    while True:
        if os.path.exists(current) and os.path.samefile(current, folder):
            return True
        parent = os.path.dirname(current)
        if parent == current:
            return False
        current = parent


def _probe_paths(folder):
    """Yields existing paths whose lookups follow a folder's rule on case.

    First the folder's entries; then the folder itself and its ancestors, while
    the parent each is looked up in is on the folder's device (a mount point
    can change the rule).
    """
    try:
        with os.scandir(folder) as entries:
            yield from (entry.path for entry in entries)
    except OSError:
        pass
    path, device = folder, os.stat(folder).st_dev
    parent = os.path.dirname(path)
    while parent != path and os.stat(parent).st_dev == device:
        yield path
        path, parent = parent, os.path.dirname(parent)


def _ignores_case(folder):
    """Whether an existing folder finds names whatever their case.

    Looks the first name with an ASCII letter up again in the other case. A
    folder with no such name to try (an empty volume root) gets a probe file
    of its own; where none can be made, case is taken as ignored, so that
    names differing only in case are refused rather than overwritten.
    """
    for path in _probe_paths(folder):
        parent, name = os.path.split(path)
        swapped = name.translate(_SWAP_CASE)
        if swapped == name:
            continue
        try:
            entry = os.lstat(path)
        except OSError:  # removed since it was listed
            continue
        if not stat.S_ISDIR(entry.st_mode) and entry.st_nlink > 1:
            continue  # a hard link may live under the swapped name too
        try:
            return os.path.samestat(entry,
                                    os.lstat(os.path.join(parent, swapped)))
        except OSError:
            return False
    try:
        handle, probe = tempfile.mkstemp(dir=folder, prefix=".case-probe-")
        os.close(handle)
    except OSError:
        return True
    try:
        parent, name = os.path.split(probe)
        return os.path.lexists(os.path.join(parent, name.translate(_SWAP_CASE)))
    finally:
        os.unlink(probe)


def _path_keys(path, folders):
    """Identifies a path's aliases: symlinks, hard links and other cases.

    A path is named below its deepest existing folder, its names folded when
    that folder ignores case (folders created in it inherit the rule).
    ``folders`` caches each such folder's identity and rule, so each folder is
    probed once.
    """
    folder, names = os.path.realpath(path), ()
    while True:
        folder, name = os.path.split(folder)
        names = (name,) + names
        if folder in folders or os.path.isdir(folder) or not name:
            break
    if folder not in folders:
        info = os.stat(folder)
        folders[folder] = (info.st_dev, info.st_ino), _ignores_case(folder)
    identity, ignores_case = folders[folder]
    if ignores_case:
        # where case is ignored, so is the Unicode form of an accent (APFS,
        # HFS+); case-sensitive APFS ignores the form too, which this misses
        names = tuple(
            unicodedata.normalize("NFC", name).casefold() for name in names)
    keys = {(identity, names)}
    if os.path.exists(path):
        info = os.stat(path)
        keys.add((info.st_dev, info.st_ino))
    return keys


def _check_destinations(jobs):
    """Refuses output collisions and writes over any input, across all jobs.

    Either would silently lose audio (same.wav and same.flac, for example, or
    a.wav and A.flac where the file system ignores case).
    """
    folders = {}
    inputs = {
        key: src for src, _, _ in jobs for key in _path_keys(src, folders)
    }
    seen = {}
    for src, voice, background in jobs:
        for dst in (voice, background):
            if dst is None:
                continue
            if os.path.isdir(dst):
                sys.exit(f"Output path is a directory: {dst}")
            keys = _path_keys(dst, folders)
            sources = keys & inputs.keys()
            if sources:
                original = inputs[next(iter(sources))]
                sys.exit(f"{original} would be overwritten by an output"
                         f" ({dst}): choose another output path.")
            shared = keys & seen.keys()
            if shared:
                previous, other = seen[next(iter(shared))]
                writers = (f"The voice and the background of {src}"
                           if previous == src else f"{previous} and {src}")
                alias = f" (the same file as {other})" if other != dst else ""
                sys.exit(f"{writers} would both be written to {dst}{alias}:"
                         " choose different output names.")
            seen.update(dict.fromkeys(keys, (src, dst)))


def _save_pair(voice_path, background_path, voice, background, sample_rate):
    """Stages both float WAVs before replacing outputs, keeping resume safe."""
    staged = []
    try:
        for destination, signal in ((voice_path, voice), (background_path,
                                                          background)):
            os.makedirs(os.path.dirname(os.path.abspath(destination)),
                        exist_ok=True)
            staged.append(audio_lib.create_staging_file(destination, ".pair-"))
            audio_lib.save_audio(staged[-1],
                                 signal,
                                 sample_rate,
                                 subtype="FLOAT")
        # A previous background must not make an interrupted replacement look
        # complete after the new voice has been committed.
        if os.path.lexists(background_path):
            os.unlink(background_path)
        os.replace(staged[0], voice_path)
        os.replace(staged[1], background_path)
    finally:
        for path in staged:
            if os.path.exists(path):
                os.unlink(path)


def _isolate(vi, x, sr, out_sr, args):
    """Isolates one file at the output rate."""
    kw = dict(
        output_sample_rate=out_sr,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
    )
    return vi.isolate(x, sr, **kw)


def main(argv: list[str] | None = None) -> int:
    """Isolates the requested audio and returns the command's exit status."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        sys.exit(f"--shard must be in [0, --num-shards); got {args.shard} of"
                 f" {args.num_shards}.")
    if args.output_sr < 0:
        sys.exit("--output-sr must be positive, or 0 to keep the input rate.")
    if args.background and args.mode != "offline":
        sys.exit("--background is available only in offline mode.")
    include_background = args.mode == "offline" and not args.no_background
    jobs, resume = _outputs(args.input, args.output, args.shard,
                            args.num_shards, include_background,
                            args.background)
    todo = [(src, voice, background)
            for src, voice, background in jobs
            if not (resume and os.path.isfile(voice) and
                    (background is None or os.path.isfile(background)))]
    print(f"{len(jobs)} file(s), {len(todo)} to do", flush=True)
    if not todo:
        return 0
    device = (torch.device(args.device)
              if args.device else isolator.default_device())
    _limit_macos_cpu_threads(device)
    vi = isolator.VoiceIsolator.load(
        args.model,
        device=device,
        precision=args.precision,
        mode=args.mode,
        lookahead_frames=args.lookahead,
        block_frames=args.block_frames,
        window_frames=args.window_frames,
        crossfade_frames=args.crossfade_frames,
        offline_passes=args.offline_passes,
    )
    subtype = "FLOAT" if args.float32 else "PCM_16"
    t0, ok = time.time(), 0
    for i, (src, dst, background) in enumerate(todo):
        try:
            x, sr = audio_lib.load_audio(src)
            out_sr = sr if args.output_sr == 0 else args.output_sr
            y = _isolate(vi, x, sr, out_sr, args)
            if background is not None:
                aligned = audio_lib.resample(x, sr, out_sr)
                aligned = audio_lib.fit_length(aligned, y.shape[-1])
                _save_pair(dst, background, y, aligned - y, out_sr)
            else:
                os.makedirs(os.path.dirname(os.path.abspath(dst)),
                            exist_ok=True)
                audio_lib.save_audio(dst, y, out_sr, subtype=subtype)
            ok += 1
        # one bad file must not stop a batch; the exit status reports it
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"FAIL {src} {e}", flush=True)
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(todo)}  {el / (i + 1):.2f}s/file",
                  flush=True)
    print(f"DONE {ok}/{len(todo)}  ({time.time() - t0:.0f}s)", flush=True)
    return 0 if ok == len(todo) else 1


if __name__ == "__main__":
    sys.exit(main())
