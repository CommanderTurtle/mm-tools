"""Shared stem-separation runtime for the YuE2 and MiniMax studios.

Drives the separation scripts (``python/separate.py`` = demucs,
``python/roformer.py`` = Mel-Band Roformer vocals) as subprocesses of the
calling interpreter and parses their line-oriented JSON progress protocol:

    {"type": "progress", "stage": "...", "pct": 0-99, "message": "..."}
    {"type": "stem", "name": "vocals"}
    {"type": "done", "stems": [...], "out_dir": "..."}
    {"type": "error", "message": "..."}

Engine selection follows the upstream app: a vocals-only request uses the
Mel-Band Roformer when its checkpoint is present; everything else runs
demucs (``htdemucs`` by default). Checkpoints resolve to the canonical
``models/stemkit`` tree so the central downloader (``models/
download_models.py``, bundle ``stemkit``) keeps them pinned; the scripts'
own one-time downloads remain the fallback when the tree is empty.

No telemetry: the subprocess scripts never touch the network except model
fetches. Upstream history (danielravina/stemkit at
2d44bc006e8f98170403ec131353aa7f66b49540) is preserved in
.git-archives/stemkit.git.tar.
"""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
import wave
from pathlib import Path
from typing import Callable, Optional, Sequence

ROOT = Path(__file__).resolve().parent
PYTHON_DIR = ROOT / "python"
SEPARATE_SCRIPT = PYTHON_DIR / "separate.py"
ROFORMER_SCRIPT = PYTHON_DIR / "roformer.py"

# Canonical checkpoint locations (models/download_models.py bundle "stemkit").
MODELS_ROOT = ROOT.parents[1] / "models" / "stemkit"
ROFORMER_CKPT = MODELS_ROOT / "roformer" / "MelBandRoformer.ckpt"
DEMUCS_CACHE_ROOT = MODELS_ROOT / "torch-hub"

ALL_STEMS = ("vocals", "drums", "bass", "other")
PRESETS = {
    "all": list(ALL_STEMS),
    "karaoke": ["drums", "bass", "other"],
    "acapella": ["vocals"],
    "drums_bass": ["drums", "bass"],
}

ProgressCallback = Callable[[str, int, str], None]


class StemSplitError(RuntimeError):
    """Raised when a separation subprocess reports an error or dies."""


def _models_env(base_env: dict[str, str]) -> dict[str, str]:
    """Point the demucs model cache at the canonical models/stemkit tree."""
    env = dict(base_env)
    env.setdefault("TORCH_HOME", str(DEMUCS_CACHE_ROOT))
    return env


def _run_json_script(
    script: Path,
    args: list[str],
    python: str | None,
    progress: Optional[ProgressCallback],
    cwd: Path | None = None,
) -> dict:
    """Run one vendor script; stream its JSON lines; return the done record."""
    command = [str(python or sys.executable), str(script), *args]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_models_env(os.environ),
        cwd=str(cwd) if cwd else None,
    )
    done: dict = {}
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # Non-JSON noise (warnings); surface it as a status message.
            if progress:
                progress("log", -1, line[:400])
            continue
        kind = event.get("type")
        if kind == "progress" and progress:
            progress(str(event.get("stage", "")), int(event.get("pct", 0)),
                     str(event.get("message", "")))
        elif kind == "error":
            process.wait()
            raise StemSplitError(str(event.get("message", "separation failed")))
        elif kind == "done":
            done = event
    returncode = process.wait()
    if returncode != 0 or not done:
        raise StemSplitError(
            f"{script.name} exited {returncode} without a done record")
    return done



def is_riff_wav(path: Path) -> bool:
    """True when ``path`` carries a RIFF header (the only layout the
    stdlib wave readers in the separation scripts accept)."""
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == b"RIFF"
    except OSError:
        return False


def write_pcm16_wav(path: Path, data, rate: int) -> None:
    """Hand-write ``data`` (frames, channels) float samples as a little-endian
    PCM_16 RIFF wav, without libsndfile.

    The libsndfile builds bundled with some soundfile wheels reject every
    write-open with "Format not recognised." while their readers work fine,
    so the stem pipeline must not depend on the writer side.
    """
    import numpy as np

    samples = np.asarray(data, dtype="<f8")
    if samples.ndim == 1:
        samples = samples[:, None]
    pcm = np.clip(samples, -1.0, 1.0) * 32767.0
    payload = pcm.astype("<i2").tobytes()
    channels = int(samples.shape[1])
    block_align = channels * 2
    header = b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, int(rate), int(rate) * block_align, block_align, 16)
    header += b"data" + struct.pack("<I", len(payload))
    Path(path).write_bytes(header + payload)

def split_wav(
    input_path: Path,
    out_dir: Path,
    stems: Optional[Sequence[str]] = None,
    *,
    model: str = "htdemucs",
    device: str = "auto",
    shifts: int = 1,
    python: str | None = None,
    progress: Optional[ProgressCallback] = None,
) -> dict:
    """Separate one wav into stems, following the upstream engine plan.

    Returns ``{"engine": "roformer"|"demucs", "model": ..., "stems": [...]}``
    with the written wavs under ``out_dir``.
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    wanted = [s for s in (list(stems) if stems is not None else list(ALL_STEMS))]
    unknown = [s for s in wanted if s not in ALL_STEMS]
    if unknown:
        raise ValueError(f"unknown stems requested: {', '.join(unknown)}")
    if not wanted:
        raise ValueError("no stems requested")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not is_riff_wav(input_path):
        # Take outputs land as FLAC; the separation scripts read RIFF only.
        import soundfile as sf

        decoded, decoded_rate = sf.read(str(input_path), always_2d=True)
        write_pcm16_wav(out_dir / "__input.wav", decoded, int(decoded_rate))
        input_path = out_dir / "__input.wav"

    use_roformer = wanted == ["vocals"] and ROFORMER_CKPT.is_file()
    if use_roformer:
        done = _run_json_script(
            ROFORMER_SCRIPT,
            ["--input", str(input_path), "--out", str(out_dir),
             "--ckpt-dir", str(ROFORMER_CKPT.parent), "--device", device],
            python, progress,
        )
        engine = "roformer"
    else:
        args = ["--input", str(input_path), "--out", str(out_dir),
                "--model", model, "--device", device, "--shifts", str(shifts)]
        if wanted != list(ALL_STEMS):
            args += ["--only", ",".join(wanted)]
        done = _run_json_script(SEPARATE_SCRIPT, args, python, progress)
        engine = "demucs"
    return {
        "engine": engine,
        "model": "mel-band-roformer" if engine == "roformer" else model,
        "stems": [s for s in wanted if (out_dir / f"{s}.wav").is_file()],
    }


def load_mono_float32(path: Path) -> tuple[list[float], int]:
    """Read a wav (16-bit int or 32-bit float) as mono float samples."""
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        width = handle.getsampwidth()
        channels = handle.getnchannels()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype("<f4") / 32768.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype="<f4")
    else:
        raise ValueError(f"unsupported sample width {width}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(float).tolist(), rate


def _active_segments(envelope: list[float], rate: int, hop: int) -> list[tuple[float, float]]:
    """Contiguous vocal-active spans from a log-energy envelope."""
    import numpy as np
    env = np.asarray(envelope, dtype=np.float64)
    peak = float(env.max()) if env.size else 0.0
    if peak <= 1e-5:
        return []
    threshold = 0.30 * peak
    active = env > threshold
    frame_seconds = hop / float(rate)
    gap_max = int(round(0.35 / frame_seconds))   # bridge gaps under 350 ms
    min_len = int(round(0.40 / frame_seconds))   # drop blips under 400 ms
    segments: list[list[int]] = []
    run_start: Optional[int] = None
    gap = 0
    for index, value in enumerate(active):
        if value:
            if run_start is None:
                run_start = index
            elif gap:
                # gap absorbed into the run; extend the start boundary back
                pass
            gap = 0
        else:
            if run_start is not None:
                gap += 1
                if gap > gap_max:
                    end = index - gap
                    if end - run_start >= min_len:
                        segments.append([run_start, end])
                    run_start = None
                    gap = 0
    if run_start is not None:
        end = len(active)
        if end - run_start >= min_len:
            segments.append([run_start, end])
    return [(start * frame_seconds, end * frame_seconds)
            for start, end in segments]


def lyric_timestamps(
    lyrics: Sequence[str],
    envelope: Sequence[float],
    rate: int,
    hop: int = 1024,
) -> list[dict]:
    """Assign lyric lines to vocal-active time slots.

    ``lyrics`` are raw lines; lines that look like section tags
    (``[Verse]``) carry no timestamp of their own. Every other line is
    placed inside a vocal-active span, in order, with each span weighted by
    its length — so lines never start during a silent intro or outro and
    their density follows where the singing actually happens.
    """
    lines = [str(line) for line in lyrics]
    timed_indexes = [
        i for i, line in enumerate(lines)
        if line.strip() and not re.fullmatch(r"\[.+\]", line.strip())
    ]
    if not timed_indexes:
        return []
    segments = _active_segments(list(envelope), rate, hop)
    total_active = sum(end - start for start, end in segments)
    if total_active <= 0:
        duration = len(envelope) * hop / float(rate)
        step = duration / len(timed_indexes)
        return [{"line": i, "start": round(index * step, 3)}
                for index, i in enumerate(timed_indexes)]
    per_line: list[float] = []
    remaining = len(timed_indexes)
    for start, end in segments:
        share = max(1, round((end - start) / total_active * len(timed_indexes)))
        share = min(share, remaining)
        span = end - start
        for slot in range(share):
            per_line.append(start + span * (slot + 0.25) / share)
        remaining -= share
        if remaining <= 0:
            break
    # A sparse-active track may yield fewer slots than lines; fall back to a
    # uniform spread across the last active span for the overflow.
    if len(per_line) < len(timed_indexes):
        last_end = segments[-1][1]
        step = max(0.5, (last_end - segments[-1][0]) / max(1, len(timed_indexes) - len(per_line)))
        cursor = per_line[-1] if per_line else 0.0
        while len(per_line) < len(timed_indexes):
            cursor = min(last_end, cursor + step)
            per_line.append(cursor)
    return [{"line": int(i), "start": round(float(t), 3)}
            for i, t in zip(timed_indexes, per_line)]


def vocal_envelope(
    vocals_path: Path,
    hops: int = 0,
    rate_override: int = 0,
) -> tuple[list[float], int, int]:
    """Log-RMS envelope of a vocal stem: (envelope, rate, hop)."""
    import numpy as np
    audio, rate = load_mono_float32(vocals_path)
    hop = 1024
    if not audio:
        return [], rate, hop
    arr = np.asarray(audio, dtype=np.float64)
    frames = int(np.ceil(len(arr) / hop))
    padded = np.zeros(frames * hop, dtype=np.float64)
    padded[: len(arr)] = arr
    reshaped = padded.reshape(frames, hop)
    rms = np.sqrt(np.mean(reshaped ** 2, axis=1) + 1e-10)
    envelope = (np.log1p(rms * 40.0)).tolist()
    return envelope, rate, hop


def vocal_timestamps_for_lyrics(
    track_path: Path,
    lyrics: Sequence[str],
    work_dir: Path,
    *,
    python: str | None = None,
    device: str = "auto",
    progress: Optional[ProgressCallback] = None,
) -> list[dict]:
    """Full pipeline: separate vocals, then align lyric lines to them."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        progress("separate", 0, "Isolating vocals for lyric timing")
    result = split_wav(track_path, work_dir / "stems", stems=["vocals"],
                       device=device, python=python, progress=progress)
    if "vocals" not in result["stems"]:
        raise StemSplitError("vocal separation produced no vocals stem")
    envelope, rate, hop = vocal_envelope(work_dir / "stems" / "vocals.wav")
    if progress:
        progress("sync", 99, "Aligning lyrics to vocal activity")
    return lyric_timestamps(lyrics, envelope, rate, hop)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="stem runtime CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_split = sub.add_parser("split", help="separate a local wav into stems")
    p_split.add_argument("--input", required=True)
    p_split.add_argument("--out", required=True)
    p_split.add_argument("--preset", default="all", choices=sorted(PRESETS))
    p_split.add_argument("--stems", default="", help="comma list; overrides --preset")
    p_split.add_argument("--model", default="htdemucs")
    p_split.add_argument("--device", default="auto")

    p_sync = sub.add_parser("timestamps", help="vocal-aligned lyric timestamps")
    p_sync.add_argument("--input", required=True, help="mixed track wav")
    p_sync.add_argument("--lyrics", required=True, help="plain-text lyrics file")
    p_sync.add_argument("--work", required=True, help="scratch directory")
    p_sync.add_argument("--device", default="auto")

    args = parser.parse_args()
    if args.command == "split":
        wanted = ([s for s in args.stems.split(",") if s.strip()]
                  if args.stems else PRESETS[args.preset])
        print(json.dumps(split_wav(Path(args.input), Path(args.out), wanted,
                                   model=args.model, device=args.device)))
    else:
        lines = Path(args.lyrics).read_text(encoding="utf-8").splitlines()
        rows = vocal_timestamps_for_lyrics(Path(args.input), lines,
                                           Path(args.work), device=args.device)
        print(json.dumps(rows))
