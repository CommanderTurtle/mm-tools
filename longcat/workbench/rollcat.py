#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Interactive LongCat TTS batch re-roller: one line of in.txt per WAV, numbered
001-longcat.wav and friends, rerolled by index until you say done. Standard
library only - vox's client scripts already proved the wire format needs
nothing else (vox/scripts/route_acceptance.py:24-57).

Wire contract (longcat/local_tts/server.py:143-195): POST /api/synthesize as
multipart/form-data, response is the whole audio/wav body. Field names match
vox's client (vox/crates/vox-local-core/src/longcat.rs:87-105).

Configuration, lowest precedence first (mirrors vox: embedded defaults, then
TOML with every field defaulted; keys verbatim from vox/src/config/defaults.toml:121-142):
  1. the DEFAULTS table below
  2. vox's own config, if present: $VOX_CONFIG, else
     $XDG_CONFIG_HOME/com.vox/vox/vox.toml, else ~/.config/com.vox/vox/vox.toml
     - only the [tts.longcat] table is read, never written
  3. rollcat.toml next to this script (or --config): [tts.longcat] uses the
     same grammar as vox, so a vox block pastes in verbatim; [rollcat] holds
     this script's own knobs (input, output, padding)
  4. $LONGCAT_URL, else $LONGCAT_PORT -> http://127.0.0.1:<port>
  5. command-line flags

Join whole line takes into one file with `c 4-11` or `c 4,5,6,9` or `c a`; the
numbered group file (001-concat.wav ...) plus its dat.json metadata record is
everything `d 1` needs to split the members back out byte-for-byte, so
de-concatenation never needs the model.

The reference clip is discovered rather than hard-coded: `<stem>.txt` next to
`<stem>.<media>` in this directory, default stem `config` - so config.txt plus
config.m4a, .wav, .mp3, anything ffmpeg reads. A missing combo is a startup
error followed by a prompt for another literal stem (e.g. "voice2"); `--voice`,
[rollcat] voice_stem and the vox voice_profiles block set it up front, and
`--no-voice` skips the voice entirely. Both halves are mandatory: the engine
refuses prompt_audio without prompt_text, and the reverse
(longcat/local_tts/core.py:199-202).

Usage: uv run rollcat.py        (from anywhere; paths resolve to this dir)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import mimetypes
import os
import re
import sys
import time
import tomllib
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
STATE_NAME = "dat.json"

DEFAULTS: dict[str, Any] = {
    "base_url": "http://127.0.0.1:8230",
    "prompt_audio_path": "",
    "prompt_text": "",
    "active_voice_profile": "",
    "steps": 16,
    "guidance_strength": 4.0,
    "guidance_method": "apg",
    "seed": 1024,
    "duration_scale": 1.0,
}
ROLLCAT_DEFAULTS: dict[str, Any] = {
    "input": "in.txt",
    "output": "output",
    "file_stem": "longcat",
    "concat_stem": "concat",
    "voice_stem": "config",
    "pad": 3,
    "timeout": 1800,
    "status_timeout": 10,
    "load_timeout": 1800,
}

# The reference clip's format is whatever ffmpeg decodes: the server pipes the
# upload through ffmpeg before the model ever sees it, resampling to the model
# rate and folding to mono (longcat/local_tts/core.py:77-109). This list only
# ranks the choice when several renderings of the same stem sit side by side.
MEDIA_SUFFIXES = (
    ".wav", ".flac", ".aiff", ".aif", ".ogg", ".opus", ".m4a", ".mp4",
    ".aac", ".mp3", ".webm", ".wma", ".amr", ".3gp",
)

# --------------------------------------------------------------------------------------
# Text corrections: what actually reaches the model. Two modes per line item,
# toggled with the `s` command: "plain" spells runs out with literal quotes,
# "string" spells them as spoken words. Both are deliberately hand-editable lists;
# add a row here and nothing else has to change.
# --------------------------------------------------------------------------------------

CAPS_MIN_RUN = 2  # CEV -> "C" "E" "V" / see ee vee. Lower to 1 to catch lone capitals.
DIGIT_MIN_RUN = 2  # 123 -> "1" "2" "3" / one two three. NOTE: 800 also splits.
LETTER_WORDS = {
    "a": "ay", "b": "bee", "c": "see", "d": "dee", "e": "ee", "f": "eff",
    "g": "gee", "h": "aitch", "i": "eye", "j": "jay", "k": "kay", "l": "el",
    "m": "em", "n": "en", "o": "oh", "p": "pee", "q": "cue", "r": "ar",
    "s": "ess", "t": "tee", "u": "you", "v": "vee", "w": "double u",
    "x": "ex", "y": "why", "z": "zee",
}
DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}
# Applied first, identical in both modes: (pattern, replacement).
EXPANSIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\betc\b\.?", re.IGNORECASE), "etcetera"),
    (re.compile(r"\betcetera\b", re.IGNORECASE), "etcetera"),  # drop a stray period
    (re.compile(r"&amp;", re.IGNORECASE), " and "),
    (re.compile(r"\blol\b", re.IGNORECASE), "lol"),  # deliberate no-op, kept visible
    (re.compile(r"\btts\b", re.IGNORECASE), "TTS"),
]
# Applied second, mode-dependent: character -> (plain word, string word).
SYMBOLS: dict[str, tuple[str, str]] = {
    "&": ("and", "and"),
    "/": ("slash", "or"),
    "%": ("percent", "percent"),
    "@": ("at", "at"),
}
_SYMBOLS_RE = re.compile("|".join(re.escape(key) for key in SYMBOLS))
_CAPS_RE = re.compile(rf"\b[A-Z]{{{CAPS_MIN_RUN},}}\b")
_DIGITS_RE = re.compile(rf"\b\d{{{DIGIT_MIN_RUN},}}\b")


def expand_runs(run: str, mode: str, words: dict[str, str]) -> str:
    """Render one capital/digit run for the active mode."""
    if mode == "string":
        parts = [words.get(char.lower(), char) for char in run]
        return " ".join(parts)
    return " ".join(f'"{char}"' for char in run)


def transform(text: str, mode: str) -> str:
    """Source line -> the exact string posted to LongCat."""
    result = text
    for pattern, replacement in EXPANSIONS:
        result = pattern.sub(replacement, result)
    result = _SYMBOLS_RE.sub(lambda match: " %s " % SYMBOLS[match.group(0)][0 if mode == "plain" else 1], result)
    result = _CAPS_RE.sub(lambda match: expand_runs(match.group(0), mode, LETTER_WORDS), result)
    result = _DIGITS_RE.sub(lambda match: expand_runs(match.group(0), mode, DIGIT_WORDS), result)
    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r" +([,.;:!?])", r"\1", result)
    return result.strip()


# --------------------------------------------------------------------------------------
# Configuration loading
# --------------------------------------------------------------------------------------


def read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SystemExit(f"rollcat: cannot parse {path}: {error}") from error


def vox_config_path() -> Path | None:
    override = os.environ.get("VOX_CONFIG", "").strip()
    if override:
        return Path(override).expanduser()
    base = Path(os.environ.get("XDG_CONFIG_HOME", "") or Path.home() / ".config")
    for candidate in (base / "com.vox" / "vox" / "vox.toml", base / "vox" / "vox.toml"):
        if candidate.is_file():
            return candidate
    return None


def load_config(config_path: Path) -> tuple[dict[str, Any], list[str]]:
    longcat = dict(DEFAULTS)
    rollcat = dict(ROLLCAT_DEFAULTS)
    sources: list[str] = []

    def absorb(section: object, origin: str) -> None:
        if not isinstance(section, dict):
            return
        longcat.update({key: value for key, value in section.items() if key in DEFAULTS})
        if isinstance(section.get("voice_profiles"), list):
            longcat["_voice_profiles"] = section["voice_profiles"]
        sources.append(origin)

    vox_path = vox_config_path()
    if vox_path:
        absorb(read_toml(vox_path).get("tts", {}).get("longcat"), f"vox {vox_path.name}")

    document = read_toml(config_path)
    absorb(document.get("tts", {}).get("longcat"), config_path.name)
    own = document.get("rollcat")
    if isinstance(own, dict):
        rollcat.update({key: value for key, value in own.items() if key in ROLLCAT_DEFAULTS})

    url = os.environ.get("LONGCAT_URL", "").strip()
    port = os.environ.get("LONGCAT_PORT", "").strip()
    if url:
        longcat["base_url"] = url
        sources.append("$LONGCAT_URL")
    elif port:
        longcat["base_url"] = f"http://127.0.0.1:{port}"
        sources.append("$LONGCAT_PORT")

    longcat["base_url"] = str(longcat["base_url"]).rstrip("/")
    longcat.setdefault("_voice_profiles", [])
    return {"longcat": longcat, "rollcat": rollcat}, sources


def pair_here(directory: Path, stem: str) -> dict[str, Any] | None:
    """`<stem>.txt` plus `<stem>.<media>` in one directory, or nothing.

    Both or neither is a hard rule: the engine rejects prompt_audio without
    prompt_text, and prompt_text without prompt_audio (longcat/local_tts/core.py:199-202).
    """
    if not stem or not directory.is_dir():
        return None
    transcript = directory / f"{stem}.txt"
    if not transcript.is_file():
        return None
    text = transcript.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return None
    found = {path.suffix.lower(): path for path in directory.glob(f"{stem}.*") if path.is_file()}
    for suffix in MEDIA_SUFFIXES:
        audio = found.get(suffix)
        if audio is not None:
            return {
                "name": stem,
                "audio_path": audio,
                "transcript_path": transcript,
                "transcript": text,
            }
    return None


def pair_from_config(
    longcat: dict[str, Any], profiles: list[dict[str, Any]], wanted: str
) -> dict[str, Any] | None:
    """A vox voice_profiles entry, or vox's flat prompt_audio_path key, if configured."""
    name = (wanted or str(longcat.get("active_voice_profile", ""))).strip()
    if name:
        chosen = next(
            (p for p in profiles if str(p.get("name", "")).strip().casefold() == name.casefold()),
            None,
        )
        if chosen is None:
            return None  # not a profile name: may still be a stem to discover
    else:
        chosen = {"audio_path": longcat.get("prompt_audio_path", ""), "transcript_path": ""}
    audio = str(chosen.get("audio_path", "")).strip()
    if not audio:
        return None
    audio_path = Path(os.path.expandvars(audio)).expanduser()
    if not audio_path.is_absolute():
        audio_path = (HERE / audio_path).resolve()
    if not audio_path.is_file():
        raise SystemExit(f"rollcat: prompt audio not found: {audio_path}")
    transcript = str(chosen.get("transcript_path", "")).strip()
    transcript_path = (
        Path(os.path.expandvars(transcript)).expanduser() if transcript else audio_path.with_suffix(".txt")
    )
    if not transcript_path.is_absolute():
        transcript_path = (HERE / transcript_path).resolve()
    if transcript_path.is_file():
        text = transcript_path.read_text(encoding="utf-8", errors="replace").strip()
    else:
        text = str(longcat.get("prompt_text", "")).strip()
    if not text:
        raise SystemExit(f"rollcat: no transcript for {audio_path.name}; put it in {transcript_path.name}")
    return {
        "name": name or audio_path.stem,
        "audio_path": audio_path,
        "transcript_path": transcript_path,
        "transcript": text,
    }


def describe_pair(stem: str) -> str:
    return f"{stem}.txt + {stem}." + "{" + ",".join(suffix[1:] for suffix in MEDIA_SUFFIXES) + "}"


def pick_voice(args: argparse.Namespace, settings: dict[str, Any], directories: list[Path]) -> dict[str, Any] | None:
    """Find the reference pair: config.txt + config.media first, then ask.

    Order: --no-voice, a configured vox profile, then the stems --voice,
    rollcat.voice_stem and finally "config", looked for in each directory given.
    """
    if args.no_voice:
        return None
    longcat = settings["longcat"]
    explicit = str(args.voice or "").strip()
    pair = pair_from_config(longcat, longcat.get("_voice_profiles", []) or [], explicit)
    if pair:
        return pair
    stems = [
        stem
        for stem in dict.fromkeys(
            [explicit, str(settings["rollcat"].get("voice_stem", "")).strip(), "config"]
        )
        if stem
    ]
    for stem in stems:
        for directory in directories:
            pair = pair_here(directory, stem)
            if pair:
                return pair
    message = f"no reference pair found: need {(' or '.join(describe_pair(stem) for stem in stems))} in " \
              f"{', '.join(str(directory) for directory in directories)}"
    if not sys.stdin.isatty():
        raise SystemExit(
            f"rollcat: {message}\n"
            "         give it --voice <stem>, set [rollcat] voice_stem in rollcat.toml, "
            "or --no-voice to go zero-shot"
        )
    print(f"rollcat: {message}")
    for _ in range(3):
        try:
            answer = input('reference stem? (e.g. "voice2" for voice2.txt + voice2.m4a; Enter or q gives up): ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not answer or answer.casefold() in {"q", "quit"}:
            break
        candidate = Path(os.path.expandvars(answer)).expanduser()
        if not candidate.is_absolute():
            candidate = directories[0] / candidate
        pair = pair_here(candidate.parent, candidate.stem)
        if pair is None and not candidate.is_absolute():
            pair = pair_here(Path.cwd() / candidate.name, Path(candidate.name).stem)
        if pair:
            return pair
        print(f"       still nothing matching {describe_pair(candidate.stem)} in {candidate.parent}")
    raise SystemExit("rollcat: no usable reference pair (a too-long clip is refused too; --no-voice skips it)")


# --------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------


def request_json(url: str, method: str = "GET", timeout: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body or b"{}")


def post_wav(base_url: str, fields: dict[str, object], files: dict[str, Path], timeout: float) -> tuple[bytes, dict[str, str]]:
    """multipart/form-data POST with stdlib only; same shape as vox's helper."""
    boundary = f"rollcat-{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")
    for name, path in files.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'.encode())
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body.extend(f"Content-Type: {mime}\r\n\r\n".encode())
        body.extend(path.read_bytes())
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    request = urllib.request.Request(
        f"{base_url}/api/synthesize",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), {key.lower(): value for key, value in response.headers.items()}


def probe(base_url: str, status_timeout: float, load_timeout: float) -> None:
    try:
        status = request_json(f"{base_url}/api/status", timeout=status_timeout)
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise SystemExit(
            f"rollcat: LongCat httpd not answering at {base_url} ({error}).\n"
            f"         Start it with ../starthttp.sh; this script will not load a model for you."
        )
    if status.get("loaded"):
        print(f"service {base_url} loaded on {status.get('device', '?')} "
              f"({status.get('dtype', '?')}), role {status.get('role', '?')}")
        return
    if status.get("role") != "http":
        print(f"note: role is {status.get('role')!r}, not 'http'; continuing anyway")
    print(f"model not loaded yet; POST {base_url}/api/load ...")
    try:
        request_json(f"{base_url}/api/load", method="POST", timeout=load_timeout)
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise SystemExit(f"rollcat: {base_url}/api/load failed: {error}")
    print("loaded")


def read_wav_info(payload: bytes, headers: dict[str, str]) -> dict[str, Any]:
    info: dict[str, Any] = {}
    raw = headers.get("x-longcat-metadata", "")
    if raw:
        try:
            info.update(json.loads(raw))
        except ValueError:
            pass
    try:
        with wave.open(io.BytesIO(payload)) as reader:
            info.setdefault("sample_rate", reader.getframerate())
            info["audio_seconds"] = round(reader.getnframes() / max(1, reader.getframerate()), 3)
            info.setdefault("sample_width", reader.getsampwidth())
            info.setdefault("channels", reader.getnchannels())
    except Exception:  # noqa: BLE001 - the header is a nicety, never a blocker
        pass
    return info


def wav_frames(path: Path) -> tuple[tuple[int, int, int], bytes]:
    """(channels, sample_width, rate) plus the raw frame bytes of a PCM WAV."""
    with wave.open(str(path), "rb") as reader:
        params = (reader.getnchannels(), reader.getsampwidth(), reader.getframerate())
        return params, reader.readframes(reader.getnframes())


def write_wav(path: Path, params: tuple[int, int, int], frames: bytes) -> None:
    """Write frames through a .part file so a failure never truncates a keeper."""
    channels, sample_width, rate = params
    temporary = path.with_name(path.name + ".part")
    with wave.open(str(temporary), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(rate)
        writer.writeframesraw(frames)
    os.replace(temporary, path)


# --------------------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------------------


def save_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"note: ignoring unreadable {path.name} ({error})")
        return None
    if not isinstance(state, dict) or not isinstance(state.get("items"), list):
        print(f"note: ignoring malformed {path.name}")
        return None
    return state


def wipe_output(output: Path, stems: tuple[str, ...]) -> int:
    """Remove this tool's own output: line WAVs, group WAVs, their .part files, state."""
    removed = 0
    for entry in sorted(output.iterdir()):
        if not entry.is_file():
            continue
        ours = entry.name == STATE_NAME or (
            entry.suffix == ".wav" and entry.stem.endswith(stems)
        ) or entry.suffix == ".part"
        if ours:
            entry.unlink()
            removed += 1
    return removed


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


class Roller:
    def __init__(self, args: argparse.Namespace, settings: dict[str, Any], voice: dict[str, Any] | None) -> None:
        self.args = args
        self.longcat = settings["longcat"]
        self.rollcat = settings["rollcat"]
        self.voice = voice
        self.base_dir = HERE
        self.input_path = self._resolve(self.rollcat["input"]) if args.input is None else self._resolve(args.input)
        self.output = self._resolve(self.rollcat["output"]) if args.output is None else self._resolve(args.output)
        self.stem = str(self.rollcat["file_stem"])
        self.concat_stem = str(self.rollcat["concat_stem"])
        self.pad = max(3, int(self.rollcat["pad"]))
        self.state_path = self.output / STATE_NAME
        self.state: dict[str, Any] = {}
        self.notes: list[str] = []

    def _resolve(self, value: object) -> Path:
        path = Path(os.path.expandvars(str(value))).expanduser()
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    def wav_name(self, index: int, total: int) -> str:
        return f"{index:0{self.width(total)}d}-{self.stem}.wav"

    def group_name(self, group: int) -> str:
        return f"{group:0{self.width(max(len(self.state.get('items', [])), group))}d}-{self.concat_stem}.wav"

    def width(self, total: int) -> int:
        return max(self.pad, len(str(max(total, 1))))

    def media_files(self) -> list[Path]:
        """Line WAVs and group WAVs currently sitting in the output directory."""
        stems = (self.stem, self.concat_stem)
        if not self.output.is_dir():
            return []
        return sorted(
            entry
            for entry in self.output.iterdir()
            if entry.is_file() and entry.suffix == ".wav" and entry.stem.endswith(stems)
        )

    @property
    def groups(self) -> list[dict[str, Any]]:
        groups = self.state.setdefault("concats", [])
        return groups if isinstance(groups, list) else []

    def group_of(self, index: int) -> dict[str, Any] | None:
        """The concat group that currently owns a line, if any."""
        for group in self.groups:
            if any(member.get("index") == index for member in group.get("members", [])):
                return group
        return None

    def parse_spec(self, spec: str, everything: list[int]) -> list[int]:
        """`4,5,6,9` / `4-11` / `4-6,9` / `a` -> sorted line or group numbers."""
        picks: set[int] = set()
        for token in re.split(r"[,\s]+", spec.strip()):
            if not token:
                continue
            if token.casefold() == "a":
                picks.update(everything)
                continue
            span = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
            if span:
                low, high = int(span.group(1)), int(span.group(2))
                picks.update(range(min(low, high), max(low, high) + 1))
                continue
            if token.isdigit():
                picks.add(int(token))
                continue
            raise ValueError(token)
        return sorted(picks)

    def read_lines(self) -> list[str]:
        if not self.input_path.is_file():
            raise SystemExit(f"rollcat: no input file: {self.input_path}")
        return [line.strip() for line in self.input_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def seed_base(self) -> int:
        return int(self.args.seed if self.args.seed is not None else self.longcat["seed"])

    def sync_items(self, lines: list[str]) -> list[int]:
        """Rebuild the item list from in.txt, keeping each slot's seed and mode.

        Returns the indexes that need rendering: new slots, slots whose source line
        changed, and slots whose WAV is missing on disk.
        """
        old = {item["index"]: item for item in self.state.get("items", []) if isinstance(item, dict)}
        items: list[dict[str, Any]] = []
        dirty: list[int] = []
        stale: dict[int, bool] = {}
        for number, line in enumerate(lines, start=1):
            prior = old.get(number) or {}
            mode = "string" if prior.get("string_mode") else "plain"
            reworded = bool(old.get(number)) and prior.get("text_raw") != line
            grouped = self.group_of(number) is not None
            item = {
                "index": number,
                "file": self.wav_name(number, len(lines)),
                "text_raw": line,
                "text_sent": transform(line, mode),
                "seed": int(prior["seed"]) if "seed" in prior else self.seed_base(),
                "rerolls": int(prior.get("rerolls", 0)),
                "string_mode": mode == "string",
                # a reword that happened while the line sat inside a group file: the
                # group still holds the old wording, so the take is out of date
                "audio_stale": bool(prior.get("audio_stale")) or (reworded and grouped),
            }
            stale[number] = item["audio_stale"]
            items.append(item)
            absent = not (self.output / item["file"]).is_file()
            if not old.get(number) or reworded or absent or item["audio_stale"]:
                dirty.append(number)
        self.notes = []
        keep: list[int] = []
        for number in dirty:
            owner = self.group_of(number)
            if owner is None:
                keep.append(number)
            elif stale[number]:
                self.notes.append(
                    f"line {number} was reworded while inside concat group {owner['group']} "
                    f"({owner['file']}): that group still holds the old wording. "
                    f"d {owner['group']} then r to re-render it"
                )
        dirty = keep
        self.state.update(
            {
                "version": 1,
                "source": self.input_path.name,
                "source_sha256": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16],
                "base_url": self.longcat["base_url"],
                "voice": None if self.voice is None else {"name": self.voice["name"], "audio_path": str(self.voice["audio_path"])},
                "params": {
                    key: self.longcat[key]
                    for key in ("steps", "guidance_strength", "guidance_method", "duration_scale")
                },
                "base_seed": self.seed_base(),
                "items": items,
            }
        )
        return dirty

    def item(self, index: int) -> dict[str, Any]:
        for item in self.state["items"]:
            if item["index"] == index:
                return item
        raise IndexError(index)

    def render(self, item: dict[str, Any], reason: str) -> bool:
        mode = "string" if item["string_mode"] else "plain"
        item["text_sent"] = transform(item["text_raw"], mode)
        fields = {
            "text": item["text_sent"],
            "steps": self.longcat["steps"],
            "guidance_strength": self.longcat["guidance_strength"],
            "guidance_method": self.longcat["guidance_method"],
            "seed": item["seed"],
            "duration_scale": self.longcat["duration_scale"],
        }
        files = {} if self.voice is None else {"prompt_audio": self.voice["audio_path"]}
        if self.voice is not None:
            fields["prompt_text"] = self.voice["transcript"]
        label = f"{item['index']:>3} {item['file']}"
        print(f"[{reason}] {label}  seed={item['seed']} {mode}  {len(item['text_sent'])} chars")
        started = time.monotonic()
        try:
            payload, headers = post_wav(
                self.longcat["base_url"], fields, files, float(self.rollcat["timeout"])
            )
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace").strip().replace("\n", " ")
            print(f"       HTTP {error.code}: {detail[:400]}  (kept {item['file']} as it was)")
            return False
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            print(f"       request failed: {error}  (kept {item['file']} as it was)")
            return False
        if payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
            print(f"       response is not a WAV ({len(payload)} bytes, starts {payload[:12]!r})")
            return False
        destination = self.output / item["file"]
        temporary = destination.with_name(destination.name + ".part")
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
        info = read_wav_info(payload, headers)
        item.update(
            {
                "bytes": len(payload),
                "sample_rate": info.get("sample_rate"),
                "audio_seconds": info.get("audio_seconds"),
                "seconds_elapsed": round(time.monotonic() - started, 2),
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "audio_stale": False,
            }
        )
        print(f"       ok {len(payload)} bytes  {info.get('audio_seconds', '?')}s  "
              f"{info.get('sample_rate', '?')} Hz  in {item['seconds_elapsed']}s")
        save_state(self.state_path, self.state)
        return True

    def render_all(self, indexes: list[int], reason: str) -> None:
        for number in indexes:
            self.render(self.item(number), reason)

    # ---- concatenation --------------------------------------------------------------

    def do_concat(self, spec: str) -> None:
        """Join whole line WAVs into one group file, keeping the join reversible."""
        try:
            numbers = self.parse_spec(spec, [item["index"] for item in self.state["items"]])
        except ValueError as error:
            print(f"concat: bad selection {error!r}  (use n, n-m, a,b,c or a)")
            return
        if not numbers:
            print("concat: nothing selected  (c 4-11 | c 4,5,6,9 | c a)")
            return
        inside = [number for number in numbers if self.group_of(number)]
        if inside:
            print("concat: already grouped: "
                  + ", ".join(f"{number} in group {self.group_of(number)['group']}" for number in inside)
                  + " - d that group first")
            return
        unknown = [number for number in numbers if number not in {item["index"] for item in self.state["items"]}]
        if unknown:
            print(f"concat: no such line(s): {', '.join(map(str, unknown))} (1..{len(self.state['items'])})")
            return
        missing = [number for number in numbers if not (self.output / self.item(number)["file"]).is_file()]
        if missing:
            print("concat: no WAV yet for: " + ", ".join(self.item(number)["file"] for number in missing))
            return
        params: tuple[int, int, int] | None = None
        payloads: list[tuple[dict[str, Any], bytes]] = []
        for number in numbers:
            path = self.output / self.item(number)["file"]
            try:
                member_params, frames = wav_frames(path)
            except Exception as error:  # noqa: BLE001 - unreadable member, keep the group out
                print(f"concat: cannot read {path.name}: {error}")
                return
            if params is None:
                params = member_params
            elif member_params != params:
                print(f"concat: refused, {path.name} is {member_params} but the group would be {params}")
                return
            payloads.append((self.item(number), frames))
        group = max(
            [int(entry.get("group", 0)) for entry in self.groups]
            + [int(self.state.get("last_group", 0))]
        ) + 1  # never reused, so "d 3" always means the same set of lines
        self.state["last_group"] = group
        name = self.group_name(group)
        destination = self.output / name
        channels, sample_width, rate = params  # type: ignore[misc]
        joined = b"".join(frames for _, frames in payloads)
        write_wav(destination, params, joined)  # type: ignore[arg-type]
        members = []
        for item, frames in payloads:
            members.append(
                {
                    "index": item["index"],
                    "file": item["file"],
                    "frames": len(frames) // (channels * sample_width),
                    "bytes": len(frames),
                    "sha256": hashlib.sha256(frames).hexdigest()[:12],
                }
            )
            (self.output / item["file"]).unlink(missing_ok=True)
        record = {
            "group": group,
            "file": name,
            "channels": channels,
            "sample_width": sample_width,
            "sample_rate": rate,
            "frames": len(joined) // (channels * sample_width),
            "bytes": destination.stat().st_size,
            "audio_seconds": round(len(joined) / (channels * sample_width * rate), 3),
            "members": members,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.groups.append(record)
        save_state(self.state_path, self.state)
        print(f"group {group} -> {name}  lines {', '.join(str(m['index']) for m in members)}  "
              f"{record['audio_seconds']}s  {record['bytes']} bytes  "
              f"(d {group} puts {', '.join(m['file'] for m in members)} back)")

    def do_unconcat(self, spec: str) -> None:
        """Split group files back into their member WAVs from the stored metadata."""
        try:
            numbers = self.parse_spec(spec, [int(entry.get("group", 0)) for entry in self.groups])
        except ValueError as error:
            print(f"deconcat: bad selection {error!r}  (use the group number, n-m, a,b,c or a)")
            return
        if not numbers:
            print("deconcat: nothing selected  (d 1 | d 1-2 | d a)")
            return
        for number in numbers:
            record = next((entry for entry in self.groups if int(entry.get("group", -1)) == number), None)
            if record is None:
                print(f"no concat group {number}" + (f"; groups: {[g['group'] for g in self.groups]}" if self.groups else ""))
                continue
            source = self.output / str(record["file"])
            if not source.is_file():
                print(f"group {number}: {source.name} is missing, cannot expand")
                continue
            params = (int(record["channels"]), int(record["sample_width"]), int(record["sample_rate"]))
            frame_size = params[0] * params[1]
            _, frames = wav_frames(source)
            restored = []
            offset = 0
            for member in record["members"]:
                end = offset + int(member["frames"]) * frame_size
                chunk = frames[offset:end]
                offset = end
                digest = hashlib.sha256(chunk).hexdigest()[:12]
                target = self.output / str(member["file"])
                write_wav(target, params, chunk)
                out_of_date = any(
                    entry["index"] == member.get("index") and entry.get("audio_stale")
                    for entry in self.state["items"]
                )
                restored.append(
                    f"{target.name}"
                    f"{' (exact)' if digest == member['sha256'] else ' (CHECKSUM MISMATCH)'}"
                    + (" [old wording - r re-renders it]" if out_of_date else "")
                )
            source.unlink(missing_ok=True)
            self.groups.remove(record)
            save_state(self.state_path, self.state)
            print(f"group {number} expanded back into: " + ", ".join(restored))

    def table(self) -> None:
        print(f"{'#':>4}  {'seed':>6}  {'mode':<7} {'state':<6} file / text")
        for item in self.state["items"]:
            owner = self.group_of(item["index"])
            on_disk = (
                f"grp{owner['group']}"
                if owner
                else ("stale" if item.get("audio_stale")
                      else ("wav" if (self.output / item["file"]).is_file() else "-"))
            )
            mode = "string" if item["string_mode"] else "plain"
            print(f"{item['index']:>4}  {item['seed']:>6}  {mode:<7} {on_disk:<6} {item['file']}")
            print(f"{'':>4}  {'':>6}  {'':<7} {'':<6}   raw  {item['text_raw']}")
            print(f"{'':>4}  {'':>6}  {'':<7} {'':<6}   sent {item['text_sent']}")
        if self.groups:
            print("concat groups:")
            for record in self.groups:
                present = "wav" if (self.output / str(record["file"])).is_file() else "MISSING"
                print(f"  group {record['group']}  {record['file']}  lines "
                      f"{','.join(str(member['index']) for member in record['members'])}  "
                      f"{record.get('audio_seconds', '?')}s  {record['bytes']} bytes  {present}")

    # ---- interactive commands -------------------------------------------------------

    def apply_command(self, line: str) -> bool:
        """Handle one command line; False means the session is over."""
        command = line.strip()
        if not command:
            return True
        lowered = command.casefold()
        if lowered in {"q", "quit", "exit", "done", "x"}:
            return False
        if lowered in {"h", "?", "help"}:
            print(HELP)
            return True
        if lowered in {"l", "p", "list", "status"}:
            self.table()
            return True
        if lowered in {"g", "all", "regen"}:
            free = [item["index"] for item in self.state["items"] if not self.group_of(item["index"])]
            grouped = len(self.state["items"]) - len(free)
            if grouped:
                print(f"skipping {grouped} line(s) held by a concat group")
            self.render_all(free, "all")
            return True
        if lowered in {"r", "reload"}:
            dirty = self.sync_items(self.read_lines())
            save_state(self.state_path, self.state)
            for note in self.notes:
                print("note: " + note)
            if dirty:
                self.render_all(dirty, "reload")
            elif not self.notes:
                print("in.txt unchanged")
            return True

        # c / d take a selection: n, n-m, a,b,c or a (a = every line, or every group)
        selection = re.match(r"^(?P<kw>[a-z]+)(?P<spec>.*)$", command, re.IGNORECASE)
        if selection:
            keyword = selection.group("kw").casefold()
            spec = selection.group("spec").strip()
            if keyword in {"c", "concat"}:
                self.do_concat(spec)
                return True
            if keyword in {"d", "deconcat", "deconcatenate", "unconcat"}:
                self.do_unconcat(spec)
                return True

        match = COMMAND_RE.match(command)
        if not match:
            print(f"unrecognized: {command!r}  (n, +n, -n, +-n, c/d selection; h for help)")
            return True
        toggle = match.group("s") is not None
        sign = match.group("sign")
        delta = 0 if sign == "+-" else (-1 if sign == "-" else 1)
        number = int(match.group("index"))
        try:
            item = self.item(number)
        except IndexError:
            print(f"no line {number}; 1..{len(self.state['items'])}")
            return True
        owner = self.group_of(number)
        if owner:
            print(f"line {number} is inside concat group {owner['group']} ({owner['file']}); "
                  f"d {owner['group']} puts its files back before you can reroll it")
            return True
        if toggle:
            item["string_mode"] = not item["string_mode"]
        item["seed"] += delta
        if delta:
            item["rerolls"] = int(item.get("rerolls", 0)) + 1
        reason = {1: "reroll", 0: "restring", -1: "seed-down"}[delta]
        if toggle:
            reason += "+s"
        self.render(item, reason)
        return True


# index command: optional `s`, then an optional seed step, then the line number.
# `+-` means "no seed change" - the only delta left is the s toggle.
COMMAND_RE = re.compile(
    r"^(?P<s>s)?\s*(?P<sign>\+-|\+|-)?\s*(?P<index>\d+)$", re.IGNORECASE
)
HELP = """commands (one per line):
  +/- reroll:
    n, +n      regenerate line n at seed + 1
    -n         regenerate line n at seed - 1 (down a seed step)
    +-n        reroll in place: the same seed
  s : toggle string mode - combine with the above (s3, s+-3, s-2). Capital and
      digit runs become spoken words (see ee vee / one two three) instead of
      quoted letters ("C" "E" "V" / "1" "2" "3").
  c / d : concat / de-concat buttons. Use a range or comma delimited list, or
      even a comma delimited range; 'a' means all files (e.g. 'c a').
        c 4,5,6,9   c 4-11   c 4-6,9   c a   -> 001-concat.wav, 002-concat.wav ...
        d 1   d 1-2   d a                    -> members come back byte-for-byte
      A grouped line is tracked by its group number: reroll 5 only after
      d <group> has expanded 005-longcat.wav back out.
  r : re-read in.txt and render new or changed lines
  g : regenerate every un-grouped line at its current seed
  l : list the lines and the concat groups
  h : this help
  q : quit (done, x, Ctrl-D work too), keeping ./output as it stands
"""


# --------------------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ./in.txt into numbered LongCat WAVs and reroll them by index.",
        epilog=HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=HERE / "rollcat.toml", help="TOML config (default: rollcat.toml beside the script)")
    parser.add_argument("--in", dest="input", help="input text file (default: config rollcat.input, else in.txt)")
    parser.add_argument("--out", dest="output", help="output directory (default: config rollcat.output, else output)")
    parser.add_argument("--url", help="LongCat base URL (default: from config, else http://127.0.0.1:8230)")
    parser.add_argument("--seed", type=int, help="base seed for brand new lines (default: config seed, else 1024)")
    parser.add_argument("--voice", help="reference stem (config, voice2, ...) or a configured vox voice profile name")
    parser.add_argument("--no-voice", action="store_true", help="zero-shot: send text with no reference audio")
    parser.add_argument("--overwrite", action="store_true", help="wipe prior output without prompting")
    parser.add_argument("--resume", action="store_true", help="keep prior output without prompting")
    parser.add_argument("--preview", action="store_true", help="print the corrected text for every line and exit (no HTTP)")
    parser.add_argument("--force", action="store_true", help="re-render lines that already have a WAV")
    parser.add_argument("--no-ask", action="store_true", help="never prompt; go straight to the command loop")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings, sources = load_config(args.config)
    if args.url:
        settings["longcat"]["base_url"] = args.url.rstrip("/")
        sources.append("--url")
    longcat = settings["longcat"]
    voice = None if args.preview else pick_voice(args, settings, list(dict.fromkeys([HERE, Path.cwd()])))

    roller = Roller(args, settings, voice)
    print(f"config: {' > '.join(sources) or 'built-in defaults'}")
    print(f"service: {longcat['base_url']}  seed={roller.seed_base()}  steps={longcat['steps']} "
          f"{longcat['guidance_method']} g={longcat['guidance_strength']} scale={longcat['duration_scale']}")
    voice_line = (
        f"{voice['name']}  <- {voice['audio_path'].name} + {voice['transcript_path'].name} "
        f"in {voice['audio_path'].parent}"
        if voice
        else "none (zero-shot)"
    )
    print(f"voice: {'not needed for --preview' if args.preview else voice_line}")
    print(f"input: {roller.input_path}\noutput: {roller.output}")

    lines = roller.read_lines()
    roller.output.mkdir(parents=True, exist_ok=True)

    if args.preview:
        for number, line in enumerate(lines, start=1):
            print(f"{number:>4} plain  {transform(line, 'plain')}")
            print(f"{'':>4} string {transform(line, 'string')}")
        return

    prior = load_state(roller.state_path)
    media = roller.media_files()
    if prior or media:
        if args.overwrite:
            choice = "o"
        elif args.resume or not sys.stdin.isatty():
            choice = "r"
        else:
            print(f"output holds {len(media)} WAV(s)" + (" and a prior dat.json" if prior else ""))
            answer = input("resume from prior state, or overwrite ./output destructively? [R/resume, O/overwrite] ").strip().casefold()
            choice = "o" if answer.startswith("o") else "r"
        if choice == "o":
            print(f"removed {wipe_output(roller.output, (roller.stem, roller.concat_stem))} file(s) from {roller.output}")
            prior = None
    roller.state = prior or {}

    probe(longcat["base_url"], float(settings["rollcat"]["status_timeout"]), float(settings["rollcat"]["load_timeout"]))

    dirty = roller.sync_items(lines)
    for note in roller.notes:
        print("note: " + note)
    if args.force:
        dirty = [item["index"] for item in roller.state["items"] if not roller.group_of(item["index"])]
    save_state(roller.state_path, roller.state)
    if dirty:
        print(f"rendering {len(dirty)} line(s) of {len(lines)}: {', '.join(str(n) for n in dirty)}")
        roller.render_all(dirty, "init")
    elif not roller.notes:
        print("everything already rendered")
    save_state(roller.state_path, roller.state)

    if args.no_ask:
        return
    print(HELP)
    while True:
        try:
            line = input("rollcat> ")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break
        if not roller.apply_command(line):
            break
    print(f"done: {roller.output}  state: {roller.state_path.name}")


if __name__ == "__main__":
    main()
