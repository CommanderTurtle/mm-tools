"""Chrisper v2 refine: re-time lyric lines from local whisper word timing.

The v1 pass places lyric lines on the rendered take's vocal-activity
envelope. This module asks the resident CrisperWhisper HTTP service
(``whisper``, ``CW2_PORT``, default 8172) to transcribe short snippets of
the take and realigns each line to the word timing it reports. Snippets
keep every transcription call bounded regardless of track length; at most
``MAX_PASSES`` passes run, spaced across the full duration.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

SNIPPET_WINDOW = 2.5  # seconds per transcription pass
MAX_PASSES = 20       # hard cap on snippet count
MIN_PASS_GAP = 10.0   # seconds between snippet centers
SEARCH_RADIUS = 12.0  # seconds around a line's baseline time
MAX_SKIPS = 2         # unmatched whisper words tolerated between two hits


DEFAULT_BASE = "http://127.0.0.1:8172"


def _discovery_urls() -> list[str]:
    """Live CrisperWhisper instances announced by whisper's launchers.

    Each launcher writes ``crisperwhisper-<role>-<uid>.json`` under
    $XDG_RUNTIME_DIR (or /tmp) while its server runs; the machine service
    (``starthttp.sh``) is preferred over the browser workbench
    (``startwithuv.sh``) when both are live.
    """
    base_dir = Path(os.getenv("XDG_RUNTIME_DIR") or "/tmp")
    urls: list[str] = []
    for role in ("http", "ui"):
        try:
            payload = json.loads((base_dir / f"crisperwhisper-{role}-{os.getuid()}.json").read_text(encoding="utf-8"))
            port = int(payload["port"])
            scheme = str(payload.get("scheme") or "http").lower()
            if scheme not in {"http", "https"}:
                scheme = "http"
            urls.append(f"{scheme}://127.0.0.1:{port}")
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return urls


def candidate_urls() -> list[str]:
    """Candidate bases, best first: an explicit ``CW2_URL`` override, then the
    live-discovered instances, then ``CW2_PORT``, then the documented default."""
    override = os.getenv("CW2_URL", "").strip().rstrip("/")
    if override:
        return [override]
    urls = _discovery_urls()
    raw_port = os.getenv("CW2_PORT", "").strip()
    if raw_port.isdigit():
        urls.append(f"http://127.0.0.1:{raw_port}")
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls or [DEFAULT_BASE]:
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


def base_url() -> str:
    return candidate_urls()[0]


def _port_of(url: str) -> int:
    parsed = urllib.parse.urlparse(url)
    return parsed.port or (443 if parsed.scheme == "https" else 80)


def status() -> dict[str, Any]:
    """Probe every candidate without touching the model; the first answer wins."""
    tried = candidate_urls()
    for url in tried:
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=2) as handle:
                payload = json.loads(handle.read().decode("utf-8"))
            return {
                "up": True,
                "loaded": bool(payload.get("loaded")),
                "model_present": bool(payload.get("model_present", True)),
                "url": url,
                "port": _port_of(url),
                "tried": tried,
            }
        except Exception:
            continue
    return {
        "up": False,
        "loaded": False,
        "model_present": False,
        "url": tried[0],
        "port": _port_of(tried[0]),
        "tried": tried,
    }


def plan_snippets(duration: float) -> list[tuple[float, float]]:
    """Bounded sample points: up to ``MAX_PASSES`` windows spread across the take."""
    if not isinstance(duration, (int, float)) or duration <= 0:
        return []
    step = max(MIN_PASS_GAP, duration / MAX_PASSES)
    window = min(SNIPPET_WINDOW, max(1.0, duration))
    starts: list[float] = []
    center = min(MIN_PASS_GAP, duration / 2.0)
    while center < duration - window / 2.0 or not starts:
        starts.append(center)
        if len(starts) >= MAX_PASSES:
            break
        center += step
    spans: list[tuple[float, float]] = []
    for center in starts:
        start = max(0.0, min(center - window / 2.0, max(0.0, duration - window)))
        end = min(duration, start + window)
        if end - start >= 0.8:
            spans.append((round(start, 3), round(end, 3)))
    return spans


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_TAG_RE = re.compile(r"\[.+\]")


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(str(text))]


def slice_wav(source: Path, start: float, end: float, dest: Path) -> None:
    """Write one bounded region of the take as a 16-bit PCM wav."""
    import soundfile as sf

    info = sf.info(source)
    frame_start = max(0, int(round(start * info.samplerate)))
    frames = max(1, int(round(end * info.samplerate)) - frame_start)
    data, _ = sf.read(source, start=frame_start, frames=frames, always_2d=True)
    from studio_api import write_pcm16_wav

    write_pcm16_wav(dest, data, int(info.samplerate))


def _post_multipart(url: str, fields: dict[str, str], filename: str, payload: bytes, timeout: float = 120.0) -> dict[str, Any]:
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode("utf-8")
    body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: audio/wav\r\n\r\n".encode("utf-8")
    body += payload
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")
    request = urllib.request.Request(
        url,
        data=bytes(body),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as handle:
        return json.loads(handle.read().decode("utf-8"))


def detect_language(audio: Path, base: str) -> str | None:
    try:
        payload = _post_multipart(f"{base}/api/detect-language", {}, f"detect{audio.suffix}", audio.read_bytes(), timeout=60.0)
        code = str(payload.get("language") or "").strip().lower()
        return code or None
    except Exception:
        return None


def transcribe_snippet(path: Path, offset: float, language: str, hotwords: str, base: str) -> list[dict[str, Any]]:
    """Transcribe one snippet; return its words rebased onto full-track time."""
    payload = _post_multipart(
        f"{base}/api/transcribe",
        {
            "operation": "verbatim",
            "language": language,
            "word_timestamps": "true",
            "hotwords": str(hotwords)[:4000],
        },
        f"snippet{path.name}",
        path.read_bytes(),
    )
    entry = (payload.get("results") or {}).get("verbatim") or {}
    rows: list[dict[str, Any]] = []
    for word in entry.get("words") or []:
        if not isinstance(word, dict):
            continue
        start = word.get("start")
        if not isinstance(start, (int, float)):
            continue
        rows.append({"word": str(word.get("word") or ""), "start": offset + float(start)})
    return rows


def _match_score(tokens: list[str], window: list[tuple[str, float]], pos: int) -> tuple[int, int]:
    """Score one candidate alignment: (hits, skips consumed)."""
    hits = 1
    cursor = pos
    expected = 1
    skips = 0
    while expected < len(tokens):
        advanced = False
        limit = min(len(window), cursor + 1 + MAX_SKIPS + 1)
        for candidate in range(cursor + 1, limit):
            word = window[candidate][0]
            target = tokens[expected]
            if word == target or (len(target) >= 4 and word.startswith(target[:4])):
                hits += 1
                expected += 1
                cursor = candidate
                advanced = True
                break
            skips += 1
        if not advanced:
            break
    return hits, skips


def _locate(tokens: list[str], anchor: float, words: list[tuple[str, float]]) -> float | None:
    """Best matching start time for one line near its baseline position."""
    if not tokens:
        return None
    window = [(word, start) for word, start in words if anchor - SEARCH_RADIUS <= start <= anchor + SEARCH_RADIUS]
    if not window:
        return None
    first = tokens[0]
    best: tuple[int, int, float] | None = None  # (hits, -skips, start)
    for pos, (word, start) in enumerate(window):
        if word != first and not (len(first) >= 4 and word.startswith(first[:4])):
            continue
        hits, skips = _match_score(tokens, window, pos)
        threshold = 1 if len(tokens) == 1 else max(2, int(round(len(tokens) * 0.6)))
        if hits < threshold:
            continue
        score = (hits, -skips, -start)
        if best is None or score > (best[0], best[1], best[2]):
            best = (hits, -skips, start)
    return best[2] if best else None


def realign(
    lines: list[str],
    baseline: list[dict[str, Any]],
    words: list[dict[str, Any]],
    duration: float,
) -> tuple[list[dict[str, Any]], int]:
    """Re-time non-tag lyric lines against whisper words.

    Baseline rows ({line, start}) seed each search window so a hallucinated
    snippet can never drag a line across the track. Lines with no confident
    match keep their baseline (or uniform) time. Returns (rows, matched).
    """
    timed = [i for i, line in enumerate(lines) if line.strip() and not _TAG_RE.fullmatch(line.strip())]
    if not timed:
        return [], 0
    by_line: dict[int, float] = {}
    for row in baseline or []:
        if isinstance(row, dict) and isinstance(row.get("line"), int) and isinstance(row.get("start"), (int, float)):
            by_line[row["line"]] = float(row["start"])
    fallback_step = max(0.5, duration / max(1, len(timed)))
    global_words = [(str(w.get("word") or "").lower(), float(w.get("start") or 0.0)) for w in words or [] if isinstance(w, dict) and w.get("word")]
    rows: list[dict[str, Any]] = []
    matched = 0
    previous = 0.0
    for index in timed:
        anchor = by_line.get(index, fallback_step * index)
        tokens = _tokens(lines[index])
        located = _locate(tokens, anchor, global_words)
        if located is None:
            start = anchor
        else:
            start = max(0.0, located - 0.05)
            matched += 1
        start = max(start, previous)
        previous = start
        rows.append({"line": index, "start": round(start, 3)})
    return rows, matched


def refine(
    audio: Path,
    duration: float,
    lines: list[str],
    baseline: list[dict[str, Any]],
    hotwords: str,
    language: str,
    work_dir: Path,
    base: str | None = None,
) -> dict[str, Any]:
    """Run the full v2 pass: snippet plan, whisper, realignment."""
    resolved_base = base or base_url()
    work_dir = Path(work_dir)
    spans = plan_snippets(duration)
    if not spans:
        raise ValueError("Track is too short to refine.")
    resolved = str(language or "auto").strip().lower()
    if resolved in {"", "auto"}:
        resolved = ""
    words: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(spans):
        snippet = work_dir / f"snippet_{index:02d}.wav"
        slice_wav(audio, start, end, snippet)
        if not resolved:
            resolved = detect_language(snippet, resolved_base) or "en"
        words.extend(transcribe_snippet(snippet, start, resolved, hotwords, resolved_base))
    rows, matched = realign(lines, baseline, words, duration)
    return {
        "rows": rows,
        "language": resolved,
        "snippets": len(spans),
        "matched": matched,
        "total": len(rows),
        "words": len(words),
    }
