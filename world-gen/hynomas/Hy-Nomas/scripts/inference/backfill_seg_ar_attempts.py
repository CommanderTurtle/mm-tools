#!/usr/bin/env python3
"""Build Dashboard attempt manifests from legacy SegAR output metadata."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from apps.seg_ar_attempt_archive import (
    UserAttemptArchive,
    build_user_attempt_index,
)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def transcript_map(root: Path) -> dict[str, dict[str, Any]]:
    values = {}
    for path in root.rglob("pe_transcripts/*.json"):
        value = load_json(path)
        if value is None:
            continue
        session_id = str(value.get("session_id") or "").strip()
        if session_id:
            values[session_id] = {"path": path, "value": value}
    return values


def turn_input(turn: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(turn, dict):
        return {}
    direct = turn.get("input_request")
    if isinstance(direct, dict):
        return direct
    message = turn.get("user_message")
    if not isinstance(message, dict):
        return {}
    content = message.get("content")
    if not isinstance(content, list):
        return {}
    candidates = []
    decoder = json.JSONDecoder()
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        text = str(item.get("text") or "")
        for match in re.finditer(r"\{", text):
            try:
                value, _ = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                continue
            keys = set(value)
            if keys & {
                "perspective_view",
                "subject_description",
                "environment_description",
                "subsequent_text_prompt",
                "action",
            }:
                candidates.append(value)
    return candidates[-1] if candidates else {}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value)


def choose_video(metadata_path: Path, metadata: dict[str, Any]) -> Path | None:
    output_dir = metadata_path.parent
    session_id = str(metadata.get("session_id") or "").strip()
    request_id = str(metadata.get("request_id") or "").strip()
    if session_id and request_id:
        cumulative = output_dir / (
            f"session_{safe_name(session_id)}_{safe_name(request_id)}.mp4"
        )
        if cumulative.is_file() and cumulative.stat().st_size > 0:
            return cumulative
    filename = str(metadata.get("video_filename") or "").strip()
    generated = output_dir / filename if filename else metadata_path.with_suffix(".mp4")
    if generated.is_file() and generated.stat().st_size > 0:
        return generated
    return None


def backfill(root: Path, *, dry_run: bool = False) -> dict[str, int]:
    root = root.expanduser().resolve()
    transcripts = transcript_map(root)
    by_session: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("gen_*.json"):
        metadata = load_json(path)
        if metadata is None:
            continue
        session_id = str(metadata.get("session_id") or "").strip()
        if session_id:
            by_session[session_id].append(path)

    counts = {
        "sessions": len(by_session),
        "metadata": 0,
        "written": 0,
        "missing_video": 0,
        "invalid_metadata": 0,
    }
    for session_id, paths in sorted(by_session.items()):
        paths.sort(key=lambda path: (path.stat().st_mtime, path.name))
        transcript_entry = transcripts.get(session_id) or {}
        transcript = transcript_entry.get("value")
        turns = (
            list(transcript.get("turns") or []) if isinstance(transcript, dict) else []
        )
        turns.sort(
            key=lambda turn: (
                (
                    int(turn.get("record_index") or 0),
                    float(turn.get("created_at") or 0.0),
                )
                if isinstance(turn, dict)
                else (0, 0.0)
            )
        )
        for index, metadata_path in enumerate(paths):
            metadata = load_json(metadata_path)
            if metadata is None:
                counts["invalid_metadata"] += 1
                continue
            counts["metadata"] += 1
            video = choose_video(metadata_path, metadata)
            if video is None:
                counts["missing_video"] += 1
                continue
            turn = turns[index] if index < len(turns) else None
            assistant_response = (
                str(turn.get("assistant_response") or "")
                if isinstance(turn, dict)
                else ""
            )
            parsed_response = (
                turn.get("assistant_parsed_response")
                if isinstance(turn, dict)
                else None
            )
            enhanced_prompt = assistant_response or (
                json.dumps(parsed_response, ensure_ascii=False)
                if parsed_response is not None
                else str(metadata.get("prompt") or "")
            )
            created_at = metadata_path.stat().st_mtime
            request_id = str(metadata.get("request_id") or metadata_path.stem)
            output_dir = metadata_path.parent
            slot_id = (
                output_dir.name
                if re.fullmatch(r"sp16-\d+", output_dir.name)
                else root.name
            )
            relative_video = video.relative_to(output_dir).as_posix()
            relative_metadata = metadata_path.relative_to(output_dir).as_posix()
            pe_input = turn_input(turn)
            pe_session = {
                "session_id": session_id,
            }
            transcript_path = transcript_entry.get("path")
            if transcript_path is not None:
                pe_session["pe_transcript_path"] = str(transcript_path)
            job = {
                "id": request_id,
                "kind": "generate",
                "status": "succeeded",
                "created_at": created_at,
                "updated_at": created_at,
                "started_at": None,
                "finished_at": created_at,
                "payload": {
                    "_user_id": "_unknown_user",
                    "_dispatch_host_id": slot_id,
                    "_pe_session": pe_session,
                    "pe_input": pe_input,
                    "user_prompts": metadata.get("user_prompts") or [],
                    "carry_user_prompt": metadata.get("carry_user_prompt") or "",
                    "camera_trajectory": metadata.get("camera_trajectory"),
                    "num_frames": metadata.get("num_frames"),
                    "segment_boundaries": metadata.get("segment_boundaries"),
                    "segment_has_camera": metadata.get("segment_has_camera"),
                    "seed": metadata.get("seed"),
                    "width": metadata.get("width"),
                    "height": metadata.get("height"),
                    "prompt": metadata.get("prompt") or "",
                },
                "state": {
                    "status": "succeeded",
                    "user_id": "_unknown_user",
                    "dispatch_host_id": slot_id,
                    "prompt_enhancement_mode": (
                        turn.get("enhancement_mode") if isinstance(turn, dict) else None
                    ),
                    "pe_request_id": (
                        turn.get("request_id") if isinstance(turn, dict) else None
                    ),
                    "pe_prompt": enhanced_prompt,
                    "pe_transcript_path": (
                        str(transcript_path) if transcript_path is not None else None
                    ),
                    "video_url": f"outputs/{relative_video}",
                    "result": {
                        "id": request_id,
                        "status": "succeeded",
                        "video_url": f"outputs/{relative_video}",
                        "metadata_url": f"outputs/{relative_metadata}",
                        "frames": metadata.get("frame_count"),
                        "fps": metadata.get("fps"),
                        "duration_seconds": metadata.get("duration_seconds"),
                        "generation_seconds": metadata.get("generation_seconds"),
                        "encoding_seconds": metadata.get("encoding_seconds"),
                    },
                },
            }
            if not dry_run:
                UserAttemptArchive(
                    output_dir,
                    directory=root / "user_attempts",
                ).write(job)
            counts["written"] += 1
    if not dry_run:
        build_user_attempt_index(root / "user_attempts")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()
    for value in args.roots:
        root = Path(value)
        if args.index_only:
            index = build_user_attempt_index(root / "user_attempts")
            print(
                json.dumps(
                    {
                        "root": str(root),
                        "index_only": True,
                        "indexed": index["count"],
                    },
                    ensure_ascii=False,
                )
            )
            continue
        print(
            json.dumps(
                {"root": str(root), **backfill(root, dry_run=args.dry_run)},
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
