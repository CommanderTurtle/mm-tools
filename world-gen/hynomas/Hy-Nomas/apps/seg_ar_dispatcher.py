"""Frontend gateway that leases and reverse-proxies eight GPU hosts."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import secrets
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import requests
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apps.seg_ar_attempt_archive import (
    list_user_attempts,
)
from apps.seg_ar_gateway import (
    GalleryStorageBusyError,
    add_gallery_item_from_source,
    read_hidden_gallery_item_ids,
    read_gallery_items,
    remove_gallery_item,
)
from apps.seg_ar_resource_manager import (
    ResourceBusyError,
    ResourceManager,
    SessionBindingConflictError,
)

logger = logging.getLogger("seg_ar_dispatcher")
SESSION_COOKIE = "nomas_dispatch_session"
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
INTERNAL_IDENTITY_HEADERS = {
    "x-nomas-user-id",
    "x-nomas-host-id",
}
EXPLICIT_SESSION_RELEASE_HEADER = "x-nomas-explicit-session-release"
SESSION_RELEASE_PATH_RE = re.compile(r"v1/sessions/[^/]+/release")
CASE_DOWNLOAD_RENDERER_VERSION = 1
CASE_DEFAULT_TURN_SECONDS = 5.0
DEFAULT_HEALTH_REQUEST_TIMEOUT_SECONDS = 45.0
CASE_CAMERA_KEYS = {
    "KeyW",
    "KeyA",
    "KeyS",
    "KeyD",
    "ArrowUp",
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
}


def _generation_session_id(body: bytes) -> str:
    """Extract the durable World id from a generation request body."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    pe_input = payload.get("pe_input")
    nested_session_id = (
        pe_input.get("session_id") if isinstance(pe_input, dict) else None
    )
    return str(nested_session_id or payload.get("session_id") or "").strip()


def _upstream_headers(
    incoming: Any,
    *,
    user_id: str,
    host_id: str,
) -> dict[str, str]:
    headers = {
        key: value
        for key, value in incoming.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
        and key.lower()
        not in {
            "host",
            "cookie",
            "content-length",
            *INTERNAL_IDENTITY_HEADERS,
        }
    }
    headers["X-Nomas-User-Id"] = quote(user_id, safe="")
    headers["X-Nomas-Host-Id"] = host_id
    return headers


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _case_attempt_prompt(attempt: dict[str, Any]) -> str:
    request = attempt.get("request")
    request = request if isinstance(request, dict) else {}
    prompts = attempt.get("prompts")
    prompts = prompts if isinstance(prompts, dict) else {}
    original_input = prompts.get("original_user_input")
    original_input = original_input if isinstance(original_input, dict) else {}

    user_prompts = request.get("user_prompts")
    if isinstance(user_prompts, list):
        for value in user_prompts:
            prompt = str(value or "").strip()
            if prompt:
                return prompt[:8000]

    action = str(original_input.get("action") or "").strip()
    if action:
        return action[:8000]

    originals = prompts.get("original_user_prompts")
    if isinstance(originals, list):
        for value in reversed(originals):
            prompt = str(value or "").strip()
            if prompt:
                return prompt[:8000]

    environment = str(original_input.get("environment_description") or "").strip()
    subject = str(original_input.get("subject_description") or "").strip()
    if environment and subject:
        return f"{subject} · {environment}"[:8000]
    if environment or subject:
        return (environment or subject)[:8000]

    carry_prompt = str(
        request.get("carry_user_prompt") or prompts.get("carry_user_prompt") or ""
    ).strip()
    if carry_prompt:
        return carry_prompt[:8000]
    return str(prompts.get("final_model_prompt") or "").strip()[:8000]


def _case_camera_overlay(
    attempt: dict[str, Any],
    *,
    fps: float,
) -> dict[str, Any] | None:
    request = attempt.get("request")
    request = request if isinstance(request, dict) else {}
    raw_value = request.get("camera_trajectory")
    if isinstance(raw_value, str):
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            return None
    else:
        value = raw_value
    if not isinstance(value, dict):
        return None

    raw_frames = value.get("frames")
    if not isinstance(raw_frames, list):
        return None
    sample_rate = _finite_number(value.get("sampleRate"))
    if sample_rate is None or sample_rate <= 0:
        sample_rate = max(fps, 0.001)
    frames = []
    for index, raw_frame in enumerate(raw_frames[:12000]):
        if not isinstance(raw_frame, dict):
            continue
        position = raw_frame.get("position")
        quaternion = raw_frame.get("quaternion")
        if not isinstance(position, dict) or not isinstance(
            quaternion,
            dict,
        ):
            continue
        coordinates = {
            axis: _finite_number(position.get(axis)) for axis in ("x", "y", "z")
        }
        rotation = {
            axis: _finite_number(quaternion.get(axis)) for axis in ("x", "y", "z", "w")
        }
        if any(value is None for value in coordinates.values()) or any(
            value is None for value in rotation.values()
        ):
            continue
        timestamp = _finite_number(raw_frame.get("t"))
        if timestamp is None:
            timestamp = index * 1000.0 / sample_rate
        field_of_view = _finite_number(raw_frame.get("fov"))
        frames.append(
            {
                "frame": int(_finite_number(raw_frame.get("frame")) or index),
                "t": max(0.0, timestamp),
                "position": coordinates,
                "quaternion": rotation,
                "fov": field_of_view if field_of_view is not None else 55.0,
            }
        )
    if not frames:
        return None

    key_events = []
    raw_events = value.get("keyEvents")
    if isinstance(raw_events, list):
        for raw_event in raw_events[:24000]:
            if not isinstance(raw_event, dict):
                continue
            timestamp = _finite_number(raw_event.get("t"))
            code = str(raw_event.get("code") or "")
            if timestamp is None or timestamp < 0 or code not in CASE_CAMERA_KEYS:
                continue
            key_events.append(
                {
                    "t": timestamp,
                    "code": code,
                    "pressed": raw_event.get("pressed") is True,
                }
            )
    key_events.sort(key=lambda event: event["t"])
    duration_ms = _finite_number(value.get("durationMs"))
    if duration_ms is None or duration_ms <= 0:
        duration_ms = max(
            1000.0 / sample_rate,
            float(frames[-1]["t"]) - float(frames[0]["t"]) + 1000.0 / sample_rate,
        )
    return {
        "name": str(value.get("name") or "Camera path")[:160],
        "duration_ms": duration_ms,
        "sample_rate": sample_rate,
        "frames": frames,
        "key_events": key_events,
    }


def _case_attempt_timing(
    attempt: dict[str, Any],
    *,
    previous_end: float,
) -> tuple[float, float, float]:
    result = attempt.get("result")
    result = result if isinstance(result, dict) else {}
    request = attempt.get("request")
    request = request if isinstance(request, dict) else {}

    fps = _finite_number(result.get("fps"))
    if fps is None or fps <= 0:
        fps = 24.0
    end = _finite_number(result.get("duration_seconds"))
    frames = _finite_number(result.get("frames"))
    if (end is None or end <= 0) and frames is not None and frames > 0:
        end = frames / fps

    start = _finite_number(result.get("playback_start_seconds"))
    if start is None or start < 0:
        segment_frames = _finite_number(result.get("segment_frames"))
        if end is not None and segment_frames is not None and segment_frames > 0:
            start = max(0.0, end - segment_frames / fps)
        else:
            start = max(0.0, previous_end)

    if end is None or end <= start:
        request_frames = _finite_number(request.get("num_frames"))
        boundaries = request.get("segment_boundaries")
        if isinstance(boundaries, list):
            finite_boundaries = [
                value
                for raw in boundaries
                if (value := _finite_number(raw)) is not None
            ]
            if finite_boundaries:
                request_frames = max(finite_boundaries)
        if request_frames is not None and request_frames > 0:
            end = start + request_frames / fps
        else:
            end = start + CASE_DEFAULT_TURN_SECONDS
    if end <= start:
        end = start + max(1.0 / fps, 0.04)
    return start, end, fps


def _build_case_overlay(
    attempts: list[dict[str, Any]],
    *,
    selected_job_id: str,
    theme: str = "dark",
) -> dict[str, Any]:
    """Rebuild the share/download timeline from archived generation records."""
    ordered = sorted(
        attempts,
        key=lambda item: float(item.get("updated_at") or item.get("created_at") or 0.0),
    )
    selected_index = next(
        (
            index
            for index, attempt in enumerate(ordered)
            if str(attempt.get("job_id") or "") == selected_job_id
        ),
        len(ordered) - 1,
    )
    if selected_index > 0:
        selected_result = ordered[selected_index].get("result")
        selected_result = selected_result if isinstance(selected_result, dict) else {}
        selected_duration = _finite_number(selected_result.get("duration_seconds"))
        selected_frames = _finite_number(selected_result.get("frames"))
        selected_fps = _finite_number(selected_result.get("fps"))
        if (
            (selected_duration is None or selected_duration <= 0)
            and selected_frames is not None
            and selected_frames > 0
            and selected_fps is not None
            and selected_fps > 0
        ):
            selected_duration = selected_frames / selected_fps
        prior_durations = []
        for prior in ordered[:selected_index]:
            prior_result = prior.get("result")
            prior_result = prior_result if isinstance(prior_result, dict) else {}
            duration = _finite_number(prior_result.get("duration_seconds"))
            frames = _finite_number(prior_result.get("frames"))
            fps = _finite_number(prior_result.get("fps"))
            if (
                (duration is None or duration <= 0)
                and frames is not None
                and frames > 0
                and fps is not None
                and fps > 0
            ):
                duration = frames / fps
            if duration is not None and duration > 0:
                prior_durations.append(duration)
        if (
            selected_result.get("playback_start_seconds") is None
            and selected_result.get("segment_frames") is None
            and selected_duration is not None
            and prior_durations
            and selected_duration <= max(prior_durations) + 0.001
        ):
            # Older archives may point every turn at a standalone segment
            # instead of a cumulative World MP4. In that case, showing only
            # the selected turn is more accurate than placing prior prompts
            # beyond the end of the video.
            ordered = [ordered[selected_index]]
            selected_index = 0
    turns = []
    previous_end = 0.0
    for attempt in ordered[: selected_index + 1]:
        start, end, fps = _case_attempt_timing(
            attempt,
            previous_end=previous_end,
        )
        previous_end = max(previous_end, end)
        camera = _case_camera_overlay(attempt, fps=fps)
        prompt = _case_attempt_prompt(attempt)
        if camera is not None and not prompt:
            prompt = "Generate from the selected camera path."
        if not prompt and camera is None:
            continue
        turn = {
            "number": len(turns) + 1,
            "prompt": prompt,
            "start_seconds": start,
            "end_seconds": end,
        }
        if camera is not None:
            turn["camera"] = camera
        turns.append(turn)
    return {
        "version": 1,
        **({"theme": theme} if theme in {"dark", "light"} else {}),
        "turns": turns,
    }


def _normalize_case_overlay(value: object) -> dict[str, Any]:
    # Keep the Cases archive adapter exactly aligned with the Gateway contract
    # without moving or duplicating the renderer/API implementation.
    from apps.seg_ar_gateway import _normalize_share_overlay

    try:
        return _normalize_share_overlay(value)
    except ValueError:
        prompt_only = dict(value) if isinstance(value, dict) else {}
        prompt_only["turns"] = [
            {
                key: item[key]
                for key in (
                    "number",
                    "prompt",
                    "start_seconds",
                    "end_seconds",
                )
                if key in item
            }
            for item in prompt_only.get("turns", [])
            if isinstance(item, dict)
        ]
        return _normalize_share_overlay(prompt_only)


def _case_shared_video_page_html(share_id: str) -> str:
    from apps.seg_ar_gateway import _shared_video_page_html

    return _shared_video_page_html(share_id)


def _burn_case_overlay_video(
    source: Path,
    destination: Path,
    overlay: dict[str, Any],
) -> dict[str, Any]:
    from apps.camera_video_overlay import burn_shared_overlay_video

    return burn_shared_overlay_video(source, destination, overlay)


def create_dispatcher_app(
    manager: ResourceManager,
    *,
    frontend_dir: Path | None,
    attempt_dir: Path | None = None,
    attempt_dirs: list[Path] | None = None,
    gallery_dir: Path | None = None,
    health_interval: float = 5,
    health_request_timeout: float = DEFAULT_HEALTH_REQUEST_TIMEOUT_SECONDS,
    proxy_timeout: float = 3600,
    disconnect_grace_seconds: float = 3,
) -> FastAPI:
    app = FastAPI(title="NOMAS multi-host dispatcher", version="1.0")
    if health_request_timeout <= 0:
        raise ValueError("health_request_timeout must be positive")
    if gallery_dir is not None:
        gallery_dir.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    host_states: dict[str, bool] = {}
    configured_attempt_dirs: list[Path] = []
    for value in [*(attempt_dirs or []), attempt_dir]:
        if value is None:
            continue
        resolved = value.expanduser().resolve()
        if resolved not in configured_attempt_dirs:
            configured_attempt_dirs.append(resolved)
    attempt_sources: dict[str, Path] = {}
    for index, directory in enumerate(configured_attempt_dirs):
        raw_source = (
            re.sub(
                r"[^A-Za-z0-9_.-]+",
                "-",
                directory.parent.name,
            ).strip(".-")
            or f"source-{index + 1}"
        )
        source_id = raw_source
        suffix = 2
        while source_id in attempt_sources:
            source_id = f"{raw_source}-{suffix}"
            suffix += 1
        attempt_sources[source_id] = directory
    cases_cache_lock = threading.RLock()
    cases_refresh_lock = threading.Lock()
    cases_cache: dict[str, Any] = {
        "loaded": False,
        "refreshing": False,
        "refreshed_at": None,
        "refresh_seconds": None,
        "error": None,
        "items": [],
        "sessions": [],
        "sources": [],
        "total": 0,
    }
    case_share_index_path = manager.path.with_suffix(".case_video_shares.json")
    case_share_lock = threading.RLock()
    case_download_lock = threading.RLock()
    gallery_write_lock = threading.RLock()
    case_download_dir = manager.path.parent / f"{manager.path.stem}_case_downloads"

    def load_case_shares() -> dict[str, dict[str, Any]]:
        if not case_share_index_path.is_file():
            return {}
        try:
            value = json.loads(case_share_index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def save_case_shares(shares: dict[str, dict[str, Any]]) -> None:
        case_share_index_path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_temporary = tempfile.mkstemp(
            prefix=f".{case_share_index_path.name}.",
            suffix=".tmp",
            dir=str(case_share_index_path.parent),
        )
        temporary = Path(raw_temporary)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(shares, handle, ensure_ascii=False, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, case_share_index_path)
        finally:
            temporary.unlink(missing_ok=True)

    def build_cases_snapshot() -> dict[str, Any]:
        started = time.monotonic()
        items: list[dict[str, Any]] = []
        sources = []
        total = 0
        for source_id, directory in attempt_sources.items():
            source_items = list_user_attempts(
                directory,
                limit=5000,
                source_id=source_id,
            )
            source_total = len(source_items)
            total += source_total
            sources.append(
                {
                    "id": source_id,
                    "label": directory.parent.name,
                    "attempt_dir": str(directory),
                    "count": source_total,
                }
            )
            items.extend(source_items)
        status_priority = {
            "succeeded": 4,
            "failed": 3,
            "cancelled": 3,
            "running": 2,
            "queued": 1,
            "enhancing": 1,
        }
        deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
        for item in items:
            session_id = str(item.get("session_id") or item.get("job_id") or "")
            job_id = str(item.get("job_id") or "")
            key = (session_id, job_id)
            existing = deduplicated.get(key)
            if existing is None:
                deduplicated[key] = item
                continue
            candidate_rank = (
                status_priority.get(str(item.get("status") or ""), 0),
                float(item.get("updated_at") or item.get("created_at") or 0.0),
            )
            existing_rank = (
                status_priority.get(str(existing.get("status") or ""), 0),
                float(existing.get("updated_at") or existing.get("created_at") or 0.0),
            )
            if candidate_rank > existing_rank:
                deduplicated[key] = item
        items = list(deduplicated.values())
        items.sort(
            key=lambda item: float(
                item.get("updated_at") or item.get("created_at") or 0.0
            ),
            reverse=True,
        )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            session_id = str(item.get("session_id") or item.get("job_id") or "")
            grouped.setdefault(session_id, []).append(item)
        sessions = []
        for session_id, attempts in grouped.items():
            attempts.sort(
                key=lambda item: float(
                    item.get("updated_at") or item.get("created_at") or 0.0
                )
            )
            latest = attempts[-1]
            user_ids = {
                str(attempt.get("user_id") or "").casefold() for attempt in attempts
            }
            searchable = json.dumps(
                {
                    "user_ids": list(user_ids),
                    "session_id": session_id,
                    "prompts": [attempt.get("prompts") for attempt in attempts],
                    "errors": [attempt.get("error") for attempt in attempts],
                },
                ensure_ascii=False,
            ).casefold()
            sessions.append(
                {
                    "session_id": session_id,
                    "updated_at": float(
                        latest.get("updated_at") or latest.get("created_at") or 0.0
                    ),
                    "latest_status": str(latest.get("status") or "").casefold(),
                    "user_ids": user_ids,
                    "searchable": searchable,
                    "attempts": attempts,
                }
            )
        sessions.sort(key=lambda session: session["updated_at"], reverse=True)
        return {
            "loaded": True,
            "refreshing": False,
            "refreshed_at": time.time(),
            "refresh_seconds": time.monotonic() - started,
            "error": None,
            "items": items,
            "sessions": sessions,
            "sources": sources,
            "total": total,
        }

    def refresh_cases_cache() -> bool:
        if not cases_refresh_lock.acquire(blocking=False):
            return False
        with cases_cache_lock:
            cases_cache["refreshing"] = True
        try:
            snapshot = build_cases_snapshot()
            with cases_cache_lock:
                cases_cache.clear()
                cases_cache.update(snapshot)
            logger.info(
                "cases cache refreshed: attempts=%d sessions=%d seconds=%.2f",
                len(snapshot["items"]),
                len(snapshot["sessions"]),
                float(snapshot["refresh_seconds"]),
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("cases cache refresh failed")
            with cases_cache_lock:
                cases_cache.update(
                    {
                        "refreshing": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            return False
        finally:
            cases_refresh_lock.release()

    def refresh_cases_cache_loop() -> None:
        while not stop_event.is_set():
            refresh_cases_cache()
            stop_event.wait(10)

    def case_attempt_context(
        source_id: str,
        job_id: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if source_id not in attempt_sources:
            raise HTTPException(status_code=404, detail="case not found")
        with cases_cache_lock:
            cache_loaded = bool(cases_cache["loaded"])
        if not cache_loaded:
            refresh_cases_cache()
        with cases_cache_lock:
            items = list(cases_cache["items"])
        selected = next(
            (
                item
                for item in items
                if str(item.get("source_id") or "") == source_id
                and str(item.get("job_id") or "") == job_id
            ),
            None,
        )
        if selected is None:
            raise HTTPException(status_code=404, detail="case not found")
        session_id = str(selected.get("session_id") or selected.get("job_id") or "")
        session_attempts = [
            item
            for item in items
            if str(item.get("source_id") or "") == source_id
            and str(item.get("session_id") or item.get("job_id") or "") == session_id
        ]
        return selected, session_attempts

    def resolve_case_video(
        source_id: str,
        attempt: dict[str, Any],
    ) -> Path:
        directory = attempt_sources.get(source_id)
        if directory is None:
            raise HTTPException(status_code=404, detail="case video not found")
        artifacts = attempt.get("artifacts")
        artifacts = artifacts if isinstance(artifacts, dict) else {}
        video_path = str(artifacts.get("video_path") or "").strip()
        if not video_path:
            raise HTTPException(status_code=404, detail="case video not found")
        deployment_root = directory.parent.resolve()
        candidate = Path(video_path).expanduser().resolve()
        if (
            candidate == deployment_root
            or deployment_root not in candidate.parents
            or candidate.suffix.lower() != ".mp4"
            or not candidate.is_file()
        ):
            raise HTTPException(status_code=404, detail="case video not found")
        return candidate

    def case_gallery_metadata(
        attempt: dict[str, Any],
        session_attempts: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        prompts = attempt.get("prompts")
        prompts = prompts if isinstance(prompts, dict) else {}
        original_input = prompts.get("original_user_input")
        original_input = original_input if isinstance(original_input, dict) else {}
        result = attempt.get("result")
        result = result if isinstance(result, dict) else {}
        request_payload = attempt.get("request")
        request_payload = request_payload if isinstance(request_payload, dict) else {}

        session_inputs: list[dict[str, Any]] = []
        for session_attempt in sorted(
            session_attempts,
            key=lambda item: float(
                item.get("created_at") or item.get("updated_at") or 0
            ),
        ):
            session_prompts = session_attempt.get("prompts")
            session_prompts = (
                session_prompts if isinstance(session_prompts, dict) else {}
            )
            session_original = session_prompts.get("original_user_input")
            if isinstance(session_original, dict):
                session_inputs.append(session_original)
            session_request = session_attempt.get("request")
            session_request = (
                session_request if isinstance(session_request, dict) else {}
            )
            session_pe_input = session_request.get("pe_input")
            if isinstance(session_pe_input, dict):
                session_inputs.append(session_pe_input)

        def first_session_value(key: str) -> str:
            return next(
                (
                    str(item.get(key) or "").strip()
                    for item in session_inputs
                    if str(item.get(key) or "").strip()
                ),
                "",
            )

        perspective = str(
            payload.get("perspective")
            or original_input.get("perspective")
            or first_session_value("perspective")
            or "third-person"
        ).strip()
        duration = result.get("duration_seconds")
        if duration in (None, ""):
            frames = _finite_number(result.get("frames"))
            fps = _finite_number(result.get("fps"))
            if frames is not None and fps is not None and fps > 0:
                duration = frames / fps
        subject = str(
            payload.get("subject")
            or original_input.get("subject_description")
            or first_session_value("subject_description")
            or ""
        ).strip()
        attributes = str(
            payload.get("attributes")
            or original_input.get("environment_description")
            or first_session_value("environment_description")
            or ""
        ).strip()
        direction = str(
            payload.get("direction") or _case_attempt_prompt(attempt) or ""
        ).strip()
        return {
            "title": payload.get("title") or subject or direction,
            "perspective": perspective,
            "subject": subject,
            "attributes": attributes,
            "direction": direction,
            "prompt_timeline": _build_case_overlay(
                session_attempts,
                selected_job_id=str(attempt.get("job_id") or ""),
            ).get("turns", []),
            "duration_seconds": payload.get(
                "duration_seconds",
                duration or 0,
            ),
            "num_frames": request_payload.get("num_frames"),
        }

    def original_case_download(
        source_id: str,
        attempt: dict[str, Any],
        source: Path,
        *,
        fallback_reason: str | None = None,
    ) -> dict[str, Any]:
        video_url = str(attempt.get("video_url") or "").strip()
        if not video_url:
            directory = attempt_sources[source_id]
            relative = source.relative_to(directory.parent.resolve())
            video_url = "cases/media/" + "/".join(
                quote(part, safe="") for part in (source_id, *relative.parts)
            )
        response = {
            "download_path": video_url,
            "filename": source.name,
            "overlay": False,
        }
        if fallback_reason:
            response["fallback_reason"] = fallback_reason
        return response

    def resolve_case_share(
        share_id: str,
    ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        with case_share_lock:
            record = load_case_shares().get(share_id)
        if not isinstance(record, dict):
            raise HTTPException(status_code=404, detail="shared case not found")
        source_id = str(record.get("source_id") or "")
        directory = attempt_sources.get(source_id)
        if directory is None:
            raise HTTPException(status_code=404, detail="shared case not found")
        deployment_root = directory.parent.resolve()
        relative_path = str(record.get("relative_path") or "")
        candidate = (deployment_root / relative_path).resolve()
        if (
            not relative_path
            or candidate == deployment_root
            or deployment_root not in candidate.parents
            or candidate.suffix.lower() != ".mp4"
            or not candidate.is_file()
        ):
            raise HTTPException(status_code=404, detail="shared case not found")
        try:
            overlay = _normalize_case_overlay(record.get("overlay"))
        except ValueError as exc:
            raise HTTPException(
                status_code=404,
                detail="shared case metadata is invalid",
            ) from exc
        return candidate, overlay, record

    def probe_hosts() -> None:
        while not stop_event.is_set():
            for host in manager.snapshot()["hosts"]:
                try:
                    response = requests.get(
                        f"{host['url']}/health",
                        timeout=health_request_timeout,
                    )
                    health = response.json()
                    online = response.ok and bool(health.get("worker_online"))
                    reported_available = health.get("available")
                    available = bool(
                        online
                        and (
                            reported_available
                            if reported_available is not None
                            else (
                                not health.get("busy")
                                and int(health.get("queued_jobs") or 0) == 0
                                and int(health.get("running_jobs") or 0) == 0
                                and int(health.get("enhancing_jobs") or 0) == 0
                                and int(health.get("reserved_session_count") or 0) == 0
                            )
                        )
                    )
                except (requests.RequestException, ValueError):
                    online = False
                    available = False
                host_id = str(host["id"])
                previous = host_states.get(host_id)
                if previous is None or previous != online:
                    logger.info(
                        "backend health changed: host=%s url=%s online=%s",
                        host_id,
                        host["url"],
                        online,
                    )
                    host_states[host_id] = online
                manager.update_host(
                    host_id,
                    online,
                    available=available,
                )
            stop_event.wait(health_interval)

    @app.on_event("startup")
    def start_monitor() -> None:
        thread = threading.Thread(
            target=probe_hosts, name="host-health-monitor", daemon=True
        )
        app.state.health_thread = thread
        thread.start()
        cases_thread = threading.Thread(
            target=refresh_cases_cache_loop,
            name="cases-cache-refresh",
            daemon=True,
        )
        app.state.cases_thread = cases_thread
        cases_thread.start()

    @app.on_event("shutdown")
    def stop_monitor() -> None:
        stop_event.set()
        thread = getattr(app.state, "health_thread", None)
        if thread is not None:
            thread.join(timeout=health_interval + 1)
        cases_thread = getattr(app.state, "cases_thread", None)
        if cases_thread is not None:
            cases_thread.join(timeout=1)

    def token_from(request: Request) -> str:
        token = request.cookies.get(SESSION_COOKIE, "")
        if not token:
            raise HTTPException(status_code=401, detail="login required")
        return token

    def refresh_cookie(response: Response, token: str) -> None:
        response.set_cookie(
            SESSION_COOKIE,
            token,
            httponly=True,
            samesite="lax",
            secure=False,
            max_age=int(manager.lease_seconds),
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "architecture": "multi_host_dispatcher",
            **manager.snapshot(),
        }

    @app.post("/auth/login")
    async def login(request: Request) -> Response:
        raw_body = await request.body()
        try:
            payload = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning(
                "login rejected: invalid JSON client=%s content_type=%r "
                "body_bytes=%d error=%s",
                request.client.host if request.client else "unknown",
                request.headers.get("content-type"),
                len(raw_body),
                exc,
            )
            raise HTTPException(
                status_code=400,
                detail="request body must be valid JSON",
            ) from exc
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400,
                detail="request JSON must be an object",
            )
        user_id = str(payload.get("user_id") or "")
        try:
            token, state = manager.login(
                user_id,
                page_id=str(payload.get("page_id") or ""),
            )
        except ResourceBusyError as exc:
            logger.info("login rejected: user=%r reason=%s", user_id, exc)
            raise HTTPException(status_code=423, detail=str(exc)) from exc
        except ValueError as exc:
            logger.info("login rejected: user=%r reason=%s", user_id, exc)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        logger.info(
            "login accepted: user=%r status=%s host=%s waitlist_position=%s",
            state["user_id"],
            state["status"],
            state.get("host_id"),
            state.get("waitlist_position"),
        )
        response = Response(
            content=json.dumps(state, separators=(",", ":")),
            media_type="application/json",
        )
        refresh_cookie(response, token)
        return response

    @app.get("/auth/status")
    def auth_status(request: Request) -> dict[str, Any]:
        state = manager.status(
            token_from(request),
            page_id=request.headers.get("x-nomas-page-id"),
        )
        if state is None:
            raise HTTPException(status_code=401, detail="session expired")
        return state

    @app.post("/auth/reconnect")
    def reconnect(request: Request, response: Response) -> dict[str, Any]:
        token = token_from(request)
        state = manager.reconnect(
            token,
            page_id=request.headers.get("x-nomas-page-id"),
        )
        if state is None:
            raise HTTPException(status_code=401, detail="session expired")
        refresh_cookie(response, token)
        return state

    @app.post("/auth/heartbeat")
    def heartbeat(request: Request, response: Response) -> dict[str, Any]:
        token = token_from(request)
        state = manager.touch(
            token,
            page_id=request.headers.get("x-nomas-page-id"),
        )
        if state is None:
            raise HTTPException(status_code=401, detail="session expired")
        refresh_cookie(response, token)
        return state

    @app.post("/auth/disconnect", status_code=202)
    async def disconnect(request: Request) -> dict[str, Any]:
        token = token_from(request)
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {}
        state = manager.disconnect(
            token,
            page_id=(
                str(payload.get("page_id") or "") if isinstance(payload, dict) else ""
            ),
            grace_seconds=disconnect_grace_seconds,
        )
        if state is None:
            raise HTTPException(status_code=401, detail="session expired")
        return {
            "status": "disconnected",
            "host_retained_until_lease_expiry": True,
            "lease_expires_at": state["lease_expires_at"],
        }

    @app.get("/auth/allocations")
    def allocations(request: Request) -> dict[str, Any]:
        token = token_from(request)
        if manager.status(token) is None:
            raise HTTPException(status_code=401, detail="session expired")
        values = manager.active_allocations()
        return {
            "allocations": values,
            "count": len(values),
            "server_time": time.time(),
        }

    @app.post("/auth/logout")
    def logout(request: Request) -> Response:
        token = request.cookies.get(SESSION_COOKIE, "")
        if token:
            manager.logout(token)
        response = Response(
            content='{"status":"logged_out"}',
            media_type="application/json",
        )
        response.delete_cookie(SESSION_COOKIE)
        return response

    @app.get("/cases", response_class=HTMLResponse)
    @app.get("/cases/", response_class=HTMLResponse)
    def cases_page() -> HTMLResponse:
        page = PROJECT_ROOT / "apps" / "seg_ar_cases.html"
        return HTMLResponse(
            page.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/v1/cases")
    def cases(
        page: int = 1,
        page_size: int = 24,
        limit: int | None = None,
        user_id: str = "",
        status: str = "",
        search: str = "",
    ) -> dict[str, Any]:
        if not attempt_sources:
            return {
                "items": [],
                "count": 0,
                "total": 0,
                "session_total": 0,
                "page": 1,
                "page_size": max(1, min(int(page_size), 100)),
                "pages": 0,
                "configured": False,
            }
        with cases_cache_lock:
            cache_loaded = bool(cases_cache["loaded"])
        if not cache_loaded:
            # Direct endpoint calls in tests do not run FastAPI startup hooks.
            # Populate synchronously in that case; production warms in the
            # dedicated background thread.
            refresh_cases_cache()
        with cases_cache_lock:
            cache_snapshot = {
                "loaded": cases_cache["loaded"],
                "refreshing": cases_cache["refreshing"],
                "refreshed_at": cases_cache["refreshed_at"],
                "refresh_seconds": cases_cache["refresh_seconds"],
                "error": cases_cache["error"],
                "sessions": cases_cache["sessions"],
                "sources": cases_cache["sources"],
                "total": cases_cache["total"],
            }
        if not cache_snapshot["loaded"]:
            return {
                "items": [],
                "count": 0,
                "total": 0,
                "session_total": 0,
                "page": 1,
                "page_size": max(1, min(int(page_size), 100)),
                "pages": 0,
                "configured": True,
                "warming_up": True,
                "cache_error": cache_snapshot["error"],
                "updated_at": time.time(),
            }
        requested_page_size = limit if limit is not None else page_size
        bounded_page_size = max(1, min(int(requested_page_size), 100))
        requested_page = max(1, int(page))
        clean_user = user_id.strip().casefold()
        clean_status = status.strip().casefold()
        clean_search = search.strip().casefold()
        sessions = []
        for session in cache_snapshot["sessions"]:
            if clean_user and not any(
                clean_user in value for value in session["user_ids"]
            ):
                continue
            if clean_status and clean_status != session["latest_status"]:
                continue
            if clean_search and clean_search not in session["searchable"]:
                continue
            sessions.append(session)
        session_total = len(sessions)
        pages = (
            (session_total + bounded_page_size - 1) // bounded_page_size
            if session_total
            else 0
        )
        current_page = min(requested_page, max(1, pages))
        start = (current_page - 1) * bounded_page_size
        page_sessions = sessions[start : start + bounded_page_size]
        page_items = [
            attempt for session in page_sessions for attempt in session["attempts"]
        ]
        page_items.sort(
            key=lambda item: float(
                item.get("updated_at") or item.get("created_at") or 0.0
            ),
            reverse=True,
        )
        return {
            "items": page_items,
            "count": len(page_items),
            "total": cache_snapshot["total"],
            "session_total": session_total,
            "page": current_page,
            "page_size": bounded_page_size,
            "pages": pages,
            "configured": True,
            "sources": cache_snapshot["sources"],
            "cache_refreshed_at": cache_snapshot["refreshed_at"],
            "cache_refresh_seconds": cache_snapshot["refresh_seconds"],
            "cache_refreshing": cache_snapshot["refreshing"],
            "cache_error": cache_snapshot["error"],
            "updated_at": time.time(),
        }

    @app.get("/cases/media/{source_id}/{path:path}")
    def case_media(source_id: str, path: str) -> FileResponse:
        if not attempt_sources:
            raise HTTPException(status_code=404, detail="case archive not configured")
        directory = attempt_sources.get(source_id)
        legacy_slot = None
        if directory is None and len(attempt_sources) == 1:
            if re.fullmatch(r"sp16-\d+", source_id):
                directory = next(iter(attempt_sources.values()))
                legacy_slot = source_id
        if directory is None:
            raise HTTPException(status_code=404, detail="case media not found")
        deployment_root = directory.parent.resolve()
        relative_path = (
            Path(legacy_slot) / unquote(path)
            if legacy_slot is not None
            else Path(unquote(path))
        )
        candidate = (deployment_root / relative_path).resolve()
        if (
            candidate == deployment_root
            or deployment_root not in candidate.parents
            or candidate.suffix.lower() != ".mp4"
            or not candidate.is_file()
        ):
            raise HTTPException(status_code=404, detail="case media not found")
        return FileResponse(
            candidate,
            media_type="video/mp4",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.post(
        "/v1/cases/{source_id}/{job_id}/shares",
        status_code=201,
    )
    def create_case_share(
        source_id: str,
        job_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        selected, session_attempts = case_attempt_context(source_id, job_id)
        source = resolve_case_video(source_id, selected)
        overlay = _normalize_case_overlay(
            _build_case_overlay(
                session_attempts,
                selected_job_id=job_id,
                theme=str(payload.get("theme") or "dark"),
            )
        )
        deployment_root = attempt_sources[source_id].parent.resolve()
        share_id = secrets.token_urlsafe(18)
        with case_share_lock:
            shares = load_case_shares()
            shares[share_id] = {
                "source_id": source_id,
                "job_id": job_id,
                "session_id": selected.get("session_id"),
                "relative_path": source.relative_to(deployment_root).as_posix(),
                "created_at": time.time(),
                "overlay": overlay,
            }
            save_case_shares(shares)
        return {
            "id": share_id,
            "share_path": f"cases/share/{share_id}",
            "overlay": bool(overlay["turns"]),
            "overlay_turns": len(overlay["turns"]),
        }

    @app.post(
        "/v1/cases/{source_id}/{job_id}/gallery",
        status_code=201,
    )
    def add_case_to_gallery(
        source_id: str,
        job_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if gallery_dir is None:
            raise HTTPException(
                status_code=503,
                detail="Gallery storage is not configured",
            )
        selected, session_attempts = case_attempt_context(source_id, job_id)
        if str(selected.get("status") or "").casefold() != "succeeded":
            raise HTTPException(
                status_code=409,
                detail="only successful cases can be added to Gallery",
            )
        source = resolve_case_video(source_id, selected)
        try:
            return add_gallery_item_from_source(
                gallery_dir,
                source,
                case_gallery_metadata(selected, session_attempts, payload),
                process_lock=gallery_write_lock,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except GalleryStorageBusyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/v1/gallery")
    def gallery_items() -> dict[str, Any]:
        items = read_gallery_items(gallery_dir)
        return {
            "items": items,
            "count": len(items),
            "hidden_ids": read_hidden_gallery_item_ids(gallery_dir),
            "configured": gallery_dir is not None,
        }

    @app.delete("/v1/gallery/{item_id}")
    def remove_from_gallery(
        item_id: str,
        delete_media: bool = False,
    ) -> dict[str, Any]:
        if gallery_dir is None:
            raise HTTPException(
                status_code=503,
                detail="Gallery storage is not configured",
            )
        try:
            return remove_gallery_item(
                gallery_dir,
                item_id,
                process_lock=gallery_write_lock,
                delete_media=delete_media,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except GalleryStorageBusyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post(
        "/v1/cases/{source_id}/{job_id}/downloads",
        status_code=201,
    )
    def create_case_download(
        source_id: str,
        job_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        selected, session_attempts = case_attempt_context(source_id, job_id)
        source = resolve_case_video(source_id, selected)
        if payload.get("include_overlay", True) is False:
            return original_case_download(source_id, selected, source)

        overlay = _normalize_case_overlay(
            _build_case_overlay(
                session_attempts,
                selected_job_id=job_id,
                theme=str(payload.get("theme") or "dark"),
            )
        )
        if not overlay["turns"]:
            return original_case_download(
                source_id,
                selected,
                source,
                fallback_reason=(
                    "Prompt/camera overlay metadata is unavailable; "
                    "returned the Original video."
                ),
            )

        source_stat = source.stat()
        fingerprint = hashlib.sha256(
            (
                f"v{CASE_DOWNLOAD_RENDERER_VERSION}:"
                f"{source}:{source_stat.st_size}:{source_stat.st_mtime_ns}:"
                + json.dumps(
                    overlay,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ).encode("utf-8")
        ).hexdigest()[:16]
        destination = case_download_dir / (
            f"{source.stem}-with-overlay-{fingerprint}.mp4"
        )
        with case_download_lock:
            if not destination.is_file():
                case_download_dir.mkdir(parents=True, exist_ok=True)
                fd, raw_temporary = tempfile.mkstemp(
                    prefix=f".case-overlay-{fingerprint}-",
                    suffix=".mp4",
                    dir=str(case_download_dir),
                )
                os.close(fd)
                temporary = Path(raw_temporary)
                try:
                    _burn_case_overlay_video(source, temporary, overlay)
                    os.replace(temporary, destination)
                except Exception as exc:  # noqa: BLE001
                    destination.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=500,
                        detail=f"Could not render overlay video: {exc}",
                    ) from exc
                finally:
                    temporary.unlink(missing_ok=True)
        return {
            "download_path": ("cases/downloads/" + quote(destination.name, safe="")),
            "filename": destination.name,
            "overlay": True,
        }

    @app.get("/cases/share/{share_id}", response_class=HTMLResponse)
    def shared_case_page(share_id: str) -> HTMLResponse:
        resolve_case_share(share_id)
        content = _case_shared_video_page_html(share_id).replace(
            'src="../hunyuan-logo.png"',
            'src="../../hunyuan-logo.png"',
        )
        return HTMLResponse(
            content=content,
            headers={
                "Cache-Control": "no-store",
                "X-Robots-Tag": "noindex, nofollow",
            },
        )

    @app.api_route(
        "/cases/public/{share_id}.mp4",
        methods=["GET", "HEAD"],
    )
    def shared_case_video(share_id: str) -> FileResponse:
        source, _, _ = resolve_case_share(share_id)
        return FileResponse(
            source,
            media_type="video/mp4",
            filename=None,
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.get("/cases/public/{share_id}.json")
    def shared_case_metadata(share_id: str) -> JSONResponse:
        _, overlay, record = resolve_case_share(share_id)
        return JSONResponse(
            {
                "id": share_id,
                "created_at": record.get("created_at"),
                "case": {
                    "source_id": record.get("source_id"),
                    "job_id": record.get("job_id"),
                    "session_id": record.get("session_id"),
                },
                "overlay": overlay,
            },
            headers={"Cache-Control": "private, max-age=300"},
        )

    @app.get("/cases/downloads/{filename}")
    def case_download_file(filename: str) -> FileResponse:
        root = case_download_dir.resolve()
        candidate = (root / unquote(filename)).resolve()
        if (
            candidate.parent != root
            or candidate.suffix.lower() != ".mp4"
            or not candidate.is_file()
        ):
            raise HTTPException(status_code=404, detail="case download not found")
        return FileResponse(
            candidate,
            media_type="video/mp4",
            filename=candidate.name,
            headers={"Cache-Control": "private, max-age=3600"},
        )

    async def proxy(request: Request, path: str) -> Response:
        token = token_from(request)
        active_generation = path == "v1/generate" or path.startswith("v1/generate/")
        if active_generation:
            # Generation polling is active use, even when the user is not
            # touching the keyboard or pointer while waiting for the GPU.
            # Keep the host lease alive until that operation finishes.
            state = manager.touch(token)
            if state is None:
                raise HTTPException(status_code=401, detail="session expired")
        else:
            state = manager.status(token)
            if state is None:
                raise HTTPException(status_code=401, detail="session expired")
        if (
            request.method == "POST"
            and SESSION_RELEASE_PATH_RE.fullmatch(path)
            and request.headers.get(EXPLICIT_SESSION_RELEASE_HEADER) != "1"
        ):
            # Bundles published before explicit release signaling called this
            # endpoint from pagehide. Ignoring unmarked requests preserves the
            # process-local KV cache across the first refresh that upgrades an
            # already-open browser tab to the fixed frontend.
            logger.info(
                "ignored legacy unmarked session release: user=%r path=%s",
                state["user_id"],
                path,
            )
            return Response(
                content=(
                    '{"status":"preserved","released":false,'
                    '"reason":"explicit release marker required"}'
                ),
                media_type="application/json",
            )
        body = await request.body()
        host = None
        if request.method == "POST" and path == "v1/generate":
            session_id = _generation_session_id(body)
            if session_id:
                try:
                    host = manager.bind_session_host(token, session_id)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                except SessionBindingConflictError as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                if host is None:
                    # The browser may still be waiting for its exclusive host.
                    state = manager.touch(token)
                    if state is None:
                        raise HTTPException(
                            status_code=401,
                            detail="session expired",
                        )
        if host is None:
            host = manager.assigned_host(token)
        if host is None:
            state = manager.touch(token)
            if state is None:
                raise HTTPException(status_code=401, detail="session expired")
            raise HTTPException(
                status_code=423,
                detail={
                    "message": "waiting for an available GPU host",
                    **state,
                },
            )
        target = f"{host['url']}/{path}"
        if request.url.query:
            target = f"{target}?{request.url.query}"
        headers = _upstream_headers(
            request.headers,
            user_id=str(state["user_id"]),
            host_id=str(host["id"]),
        )
        try:
            upstream = await run_in_threadpool(
                requests.request,
                request.method,
                target,
                headers=headers,
                data=body,
                stream=True,
                timeout=proxy_timeout,
            )
        except requests.RequestException as exc:
            manager.update_host(str(host["id"]), False)
            raise HTTPException(
                status_code=502, detail=f"assigned host is unavailable: {exc}"
            ) from exc
        response_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
            and key.lower() not in {"content-length", "content-encoding"}
        }
        response = StreamingResponse(
            upstream.iter_content(chunk_size=1024 * 1024),
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=upstream.headers.get("content-type"),
            background=BackgroundTask(upstream.close),
        )
        if active_generation:
            refresh_cookie(response, token)
        return response

    async def proxy_with_host_fallback(
        request: Request,
        path: str,
    ) -> Response:
        """Find durable jobs/assets even if a lease moved during host recovery."""
        token = token_from(request)
        state = manager.touch(
            token,
            page_id=request.headers.get("x-nomas-page-id"),
        )
        if state is None:
            raise HTTPException(status_code=401, detail="session expired")
        assigned = manager.assigned_host(token)
        snapshot_hosts = manager.snapshot()["hosts"]
        candidates = []
        seen = set()
        for host in [assigned, *snapshot_hosts]:
            if not isinstance(host, dict):
                continue
            host_id = str(host.get("id") or "")
            if not host_id or host_id in seen or not host.get("online"):
                continue
            seen.add(host_id)
            candidates.append(host)
        last_error: Exception | None = None
        for host in candidates:
            target = f"{host['url']}/{path}"
            if request.url.query:
                target = f"{target}?{request.url.query}"
            try:
                upstream = await run_in_threadpool(
                    requests.request,
                    request.method,
                    target,
                    headers=_upstream_headers(
                        request.headers,
                        user_id=str(state["user_id"]),
                        host_id=str(host["id"]),
                    ),
                    stream=True,
                    timeout=proxy_timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                manager.update_host(str(host["id"]), False)
                continue
            if upstream.status_code == 404:
                upstream.close()
                continue
            response_headers = {
                key: value
                for key, value in upstream.headers.items()
                if key.lower() not in HOP_BY_HOP_HEADERS
                and key.lower() not in {"content-length", "content-encoding"}
            }
            response = StreamingResponse(
                upstream.iter_content(chunk_size=1024 * 1024),
                status_code=upstream.status_code,
                headers=response_headers,
                media_type=upstream.headers.get("content-type"),
                background=BackgroundTask(upstream.close),
            )
            refresh_cookie(response, token)
            return response
        if last_error is not None and not candidates:
            raise HTTPException(
                status_code=502,
                detail=f"no inference host is available: {last_error}",
            )
        raise HTTPException(status_code=404, detail="generation job not found")

    @app.get("/v1/generate/{job_id}")
    async def proxy_generation_status(
        job_id: str,
        request: Request,
    ) -> Response:
        return await proxy_with_host_fallback(
            request,
            f"v1/generate/{job_id}",
        )

    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def proxy_api(request: Request, path: str) -> Response:
        return await proxy(request, f"v1/{path}")

    @app.api_route("/outputs/{path:path}", methods=["GET", "HEAD"])
    async def proxy_output(request: Request, path: str) -> Response:
        return await proxy_with_host_fallback(request, f"outputs/{path}")

    async def proxy_public(request: Request, path: str) -> Response:
        """Resolve a public share without requiring an authoring lease.

        Share ids are stored by the backend that generated them. With multiple
        8-GPU nodes, try the online backends until the owning node answers with
        something other than 404.
        """
        hosts = [host for host in manager.snapshot()["hosts"] if host.get("online")]
        last_error: Exception | None = None
        for host in hosts:
            target = f"{host['url']}/{path}"
            if request.url.query:
                target = f"{target}?{request.url.query}"
            try:
                upstream = await run_in_threadpool(
                    requests.request,
                    request.method,
                    target,
                    headers={
                        key: value
                        for key, value in request.headers.items()
                        if key.lower() not in HOP_BY_HOP_HEADERS
                        and key.lower()
                        not in {"host", "cookie", "authorization", "content-length"}
                    },
                    stream=True,
                    timeout=proxy_timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                manager.update_host(str(host["id"]), False)
                continue
            if upstream.status_code == 404:
                upstream.close()
                continue
            response_headers = {
                key: value
                for key, value in upstream.headers.items()
                if key.lower() not in HOP_BY_HOP_HEADERS
                and key.lower() not in {"content-length", "content-encoding"}
            }
            return StreamingResponse(
                upstream.iter_content(chunk_size=1024 * 1024),
                status_code=upstream.status_code,
                headers=response_headers,
                media_type=upstream.headers.get("content-type"),
                background=BackgroundTask(upstream.close),
            )
        if last_error is not None:
            raise HTTPException(
                status_code=502,
                detail=f"no public share backend is available: {last_error}",
            )
        raise HTTPException(status_code=404, detail="shared video not found")

    @app.api_route("/share/{path:path}", methods=["GET", "HEAD"])
    async def proxy_share(request: Request, path: str) -> Response:
        return await proxy_public(request, f"share/{path}")

    @app.api_route("/public/{path:path}", methods=["GET", "HEAD"])
    async def proxy_public_asset(request: Request, path: str) -> Response:
        return await proxy_public(request, f"public/{path}")

    @app.api_route("/gallery/{path:path}", methods=["GET", "HEAD"])
    async def gallery_asset(request: Request, path: str) -> Response:
        if gallery_dir is not None:
            gallery_root = gallery_dir.resolve()
            candidate = (gallery_root / unquote(path)).resolve()
            if (
                candidate != gallery_root
                and candidate.is_relative_to(gallery_root)
                and candidate.suffix.lower() == ".mp4"
                and candidate.is_file()
            ):
                return FileResponse(
                    candidate,
                    media_type="video/mp4",
                    headers={"Cache-Control": "public, max-age=3600"},
                )
        return await proxy_public(request, f"gallery/{path}")

    if frontend_dir is not None and frontend_dir.is_dir():
        app.mount(
            "/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend"
        )
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NOMAS multi-host dispatcher")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--state-db", default="/tmp/nomas_seg_ar_resources.json")
    parser.add_argument(
        "--attempt-dir",
        action="append",
        default=[],
    )
    parser.add_argument("--hosts-file", required=True)
    parser.add_argument("--lease-seconds", type=float, default=1800)
    parser.add_argument("--host-stale-seconds", type=float, default=30)
    parser.add_argument("--health-interval", type=float, default=5)
    parser.add_argument(
        "--health-request-timeout",
        type=float,
        default=DEFAULT_HEALTH_REQUEST_TIMEOUT_SECONDS,
        help=(
            "Per-Gateway /health request timeout in seconds. This remains "
            "independent from --health-interval because a healthy Gateway "
            "can respond slowly while reading a large durable Job DB."
        ),
    )
    parser.add_argument("--proxy-timeout", type=float, default=3600)
    parser.add_argument("--disconnect-grace-seconds", type=float, default=3)
    parser.add_argument(
        "--allow-waitlist",
        action="store_true",
        help=(
            "Keep users in a FIFO waitlist when every 8-GPU host is occupied. "
            "By default a busy deployment fails fast so nobody waits on a "
            "stalled page."
        ),
    )
    parser.add_argument(
        "--frontend-dir",
        default=str(PROJECT_ROOT / "apps" / "frontend" / "dist"),
    )
    parser.add_argument(
        "--gallery-dir",
        default="",
        help=(
            "Optional shared Gallery media directory. Dispatcher-backed Cases "
            "copy selected archived MP4s here and update the adjacent index."
        ),
    )
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    hosts_value = json.loads(
        Path(args.hosts_file).expanduser().read_text(encoding="utf-8")
    )
    hosts = hosts_value.get("hosts") if isinstance(hosts_value, dict) else hosts_value
    if not isinstance(hosts, list):
        raise ValueError("hosts file must be a list or an object containing 'hosts'")
    manager = ResourceManager(
        args.state_db,
        hosts,
        lease_seconds=args.lease_seconds,
        host_stale_seconds=args.host_stale_seconds,
        allow_waitlist=args.allow_waitlist,
    )
    app = create_dispatcher_app(
        manager,
        frontend_dir=Path(args.frontend_dir).expanduser().resolve(),
        attempt_dirs=[
            Path(value).expanduser().resolve()
            for value in (
                args.attempt_dir
                or [
                    item
                    for item in os.environ.get(
                        "NOMAS_USER_ATTEMPT_DIRS",
                        os.environ.get("NOMAS_USER_ATTEMPT_DIR", ""),
                    ).split(os.pathsep)
                    if item
                ]
            )
        ],
        gallery_dir=(
            Path(args.gallery_dir).expanduser().resolve() if args.gallery_dir else None
        ),
        health_interval=args.health_interval,
        health_request_timeout=args.health_request_timeout,
        proxy_timeout=args.proxy_timeout,
        disconnect_grace_seconds=args.disconnect_grace_seconds,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
