"""Persist user-visible generation attempts beside the shared video outputs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict
from urllib.parse import quote, unquote, urlsplit

from apps.durable_file import replace_or_overwrite

ATTEMPT_INDEX_NAME = ".cases_index.json"


def _safe_component(value: object, fallback: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return fallback
    if raw == fallback:
        return fallback
    clean = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in raw
    ).strip("._")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{clean[:80] or fallback}-{digest}"


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _manifest_paths(root: Path) -> list[Path]:
    paths_with_mtime = []
    for path in root.glob("*/*/*.json"):
        if not path.is_file() or path.name.startswith("."):
            continue
        try:
            modified_at = path.stat().st_mtime
        except OSError:
            continue
        paths_with_mtime.append((modified_at, path))
    return [path for _, path in sorted(paths_with_mtime, reverse=True)]


def _read_manifest(path: Path) -> Dict[str, Any] | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    value = _json_clone(record)
    value["archive_path"] = str(path)
    return value


def build_user_attempt_index(directory: Path) -> dict[str, Any]:
    """Build one compact snapshot for a completed historical deployment."""

    root = directory.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    items = []
    for path in _manifest_paths(root):
        record = _read_manifest(path)
        if record is not None:
            items.append(record)
    index = {
        "schema_version": 1,
        "generated_at": time.time(),
        "attempt_dir": str(root),
        "count": len(items),
        "items": items,
    }
    destination = root / ATTEMPT_INDEX_NAME
    fd, raw_temporary = tempfile.mkstemp(
        prefix=f".{ATTEMPT_INDEX_NAME}.",
        suffix=".tmp",
        dir=str(root),
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(index, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        replace_or_overwrite(temporary, destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return index


def _load_attempt_items(root: Path) -> list[Dict[str, Any]]:
    index_path = root / ATTEMPT_INDEX_NAME
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            index = None
        if isinstance(index, dict) and isinstance(index.get("items"), list):
            return [
                _json_clone(item) for item in index["items"] if isinstance(item, dict)
            ]
    items = []
    for path in _manifest_paths(root):
        record = _read_manifest(path)
        if record is not None:
            items.append(record)
    return items


def count_user_attempts(directory: Path) -> int:
    root = directory.expanduser().resolve()
    if not root.is_dir():
        return 0
    index_path = root / ATTEMPT_INDEX_NAME
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            index = None
        if isinstance(index, dict):
            try:
                return max(0, int(index.get("count") or 0))
            except (TypeError, ValueError):
                pass
    return len(_manifest_paths(root))


class UserAttemptArchive:
    """Write one incrementally updated JSON manifest per generation attempt."""

    def __init__(self, output_dir: Path, directory: Path | None = None) -> None:
        self.output_dir = output_dir.expanduser().resolve()
        configured = str(os.environ.get("NOMAS_USER_ATTEMPT_DIR") or "").strip()
        if directory is not None:
            root = directory
        elif configured:
            root = Path(configured)
        elif re.fullmatch(r"sp16-\d+", self.output_dir.name):
            root = self.output_dir.parent / "user_attempts"
        else:
            root = self.output_dir / "user_attempts"
        self.directory = root.expanduser().resolve()

    def _artifact_path(self, url: object) -> str | None:
        raw = str(url or "").strip()
        if not raw:
            return None
        clean = unquote(urlsplit(raw).path).lstrip("/")
        if not clean.startswith("outputs/"):
            return None
        candidate = (self.output_dir / clean[len("outputs/") :]).resolve()
        if candidate == self.output_dir or self.output_dir not in candidate.parents:
            return None
        return str(candidate)

    def write(self, job: Dict[str, Any]) -> Path | None:
        if str(job.get("kind") or "") != "generate":
            return None
        payload = job.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        state = job.get("state")
        state = state if isinstance(state, dict) else {}

        user_id = (
            str(state.get("user_id") or payload.get("_user_id") or "").strip()
            or "_unknown_user"
        )
        pe_input = payload.get("pe_input")
        pe_input = pe_input if isinstance(pe_input, dict) else {}
        pe_session = payload.get("_pe_session")
        pe_session = pe_session if isinstance(pe_session, dict) else {}
        session_id = str(
            pe_input.get("session_id")
            or pe_session.get("session_id")
            or payload.get("session_id")
            or "unknown-session"
        ).strip()
        job_id = str(job.get("id") or "unknown-job").strip()
        slot_id = str(
            state.get("dispatch_host_id")
            or payload.get("_dispatch_host_id")
            or os.environ.get("NOMAS_SLOT_ID")
            or self.output_dir.name
        ).strip()

        destination_dir = (
            self.directory
            / _safe_component(user_id, "_unknown_user")
            / _safe_component(session_id, "unknown-session")
        )
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / (
            f"{_safe_component(job_id, 'unknown-job')}.json"
        )
        lock_path = destination.with_suffix(".json.lock")

        result = state.get("result")
        result = result if isinstance(result, dict) else {}
        video_url = state.get("video_url") or result.get("video_url")
        metadata_url = result.get("metadata_url")
        transcript_path = state.get("pe_transcript_path") or pe_session.get(
            "pe_transcript_path"
        )
        original_user_prompts = payload.get("user_prompts")
        if not isinstance(original_user_prompts, list):
            original_user_prompts = []

        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                existing: Dict[str, Any] = {}
                if destination.is_file():
                    try:
                        loaded = json.loads(destination.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        loaded = None
                    if isinstance(loaded, dict):
                        existing = loaded

                request_record = existing.get("request")
                request_record = (
                    dict(request_record) if isinstance(request_record, dict) else {}
                )
                if pe_input:
                    request_record["pe_input"] = _json_clone(pe_input)
                for key in (
                    "user_prompts",
                    "carry_user_prompt",
                    "camera_trajectory",
                    "num_frames",
                    "num_inference_steps",
                    "segment_boundaries",
                    "segment_has_camera",
                    "seed",
                    "width",
                    "height",
                ):
                    if key in payload:
                        request_record[key] = _json_clone(payload[key])

                prompt_record = existing.get("prompts")
                prompt_record = (
                    dict(prompt_record) if isinstance(prompt_record, dict) else {}
                )
                if pe_input:
                    prompt_record["original_user_input"] = _json_clone(pe_input)
                elif pe_session:
                    restored_input = {
                        "perspective": pe_session.get(
                            "requested_perspective",
                            pe_session.get("perspective"),
                        ),
                        "subject_description": pe_session.get("subject"),
                        "environment_description": pe_session.get(
                            "environment_description"
                        ),
                    }
                    restored_input = {
                        key: value
                        for key, value in restored_input.items()
                        if value not in (None, "")
                    }
                    if restored_input:
                        prompt_record.setdefault(
                            "original_user_input",
                            _json_clone(restored_input),
                        )
                if original_user_prompts:
                    prompt_record["original_user_prompts"] = _json_clone(
                        original_user_prompts
                    )
                if payload.get("carry_user_prompt"):
                    prompt_record["carry_user_prompt"] = str(
                        payload["carry_user_prompt"]
                    )
                if state.get("pe_prompt"):
                    prompt_record["k3_enhanced_prompt"] = str(state["pe_prompt"])
                if payload.get("prompt"):
                    prompt_record["final_model_prompt"] = str(payload["prompt"])

                artifacts = existing.get("artifacts")
                artifacts = dict(artifacts) if isinstance(artifacts, dict) else {}
                if transcript_path:
                    artifacts["pe_transcript_path"] = str(transcript_path)
                if video_url:
                    artifacts["video_url"] = str(video_url)
                    artifacts["video_path"] = self._artifact_path(video_url)
                if metadata_url:
                    artifacts["metadata_url"] = str(metadata_url)
                    artifacts["metadata_path"] = self._artifact_path(metadata_url)
                for key in (
                    "stream_init_url",
                    "stream_segment_url",
                ):
                    if result.get(key):
                        artifacts[key] = str(result[key])
                        artifacts[f"{key.removesuffix('_url')}_path"] = (
                            self._artifact_path(result[key])
                        )

                status = str(job.get("status") or state.get("status") or "unknown")
                now = time.time()
                events = existing.get("events")
                events = list(events) if isinstance(events, list) else []
                event = {
                    "status": status,
                    "updated_at": float(job.get("updated_at") or now),
                }
                if (
                    not events
                    or not isinstance(events[-1], dict)
                    or events[-1].get("status") != status
                ):
                    events.append(event)

                record = {
                    "schema_version": 1,
                    "user_id": user_id,
                    "session_id": session_id,
                    "job_id": job_id,
                    "slot_id": slot_id,
                    "status": status,
                    "created_at": job.get("created_at"),
                    "updated_at": job.get("updated_at") or now,
                    "started_at": job.get("started_at"),
                    "finished_at": job.get("finished_at"),
                    "request": request_record,
                    "prompts": prompt_record,
                    "prompt_enhancement": {
                        "mode": state.get("prompt_enhancement_mode"),
                        "request_id": state.get("pe_request_id"),
                        "reasoning_effort": state.get(
                            "prompt_enhancer_reasoning_effort",
                            pe_input.get("prompt_enhancer_reasoning_effort"),
                        ),
                        "seed": state.get(
                            "prompt_enhancer_seed",
                            pe_input.get("prompt_enhancer_seed"),
                        ),
                        "seed_source": state.get(
                            "prompt_enhancer_seed_source",
                            pe_input.get("prompt_enhancer_seed_source"),
                        ),
                        "reasoning": state.get("pe_reasoning") or "",
                    },
                    "result": _json_clone(result) if result else None,
                    "error": state.get("error"),
                    "error_code": state.get("error_code"),
                    "artifacts": artifacts,
                    "events": events,
                }
                fd, raw_temporary = tempfile.mkstemp(
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    dir=str(destination_dir),
                )
                temporary = Path(raw_temporary)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        json.dump(record, handle, ensure_ascii=False, indent=2)
                        handle.write("\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    replace_or_overwrite(temporary, destination)
                finally:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return destination


def list_user_attempts(
    directory: Path,
    *,
    limit: int = 100,
    user_id: str = "",
    status: str = "",
    search: str = "",
    source_id: str = "",
) -> list[Dict[str, Any]]:
    """Read the newest attempt manifests and attach browser-safe media URLs."""

    root = directory.expanduser().resolve()
    if not root.is_dir():
        return []
    deployment_root = root.parent
    clean_user = user_id.strip().casefold()
    clean_status = status.strip().casefold()
    clean_search = search.strip().casefold()
    results: list[Dict[str, Any]] = []
    records = _load_attempt_items(root)
    records.sort(
        key=lambda item: float(item.get("updated_at") or item.get("created_at") or 0.0),
        reverse=True,
    )
    for record in records:
        if clean_user and clean_user not in str(record.get("user_id") or "").casefold():
            continue
        if clean_status and clean_status != str(record.get("status") or "").casefold():
            continue
        if clean_search:
            searchable = json.dumps(
                {
                    "user_id": record.get("user_id"),
                    "session_id": record.get("session_id"),
                    "prompts": record.get("prompts"),
                    "error": record.get("error"),
                },
                ensure_ascii=False,
            ).casefold()
            if clean_search not in searchable:
                continue

        public = _json_clone(record)
        if source_id:
            public["source_id"] = source_id
            public["source_label"] = deployment_root.name
        artifacts = public.get("artifacts")
        artifacts = artifacts if isinstance(artifacts, dict) else {}
        video_path = str(artifacts.get("video_path") or "").strip()
        if video_path:
            candidate = Path(video_path).expanduser().resolve()
            try:
                relative = candidate.relative_to(deployment_root)
            except ValueError:
                relative = None
            if relative is not None and candidate.suffix.lower() == ".mp4":
                media_parts = [
                    *([source_id] if source_id else []),
                    *relative.parts,
                ]
                public["video_url"] = "cases/media/" + "/".join(
                    quote(part, safe="") for part in media_parts
                )
        results.append(public)
        if len(results) >= max(1, min(int(limit), 5000)):
            break
    return results
