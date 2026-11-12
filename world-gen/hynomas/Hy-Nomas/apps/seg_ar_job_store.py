"""Durable file-backed queue shared by the web API and GPU worker replicas."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


class JobStore:
    """Small durable multi-worker queue with process-safe file locking.

    The runtime CPython image does not ship ``_sqlite3``, so the queue uses one
    atomically replaced JSON state file protected by ``flock``. The expected
    queue is intentionally small, making this simpler and more portable than
    adding a database dependency to the base image.

    Backward compatibility matters here: older deployments stored one
    ``worker`` record. New deployments additionally keep a ``workers`` map and
    a session-to-worker affinity map. ``worker()`` still returns one aggregate
    record with the fields consumed by the original gateway/frontend.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked_state(write=True):
            pass

    @staticmethod
    def _empty_state() -> Dict[str, Any]:
        return {
            "jobs": {},
            "worker": None,
            "workers": {},
            "sessions": {},
            "session_workers": {},
            "errors": [],
        }

    def _read_unlocked(self) -> Dict[str, Any]:
        if not self.path.is_file():
            return self._empty_state()
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._empty_state()
        if not isinstance(value, dict):
            return self._empty_state()
        value.setdefault("jobs", {})
        value.setdefault("worker", None)
        value.setdefault("workers", {})
        value.setdefault("sessions", {})
        value.setdefault("session_workers", {})
        value.setdefault("errors", [])
        legacy_worker = value.get("worker")
        if (
            not value["workers"]
            and isinstance(legacy_worker, dict)
            and legacy_worker.get("worker_id")
        ):
            value["workers"][str(legacy_worker["worker_id"])] = legacy_worker
        return value

    def _write_unlocked(self, state: Dict[str, Any]) -> None:
        fd, raw_path = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        tmp_path = Path(raw_path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        finally:
            tmp_path.unlink(missing_ok=True)

    @contextmanager
    def _locked_state(self, *, write: bool) -> Iterator[Dict[str, Any]]:
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX if write else fcntl.LOCK_SH)
            state = self._read_unlocked()
            try:
                yield state
                if write:
                    self._write_unlocked(state)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _public_job(
        state: Dict[str, Any],
        job: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = dict(job.get("state") or {})
        result.update(
            {
                "id": job["id"],
                "kind": job["kind"],
                "status": job["status"],
                "created_at": job["created_at"],
                "updated_at": job["updated_at"],
                "worker_id": job.get("worker_id"),
            }
        )
        if job["status"] == "queued":
            ahead = sum(
                1
                for candidate in state["jobs"].values()
                if candidate["status"] in {"queued", "running"}
                and candidate["created_at"] < job["created_at"]
            )
            result["queue_position"] = ahead
        else:
            result["queue_position"] = 0
        return result

    @staticmethod
    def _job_session_id(job: Dict[str, Any]) -> Optional[str]:
        payload = job.get("payload")
        if not isinstance(payload, dict):
            return None
        if job.get("kind") in {"rollback", "session_release"}:
            value = str(payload.get("session_id") or "").strip()
            return value or None
        if job.get("kind") != "generate":
            return None
        pe_session = payload.get("_pe_session")
        if isinstance(pe_session, dict):
            value = str(pe_session.get("session_id") or "").strip()
            if value:
                return value
        if not bool(payload.get("preserve_kv_cache")):
            return None
        value = str(payload.get("session_id") or "").strip()
        return value or None

    @staticmethod
    def _projected_session_frames(
        state: Dict[str, Any],
        job: Dict[str, Any],
    ) -> int:
        session_id = JobStore._job_session_id(job)
        if not session_id or job.get("kind") != "generate":
            return 0
        payload = job.get("payload")
        if not isinstance(payload, dict):
            return 0
        pe_session = payload.get("_pe_session")
        if isinstance(pe_session, dict):
            lengths = pe_session.get("segment_frame_lengths")
            if isinstance(lengths, list):
                return sum(max(0, int(value)) for value in lengths)
        existing = state["sessions"].get(session_id)
        prior = 0
        if isinstance(existing, dict):
            lengths = existing.get("segment_frame_lengths")
            if isinstance(lengths, list):
                prior = sum(max(0, int(value)) for value in lengths)
        return prior + max(0, int(payload.get("num_frames") or 0))

    @staticmethod
    def _worker_record(
        worker: Dict[str, Any],
        *,
        stale_after: float,
        now: float,
    ) -> Dict[str, Any]:
        heartbeat_at = float(worker.get("heartbeat_at") or 0.0)
        age = max(0.0, now - heartbeat_at)
        status = str(worker.get("status") or "offline")
        online = age <= stale_after and status != "offline"
        return {
            **worker,
            "online": online,
            "status": status if online else "offline",
            "heartbeat_age_seconds": age,
        }

    def submit(
        self,
        kind: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        status: str = "queued",
    ) -> Dict[str, Any]:
        now = time.time()
        job_id = f"{kind}_{uuid.uuid4().hex[:12]}"
        job = {
            "id": job_id,
            "kind": kind,
            "status": status,
            "payload": payload or {},
            "state": {
                "completed_segments": 0,
                "total_segments": 0,
                "segment_updates": [],
                "captions": [],
                "video_url": None,
                "error": None,
                "status": status,
            },
            "worker_id": None,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
        }
        with self._locked_state(write=True) as state:
            state["jobs"][job_id] = job
            return self._public_job(state, job)

    def prepare(
        self,
        job_id: str,
        payload: Dict[str, Any],
        **state_changes: Any,
    ) -> Dict[str, Any]:
        """Atomically publish a prepared payload to the GPU queue."""
        with self._locked_state(write=True) as state:
            job = state["jobs"].get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["status"] in TERMINAL_STATUSES:
                return self._public_job(state, job)
            job["payload"] = dict(payload)
            job["status"] = "queued"
            job["updated_at"] = time.time()
            job["state"].update(state_changes)
            job["state"].update(
                {
                    "status": "queued",
                    "queue_position": 0,
                    "error": None,
                }
            )
            return self._public_job(state, job)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._locked_state(write=False) as state:
            job = state["jobs"].get(job_id)
            return self._public_job(state, job) if job is not None else None

    def claim_next(
        self,
        worker_id: str,
        *,
        active_session_id: Optional[str] = None,
        max_concurrent_jobs: int = 1,
        available_free_bytes: Optional[int] = None,
        min_free_bytes: int = 0,
        admission_settle_seconds: float = 0.0,
        long_session_frame_threshold: int = 0,
        max_long_sessions: int = 0,
        exclusive_long_session: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Claim the oldest job eligible for this worker.

        Interactive sessions are sticky because their DiT KV cache, RNG state,
        and PixelUMM streaming state live inside one model process group. A worker
        holding a session may only execute jobs for that session. An unreserved
        worker may claim standalone work or atomically adopt a new session.
        """
        with self._locked_state(write=True) as state:
            running = [
                job for job in state["jobs"].values() if job.get("status") == "running"
            ]
            if len(running) >= max(1, int(max_concurrent_jobs)):
                return None
            if available_free_bytes is not None and int(available_free_bytes) < max(
                0, int(min_free_bytes)
            ):
                return None
            if running and admission_settle_seconds > 0:
                newest_start = max(
                    float(job.get("started_at") or 0.0) for job in running
                )
                if time.time() - newest_start < admission_settle_seconds:
                    return None
            queued = sorted(
                (job for job in state["jobs"].values() if job["status"] == "queued"),
                key=lambda job: job["created_at"],
            )
            long_sessions: set[str] = set()
            if long_session_frame_threshold > 0 and max_long_sessions > 0:
                for session_id in state["session_workers"]:
                    session = state["sessions"].get(session_id)
                    lengths = (
                        session.get("segment_frame_lengths")
                        if isinstance(session, dict)
                        else None
                    )
                    frames = (
                        sum(max(0, int(value)) for value in lengths)
                        if isinstance(lengths, list)
                        else 0
                    )
                    if frames >= long_session_frame_threshold:
                        long_sessions.add(str(session_id))
                for running_job in running:
                    running_session = self._job_session_id(running_job)
                    if (
                        running_session
                        and self._projected_session_frames(
                            state,
                            running_job,
                        )
                        >= long_session_frame_threshold
                    ):
                        long_sessions.add(running_session)
            if (
                exclusive_long_session
                and long_sessions
                and active_session_id not in long_sessions
            ):
                for candidate in queued:
                    candidate_session = self._job_session_id(candidate)
                    if candidate_session not in long_sessions:
                        projected_frames = self._projected_session_frames(
                            state,
                            candidate,
                        )
                        candidate["state"]["queue_reason"] = (
                            "long_session_capacity"
                            if (
                                candidate_session
                                and projected_frames >= long_session_frame_threshold
                            )
                            else "long_session_memory_guard"
                        )
                return None
            job = None
            selected_session_id: Optional[str] = None
            for candidate in queued:
                session_id = self._job_session_id(candidate)
                assigned_worker = (
                    state["session_workers"].get(session_id) if session_id else None
                )
                projected_frames = self._projected_session_frames(
                    state,
                    candidate,
                )
                if (
                    session_id
                    and long_session_frame_threshold > 0
                    and max_long_sessions > 0
                    and projected_frames >= long_session_frame_threshold
                    and session_id not in long_sessions
                    and len(long_sessions) >= max_long_sessions
                ):
                    candidate["state"]["queue_reason"] = "long_session_capacity"
                    candidate["state"]["projected_session_frames"] = projected_frames
                    continue
                if active_session_id:
                    if session_id != active_session_id:
                        continue
                    if assigned_worker not in (None, worker_id):
                        continue
                else:
                    if assigned_worker not in (None, worker_id):
                        continue
                    # A stale assignment to the same worker is removed when
                    # that worker starts. Reaching this branch therefore means
                    # either a new session or a valid same-process assignment.
                job = candidate
                selected_session_id = session_id
                break
            if job is None:
                return None
            now = time.time()
            job.update(
                {
                    "status": "running",
                    "worker_id": worker_id,
                    "started_at": now,
                    "updated_at": now,
                }
            )
            job["state"].update({"status": "running", "queue_position": 0})
            job["state"].pop("queue_reason", None)
            if selected_session_id:
                state["session_workers"][selected_session_id] = worker_id
            result = self._public_job(state, job)
            result["payload"] = dict(job.get("payload") or {})
            result["session_id"] = selected_session_id
            return result

    def patch(self, job_id: str, **changes: Any) -> Dict[str, Any]:
        with self._locked_state(write=True) as state:
            job = state["jobs"].get(job_id)
            if job is None:
                raise KeyError(job_id)
            status = str(changes.pop("status", job["status"]))
            now = time.time()
            job["status"] = status
            job["updated_at"] = now
            if status in TERMINAL_STATUSES:
                job["finished_at"] = now
            job["state"].update(changes)
            job["state"]["status"] = status
            return self._public_job(state, job)

    def append_segment(
        self,
        job_id: str,
        update: Dict[str, Any],
    ) -> Dict[str, Any]:
        with self._locked_state(write=True) as state:
            job = state["jobs"].get(job_id)
            if job is None:
                raise KeyError(job_id)
            updates = list(job["state"].get("segment_updates") or [])
            updates.append(update)
            job["status"] = "running"
            job["updated_at"] = time.time()
            job["state"].update(
                {
                    "status": "running",
                    "segment_updates": updates,
                    "completed_segments": len(updates),
                    "video_url": update.get("video_url"),
                }
            )
            return self._public_job(state, job)

    def recover_interrupted(self, worker_id: Optional[str] = None) -> int:
        now = time.time()
        recovered = 0
        with self._locked_state(write=True) as state:
            for job in state["jobs"].values():
                if job["status"] != "running":
                    continue
                if worker_id is not None and job.get("worker_id") != worker_id:
                    continue
                recovered += 1
                job.update(
                    {
                        "status": "queued",
                        "worker_id": None,
                        "started_at": None,
                        "updated_at": now,
                    }
                )
                job["state"].update(
                    {
                        "status": "queued",
                        "error": None,
                        "queue_position": 0,
                    }
                )
        return recovered

    def reset_worker(self, worker_id: str) -> Dict[str, int]:
        """Remove stale state for a worker process that has just restarted.

        Standalone interrupted jobs are safe to requeue. Interactive jobs and
        sessions cannot be resumed because their exact CUDA KV/RNG/PixelUMM state was
        process-local, so they are failed/invalidated rather than silently
        producing a numerically different continuation.
        """
        now = time.time()
        requeued = 0
        failed = 0
        released_sessions = 0
        with self._locked_state(write=True) as state:
            affected_sessions = {
                session_id
                for session_id, assigned in state["session_workers"].items()
                if assigned == worker_id
            }
            for job in state["jobs"].values():
                if job.get("status") != "running" or job.get("worker_id") != worker_id:
                    continue
                session_id = self._job_session_id(job)
                job["worker_id"] = None
                job["updated_at"] = now
                job["started_at"] = None
                if session_id:
                    failed += 1
                    affected_sessions.add(session_id)
                    job["status"] = "failed"
                    job["finished_at"] = now
                    job["state"].update(
                        {
                            "status": "failed",
                            "error": (
                                "the assigned GPU worker restarted; the exact "
                                "interactive cache was discarded, so this "
                                "continuation was not replayed"
                            ),
                        }
                    )
                else:
                    requeued += 1
                    job["status"] = "queued"
                    job["state"].update(
                        {
                            "status": "queued",
                            "error": None,
                            "queue_position": 0,
                        }
                    )
            for session_id in affected_sessions:
                state["session_workers"].pop(session_id, None)
                if state["sessions"].pop(session_id, None) is not None:
                    released_sessions += 1
            state["workers"].pop(worker_id, None)
            legacy = state.get("worker")
            if isinstance(legacy, dict) and legacy.get("worker_id") == worker_id:
                state["worker"] = None
        return {
            "requeued_jobs": requeued,
            "failed_interactive_jobs": failed,
            "released_sessions": released_sessions,
        }

    def heartbeat(
        self,
        worker_id: str,
        status: str,
        config: Dict[str, Any],
        *,
        active_session_id: Optional[str] = None,
        resources: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._locked_state(write=True) as state:
            record = {
                "worker_id": worker_id,
                "status": status,
                "heartbeat_at": time.time(),
                "config": config,
                "active_session_id": active_session_id,
                "resources": dict(resources or {}),
            }
            state["workers"][worker_id] = record
            state["worker"] = record

    def workers(self, stale_after: float = 30.0) -> list[Dict[str, Any]]:
        with self._locked_state(write=False) as state:
            now = time.time()
            return sorted(
                (
                    self._worker_record(
                        worker,
                        stale_after=stale_after,
                        now=now,
                    )
                    for worker in state["workers"].values()
                    if isinstance(worker, dict)
                ),
                key=lambda worker: str(worker.get("worker_id") or ""),
            )

    def worker(self, stale_after: float = 30.0) -> Dict[str, Any]:
        workers = self.workers(stale_after=stale_after)
        online = [worker for worker in workers if worker["online"]]
        if not online:
            latest = max(
                workers,
                key=lambda worker: float(worker.get("heartbeat_at") or 0.0),
                default={},
            )
            return {
                **latest,
                "online": False,
                "status": "offline",
                "heartbeat_at": latest.get("heartbeat_at"),
                "config": dict(latest.get("config") or {}),
                "worker_count": len(workers),
                "online_worker_count": 0,
                "busy_worker_count": 0,
                "reserved_session_count": 0,
            }
        ready = [worker for worker in online if worker["status"] == "ready"]
        busy = [worker for worker in online if worker["status"] == "busy"]
        representative = ready[0] if ready else online[0]
        return {
            **representative,
            "online": True,
            "status": "ready"
            if ready
            else ("busy" if busy else representative["status"]),
            "config": dict(representative.get("config") or {}),
            "worker_count": len(workers),
            "online_worker_count": len(online),
            "busy_worker_count": len(busy),
            "reserved_session_count": sum(
                1 for worker in online if worker.get("active_session_id")
            ),
        }

    def clear_worker(self, worker_id: Optional[str] = None) -> None:
        with self._locked_state(write=True) as state:
            if worker_id is None:
                state["workers"] = {}
                state["worker"] = None
                return
            state["workers"].pop(worker_id, None)
            legacy = state.get("worker")
            if isinstance(legacy, dict) and legacy.get("worker_id") == worker_id:
                state["worker"] = None

    def active_job_count(self) -> int:
        with self._locked_state(write=False) as state:
            return sum(
                1
                for job in state["jobs"].values()
                if job.get("status") in {"enhancing", "queued", "running"}
            )

    def list_jobs(self, limit: int = 100) -> list[Dict[str, Any]]:
        with self._locked_state(write=False) as state:
            jobs = sorted(
                state["jobs"].values(),
                key=lambda job: float(job.get("created_at") or 0.0),
                reverse=True,
            )
            return [self._public_job(state, job) for job in jobs[: max(1, int(limit))]]

    @staticmethod
    def _generation_attempt_record(job: Dict[str, Any]) -> Dict[str, Any]:
        """Project a job to the fields needed by the shared attempt archive."""

        payload = job.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        state = job.get("state")
        state = state if isinstance(state, dict) else {}
        pe_session = payload.get("_pe_session")
        pe_session = pe_session if isinstance(pe_session, dict) else {}
        projected_payload = {
            key: payload[key]
            for key in (
                "pe_input",
                "session_id",
                "user_prompts",
                "carry_user_prompt",
                "camera_trajectory",
                "num_frames",
                "segment_boundaries",
                "segment_has_camera",
                "seed",
                "width",
                "height",
                "prompt",
            )
            if key in payload
        }
        if pe_session:
            projected_payload["_pe_session"] = {
                key: pe_session[key]
                for key in (
                    "session_id",
                    "pe_transcript_path",
                    "requested_perspective",
                    "perspective",
                    "style",
                    "subject",
                    "environment_description",
                )
                if key in pe_session
            }
        projected_state = {
            key: state[key]
            for key in (
                "status",
                "user_id",
                "dispatch_host_id",
                "prompt_enhancement_mode",
                "prompt_enhancer_reasoning_effort",
                "prompt_enhancer_seed",
                "prompt_enhancer_seed_source",
                "pe_request_id",
                "pe_prompt",
                "pe_reasoning",
                "pe_transcript_path",
                "result",
                "video_url",
                "error",
                "error_code",
            )
            if key in state
        }
        return json.loads(
            json.dumps(
                {
                    key: job.get(key)
                    for key in (
                        "id",
                        "kind",
                        "status",
                        "worker_id",
                        "created_at",
                        "updated_at",
                        "started_at",
                        "finished_at",
                    )
                }
                | {
                    "payload": projected_payload,
                    "state": projected_state,
                },
                ensure_ascii=False,
            )
        )

    def generation_attempt_records(self) -> list[Dict[str, Any]]:
        """Return compact generation records for shared audit export."""

        with self._locked_state(write=False) as state:
            records = [
                self._generation_attempt_record(job)
                for job in state["jobs"].values()
                if job.get("kind") == "generate"
            ]
        return sorted(
            records,
            key=lambda job: float(job.get("created_at") or 0.0),
        )

    def changed_generation_attempt_records(
        self,
        known_updates: Dict[str, float],
    ) -> list[Dict[str, Any]]:
        """Return compact attempt records whose timestamp has changed."""

        with self._locked_state(write=False) as state:
            records = []
            for job in state["jobs"].values():
                if job.get("kind") != "generate":
                    continue
                job_id = str(job.get("id") or "")
                updated_at = float(job.get("updated_at") or 0.0)
                if job_id and known_updates.get(job_id) == updated_at:
                    continue
                records.append(self._generation_attempt_record(job))
        return sorted(
            records,
            key=lambda job: float(job.get("updated_at") or 0.0),
        )

    def generation_attempt_record(
        self,
        job_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return one compact generation record for shared audit export."""

        with self._locked_state(write=False) as state:
            job = state["jobs"].get(job_id)
            if not isinstance(job, dict) or job.get("kind") != "generate":
                return None
            return self._generation_attempt_record(job)

    def session_job_records(self, session_id: str) -> list[Dict[str, Any]]:
        """Return JSON-safe raw records for one interactive session."""

        clean_session_id = str(session_id or "").strip()
        if not clean_session_id:
            return []
        with self._locked_state(write=False) as state:
            records = []
            for job in state["jobs"].values():
                payload = job.get("payload")
                if not isinstance(payload, dict):
                    continue
                payload_session_id = str(payload.get("session_id") or "").strip()
                pe_session = payload.get("_pe_session")
                pe_session_id = (
                    str(pe_session.get("session_id") or "").strip()
                    if isinstance(pe_session, dict)
                    else ""
                )
                if clean_session_id not in {
                    payload_session_id,
                    pe_session_id,
                }:
                    continue
                records.append(json.loads(json.dumps(job, ensure_ascii=False)))
        return sorted(
            records,
            key=lambda job: float(job.get("created_at") or 0.0),
        )

    def cancel(self, job_id: str) -> Dict[str, Any]:
        with self._locked_state(write=True) as state:
            job = state["jobs"].get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["status"] not in {"enhancing", "queued"}:
                raise ValueError(
                    f"only enhancing/queued jobs may be cancelled; "
                    f"{job_id} is {job['status']}"
                )
            now = time.time()
            job["status"] = "cancelled"
            job["updated_at"] = now
            job["finished_at"] = now
            job["state"].update(
                {
                    "status": "cancelled",
                    "error": "cancelled by operator",
                }
            )
            return self._public_job(state, job)

    def timeout_running_jobs(self, timeout_seconds: float) -> Dict[str, Any]:
        """Fail work that exceeded the shared-demo wall-clock budget.

        The worker-pool supervisor calls this before terminating the affected
        process group. Interactive CUDA state is process-local, so the owning
        session is invalidated at the same time instead of allowing a later
        request to continue from an incomplete generation.
        """
        limit = max(0.0, float(timeout_seconds))
        if limit <= 0:
            return {"job_ids": [], "worker_ids": [], "session_ids": []}

        now = time.time()
        timed_out_jobs: list[str] = []
        worker_ids: set[str] = set()
        session_ids: set[str] = set()
        with self._locked_state(write=True) as state:
            for job in state["jobs"].values():
                if (
                    job.get("status") not in {"enhancing", "running"}
                    or job.get("kind") != "generate"
                ):
                    continue
                started_at = float(
                    job.get("started_at")
                    or (
                        job.get("created_at")
                        if job.get("status") == "enhancing"
                        else 0.0
                    )
                    or 0.0
                )
                if started_at <= 0 or now - started_at < limit:
                    continue

                job_id = str(job["id"])
                worker_id = str(job.get("worker_id") or "").strip()
                was_running = job.get("status") == "running"
                session_id = self._job_session_id(job)
                elapsed = max(0.0, now - started_at)
                message = (
                    "Generation exceeded the shared-demo safety timeout "
                    f"of {limit:g} seconds and was stopped after "
                    f"{elapsed:.0f} seconds. Please create a new World and "
                    "try a shorter generation."
                )
                job.update(
                    {
                        "status": "failed",
                        "updated_at": now,
                        "finished_at": now,
                    }
                )
                job["state"].update(
                    {
                        "status": "failed",
                        "error": message,
                        "error_code": "GENERATION_TIMEOUT",
                        "recoverable": True,
                    }
                )
                timed_out_jobs.append(job_id)
                if worker_id and was_running:
                    worker_ids.add(worker_id)
                if session_id:
                    session_ids.add(session_id)
                    state["session_workers"].pop(session_id, None)
                    state["sessions"].pop(session_id, None)
                errors = list(state.get("errors") or [])
                errors.append(
                    {
                        "id": f"error_{uuid.uuid4().hex[:12]}",
                        "code": "GENERATION_TIMEOUT",
                        "message": message,
                        "job_id": job_id,
                        "worker_id": worker_id or None,
                        "recoverable": True,
                        "created_at": now,
                    }
                )
                state["errors"] = errors[-200:]

        return {
            "job_ids": timed_out_jobs,
            "worker_ids": sorted(worker_ids),
            "session_ids": sorted(session_ids),
        }

    def session_worker(self, session_id: str) -> Optional[str]:
        with self._locked_state(write=False) as state:
            value = state["session_workers"].get(session_id)
            return str(value) if value else None

    def release_session(
        self,
        session_id: str,
        *,
        worker_id: Optional[str] = None,
        delete_session: bool = True,
    ) -> bool:
        with self._locked_state(write=True) as state:
            assigned = state["session_workers"].get(session_id)
            if worker_id is not None and assigned not in (None, worker_id):
                return False
            existed = assigned is not None or session_id in state["sessions"]
            state["session_workers"].pop(session_id, None)
            if delete_session:
                state["sessions"].pop(session_id, None)
            return existed

    def scheduler_snapshot(self) -> Dict[str, Any]:
        with self._locked_state(write=False) as state:
            now = time.time()
            workers = [
                self._worker_record(worker, stale_after=30.0, now=now)
                for worker in state["workers"].values()
                if isinstance(worker, dict)
            ]
            counts: Dict[str, int] = {}
            for job in state["jobs"].values():
                status = str(job.get("status") or "unknown")
                counts[status] = counts.get(status, 0) + 1
            online = [worker for worker in workers if worker["online"]]
            return {
                "workers": sorted(
                    workers,
                    key=lambda worker: str(worker.get("worker_id") or ""),
                ),
                "worker_count": len(workers),
                "online_worker_count": len(online),
                "busy_worker_count": sum(
                    worker["status"] == "busy" for worker in online
                ),
                "reserved_session_count": len(state["session_workers"]),
                "session_workers": dict(state["session_workers"]),
                "job_counts": counts,
                "queued_jobs": counts.get("queued", 0),
                "running_jobs": counts.get("running", 0),
                "enhancing_jobs": counts.get("enhancing", 0),
            }

    def record_error(
        self,
        *,
        code: str,
        message: str,
        job_id: Optional[str] = None,
        worker_id: Optional[str] = None,
        recoverable: bool = True,
    ) -> Dict[str, Any]:
        event = {
            "id": f"error_{uuid.uuid4().hex[:12]}",
            "code": str(code),
            "message": str(message)[:2000],
            "job_id": job_id,
            "worker_id": worker_id,
            "recoverable": bool(recoverable),
            "created_at": time.time(),
        }
        with self._locked_state(write=True) as state:
            errors = list(state.get("errors") or [])
            errors.append(event)
            state["errors"] = errors[-200:]
        return event

    def recent_errors(self, limit: int = 25) -> list[Dict[str, Any]]:
        with self._locked_state(write=False) as state:
            errors = [
                dict(error)
                for error in state.get("errors") or []
                if isinstance(error, dict)
            ]
            return list(reversed(errors[-max(1, int(limit)) :]))

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._locked_state(write=False) as state:
            value = state["sessions"].get(session_id)
            return dict(value) if isinstance(value, dict) else None

    def put_session(self, session_id: str, value: Dict[str, Any]) -> None:
        with self._locked_state(write=True) as state:
            state["sessions"][session_id] = dict(value)

    def delete_session(self, session_id: str) -> None:
        with self._locked_state(write=True) as state:
            state["sessions"].pop(session_id, None)
            state["session_workers"].pop(session_id, None)

    def clear_sessions(self) -> int:
        with self._locked_state(write=True) as state:
            count = len(state["sessions"])
            state["sessions"] = {}
            state["session_workers"] = {}
            return count
