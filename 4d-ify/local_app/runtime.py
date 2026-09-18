from __future__ import annotations

import json
import os
import queue
import shutil
import sqlite3
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


TERMINAL_STATES = frozenset({"complete", "failed", "cancelled", "interrupted"})
ACTIVE_STATES = frozenset({"queued", "running", "cancelling"})


def _now() -> float:
    return time.time()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _decode(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


@dataclass(slots=True)
class StudioOutput:
    path: str | Path
    kind: str
    label: str = "Output"
    media_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StudioAdapter:
    """Small contract implemented by a project's native inference adapter."""

    def __init__(self, project_root: Path, runtime_root: Path) -> None:
        self.project_root = project_root
        self.runtime_root = runtime_root

    def health(self) -> dict[str, Any]:
        return {"ready": True, "loaded": False, "details": []}

    def validate(self, request: dict[str, Any], resolve_asset: Callable[[str], Path]) -> None:
        del request, resolve_asset

    def run(self, request: dict[str, Any], context: "StudioContext") -> list[StudioOutput]:
        raise NotImplementedError

    def load(self) -> dict[str, Any]:
        return self.health()

    def unload(self) -> dict[str, Any]:
        return self.health()

    def close(self) -> None:
        pass

    def vox_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("This studio does not expose an OpenAI-compatible audio route.")


class JobStore:
    def __init__(self, database: Path) -> None:
        database.parent.mkdir(parents=True, exist_ok=True)
        self.database = database
        self._local = threading.local()
        self._schema()

    def _connect(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self.database, timeout=30, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA foreign_keys=ON")
            self._local.connection = connection
        return connection

    def _schema(self) -> None:
        connection = sqlite3.connect(self.database)
        try:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    request_json TEXT NOT NULL,
                    outputs_json TEXT NOT NULL DEFAULT '[]',
                    log_json TEXT NOT NULL DEFAULT '[]',
                    error TEXT,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created_at DESC);
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    stored_name TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS assets_created ON assets(created_at DESC);
                """
            )
            connection.execute(
                "UPDATE jobs SET status='interrupted', stage='Interrupted by server restart', "
                "finished_at=?, updated_at=? WHERE status IN ('running','cancelling')",
                (_now(), _now()),
            )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _public_job(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "mode": row["mode"],
            "status": row["status"],
            "stage": row["stage"],
            "progress": row["progress"],
            "request": _decode(row["request_json"], {}),
            "outputs": _decode(row["outputs_json"], []),
            "log": _decode(row["log_json"], []),
            "error": row["error"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "updated_at": row["updated_at"],
        }

    def create_job(self, mode: str, request: dict[str, Any]) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        now = _now()
        self._connect().execute(
            "INSERT INTO jobs(id,mode,status,stage,progress,request_json,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (job_id, mode, "queued", "Waiting for the GPU", 0.0, _json(request), now, now),
        )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any]:
        row = self._connect().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._public_job(row)

    def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._connect().execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 500)),)
        ).fetchall()
        return [self._public_job(row) for row in rows]

    def update_job(self, job_id: str, **values: Any) -> dict[str, Any]:
        if not values:
            return self.get_job(job_id)
        permitted = {
            "status", "stage", "progress", "outputs_json", "log_json", "error",
            "started_at", "finished_at",
        }
        unknown = set(values) - permitted
        if unknown:
            raise ValueError(f"Unsupported job fields: {sorted(unknown)}")
        values["updated_at"] = _now()
        setters = ",".join(f"{key}=?" for key in values)
        self._connect().execute(
            f"UPDATE jobs SET {setters} WHERE id=?", (*values.values(), job_id)
        )
        return self.get_job(job_id)

    def append_log(self, job_id: str, level: str, message: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        entries = job["log"][-499:]
        entries.append({"at": _now(), "level": level, "message": str(message)[:6000]})
        return self.update_job(job_id, log_json=_json(entries))

    def delete_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job["status"] not in TERMINAL_STATES:
            raise RuntimeError("Only finished jobs can be removed from the library.")
        self._connect().execute("DELETE FROM jobs WHERE id=?", (job_id,))

    def add_asset(self, asset_id: str, name: str, stored_name: str, media_type: str, size: int) -> dict[str, Any]:
        now = _now()
        self._connect().execute(
            "INSERT INTO assets(id,name,stored_name,media_type,size,created_at) VALUES(?,?,?,?,?,?)",
            (asset_id, name, stored_name, media_type, size, now),
        )
        return self.get_asset(asset_id)

    def get_asset(self, asset_id: str) -> dict[str, Any]:
        row = self._connect().execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone()
        if row is None:
            raise KeyError(asset_id)
        return dict(row)

    def list_assets(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._connect().execute(
            "SELECT * FROM assets ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 500)),)
        ).fetchall()
        return [dict(row) for row in rows]

    def delete_asset(self, asset_id: str) -> dict[str, Any]:
        asset = self.get_asset(asset_id)
        self._connect().execute("DELETE FROM assets WHERE id=?", (asset_id,))
        return asset


class StudioContext:
    def __init__(
        self,
        job_id: str,
        output_dir: Path,
        assets_dir: Path,
        store: JobStore,
        cancel_event: threading.Event,
    ) -> None:
        self.job_id = job_id
        self.output_dir = output_dir
        self.assets_dir = assets_dir
        self.store = store
        self.cancel_event = cancel_event
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def update(self, stage: str, progress: float | None = None, message: str | None = None) -> None:
        values: dict[str, Any] = {"stage": str(stage)[:500]}
        if progress is not None:
            values["progress"] = max(0.0, min(float(progress), 1.0))
        self.store.update_job(self.job_id, **values)
        if message:
            self.log(message)
        self.check_cancelled()

    def log(self, message: str, level: str = "info") -> None:
        self.store.append_log(self.job_id, level, message)

    def asset(self, asset_id: str) -> Path:
        asset = self.store.get_asset(asset_id)
        path = (self.assets_dir / asset["stored_name"]).resolve()
        if self.assets_dir.resolve() not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Uploaded asset {asset_id!r} is unavailable.")
        return path

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelled("Cancelled by user")

    def run_process(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        progress_parser: Callable[[str], tuple[str, float] | None] | None = None,
    ) -> None:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            while True:
                if self.cancel_event.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise JobCancelled("Cancelled by user")
                line = process.stdout.readline()
                if line:
                    clean = line.rstrip()
                    if clean:
                        self.log(clean)
                        parsed = progress_parser(clean) if progress_parser else None
                        if parsed:
                            self.update(parsed[0], parsed[1])
                if process.poll() is not None:
                    for tail in process.stdout:
                        clean = tail.rstrip()
                        if clean:
                            self.log(clean)
                    break
                if not line:
                    time.sleep(0.05)
            if process.returncode:
                raise RuntimeError(f"Native runtime exited with code {process.returncode}.")
        finally:
            if process.poll() is None:
                process.kill()


class JobCancelled(RuntimeError):
    pass


class JobRunner:
    def __init__(self, store: JobStore, adapter: StudioAdapter, runtime_root: Path) -> None:
        self.store = store
        self.adapter = adapter
        self.runtime_root = runtime_root
        self.assets_dir = runtime_root / "assets"
        self.outputs_dir = runtime_root / "outputs"
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._work, name="mm-tools-studio-gpu", daemon=True)
        self._thread.start()
        for job in reversed(store.list_jobs(500)):
            if job["status"] == "queued":
                self._queue.put(job["id"])

    @property
    def busy(self) -> bool:
        return any(job["status"] in {"running", "cancelling"} for job in self.store.list_jobs(20))

    def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        mode = str(request.get("mode", "")).strip()
        if not mode:
            raise ValueError("A studio mode is required.")
        self.adapter.validate(request, self.resolve_asset)
        job = self.store.create_job(mode, request)
        self._queue.put(job["id"])
        return job

    def model_action(self, action: str) -> dict[str, Any]:
        """Change model residency without racing the single-GPU worker."""
        if action not in {"load", "unload"}:
            raise ValueError(f"Unknown model action: {action}")
        with self._lock:
            callback = self.adapter.load if action == "load" else self.adapter.unload
            return callback()

    def resolve_asset(self, asset_id: str) -> Path:
        asset = self.store.get_asset(asset_id)
        path = (self.assets_dir / asset["stored_name"]).resolve()
        if self.assets_dir.resolve() not in path.parents or not path.is_file():
            raise FileNotFoundError(asset_id)
        return path

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if job["status"] == "queued":
            event = self._cancel.setdefault(job_id, threading.Event())
            event.set()
            return self.store.update_job(
                job_id,
                status="cancelled",
                stage="Cancelled before generation",
                finished_at=_now(),
            )
        if job["status"] == "running":
            self._cancel.setdefault(job_id, threading.Event()).set()
            return self.store.update_job(job_id, status="cancelling", stage="Stopping safely")
        return job

    def _public_outputs(self, job_id: str, outputs: list[StudioOutput], output_dir: Path) -> list[dict[str, Any]]:
        public: list[dict[str, Any]] = []
        root = output_dir.resolve()
        for index, output in enumerate(outputs):
            source = Path(output.path).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Adapter output does not exist: {source.name}")
            if root not in source.parents:
                destination = output_dir / source.name
                if destination.resolve() != source:
                    shutil.copy2(source, destination)
                source = destination.resolve()
            relative = source.relative_to(root).as_posix()
            public.append(
                {
                    "id": f"{index}-{source.name}",
                    "name": source.name,
                    "relative": relative,
                    "kind": output.kind,
                    "label": output.label,
                    "media_type": output.media_type,
                    "size": source.stat().st_size,
                    "metadata": output.metadata,
                    "url": f"/api/jobs/{job_id}/outputs/{relative}",
                }
            )
        return public

    def _work(self) -> None:
        while not self._closed.is_set():
            job_id = self._queue.get()
            if job_id is None:
                return
            try:
                job = self.store.get_job(job_id)
            except KeyError:
                continue
            if job["status"] != "queued":
                continue
            cancel = self._cancel.setdefault(job_id, threading.Event())
            output_dir = self.outputs_dir / job_id
            context = StudioContext(job_id, output_dir, self.assets_dir, self.store, cancel)
            try:
                self.store.update_job(
                    job_id,
                    status="running",
                    stage="Preparing native runtime",
                    progress=0.01,
                    started_at=_now(),
                    error=None,
                )
                context.log(f"Started {job['mode']} on the private GPU queue.")
                with self._lock:
                    outputs = self.adapter.run(job["request"], context)
                context.check_cancelled()
                public = self._public_outputs(job_id, outputs, output_dir)
                manifest = {
                    "job": job_id,
                    "mode": job["mode"],
                    "created_at": job["created_at"],
                    "finished_at": _now(),
                    "request": job["request"],
                    "outputs": public,
                }
                (output_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                self.store.update_job(
                    job_id,
                    status="complete",
                    stage="Ready",
                    progress=1.0,
                    outputs_json=_json(public),
                    finished_at=_now(),
                )
                context.log(f"Completed with {len(public)} artifact(s).")
            except JobCancelled as exc:
                self.store.update_job(
                    job_id,
                    status="cancelled",
                    stage="Cancelled",
                    error=str(exc),
                    finished_at=_now(),
                )
                context.log(str(exc), "warning")
            except Exception as exc:  # adapter errors must survive in durable history
                detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                self.store.update_job(
                    job_id,
                    status="failed",
                    stage="Generation failed",
                    error=str(exc)[:8000],
                    finished_at=_now(),
                )
                context.log(detail[-16000:], "error")
            finally:
                self._cancel.pop(job_id, None)

    def close(self) -> None:
        self._closed.set()
        for event in self._cancel.values():
            event.set()
        self._queue.put(None)
        self._thread.join(timeout=10)
        with self._lock:
            self.adapter.close()


def gpu_snapshot() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        rows = []
        for line in result.stdout.splitlines():
            name, total, used, free, driver = (part.strip() for part in line.split(",", 4))
            rows.append(
                {
                    "name": name,
                    "memory_total_mib": int(total),
                    "memory_used_mib": int(used),
                    "memory_free_mib": int(free),
                    "driver": driver,
                }
            )
        return {"available": bool(rows), "devices": rows}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"available": False, "devices": []}
