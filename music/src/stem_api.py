"""Embedded stem-separation routes for the YuE2 and MiniMax studios.

``install_stem_routes`` mounts the private stem pane API (``/api/stems/*``)
onto a studio FastAPI application: track upload, split jobs with progress
polling, and CORS-open stem delivery. Separation runs through ``studio_api``
inside the calling interpreter; uploads land under the service state dir,
splits beside them. No telemetry.
"""

from __future__ import annotations

import sys
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import studio_api  # noqa: E402

MAX_CONCURRENT_SPLITS = 2


class StemService:
    """In-process stem separation state mounted into one music studio."""

    def __init__(self, state_dir: Path, resolver: Callable[[Any], Path]) -> None:
        self.state_dir = Path(state_dir)
        self.upload_dir = self.state_dir / "uploads"
        self.split_dir = self.state_dir / "splits"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self._resolver = resolver
        self._sem = threading.Semaphore(MAX_CONCURRENT_SPLITS)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._jobs_lock = threading.Lock()

    def upload_path(self, asset_id: str) -> Path:
        """Locate a saved upload by asset id; raise FileNotFoundError if gone."""
        safe = "".join(c for c in str(asset_id) if c.isalnum())
        path = self.upload_dir / f"{safe}.wav"
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return path

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._jobs_lock:
            record = self._jobs.get(job_id)
            return dict(record) if record is not None else None

    def stem_path(self, job_id: str, name: str) -> Path | None:
        safe = "".join(c for c in name if c.isalnum() or c in "-_.")
        path = self.split_dir / job_id / f"{safe}.wav"
        return path if path.is_file() else None

    def start_split(self, ref: Any, stems: list[str], model: str,
                    device: str) -> str:
        try:
            input_path = Path(self._resolver(ref))
        except (ValueError, FileNotFoundError) as exc:
            raise LookupError(str(exc)) from exc
        if not input_path.is_file():
            raise LookupError("Select an uploaded track or a finished output.")
        job: dict[str, Any] = {
            "id": uuid.uuid4().hex[:16],
            "status": "queued",
            "stage": "queued",
            "pct": 0,
            "message": "",
            "stems": [],
            "engine": None,
            "model": None,
            "input_name": input_path.name[:80],
            "error": "",
            "created_at": time.time(),
            "stems_request": stems,
        }
        job.update({"model": model, "device": device})
        with self._jobs_lock:
            self._jobs[job["id"]] = job
        threading.Thread(target=self._run_split, args=(job, input_path),
                         daemon=True).start()
        return job["id"]

    def _set_job(self, job_id: str, **fields: Any) -> None:
        with self._jobs_lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)

    def _run_split(self, job: dict[str, Any], input_path: Path) -> None:
        def progress(stage: str, pct: int, message: str) -> None:
            if pct >= 0:
                self._set_job(job["id"], stage=stage or "separate",
                              pct=max(0, min(99, pct)), message=message)

        try:
            with self._sem:
                self._set_job(job["id"], status="running")
                out_dir = self.split_dir / job["id"]
                result = studio_api.split_wav(
                    input_path, out_dir,
                    stems=job.get("stems_request") or None,
                    model=str(job.get("model") or "htdemucs"),
                    device=str(job.get("device") or "auto"),
                    progress=progress,
                )
            self._set_job(job["id"], status="done", pct=100, message="",
                          stems=result["stems"], engine=result["engine"],
                          model=result["model"])
        except Exception as exc:  # surfaced to the UI as the job error
            self._set_job(job["id"], status="failed", message=str(exc)[:800],
                          error=str(exc)[:800])


def install_stem_routes(app: FastAPI, service: StemService) -> None:
    """Mount the private stem pane routes onto a studio application."""
    router = APIRouter(prefix="/api/stems")

    @router.post("/upload")
    async def upload(file: UploadFile) -> dict[str, Any]:
        asset_id = uuid.uuid4().hex[:16]
        destination = service.upload_dir / f"{asset_id}.wav"
        data = await file.read()
        destination.write_bytes(data)
        try:
            with wave.open(str(destination), "rb") as handle:
                seconds = round(handle.getnframes()
                                / max(1, handle.getframerate()), 3)
                rate = handle.getframerate()
        except Exception:
            destination.unlink(missing_ok=True)
            raise HTTPException(400, "Upload must be a PCM WAV file.")
        return {"asset_id": asset_id,
                "name": Path(file.filename or "track").stem[:80],
                "seconds": seconds, "rate": rate}

    @router.post("/split")
    async def split(request: Request) -> dict[str, Any]:
        payload = await request.json()
        stems = payload.get("stems")
        if isinstance(stems, list):
            stems = [str(s) for s in stems]
        elif isinstance(payload.get("preset"), str):
            stems = list(studio_api.PRESETS.get(payload["preset"])
                         or studio_api.ALL_STEMS)
        else:
            stems = list(studio_api.ALL_STEMS)
        if not stems:
            raise HTTPException(400, "Select at least one stem.")
        try:
            job_id = service.start_split(
                payload.get("input"), stems,
                model=str(payload.get("model", "htdemucs")),
                device=str(payload.get("device", "auto")),
            )
        except LookupError as exc:
            raise HTTPException(404, str(exc))
        return {"job_id": job_id}

    @router.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        record = service.get_job(job_id)
        if record is None:
            raise HTTPException(404, "Unknown split job.")
        return record

    @router.get("/jobs/{job_id}/{name}", response_class=FileResponse,
               response_model=None)
    async def stem(job_id: str, name: str) -> Any:
        path = service.stem_path(job_id, name)
        if path is None:
            raise HTTPException(404, "Stem not found.")
        return FileResponse(path, media_type="audio/wav",
                            headers={"Access-Control-Allow-Origin": "*"})

    app.include_router(router)
