"""Standalone stem studio: split local tracks into stems, one port away.

Serves the loopback-only web player (``web``) and drives the separation
scripts through ``studio_api``. Everything stays on this machine: uploads
land under ``.runtime/uploads``, splits under ``.runtime/splits``. No
telemetry.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import threading
import time
import uuid
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import studio_api  # noqa: E402

RUNTIME_ROOT = ROOT / ".runtime"
UPLOAD_DIR = RUNTIME_ROOT / "uploads"
SPLIT_DIR = RUNTIME_ROOT / "splits"
WEB_DIR = ROOT / "web"

MAX_CONCURRENT_SPLITS = 2
_split_sem = threading.Semaphore(MAX_CONCURRENT_SPLITS)
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _new_job(stage: str) -> dict[str, Any]:
    return {
        "id": uuid.uuid4().hex[:16],
        "status": "queued",
        "stage": stage,
        "pct": 0,
        "message": "",
        "stems": [],
        "engine": None,
        "model": None,
        "input_name": "",
        "error": "",
        "created_at": time.time(),
    }


def _set_job(job_id: str, **fields: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def _run_split(job: dict[str, Any], input_path: Path) -> None:
    def progress(stage: str, pct: int, message: str) -> None:
        if pct >= 0:
            _set_job(job["id"], stage=stage or "separate",
                     pct=max(0, min(99, pct)), message=message)

    try:
        with _split_sem:
            _set_job(job["id"], status="running")
            out_dir = SPLIT_DIR / job["id"]
            result = studio_api.split_wav(
                input_path, out_dir,
                stems=job.get("stems") or None,
                model=str(job.get("model") or "htdemucs"),
                device=str(job.get("device") or "auto"),
                progress=progress,
            )
        _set_job(job["id"], status="done", pct=100, message="",
                 stems=result["stems"], engine=result["engine"],
                 model=result["model"])
    except Exception as exc:  # surfaced to the UI as the job error
        _set_job(job["id"], status="failed", message=str(exc)[:800],
                 error=str(exc)[:800])


def engine_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "roformer_ckpt": bool(studio_api.ROFORMER_CKPT.is_file()),
        "gpu": False,
        "demucs": False,
        "torch": False,
    }
    try:
        import torch

        status["torch"] = True
        status["gpu"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    try:
        import demucs  # noqa: F401

        status["demucs"] = True
    except Exception:
        pass
    return status


@asynccontextmanager
async def lifespan(_app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Stem Studio", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=FileResponse, response_model=None)
async def index() -> Any:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "engines": engine_status()}


@app.get("/api/engines")
async def engines() -> dict[str, Any]:
    return engine_status()


@app.post("/api/upload")
async def upload(file: UploadFile) -> dict[str, Any]:
    asset_id = uuid.uuid4().hex[:16]
    destination = UPLOAD_DIR / f"{asset_id}.wav"
    data = await file.read()
    destination.write_bytes(data)
    try:
        with wave.open(str(destination), "rb") as handle:
            seconds = round(handle.getnframes() / max(1, handle.getframerate()), 3)
            rate = handle.getframerate()
    except Exception:
        destination.unlink(missing_ok=True)
        raise HTTPException(400, "Upload must be a PCM WAV file.")
    return {"asset_id": asset_id,
            "name": Path(file.filename or "track").stem[:80],
            "seconds": seconds, "rate": rate}


@app.post("/api/split")
async def split(request: Request) -> dict[str, Any]:
    payload = await request.json()
    asset_id = str(payload.get("input", "")).strip()
    input_path = UPLOAD_DIR / f"{asset_id}.wav"
    if not input_path.is_file():
        raise HTTPException(404, "Unknown uploaded track; upload it again.")
    stems = payload.get("stems")
    if isinstance(stems, list):
        stems = [str(s) for s in stems]
    elif isinstance(payload.get("preset"), str):
        stems = studio_api.PRESETS.get(payload["preset"])
    else:
        stems = list(studio_api.ALL_STEMS)
    job = _new_job("queued")
    job.update({
        "stems_request": stems,
        "model": str(payload.get("model", "htdemucs")),
        "device": str(payload.get("device", "auto")),
        "input_name": input_path.stem,
    })
    with _jobs_lock:
        _jobs[job["id"]] = job
    threading.Thread(target=_run_split, args=(job, input_path),
                     daemon=True).start()
    return {"job_id": job["id"]}


@app.get("/api/jobs")
async def jobs() -> list[dict[str, Any]]:
    with _jobs_lock:
        return sorted(_jobs.values(), key=lambda item: item["created_at"],
                      reverse=True)


@app.get("/api/jobs/{job_id}")
async def job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        record = _jobs.get(job_id)
        if record is None:
            raise HTTPException(404, "Unknown split job.")
        return dict(record)


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str) -> dict[str, bool]:
    with _jobs_lock:
        record = _jobs.pop(job_id, None)
    if record is not None:
        shutil.rmtree(SPLIT_DIR / job_id, ignore_errors=True)
    return {"deleted": record is not None}


@app.get("/api/stems/{job_id}/{name}", response_class=FileResponse,
         response_model=None)
async def stem(job_id: str, name: str) -> Any:
    safe = "".join(c for c in name if c.isalnum() or c in "-_.")
    path = SPLIT_DIR / job_id / f"{safe}.wav"
    if not path.is_file():
        raise HTTPException(404, "Stem not found.")
    return FileResponse(path, media_type="audio/wav",
                        headers={"Access-Control-Allow-Origin": "*"})


app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the private stem studio.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8271)
    args = parser.parse_args()
    import uvicorn

    print(f"Stem Studio: http://{args.host}:{args.port}")
    print("Separation runs locally (demucs / Mel-Band Roformer). No telemetry.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                access_log=False)


if __name__ == "__main__":
    main()
