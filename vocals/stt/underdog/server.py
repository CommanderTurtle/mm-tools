from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

# Set these before any model-library import, including direct uvicorn launches.
for name in (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
    "DO_NOT_TRACK",
):
    os.environ[name] = "1"

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core import (
    DEFAULT_MODEL,
    MAX_SECONDS,
    ROOT,
    VARIANTS,
    Cancelled,
    ModelManager,
    normalize_audio,
    stop_decoder,
)

ROLE = os.getenv("BARK_APP_ROLE", "ui")
if ROLE not in {"ui", "http"}:
    raise ValueError("BARK_APP_ROLE must be ui or http")
MAX_UPLOAD = int(os.getenv("BARK_MAX_UPLOAD_MB", "1024")) * 1024 * 1024
if MAX_UPLOAD <= 0:
    raise ValueError("BARK_MAX_UPLOAD_MB must be positive")
manager = ModelManager()
operation_lock = asyncio.Lock()
active_request = ""


@asynccontextmanager
async def lifespan(app):
    try:
        if ROLE == "http" and os.getenv("BARK_HTTP_AUTOLOAD", "1") == "1":
            await run_in_threadpool(manager.load, DEFAULT_MODEL)
        yield
    finally:
        await run_in_threadpool(manager.close)


app = FastAPI(title="Underdog", lifespan=lifespan)
origins = os.getenv("BARK_CORS_ORIGINS", "").strip()
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[x.strip() for x in origins.split(",") if x.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

if ROLE == "ui":
    app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")

    @app.get("/")
    async def index():
        return FileResponse(ROOT / "web/index.html")


def choose(variant):
    selected = variant or DEFAULT_MODEL
    if selected not in VARIANTS:
        raise HTTPException(400, "variant must be base, plus, clean or pro")
    if ROLE == "http" and selected != DEFAULT_MODEL:
        raise HTTPException(400, "HTTP-only mode uses BARK_MODEL from .env")
    return selected


@asynccontextmanager
async def exclusive():
    if manager.stopping:
        raise HTTPException(503, "Service is shutting down")
    if operation_lock.locked():
        raise HTTPException(423, "A model operation is active; wait for it to finish")
    async with operation_lock:
        try:
            yield
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except Cancelled as exc:
            raise HTTPException(409, str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, f"Bark failed: {exc}") from exc


@app.get("/api/health")
async def health():
    status = manager.status()
    selected = next(row for row in status["models"] if row["id"] == DEFAULT_MODEL)
    return {
        "ok": True,
        "role": ROLE,
        "default_model": DEFAULT_MODEL,
        "loaded": selected["loaded"],
        "model_present": selected["model_present"],
        "compute_type": selected["compute_type"],
        "busy": operation_lock.locked(),
        "request_id": active_request,
        "max_upload_mb": MAX_UPLOAD // 1048576,
        "max_audio_seconds": MAX_SECONDS,
        **status,
    }


@app.post("/api/load")
async def load(variant: str = Form("")):
    selected = choose(variant)
    async with exclusive():
        await run_in_threadpool(manager.load, selected)
    return {"ok": True, "variant": selected, "loaded": True}


@app.post("/api/unload")
async def unload(variant: str = Form("")):
    selected = choose(variant)
    async with exclusive():
        await run_in_threadpool(manager.unload, selected)
    return {"ok": True, "variant": selected, "loaded": False}


@app.post("/api/cancel")
async def cancel(request_id: str = ""):
    if not active_request or (request_id and request_id != active_request):
        raise HTTPException(409, "This transcription is no longer active")
    manager.cancel.set()
    stop_decoder()
    return {"ok": True, "message": "Cancellation requested at the next chunk boundary"}


def copy_upload(file, target):
    total = 0
    with target.open("wb") as stream:
        while block := file.file.read(1024 * 1024):
            total += len(block)
            if total > MAX_UPLOAD:
                raise HTTPException(413, "Audio exceeds BARK_MAX_UPLOAD_MB")
            stream.write(block)


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    variant: str = Form(""),
    language: str = Form("en"),
    operation: str = Form("auto"),
    chunk_duration: int = Form(30),
    request_id: str = Query("", pattern=r"^(?:[0-9a-f]{32})?$"),
):
    global active_request
    try:
        selected = choose(variant)
        native_operation = VARIANTS[selected]["operation"]
        if language.lower() not in {"en", "english"}:
            raise HTTPException(400, "Bark 3.0 supports English only")
        if operation not in {"auto", native_operation}:
            raise HTTPException(
                400,
                f"{selected} provides {native_operation}; select Clean for intended speech",
            )
        if not 1 <= chunk_duration <= 30:
            raise HTTPException(400, "chunk_duration must be between 1 and 30 seconds")
        async with exclusive():
            manager.cancel.clear()
            active_request = request_id or uuid.uuid4().hex
            try:
                with tempfile.TemporaryDirectory(prefix="underdog-") as temp:
                    source, wav = Path(temp) / "upload", Path(temp) / "audio.wav"
                    await run_in_threadpool(copy_upload, file, source)
                    await run_in_threadpool(
                        normalize_audio, source, wav, manager.cancel
                    )
                    if manager.cancel.is_set():
                        raise Cancelled("Transcription cancelled")
                    result = await run_in_threadpool(
                        manager.transcribe, wav, selected, chunk_duration
                    )
                    return {
                        "ok": True,
                        "operation": native_operation,
                        "results": {native_operation: result},
                    }
            finally:
                active_request = ""
    finally:
        await file.close()
