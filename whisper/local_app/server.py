from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import __version__
from .core import manager, normalize_audio


ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
MAX_UPLOAD = int(os.getenv("CW2_MAX_UPLOAD_MB", "1024")) * 1024 * 1024

def _autoload_enabled() -> bool:
    return os.getenv("CW2_AUTOLOAD", "1").strip().lower() not in {"0", "false", "no"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if _autoload_enabled():
        await run_in_threadpool(manager.load)
    try:
        yield
    finally:
        await run_in_threadpool(manager.unload)


app = FastAPI(title="CrisperWhisper Local", version=__version__, lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=WEB), name="assets")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health() -> dict:
    cfg = manager.config
    return {
        "ok": True,
        "loaded": manager.loaded,
        "backend": cfg.backend,
        "device": cfg.device,
        "compute_type": cfg.compute_type,
        "model_path": str(cfg.model_path),
        "model_present": cfg.model_path.is_dir(),
    }


@app.post("/api/unload")
def unload() -> dict:
    manager.unload()
    return {"ok": True, "loaded": False}


@app.post("/api/load")
def load() -> dict:
    manager.load()
    return {"ok": True, "loaded": True}


def _copy_upload(upload: UploadFile, target: Path) -> None:
    total = 0
    with target.open("wb") as output:
        while chunk := upload.file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD:
                raise ValueError(f"Upload exceeds {MAX_UPLOAD // (1024 * 1024)} MiB limit.")
            output.write(chunk)


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    operation: str = Form("both"),
    language: str = Form("en"),
    transcript: str = Form(""),
    word_timestamps: bool = Form(True),
    strategy: str = Form("continuation"),
    chunk_duration: float = Form(30.0),
    stride: float = Form(26.0),
    context_words: int = Form(12),
    max_new_tokens: int = Form(256),
    hotwords: str = Form(""),
) -> dict:
    if operation not in {"both", "verbatim", "intended", "verbatimize", "align"}:
        raise HTTPException(400, "Unsupported operation.")
    if not 1 <= chunk_duration <= 30:
        raise HTTPException(400, "Chunk duration must be between 1 and 30 seconds.")
    if not 0 < stride <= chunk_duration:
        raise HTTPException(400, "Stride must be positive and no longer than the chunk.")

    suffix = Path(file.filename or "audio").suffix[:12]
    try:
        with tempfile.TemporaryDirectory(prefix="cw2-") as temp:
            temp_dir = Path(temp)
            source = temp_dir / f"source{suffix}"
            wav = temp_dir / "audio.wav"
            await run_in_threadpool(_copy_upload, file, source)
            await run_in_threadpool(normalize_audio, source, wav)
            result = await run_in_threadpool(
                manager.run,
                wav,
                operation=operation,
                language=language,
                transcript=transcript,
                word_timestamps=word_timestamps,
                strategy=strategy,
                chunk_duration=chunk_duration,
                stride=stride,
                context_words=context_words,
                max_new_tokens=max_new_tokens,
                hotwords=[part.strip() for part in hotwords.split(",") if part.strip()],
            )
            return {"ok": True, "operation": operation, "results": result}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Transcription failed: {exc}") from exc
    finally:
        await file.close()
