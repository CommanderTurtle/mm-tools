from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .core import LongCatEngine, SynthesisOptions

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
engine = LongCatEngine()
APP_ROLE = os.getenv("LONGCAT_APP_ROLE", "http")
SECONDARY_PORT = int(os.getenv("LONGCAT_PORT", "8230"))
ROUTER_URL = os.getenv("LONGCAT_ROUTER_URL", "http://127.0.0.1:8182").rstrip("/")


def _autoload_enabled() -> bool:
    return os.getenv("LONGCAT_AUTOLOAD", "1").strip().lower() not in {"0", "false", "no"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if _autoload_enabled():
        await asyncio.to_thread(engine.load)
    try:
        yield
    finally:
        await asyncio.to_thread(engine.unload)


app = FastAPI(title="LongCat Local Voice Lab", docs_url="/api/docs", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-LongCat-Metadata"],
)
MAX_UPLOAD = int(os.getenv("LONGCAT_MAX_UPLOAD_MB", "128")) * 1024 * 1024


async def _copy_upload(upload: UploadFile, destination: Path) -> None:
    total = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD:
                raise ValueError(
                    f"Reference audio exceeds {MAX_UPLOAD // (1024 * 1024)} MiB."
                )
            output.write(chunk)


@app.get("/api/status")
async def status() -> dict:
    return {**engine.status(), "role": APP_ROLE}


@app.get("/api/ui-config")
async def ui_config() -> dict:
    return {
        "role": APP_ROLE,
        "secondary_port": SECONDARY_PORT,
        "secondary_scheme": "http",
        "router_url": ROUTER_URL,
    }


@app.post("/api/load")
async def load() -> dict:
    try:
        await asyncio.to_thread(engine.load)
        return engine.status()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/api/unload")
async def unload() -> dict:
    try:
        await asyncio.to_thread(engine.unload)
        return engine.status()
    except Exception as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/synthesize")
async def synthesize(
    text: str = Form(...),
    prompt_text: str = Form(""),
    prompt_audio: UploadFile | None = File(None),
    steps: int = Form(16),
    guidance_strength: float = Form(4.0),
    guidance_method: str = Form("apg"),
    seed: int = Form(1024),
    duration_scale: float = Form(1.0),
) -> Response:
    temporary: Path | None = None
    try:
        if prompt_audio is not None and prompt_audio.filename:
            suffix = Path(prompt_audio.filename).suffix or ".audio"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                temporary = Path(handle.name)
            await _copy_upload(prompt_audio, temporary)
        options = SynthesisOptions(
            steps=steps,
            guidance_strength=guidance_strength,
            guidance_method=guidance_method,
            seed=seed,
            duration_scale=duration_scale,
        )
        result = await asyncio.to_thread(
            engine.synthesize,
            text,
            prompt_audio=temporary,
            prompt_text=prompt_text or None,
            options=options,
        )
        audio = io.BytesIO()
        sf.write(audio, result.waveform, result.sample_rate, format="WAV", subtype="PCM_24")
        metadata = result.metadata()
        return Response(
            content=audio.getvalue(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="longcat.wav"',
                "X-LongCat-Metadata": json.dumps(metadata, separators=(",", ":")),
                "Cache-Control": "no-store",
            },
        )
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        if prompt_audio is not None:
            await prompt_audio.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(WEB / "index.html")


app.mount("/assets", StaticFiles(directory=WEB), name="assets")
