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
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .core import LongCatEngine, SynthesisOptions

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
engine = LongCatEngine()


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
    return engine.status()


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
