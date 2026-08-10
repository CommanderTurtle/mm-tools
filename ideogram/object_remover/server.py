from __future__ import annotations

import asyncio
import os
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .core import decode_image, encode_png, fuzzy_mask, remove_object
from .model_backend import ModelUnavailable, models


WEB = Path(__file__).resolve().parent / "web"
MAX_UPLOAD = int(os.getenv("OBJECT_REMOVER_MAX_UPLOAD_MB", "128")) * 1024 * 1024
app = FastAPI(title="Local Object Remover")
app.mount("/assets", StaticFiles(directory=WEB), name="assets")


async def read_upload(upload: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(1024 * 1024):
        total += len(chunk)
        if total > MAX_UPLOAD:
            raise ValueError(
                f"Upload exceeds {MAX_UPLOAD // (1024 * 1024)} MiB."
            )
        chunks.append(chunk)
    return b"".join(chunks)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/models")
def model_status() -> dict:
    return models.status()


@app.post("/api/models/{engine}/load")
async def load_model(engine: str) -> dict:
    try:
        return await asyncio.to_thread(models.load, engine)
    except (ModelUnavailable, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Could not load {engine}: {exc}") from exc


@app.post("/api/models/{engine}/unload")
async def unload_model(engine: str) -> dict:
    try:
        return await asyncio.to_thread(models.unload, engine)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/remove")
async def remove(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    engine: str = Form("objectclear"),
    method: str = Form("telea"),
    radius: float = Form(5),
    grow: int = Form(5),
    feather: int = Form(4),
    steps: int = Form(20),
    guidance: float = Form(2.5),
    seed: int = Form(42),
) -> Response:
    try:
        image_bytes = await read_upload(image)
        mask_bytes = await read_upload(mask)
        if engine == "objectclear":
            result = await asyncio.to_thread(
                models.remove_object,
                image_bytes,
                mask_bytes,
                steps=max(4, min(steps, 60)),
                guidance=max(0.0, min(guidance, 10.0)),
                seed=seed,
            )
            return Response(result, media_type="image/png")
        if engine != "opencv":
            raise ValueError("engine must be objectclear or opencv")
        if method not in {"telea", "navier-stokes"}:
            raise ValueError("method must be telea or navier-stokes")
        source = decode_image(image_bytes)
        selection = decode_image(mask_bytes)
        result = remove_object(
            source,
            selection,
            method=method,
            radius=max(1.0, min(radius, 100.0)),
            grow=max(0, min(grow, 64)),
            feather=max(0, min(feather, 64)),
        )
        return Response(encode_png(result), media_type="image/png")
    except ModelUnavailable as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ValueError, cv2.error) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Object removal failed: {exc}") from exc
    finally:
        await image.close(); await mask.close()


@app.post("/api/background")
async def background(image: UploadFile = File(...)) -> Response:
    try:
        result = await asyncio.to_thread(models.remove_background, await read_upload(image))
        return Response(result, media_type="image/png")
    except ModelUnavailable as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Background removal failed: {exc}") from exc
    finally:
        await image.close()


@app.post("/api/fuzzy")
async def fuzzy(
    image: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    tolerance: int = Form(24),
) -> Response:
    try:
        source = decode_image(await read_upload(image))
        mask = fuzzy_mask(source, x, y, max(0, min(tolerance, 255)))
        return Response(encode_png(mask), media_type="image/png")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except cv2.error as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        await image.close()
