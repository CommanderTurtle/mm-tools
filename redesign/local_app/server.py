from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .diffusers_runtime import runtime
from .jobs import OUTPUT_ROOT, ROOT, manager
from .editor import build_editor_document, export_document, validate_document


WEB = Path(__file__).resolve().parent / "web"
MAX_UPLOAD = int(os.getenv("REDESIGN_MAX_UPLOAD_MB", "80")) * 1024 * 1024
app = FastAPI(title="ReDesign Local Workbench")
app.mount("/assets", StaticFiles(directory=WEB), name="assets")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/config")
def config() -> dict:
    model = runtime.status()
    return {
        "backend": "diffusers",
        "model": model,
        "venv_present": (ROOT / ".venv/bin/python").is_file(),
        "output_root": str(OUTPUT_ROOT),
        "gpu_note": (
            "Direct local QwenImageLayeredPipeline with transformer group offload. "
            "No controller, vLLM, OpenAI-compatible endpoint, or ComfyUI process is contacted."
        ),
    }


@app.get("/api/model")
def model_status() -> dict:
    return runtime.status()


@app.post("/api/model/load")
def model_load() -> dict:
    try:
        return runtime.load_async()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/model/unload")
def model_unload() -> dict:
    try:
        return runtime.unload_async()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/jobs")
def create_job(
    image: UploadFile = File(...),
    layers: int = Form(4),
    steps: int = Form(50),
    resolution: int = Form(640),
    cfg: float = Form(4.0),
    seed: int = Form(777),
) -> dict:
    try:
        return manager.create(
            image.file,
            image.filename or "design.png",
            layers=layers,
            steps=steps,
            resolution=resolution,
            cfg=cfg,
            seed=seed,
            max_bytes=MAX_UPLOAD,
        ).public()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        image.file.close()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, after: int = 0) -> dict:
    try:
        job = manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    lines = [line for sequence, line in job.logs if sequence > after]
    return {**job.public(), "logs": lines, "log_cursor": job.log_cursor}


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str) -> dict:
    try:
        return manager.stop(job_id).public()
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


def safe_artifact(job_id: str, relative: str) -> Path:
    try:
        root = Path(manager.get(job_id).output_dir).resolve()
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    target = (root / relative).resolve()
    if root not in target.parents or not target.is_file() or target.is_symlink():
        raise HTTPException(404, "Artifact not found.")
    return target


@app.get("/api/jobs/{job_id}/artifact")
def artifact(job_id: str, path: str) -> FileResponse:
    return FileResponse(safe_artifact(job_id, path))


@app.get("/api/jobs/{job_id}/input")
def job_input(job_id: str) -> FileResponse:
    try:
        target = Path(manager.get(job_id).input_path)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(target)


@app.get("/api/jobs/{job_id}/layers")
def layers(job_id: str) -> dict:
    try:
        job = manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    candidates = list(Path(job.output_dir).rglob("parse.json"))
    if not candidates:
        return {"elements": []}
    try:
        with Image.open(job.input_path) as source:
            width, height = source.size
        document = build_editor_document(
            root=Path(job.output_dir).resolve(),
            parse_path=candidates[0],
            canvas_width=width,
            canvas_height=height,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(500, f"Could not read parse.json: {exc}") from exc
    for layer in document["layers"]:
        if layer.get("asset_path"):
            layer["asset_url"] = f"/api/jobs/{job_id}/artifact?path={quote(layer['asset_path'])}"
    return document


@app.post("/api/jobs/{job_id}/editor")
def save_editor(job_id: str, document: dict = Body(...)) -> dict:
    try:
        job = manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    try:
        result = export_document(Path(job.output_dir).resolve(), validate_document(document))
    except (OSError, ValueError) as exc:
        raise HTTPException(400, f"Could not export editable document: {exc}") from exc
    return {"ok": True, **result, "artifacts": job.public()["artifacts"]}
