from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .jobs import OUTPUT_ROOT, ROOT, endpoint_is_local, manager
from .editor import build_editor_document, export_document, validate_document


WEB = Path(__file__).resolve().parent / "web"
MAX_UPLOAD = int(os.getenv("REDESIGN_MAX_UPLOAD_MB", "80")) * 1024 * 1024
QWEN_ROOT = ROOT.parent / "models" / "qwen"
DEFAULT_DIFFUSERS = QWEN_ROOT / "suzukimain--extraint4stuff--Qwen-Image-Layered-Control-SDNQ-int4"
DEFAULT_NATIVE_FP8 = QWEN_ROOT / "T5B--qwen-image-layered-fp8" / "qwen_image_layered_fp8_e4m3fn.safetensors"
DEFAULT_NATIVE_COMPONENTS = QWEN_ROOT / "diffusers--hfstaff--Qwen-Image-Layered-modular"
DEFAULT_NATIVE_TEXT_ENCODER = DEFAULT_DIFFUSERS / "text_encoder"
DEFAULT_COMFY_SOURCES = {
    "diffusion": QWEN_ROOT / "appmana--diffusion--qwen-image-layered-int8convrot" / "qwen_image_layered_int8convrot.safetensors",
    "text_encoder": QWEN_ROOT / "comfy-org--text--qwen_2.5_vl_7b_fp8_scaled.safetensors" / "split_files" / "text_encoders" / "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "vae": QWEN_ROOT / "comfy-org--vae--qwen_image_layered_vae.safetensors" / "split_files" / "vae" / "qwen_image_layered_vae.safetensors",
}
app = FastAPI(title="ReDesign Local Workbench")
app.mount("/assets", StaticFiles(directory=WEB), name="assets")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/config")
def config() -> dict:
    backend = os.getenv("REDESIGN_QWEN_BACKEND", "native-fp8").strip().lower()
    model_value = os.getenv("REDESIGN_QWEN_MODEL", "").strip() or str(DEFAULT_NATIVE_FP8)
    model = Path(model_value).expanduser() if model_value else None
    comfy_sources = {
        "diffusion": os.getenv("REDESIGN_QWEN_INT8_PATH", str(DEFAULT_COMFY_SOURCES["diffusion"])),
        "text_encoder": os.getenv("REDESIGN_QWEN_TEXT_ENCODER_PATH", str(DEFAULT_COMFY_SOURCES["text_encoder"])),
        "vae": os.getenv("REDESIGN_QWEN_VAE_PATH", str(DEFAULT_COMFY_SOURCES["vae"])),
    }
    comfy_present = all(
        value and Path(os.path.expandvars(value)).expanduser().is_file()
        for value in comfy_sources.values()
    )
    native_present = bool(
        model
        and model.is_file()
        and Path(os.getenv("REDESIGN_QWEN_COMPONENTS", str(DEFAULT_NATIVE_COMPONENTS))).expanduser().joinpath("modular_model_index.json").is_file()
        and Path(os.getenv("REDESIGN_QWEN_TEXT_ENCODER_COMPONENTS", str(DEFAULT_NATIVE_TEXT_ENCODER))).expanduser().joinpath("config.json").is_file()
    )
    diffusers_present = bool(model and model.is_dir() and (model / "model_index.json").is_file())
    endpoint = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    return {
        "controller_url": endpoint,
        "controller_local": endpoint_is_local(endpoint),
        "controller_model": os.getenv("VLM_MODEL", ""),
        "qwen_backend": backend,
        "comfy_url": os.getenv("REDESIGN_COMFY_URL", "http://127.0.0.1:8188"),
        "qwen_model": str(model) if model else "",
        "qwen_present": (
            comfy_present if backend in {"comfy", "comfyui"}
            else diffusers_present if backend == "diffusers"
            else native_present
        ),
        "native_present": native_present,
        "comfy_models_present": comfy_present,
        "diffusers_present": diffusers_present,
        "comfy_sources": comfy_sources,
        "venv_present": (ROOT / ".venv/bin/python").is_file(),
        "output_root": str(OUTPUT_ROOT),
        "gpu_note": "Preferred: native mixed FP8/BF16 Diffusers with module offload. ComfyUI INT8 and SDNQ are compatibility fallbacks.",
    }


@app.post("/api/jobs")
def create_job(
    image: UploadFile = File(...),
    controller_url: str = Form(...),
    controller_model: str = Form(...),
    qwen_backend: str = Form("native-fp8"),
    comfy_url: str = Form("http://127.0.0.1:8188"),
    qwen_model: str = Form(""),
    qwen_gpus: str = Form("0"),
    qwen_pair_size: int = Form(1),
    tool_gpus: str = Form("0"),
    workers: int = Form(1),
    cpu_offload: bool = Form(False),
) -> dict:
    try:
        return manager.create(
            image.file,
            image.filename or "design.png",
            controller_url=controller_url,
            controller_model=controller_model,
            qwen_backend=qwen_backend,
            comfy_url=comfy_url,
            qwen_model=qwen_model,
            qwen_gpus=qwen_gpus,
            qwen_pair_size=max(1, qwen_pair_size),
            tool_gpus=tool_gpus,
            workers=max(1, min(workers, 8)),
            cpu_offload=cpu_offload,
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
