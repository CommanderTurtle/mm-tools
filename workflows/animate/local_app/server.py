from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import mimetypes
import os
import re
import secrets
import shutil
import signal
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from local_app.runtime import ACTIVE_STATES, TERMINAL_STATES, JobRunner, JobStore, StudioAdapter, gpu_snapshot


ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent / "web"
SAFE_NAME = re.compile(r"[^A-Za-z0-9._()\[\] -]+")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Studio manifest must be a JSON object.")
    return value


def _load_adapter(path: Path, project_root: Path, runtime_root: Path) -> StudioAdapter:
    spec = importlib.util.spec_from_file_location(f"mmtools_adapter_{path.parent.name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load studio adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter_type = getattr(module, "Adapter", None)
    if not isinstance(adapter_type, type) or not issubclass(adapter_type, StudioAdapter):
        raise TypeError(f"{path} must expose Adapter(StudioAdapter).")
    return adapter_type(project_root, runtime_root)


def _within(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _clean_name(value: str) -> str:
    name = SAFE_NAME.sub("_", Path(value).name).strip(" ._")
    return name[:180] or "asset.bin"


def _mime(path: Path, explicit: str | None = None) -> str:
    if explicit and explicit != "application/octet-stream":
        return explicit.split(";", 1)[0]
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _ratio(value: Any) -> float:
    text = str(value or "").strip()
    if not text or text in {"0/0", "N/A"}:
        return 0.0
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    return float(text)


def _probe_video(path: Path) -> dict[str, Any]:
    """Read local container metadata without decoding or uploading anywhere."""

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:stream_tags=rotate:stream_side_data=rotation:format=duration",
                "-of", "json", str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(result.stdout)
        stream = (payload.get("streams") or [])[0]
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError, IndexError) as exc:
        raise ValueError("The video metadata could not be read with ffprobe.") from exc

    fps = _ratio(stream.get("avg_frame_rate")) or _ratio(stream.get("r_frame_rate"))
    duration = float(stream.get("duration") or (payload.get("format") or {}).get("duration") or 0)
    raw_frames = str(stream.get("nb_frames") or "").strip()
    source_frames = int(raw_frames) if raw_frames.isdigit() else int(round(duration * fps))
    width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
    side_data = stream.get("side_data_list") or []
    rotation = next((item.get("rotation") for item in side_data if item.get("rotation") is not None), None)
    if rotation is None:
        rotation = (stream.get("tags") or {}).get("rotate", 0)
    if abs(int(float(rotation or 0))) % 180 == 90:
        width, height = height, width
    if fps <= 0 or duration <= 0 or source_frames <= 0 or width <= 0 or height <= 0:
        raise ValueError("The selected file does not expose a usable video duration and frame rate.")

    # Preserve the source dimensions whenever Animate supports them. Larger or
    # unusually small clips are fitted to the 256–2160 canvas while retaining
    # their aspect ratio, then aligned to the model's eight-pixel contract.
    scale = min(1.0, 2160 / width, 2160 / height)
    if min(width * scale, height * scale) < 256:
        scale = max(scale, 256 / min(width, height))
    output_width = max(256, min(2160, int(round(width * scale / 8)) * 8))
    output_height = max(256, min(2160, int(round(height * scale / 8)) * 8))
    bounded = max(17, min(1921, source_frames))
    wan_frames = min(1921, bounded + ((1 - bounded) % 4))
    return {
        "duration": duration,
        "fps": fps,
        "source_frames": source_frames,
        "wan_frames": wan_frames,
        "source_width": width,
        "source_height": height,
        "output_width": output_width,
        "output_height": output_height,
        "limited": source_frames > 1921,
    }


def build_application(manifest: dict[str, Any], project_root: Path, adapter_path: Path) -> FastAPI:
    project_id = str(manifest.get("id") or project_root.name)
    runtime_root = project_root / ".runtime" / "studio"
    runtime_root.mkdir(parents=True, exist_ok=True)
    store = JobStore(runtime_root / "studio.sqlite3")
    adapter = _load_adapter(adapter_path, project_root, runtime_root)
    runner = JobRunner(store, adapter, runtime_root)
    token = os.getenv("MM_STUDIO_TOKEN", "").strip()
    api_only = os.getenv("MM_STUDIO_API_ONLY", "0").strip() == "1"
    max_upload = int(manifest.get("max_upload_bytes", 8 * 1024**3))
    generation = 0

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal generation
        generation += 1
        try:
            yield
        finally:
            runner.close()

    app = FastAPI(
        title=str(manifest.get("title", project_id)),
        version="1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def private_headers(request: Request, call_next):
        if token and request.url.path.startswith(("/api/", "/v1/")):
            supplied = request.headers.get("authorization", "")
            if not secrets.compare_digest(supplied, f"Bearer {token}"):
                return JSONResponse({"detail": "A local studio bearer token is required."}, status_code=401)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api/") else "no-cache"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = "camera=(), geolocation=(), payment=(), usb=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; "
            "style-src 'self'; script-src 'self'; connect-src 'self'; worker-src 'self' blob:; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    def public_asset(asset: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": asset["id"],
            "name": asset["name"],
            "media_type": asset["media_type"],
            "size": asset["size"],
            "created_at": asset["created_at"],
            "url": f"/api/assets/{asset['id']}/content",
        }

    @app.get("/", response_class=HTMLResponse, response_model=None)
    async def index() -> Any:
        if api_only:
            return JSONResponse(
                {
                    "service": manifest.get("id"),
                    "mode": "api-only",
                    "health": "/api/health",
                    "jobs": "/api/jobs",
                    "openai_audio": "/v1/audio/speech",
                }
            )
        return FileResponse(WEB / "index.html")

    @app.get("/api/meta")
    async def meta() -> dict[str, Any]:
        return {
            "manifest": manifest,
            "privacy": {
                "local_only": True,
                "telemetry": False,
                "cors": False,
                "token_enabled": bool(token),
            },
            "runtime_generation": generation,
            "jobs": store.list_jobs(80),
            "assets": [public_asset(asset) for asset in store.list_assets(80)],
        }

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        try:
            adapter_health = adapter.health()
        except Exception as exc:
            adapter_health = {"ready": False, "loaded": False, "details": [str(exc)]}
        disk = shutil.disk_usage(runtime_root)
        return {
            "ok": bool(adapter_health.get("ready")),
            "busy": runner.busy,
            "adapter": adapter_health,
            "gpu": gpu_snapshot(),
            "disk": {"free": disk.free, "total": disk.total},
            "queue": sum(job["status"] == "queued" for job in store.list_jobs(500)),
        }

    @app.post("/api/models/load")
    async def load_models() -> dict[str, Any]:
        if runner.busy:
            raise HTTPException(409, "Wait for the active GPU job before changing model residency.")
        try:
            return await asyncio.to_thread(runner.model_action, "load")
        except Exception as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/models/unload")
    async def unload_models() -> dict[str, Any]:
        if runner.busy:
            raise HTTPException(409, "Wait for the active GPU job before changing model residency.")
        try:
            return await asyncio.to_thread(runner.model_action, "unload")
        except Exception as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/assets")
    async def upload_asset(
        request: Request,
        x_file_name: str = Header(default="asset.bin"),
        content_length: int | None = Header(default=None),
    ) -> dict[str, Any]:
        if content_length is not None and content_length > max_upload:
            raise HTTPException(413, f"The studio upload limit is {max_upload} bytes.")
        asset_id = secrets.token_hex(16)
        original = _clean_name(x_file_name)
        stored = f"{asset_id}-{original}"
        destination = runner.assets_dir / stored
        size = 0
        digest = hashlib.sha256()
        try:
            with destination.open("xb") as handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > max_upload:
                        raise HTTPException(413, f"The studio upload limit is {max_upload} bytes.")
                    digest.update(chunk)
                    handle.write(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        if not size:
            destination.unlink(missing_ok=True)
            raise HTTPException(400, "Empty uploads are not accepted.")
        media_type = _mime(destination, request.headers.get("content-type"))
        asset = store.add_asset(asset_id, original, stored, media_type, size)
        result = public_asset(asset)
        result["sha256"] = digest.hexdigest()
        return result

    @app.get("/api/assets")
    async def assets() -> list[dict[str, Any]]:
        return [public_asset(asset) for asset in store.list_assets(200)]

    @app.get("/api/assets/{asset_id}/content")
    async def asset_content(asset_id: str) -> FileResponse:
        try:
            asset = store.get_asset(asset_id)
        except KeyError as exc:
            raise HTTPException(404, "Asset not found.") from exc
        path = runner.assets_dir / asset["stored_name"]
        if not _within(runner.assets_dir, path) or not path.is_file():
            raise HTTPException(404, "Asset file is missing.")
        return FileResponse(path, media_type=asset["media_type"], filename=asset["name"])

    @app.get("/api/assets/{asset_id}/probe")
    async def probe_asset(asset_id: str) -> dict[str, Any]:
        try:
            asset = store.get_asset(asset_id)
        except KeyError as exc:
            raise HTTPException(404, "Asset not found.") from exc
        path = runner.assets_dir / asset["stored_name"]
        if not _within(runner.assets_dir, path) or not path.is_file():
            raise HTTPException(404, "Asset file is missing.")
        if not str(asset["media_type"]).startswith("video/"):
            raise HTTPException(422, "Only video assets have frame metadata.")
        try:
            return await asyncio.to_thread(_probe_video, path)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.delete("/api/assets/{asset_id}")
    async def delete_asset(asset_id: str) -> dict[str, bool]:
        for job in store.list_jobs(500):
            if job["status"] in ACTIVE_STATES and asset_id in _asset_ids(job["request"]):
                raise HTTPException(409, "That upload belongs to an active job.")
        try:
            asset = store.delete_asset(asset_id)
        except KeyError as exc:
            raise HTTPException(404, "Asset not found.") from exc
        (runner.assets_dir / asset["stored_name"]).unlink(missing_ok=True)
        return {"deleted": True}

    @app.post("/api/jobs")
    async def create_job(request: Request) -> dict[str, Any]:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("Job payload must be an object.")
            return runner.submit(payload)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/jobs")
    async def jobs(limit: int = 100) -> list[dict[str, Any]]:
        return store.list_jobs(limit)

    @app.get("/api/jobs/{job_id}")
    async def job(job_id: str) -> dict[str, Any]:
        try:
            return store.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Job not found.") from exc

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return runner.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Job not found.") from exc

    @app.delete("/api/jobs/{job_id}")
    async def delete_job(job_id: str) -> dict[str, bool]:
        try:
            store.delete_job(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Job not found.") from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"deleted": True}

    @app.get("/api/jobs/{job_id}/outputs/{relative:path}")
    async def job_output(job_id: str, relative: str, download: bool = False) -> FileResponse:
        try:
            job = store.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Job not found.") from exc
        output = next((item for item in job["outputs"] if item.get("relative") == relative), None)
        if output is None:
            raise HTTPException(404, "Output not found in this job.")
        root = runner.outputs_dir / job_id
        path = root / relative
        if not _within(root, path) or not path.is_file():
            raise HTTPException(404, "Output file is missing.")
        return FileResponse(
            path,
            media_type=output.get("media_type") or _mime(path),
            filename=path.name if download else None,
        )

    @app.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        async def stream():
            revision = ""
            while not await request.is_disconnected():
                jobs_now = store.list_jobs(80)
                signature = "|".join(
                    f"{item['id']}:{item['updated_at']}:{item['status']}" for item in jobs_now
                )
                if signature != revision:
                    revision = signature
                    payload = _safe_event({"jobs": jobs_now, "at": time.time()})
                    yield f"event: jobs\ndata: {payload}\n\n"
                else:
                    yield ": keepalive\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/audio/speech", response_model=None)
    async def vox_speech(request: Request) -> Any:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("Audio request must be an object.")
            studio_request = adapter.vox_request(payload)
            job = runner.submit(studio_request)
        except (ValueError, KeyError, NotImplementedError) as exc:
            raise HTTPException(422, str(exc)) from exc
        if payload.get("async"):
            return JSONResponse({"id": job["id"], "status": job["status"]}, status_code=202)
        timeout = min(max(float(payload.get("timeout", 1800)), 1.0), 7200.0)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = store.get_job(job["id"])
            if current["status"] in TERMINAL_STATES:
                break
            await asyncio.sleep(0.5)
        else:
            return JSONResponse({"id": job["id"], "status": "processing"}, status_code=202)
        if current["status"] != "complete" or not current["outputs"]:
            raise HTTPException(500, current.get("error") or "Speech generation did not complete.")
        audio = next((item for item in current["outputs"] if item.get("kind") == "audio"), current["outputs"][0])
        path = runner.outputs_dir / job["id"] / audio["relative"]
        return FileResponse(path, media_type=audio.get("media_type") or _mime(path), filename=path.name)

    app.mount("/assets", StaticFiles(directory=WEB), name="assets")
    return app


def _asset_ids(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith("_asset") and isinstance(child, str):
                result.add(child)
            elif key.endswith("_assets") and isinstance(child, list):
                result.update(item for item in child if isinstance(item, str))
            else:
                result.update(_asset_ids(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_asset_ids(child))
    return result


def _safe_event(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("\n", "\\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one private mm-tools studio.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--host", default=os.getenv("MM_STUDIO_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MM_STUDIO_PORT", "8260")))
    args = parser.parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    adapter_path = args.adapter.expanduser().resolve()
    project_root = (args.project_root or manifest_path.parent.parent).expanduser().resolve()
    application = build_application(_load_json(manifest_path), project_root, adapter_path)
    uvicorn.run(application, host=args.host, port=args.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
