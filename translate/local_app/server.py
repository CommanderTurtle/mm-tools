from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from server import (
    ARBITER_ID,
    DEFAULT_SYSTEM_PROMPT,
    DETECTOR_ID,
    LANGUAGE_NAMES,
    MODEL_ID,
    VISION_ID,
    TranslationEngine,
    _env_path,
)


WEB = Path(__file__).resolve().parents[1] / "web"
EXTERNAL_HOST = os.getenv("TRANSLATE_EXTERNAL_HOST", "127.0.0.1")
EXTERNAL_PORT = int(os.getenv("TRANSLATE_PORT", "8176"))
EXTERNAL_TIMEOUT = float(os.getenv("TRANSLATE_EXTERNAL_TIMEOUT", "10"))
VISION_EXTERNAL_TIMEOUT = float(
    os.getenv("TRANSLATE_VISION_EXTERNAL_TIMEOUT", "180")
)
MAX_BODY_BYTES = int(os.getenv("TRANSLATE_MAX_BODY_BYTES", "1048576"))
MAX_VISION_BODY_BYTES = int(
    os.getenv("TRANSLATE_MAX_VISION_BODY_BYTES", "33554432")
)


class RequestBodyLimitMiddleware:
    """Bound UI API request bodies even when Content-Length is absent or false."""

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        max_body_bytes: int,
        max_vision_body_bytes: int,
    ) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.max_vision_body_bytes = max_vision_body_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path == "/api/vision":
            limit = self.max_vision_body_bytes
        elif path == "/api/translate":
            limit = self.max_body_bytes
        else:
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                declared_size = int(declared)
            except ValueError:
                await JSONResponse(
                    {"detail": "Content-Length must be an integer"}, status_code=400
                )(scope, receive, send)
                return
            if declared_size < 0:
                await JSONResponse(
                    {"detail": "Content-Length must not be negative"}, status_code=400
                )(scope, receive, send)
                return
            if declared_size > limit:
                await JSONResponse(
                    {"detail": f"request body exceeds the {limit}-byte limit"},
                    status_code=413,
                )(scope, receive, send)
                return

        messages: list[dict[str, Any]] = []
        total = 0
        while True:
            message = await receive()
            messages.append(message)
            if message.get("type") != "http.request":
                break
            total += len(message.get("body", b""))
            if total > limit:
                await JSONResponse(
                    {"detail": f"request body exceeds the {limit}-byte limit"},
                    status_code=413,
                )(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        async def replay() -> dict[str, Any]:
            if messages:
                return messages.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay, send)


def _new_engine() -> TranslationEngine:
    return TranslationEngine(
        model_path=_env_path(
            "TRANSLATE_MODEL_PATH",
            "~/multimedia/models/text-only/mradermacher--EraX-Translator-V1.0-GGUF/"
            "EraX-Translator-V1.0.Q8_0.gguf",
        ),
        detector_path=_env_path(
            "TRANSLATE_DETECTOR_PATH",
            "~/multimedia/models/text-only/"
            "papluca--xlm-roberta-base-language-detection",
        ),
        arbiter_model_path=_env_path(
            "TRANSLATE_ARBITER_MODEL_PATH",
            "~/multimedia/models/text-only/anhbn--EraX-VL-7B-V1.5-Openvino-INT4",
        ),
        arbiter_runtime_path=_env_path(
            "TRANSLATE_ARBITER_RUNTIME_PATH",
            "~/multimedia/translate/.runtime/erax-vl-openvino",
        ),
        arbiter_device=os.getenv("TRANSLATE_ARBITER_DEVICE", "CPU"),
        arbiter_max_tokens=int(os.getenv("TRANSLATE_ARBITER_MAX_TOKENS", "16")),
        arbiter_finalists=int(os.getenv("TRANSLATE_ARBITER_FINALISTS", "4")),
        vision_max_tokens=int(os.getenv("TRANSLATE_VISION_MAX_TOKENS", "512")),
        max_image_bytes=int(os.getenv("TRANSLATE_MAX_IMAGE_BYTES", "20971520")),
        max_image_pixels=int(os.getenv("TRANSLATE_MAX_IMAGE_PIXELS", "50000000")),
        max_tokens=int(os.getenv("TRANSLATE_MAX_TOKENS", "512")),
        n_ctx=int(os.getenv("TRANSLATE_CONTEXT", "2048")),
        n_gpu_layers=int(os.getenv("TRANSLATE_GPU_LAYERS", "-1")),
        n_threads=int(
            os.getenv(
                "TRANSLATE_THREADS", str(max(1, (os.cpu_count() or 4) // 2))
            )
        ),
    )


engine = _new_engine()


def _autoload() -> bool:
    return os.getenv("TRANSLATE_UI_AUTOLOAD", "0").strip().lower() not in {
        "0",
        "false",
        "no",
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    if _autoload():
        await run_in_threadpool(engine.load)
    try:
        yield
    finally:
        await run_in_threadpool(engine.unload)


app = FastAPI(title="Translate Local Workbench", lifespan=lifespan)
app.add_middleware(
    RequestBodyLimitMiddleware,
    max_body_bytes=MAX_BODY_BYTES,
    max_vision_body_bytes=MAX_VISION_BODY_BYTES,
)
app.mount("/assets", StaticFiles(directory=WEB), name="assets")


class TranslationRequest(BaseModel):
    text: str = Field(min_length=1)
    source_language: str = "auto"
    target_language: str = "English"
    max_tokens: int | None = Field(default=None, ge=1, le=2048)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    backend: Literal["local", "external"] = "local"


class VisionRequest(BaseModel):
    image_data_url: str = Field(min_length=1)
    mode: Literal["explain", "translate", "custom"] = "explain"
    prompt: str = ""
    source_language: str = "auto"
    target_language: str = "English"
    max_tokens: int | None = Field(default=None, ge=1, le=2048)
    backend: Literal["local", "external"] = "local"


def _external_url(path: str) -> str:
    return f"http://{EXTERNAL_HOST}:{EXTERNAL_PORT}{path}"


def _external_request(
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = EXTERNAL_TIMEOUT,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    api_key = os.getenv("TRANSLATE_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        _external_url(path), data=data, headers=headers, method="POST" if data else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace").strip()
        try:
            parsed = json.loads(message)
            message = str(parsed.get("error") or parsed.get("detail") or message)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise RuntimeError(message or f"External service returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"External translator is unavailable: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("External translator returned an invalid response")
    return value


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "loaded": engine.loaded,
        "model": MODEL_ID,
        "detector": DETECTOR_ID,
        "arbiter": ARBITER_ID,
        "runtime": "UI-local · llama.cpp + OpenVINO INT4 + transformers-cpu",
        "model_present": engine.model_path.is_file(),
        "vision": {"model": VISION_ID, "modes": ["explain", "translate", "custom"]},
        "cloud": False,
    }


@app.get("/api/languages")
def languages() -> dict[str, dict[str, str]]:
    return {"languages": LANGUAGE_NAMES}


@app.get("/api/external-health")
def external_health() -> dict[str, Any]:
    try:
        state = _external_request("/health")
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"ok": True, "endpoint": _external_url(""), **state}


@app.post("/api/load")
def load() -> dict[str, Any]:
    engine.load()
    return {"ok": True, "loaded": True}


@app.post("/api/unload")
def unload() -> dict[str, Any]:
    engine.unload()
    return {"ok": True, "loaded": False}


@app.post("/api/translate")
def translate(request: TranslationRequest) -> dict[str, Any]:
    if request.backend == "external":
        try:
            result = _external_request(
                "/translate",
                {
                    "text": request.text,
                    "source_language": request.source_language,
                    "target_language": request.target_language,
                    "system_prompt": request.system_prompt,
                    "max_tokens": request.max_tokens,
                },
            )
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {"backend": "external", **result}

    try:
        value, source, confidence = engine.translate(
            request.text,
            source_language=request.source_language,
            target_language=request.target_language,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Translation failed: {exc}") from exc
    return {
        "backend": "local",
        "translation": value,
        "source_language": source,
        "source_confidence": confidence,
        "model": MODEL_ID,
    }


@app.post("/api/vision")
def vision(request: VisionRequest) -> dict[str, Any]:
    payload = request.model_dump(exclude={"backend"})
    if request.backend == "external":
        try:
            result = _external_request(
                "/vision", payload, timeout=VISION_EXTERNAL_TIMEOUT
            )
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {"backend": "external", **result}

    try:
        output, mode = engine.analyze_image(**payload)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Image inference failed: {exc}") from exc
    return {
        "backend": "local",
        "output": output,
        "mode": mode,
        "model": VISION_ID,
    }
