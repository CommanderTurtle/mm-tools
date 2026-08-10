#!/usr/bin/env python3
"""Private EraX translation and XLM-R language-detection HTTP service."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


MODEL_ID = "erax-translator-v1.0-q6-k"
DETECTOR_ID = "xlm-roberta-base-language-detection"
DEFAULT_SYSTEM_PROMPT = (
    "You are a precise multilingual translation system. Translate the entire "
    "source faithfully into the requested language. Preserve names, numbers, "
    "formatting, profanity, and tone. Never add commentary, an introduction, "
    "a label, or an explanation. Return only the translated text."
)
LANGUAGE_NAMES = {
    "ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "nl": "Dutch", "pl": "Polish",
    "pt": "Portuguese", "ru": "Russian", "sw": "Swahili", "th": "Thai",
    "tr": "Turkish", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese",
    "ko": "Korean", "uk": "Ukrainian", "yue": "Cantonese",
}
_HEADER = re.compile(
    r"^(?:\*\*|__)?\s*(?:assistant|translation|translated text|output|result|"
    r"final answer|[a-z ]+ translation)\s*:\s*(?:\*\*|__)?\s*",
    re.IGNORECASE,
)


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


class TranslationEngine:
    def __init__(
        self,
        model_path: Path,
        detector_path: Path,
        max_tokens: int,
        n_ctx: int,
        n_gpu_layers: int,
        n_threads: int,
    ) -> None:
        self.model_path = model_path
        self.detector_path = detector_path
        self.max_tokens = max(1, max_tokens)
        self.n_ctx = max(1024, n_ctx)
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = max(1, n_threads)
        self._llm: Any | None = None
        self._detector_tokenizer: Any | None = None
        self._detector: Any | None = None
        self._torch: Any | None = None
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._llm is not None and self._detector is not None

    def load(self) -> None:
        with self._lock:
            if self.loaded:
                return
            if not self.model_path.is_file():
                raise RuntimeError(f"EraX GGUF not found: {self.model_path}")
            if not (self.detector_path / "config.json").is_file():
                raise RuntimeError(f"language detector not found: {self.detector_path}")

            from llama_cpp import Llama
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_batch=min(1024, self.n_ctx),
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                seed=0,
                verbose=False,
            )
            self._detector_tokenizer = AutoTokenizer.from_pretrained(
                self.detector_path, local_files_only=True
            )
            self._detector = AutoModelForSequenceClassification.from_pretrained(
                self.detector_path, local_files_only=True
            ).to("cpu").eval()
            torch.set_num_threads(self.n_threads)
            self._torch = torch

    def unload(self) -> None:
        with self._lock:
            self._llm = None
            self._detector = None
            self._detector_tokenizer = None
            self._torch = None
            gc.collect()

    def detect_language(self, text: str) -> tuple[str, float]:
        text = text.strip()
        if not text:
            raise ValueError("text is empty")
        with self._lock:
            self.load()
            assert self._detector is not None
            assert self._detector_tokenizer is not None
            assert self._torch is not None
            inputs = self._detector_tokenizer(
                text, truncation=True, max_length=512, return_tensors="pt"
            )
            with self._torch.inference_mode():
                logits = self._detector(**inputs).logits
                scores = self._torch.softmax(logits, dim=-1)[0]
            confidence, index = self._torch.max(scores, dim=0)
            language = self._detector.config.id2label[int(index.item())]
            return str(language), float(confidence.item())

    def translate(
        self,
        text: str,
        source_language: str = "auto",
        target_language: str = "English",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int | None = None,
    ) -> tuple[str, str, float | None]:
        text = text.strip()
        if not text:
            return "", source_language, None
        with self._lock:
            self.load()
            assert self._llm is not None
            confidence: float | None = None
            if source_language.lower() in {"auto", "detect"}:
                source_code, confidence = self.detect_language(text)
                source = LANGUAGE_NAMES.get(source_code, source_code)
            else:
                source_code = source_language
                source = LANGUAGE_NAMES.get(source_language.lower(), source_language)
            target = LANGUAGE_NAMES.get(target_language.lower(), target_language)
            limit = min(max(1, max_tokens or self.max_tokens), self.max_tokens)
            response = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {
                        "role": "user",
                        "content": f"Translate from {source} to {target}:\n\n{text}",
                    },
                ],
                max_tokens=limit,
                temperature=0.2,
                top_p=0.95,
                top_k=64,
                min_p=0.1,
                repeat_penalty=1.05,
            )
            content = response["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise RuntimeError("EraX returned no text")
            translated = _clean_output(content)
            if not translated:
                raise RuntimeError("EraX returned an empty translation")
            return translated, source_code, confidence


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "LocalTranslate/2.0"

    @property
    def app(self) -> "TranslationServer":
        return self.server  # type: ignore[return-value]

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

    def _authorized(self) -> bool:
        expected = self.app.api_key
        return not expected or self.headers.get("Authorization", "") == f"Bearer {expected}"

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        size = int(self.headers.get("Content-Length", "0"))
        if size > self.app.max_body_bytes:
            raise ValueError("request body is too large")
        value = json.loads(self.rfile.read(size) or b"{}")
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        if self.path == "/health":
            self._json(HTTPStatus.OK, {
                "status": "ok", "loaded": self.app.engine.loaded,
                "model": MODEL_ID, "detector": DETECTOR_ID,
                "runtime": "llama.cpp + transformers-cpu", "cloud": False,
            })
        elif self.path == "/v1/models":
            self._json(HTTPStatus.OK, {
                "object": "list", "data": [{"id": MODEL_ID, "object": "model"}]
            })
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        try:
            body = self._body()
            if self.path == "/load":
                self.app.engine.load()
                result: dict[str, Any] = {"status": "loaded", "model": MODEL_ID}
            elif self.path == "/unload":
                self.app.engine.unload()
                result = {"status": "unloaded", "model": MODEL_ID}
            elif self.path == "/detect":
                language, confidence = self.app.engine.detect_language(str(body.get("text", "")))
                result = {"language": language, "confidence": confidence, "detector": DETECTOR_ID}
            elif self.path == "/translate":
                translation, language, confidence = self.app.engine.translate(
                    text=str(body.get("text", "")),
                    source_language=str(body.get("source_language", "auto")),
                    target_language=str(body.get("target_language", "English")),
                    system_prompt=str(body.get("system_prompt", DEFAULT_SYSTEM_PROMPT)),
                    max_tokens=_optional_int(body.get("max_tokens")),
                )
                result = {
                    "translation": translation, "source_language": language,
                    "source_confidence": confidence, "model": MODEL_ID,
                }
            elif self.path == "/v1/chat/completions":
                text, source, target = _translation_request(body.get("messages", []))
                content, _, _ = self.app.engine.translate(
                    text, source, target, max_tokens=_optional_int(body.get("max_tokens"))
                )
                result = {
                    "id": f"vox-{int(time.time() * 1000)}", "object": "chat.completion",
                    "created": int(time.time()), "model": MODEL_ID,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                }
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._json(HTTPStatus.OK, result)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


class TranslationServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], engine: TranslationEngine, api_key: str, max_body_bytes: int) -> None:
        super().__init__(address, ApiHandler)
        self.engine = engine
        self.api_key = api_key
        self.max_body_bytes = max_body_bytes


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _clean_output(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines.pop()
        cleaned = "\n".join(lines).strip()
    return _HEADER.sub("", cleaned, count=1).strip()


def _translation_request(messages: object) -> tuple[str, str, str]:
    if not isinstance(messages, list):
        raise ValueError("messages must be an array")
    text = ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            text = str(message.get("content", "")).strip()
            break
    if not text:
        raise ValueError("messages contain no user text")
    source = "auto"
    target = "English"
    source_match = re.search(r"source language is ([^.]+)", text, re.IGNORECASE)
    target_match = re.search(r"target language is ([^.]+)", text, re.IGNORECASE)
    if source_match:
        source = source_match.group(1).strip()
    if target_match:
        target = target_match.group(1).strip()
    block = re.search(r"<source_text>\s*(.*?)\s*</source_text>", text, re.DOTALL | re.IGNORECASE)
    return (block.group(1).strip() if block else text), source, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("TRANSLATE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TRANSLATE_PORT", "8176")))
    args = parser.parse_args()
    engine = TranslationEngine(
        model_path=_env_path("TRANSLATE_MODEL_PATH", "~/multimedia/models/text-only/anhbn--raX-Translator-V1.0-GGUF/EraX-Translator-V1.0.Q6_K.gguf"),
        detector_path=_env_path("TRANSLATE_DETECTOR_PATH", "~/multimedia/models/text-only/papluca--xlm-roberta-base-language-detection"),
        max_tokens=int(os.environ.get("TRANSLATE_MAX_TOKENS", "512")),
        n_ctx=int(os.environ.get("TRANSLATE_CONTEXT", "2048")),
        n_gpu_layers=int(os.environ.get("TRANSLATE_GPU_LAYERS", "-1")),
        n_threads=int(os.environ.get("TRANSLATE_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))),
    )
    server = TranslationServer(
        (args.host, args.port), engine, os.environ.get("TRANSLATE_API_KEY", ""),
        int(os.environ.get("TRANSLATE_MAX_BODY_BYTES", "1048576")),
    )
    if os.environ.get("TRANSLATE_AUTOLOAD", "1").strip().lower() not in {"0", "false", "no"}:
        engine.load()
    print(f"Local EraX translator listening on http://{args.host}:{args.port} (loaded={engine.loaded})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.unload()


if __name__ == "__main__":
    main()
