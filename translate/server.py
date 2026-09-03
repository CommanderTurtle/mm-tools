#!/usr/bin/env python3
"""Private EraX translation, fragment arbitration, and language HTTP service."""

from __future__ import annotations

import argparse
import base64
import binascii
import gc
import io
import json
import os
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


MODEL_ID = "erax-translator-v1.0-q8-0"
DETECTOR_ID = "xlm-roberta-base-language-detection"
ARBITER_ID = "erax-vl-7b-v1.5-openvino-int4"
VISION_ID = ARBITER_ID
VISION_MODES = ("explain", "translate", "custom")
DEFAULT_SYSTEM_PROMPT = (
    "Bạn là hệ thống dịch thuật đa ngôn ngữ. Dịch đầy đủ và sát nghĩa sang "
    "ngôn ngữ được yêu cầu. Giữ nguyên tên, số, định dạng, lời tục và giọng "
    "điệu. Chỉ trả về bản dịch; không nhãn, giải thích hay bình luận."
)
LANGUAGE_NAMES = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese",
    "az": "Azerbaijani", "ba": "Bashkir", "be": "Belarusian", "bg": "Bulgarian",
    "bn": "Bengali", "bo": "Tibetan", "br": "Breton", "bs": "Bosnian",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish",
    "fo": "Faroese", "fr": "French", "gl": "Galician", "gu": "Gujarati",
    "ha": "Hausa", "haw": "Hawaiian", "he": "Hebrew", "hi": "Hindi",
    "hr": "Croatian", "ht": "Haitian Creole", "hu": "Hungarian", "hy": "Armenian",
    "id": "Indonesian", "is": "Icelandic", "it": "Italian", "ja": "Japanese",
    "jw": "Javanese", "ka": "Georgian", "kk": "Kazakh", "km": "Khmer",
    "kn": "Kannada", "ko": "Korean", "la": "Latin", "lb": "Luxembourgish",
    "ln": "Lingala", "lo": "Lao", "lt": "Lithuanian", "lv": "Latvian",
    "mg": "Malagasy", "mi": "Maori", "mk": "Macedonian", "ml": "Malayalam",
    "mn": "Mongolian", "mr": "Marathi", "ms": "Malay", "mt": "Maltese",
    "my": "Burmese", "ne": "Nepali", "nl": "Dutch", "nn": "Nynorsk",
    "no": "Norwegian", "oc": "Occitan", "pa": "Punjabi", "pl": "Polish",
    "ps": "Pashto", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "sa": "Sanskrit", "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak",
    "sl": "Slovenian", "sn": "Shona", "so": "Somali", "sq": "Albanian",
    "sr": "Serbian", "su": "Sundanese", "sv": "Swedish", "sw": "Swahili",
    "ta": "Tamil", "te": "Telugu", "tg": "Tajik", "th": "Thai",
    "tk": "Turkmen", "tl": "Tagalog", "tr": "Turkish", "tt": "Tatar",
    "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese",
    "yi": "Yiddish", "yo": "Yoruba", "yue": "Cantonese", "zh": "Chinese",
}
_VI_TARGET_NAMES = {
    "en": "Anh", "english": "Anh",
    "vi": "Việt", "vietnamese": "Việt",
    "de": "Đức", "german": "Đức",
    "fr": "Pháp", "french": "Pháp",
    "es": "Tây Ban Nha", "spanish": "Tây Ban Nha",
    "pt": "Bồ Đào Nha", "portuguese": "Bồ Đào Nha",
    "it": "Ý", "italian": "Ý",
    "nl": "Hà Lan", "dutch": "Hà Lan",
    "ru": "Nga", "russian": "Nga",
    "uk": "Ukraina", "ukrainian": "Ukraina",
    "zh": "Hoa", "chinese": "Hoa",
    "yue": "Quảng Đông", "cantonese": "Quảng Đông",
    "ja": "Nhật", "japanese": "Nhật",
    "ko": "Hàn", "korean": "Hàn",
    "hi": "Hindi", "hindi": "Hindi",
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
        arbiter_model_path: Path,
        arbiter_runtime_path: Path,
        arbiter_device: str,
        arbiter_max_tokens: int,
        arbiter_finalists: int,
        vision_max_tokens: int,
        max_image_bytes: int,
        max_image_pixels: int,
        max_tokens: int,
        n_ctx: int,
        n_gpu_layers: int,
        n_threads: int,
    ) -> None:
        self.model_path = model_path
        self.detector_path = detector_path
        self.arbiter_model_path = arbiter_model_path
        self.arbiter_runtime_path = arbiter_runtime_path
        self.arbiter_device = arbiter_device
        self.arbiter_max_tokens = max(8, arbiter_max_tokens)
        self.arbiter_finalists = min(8, max(2, arbiter_finalists))
        self.vision_max_tokens = max(16, vision_max_tokens)
        self.max_image_bytes = max(1, max_image_bytes)
        self.max_image_pixels = max(1, max_image_pixels)
        self.max_tokens = max(1, max_tokens)
        self.n_ctx = max(1024, n_ctx)
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = max(1, n_threads)
        self._llm: Any | None = None
        self._detector_tokenizer: Any | None = None
        self._detector: Any | None = None
        self._arbiter: Any | None = None
        self._openvino: Any | None = None
        self._openvino_genai: Any | None = None
        self._torch: Any | None = None
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return (
            self._llm is not None
            and self._detector is not None
            and self._arbiter is not None
        )

    def load(self) -> None:
        with self._lock:
            if self.loaded:
                return
            if not self.model_path.is_file():
                raise RuntimeError(f"EraX GGUF not found: {self.model_path}")
            if not (self.detector_path / "config.json").is_file():
                raise RuntimeError(f"language detector not found: {self.detector_path}")
            if not (self.arbiter_model_path / "openvino_language_model.xml").is_file():
                raise RuntimeError(f"EraX-VL OpenVINO model not found: {self.arbiter_model_path}")

            from llama_cpp import Llama
            import openvino
            import openvino_genai
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            os.environ.setdefault("DO_NOT_TRACK", "1")
            os.environ.setdefault("SCARF_NO_ANALYTICS", "1")
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
            arbiter_runtime = self._prepare_arbiter_runtime()
            self._arbiter = openvino_genai.VLMPipeline(
                arbiter_runtime, self.arbiter_device
            )
            self._openvino = openvino
            self._openvino_genai = openvino_genai
            torch.set_num_threads(self.n_threads)
            self._torch = torch

    def unload(self) -> None:
        with self._lock:
            self._llm = None
            self._detector = None
            self._detector_tokenizer = None
            self._arbiter = None
            self._openvino = None
            self._openvino_genai = None
            self._torch = None
            gc.collect()

    def detect_language(self, text: str) -> tuple[str, float]:
        text = text.strip()
        if not text:
            raise ValueError("text is empty")
        return self.detect_languages([text])[0]

    def detect_languages(self, texts: list[str]) -> list[tuple[str, float]]:
        """Classify a text batch in one inexpensive XLM-R forward pass."""
        cleaned = [text.strip() for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise ValueError("texts contain an empty value")
        with self._lock:
            self.load()
            assert self._detector is not None
            assert self._detector_tokenizer is not None
            assert self._torch is not None
            inputs = self._detector_tokenizer(
                cleaned,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with self._torch.inference_mode():
                logits = self._detector(**inputs).logits
                scores = self._torch.softmax(logits, dim=-1)
            confidence, index = self._torch.max(scores, dim=1)
            labels = self._detector.config.id2label
            return [
                (str(labels[int(row.item())]), float(score.item()))
                for row, score in zip(index, confidence)
            ]

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
            # EraX is itself multilingual.  A source token is useful metadata,
            # but it must never gate translation; only Crisper needs the
            # language selected by the optional MITM lane.
            source_code = source_language
            confidence: float | None = None
            target_key = target_language.strip().lower()
            target = _VI_TARGET_NAMES.get(
                target_key, LANGUAGE_NAMES.get(target_key, target_language)
            )
            limit = min(max(1, max_tokens or self.max_tokens), self.max_tokens)
            translated = self._translate_with_erax(
                text, target, system_prompt, limit
            )
            if self._translation_needs_fallback(text, translated, target_key):
                target_code = _language_code(target_key)
                vl_target = LANGUAGE_NAMES.get(target_code or "", target_language)
                translated = self._translate_with_vl(text, vl_target, limit)
            return translated, source_code, confidence

    def _translate_with_erax(
        self, text: str, target: str, system_prompt: str, limit: int
    ) -> str:
        assert self._llm is not None
        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {
                    "role": "user",
                    "content": f"{text}\n\nDịch sang tiếng {target}.",
                },
            ],
            max_tokens=limit,
            temperature=0.2,
            top_p=0.95,
            top_k=64,
            min_p=0.1,
            repeat_penalty=1.05,
            stop=["<end_of_turn>", "<eos>"],
        )
        content = response["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise RuntimeError("EraX returned no text")
        translated = _clean_output(content)
        if not translated:
            raise RuntimeError("EraX returned an empty translation")
        return translated

    def _translation_needs_fallback(
        self, source: str, translated: str, target_key: str
    ) -> bool:
        target_code = _language_code(target_key)
        if _comparable_text(source) == _comparable_text(translated):
            if target_code is None:
                return True
            detected, _ = self.detect_language(source)
            return detected != target_code
        if target_code is not None:
            assert self._detector is not None
            supported = {
                str(label) for label in self._detector.config.id2label.values()
            }
            if target_code not in supported:
                return False
            detected, confidence = self.detect_language(translated)
            return confidence >= 0.80 and detected != target_code
        return False

    def _translate_with_vl(self, text: str, target: str, limit: int) -> str:
        """Use the resident INT4 model only when the tiny translator no-ops."""
        assert self._arbiter is not None
        assert self._openvino_genai is not None
        config = self._openvino_genai.GenerationConfig()
        config.max_new_tokens = min(128, limit)
        config.do_sample = False
        prompt = (
            f"Translate the text into {target}. Return only the translation.\n"
            f"Text: {text}"
        )
        generated = self._arbiter.generate(prompt, generation_config=config)
        translated = _clean_output(_openvino_text(generated))
        if not translated:
            raise RuntimeError("EraX-VL returned an empty translation")
        return translated

    def analyze_image(
        self,
        image_data_url: str,
        mode: str = "explain",
        prompt: str = "",
        source_language: str = "auto",
        target_language: str = "English",
        max_tokens: int | None = None,
    ) -> tuple[str, str]:
        """Run one stateless image request through the resident EraX-VL lane."""
        instruction = _vision_prompt(
            mode, prompt, source_language, target_language
        )
        image = _decode_image_array(
            image_data_url, self.max_image_bytes, self.max_image_pixels
        )
        with self._lock:
            self.load()
            assert self._arbiter is not None
            assert self._openvino is not None
            assert self._openvino_genai is not None
            config = self._openvino_genai.GenerationConfig()
            config.max_new_tokens = min(
                max(1, max_tokens or self.vision_max_tokens),
                self.vision_max_tokens,
            )
            config.do_sample = False
            generated = self._arbiter.generate(
                instruction,
                images=[self._openvino.Tensor(image)],
                generation_config=config,
            )
        output = _openvino_text(generated).strip()
        if not output:
            raise RuntimeError("EraX-VL returned an empty image response")
        return output, mode.strip().lower()

    def arbitrate_candidates(self, candidates: object) -> dict[str, Any]:
        """Resolve one utterance without turning detection into sticky state.

        Crisper has already produced every language row in one decoder batch.
        XLM-R is used here only as a cheap prefill reducer.  EraX-VL remains
        the fragment/coherence arbiter, and a second tiny EraX-VL comparison
        runs only when more than one translated finalist survives.
        """
        normalized = _normalize_candidates(candidates)
        with self._lock:
            self.load()
            assert self._arbiter is not None
            assert self._openvino_genai is not None
            classifications = self.detect_languages(
                [candidate["text"] for candidate in normalized]
            )
            for candidate, (language, confidence) in zip(normalized, classifications):
                candidate["classifier_language"] = language
                candidate["classifier_confidence"] = confidence

            matched = [
                candidate for candidate in normalized
                if candidate["classifier_language"] == candidate["language_prompt"]
            ]
            # XLM-R covers 20 languages while Crisper exposes 98.  Preserve the
            # four strongest acoustic candidates so the classifier can reduce
            # prefill without making the other 78 languages unreachable.
            acoustic = sorted(
                normalized,
                key=lambda candidate: float(candidate.get("acoustic_probability", 0.0)),
                reverse=True,
            )[:4]
            pool: list[dict[str, Any]] = []
            seen: set[int] = set()
            for candidate in [*matched, *acoustic]:
                source_index = int(candidate["source_index"])
                if source_index not in seen:
                    pool.append(candidate)
                    seen.add(source_index)
            if not pool:
                pool = normalized
            prompt = _fragment_arbiter_prompt(pool, self.arbiter_finalists)
            config = self._openvino_genai.GenerationConfig()
            config.max_new_tokens = self.arbiter_max_tokens
            config.do_sample = False
            generated = self._arbiter.generate(prompt, generation_config=config)
            raw = _openvino_text(generated)
            finalist_indexes = _parse_candidate_indexes(
                raw, len(pool), self.arbiter_finalists
            )
            if not finalist_indexes:
                retry = self._arbiter.generate(
                    prompt + "\nAnswer now with comma-separated integers only.",
                    generation_config=config,
                )
                raw = _openvino_text(retry)
                finalist_indexes = _parse_candidate_indexes(
                    raw, len(pool), self.arbiter_finalists
                )
            if not finalist_indexes:
                finalist_indexes = [max(
                    range(len(pool)),
                    key=lambda index: float(pool[index].get("classifier_confidence", 0.0)),
                )]

            finalists = [pool[index] for index in finalist_indexes]
            ambiguity_output: str | None = None
            acoustic = max(
                pool,
                key=lambda candidate: float(candidate.get("acoustic_probability", 0.0)),
            )
            acoustic_probability = float(acoustic.get("acoustic_probability", 0.0))
            # The acoustic prior identifies the spoken language, while these
            # deliberately short candidate decodes identify viable text.  A
            # 24-token candidate may be clipped even when its language prior
            # is overwhelming; the selected token is followed by a separate
            # full Crisper pass, so transcript completeness must not veto a
            # decisive acoustic language result here.
            acoustic_is_decisive = acoustic_probability >= 0.90
            if acoustic_is_decisive:
                selected = acoustic
                ambiguity_output = f"acoustic-prior:{acoustic_probability:.6f}"
            elif len(finalists) > 1:
                for finalist in finalists:
                    translation, _, _ = self.translate(
                        finalist["text"], target_language="English", max_tokens=128
                    )
                    finalist["comparison_translation"] = translation
                ambiguity_prompt = _ambiguity_prompt(finalists)
                judged = self._arbiter.generate(
                    ambiguity_prompt, generation_config=config
                )
                ambiguity_output = _openvino_text(judged)
                selected_index = _parse_candidate_index(
                    ambiguity_output, len(finalists)
                )
                if selected_index is None:
                    selected_index = 0
                selected = finalists[selected_index]
            else:
                selected = finalists[0]
            return {
                "selected_index": selected["source_index"],
                "selected_candidate": selected,
                "text": selected["text"],
                "language": selected["language_prompt"],
                "language_confidence": (
                    selected.get("classifier_confidence")
                    if selected.get("classifier_language") == selected["language_prompt"]
                    else selected.get("acoustic_probability")
                ),
                "arbiter": ARBITER_ID,
                "arbiter_output": raw,
                "ambiguity_output": ambiguity_output,
                "finalist_count": len(finalists),
                "detector": DETECTOR_ID,
            }

    def _prepare_arbiter_runtime(self) -> Path:
        """Compose an immutable model view with locally converted tokenizers."""
        runtime = self.arbiter_runtime_path
        runtime.mkdir(parents=True, exist_ok=True)
        for source in self.arbiter_model_path.iterdir():
            target = runtime / source.name
            if target.exists() or target.is_symlink():
                continue
            target.symlink_to(source, target_is_directory=source.is_dir())

        tokenizer_xml = runtime / "openvino_tokenizer.xml"
        detokenizer_xml = runtime / "openvino_detokenizer.xml"
        if not tokenizer_xml.is_file() or not detokenizer_xml.is_file():
            import openvino as ov
            from openvino_tokenizers import convert_tokenizer
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.arbiter_model_path, local_files_only=True
            )
            ov_tokenizer, ov_detokenizer = convert_tokenizer(
                tokenizer, with_detokenizer=True
            )
            ov.save_model(ov_tokenizer, tokenizer_xml)
            ov.save_model(ov_detokenizer, detokenizer_xml)
        return runtime


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

    def _body(self, max_bytes: int | None = None) -> dict[str, Any]:
        size = int(self.headers.get("Content-Length", "0"))
        if size < 0:
            raise ValueError("Content-Length must not be negative")
        limit = self.app.max_body_bytes if max_bytes is None else max_bytes
        if size > limit:
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
                "model": MODEL_ID, "detector": DETECTOR_ID, "arbiter": ARBITER_ID,
                "runtime": "llama.cpp + OpenVINO INT4 + transformers-cpu",
                "vision": {"model": VISION_ID, "modes": list(VISION_MODES)},
                "cloud": False,
            })
        elif self.path == "/v1/models":
            self._json(HTTPStatus.OK, {
                "object": "list", "data": [
                    {"id": MODEL_ID, "object": "model", "capabilities": ["translation"]},
                    {"id": VISION_ID, "object": "model", "capabilities": ["vision", "ocr", "vqa"]},
                ]
            })
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        try:
            body = self._body(
                self.app.max_vision_body_bytes
                if self.path == "/vision"
                else None
            )
            if self.path == "/load":
                self.app.engine.load()
                result: dict[str, Any] = {"status": "loaded", "model": MODEL_ID}
            elif self.path == "/unload":
                self.app.engine.unload()
                result = {"status": "unloaded", "model": MODEL_ID}
            elif self.path == "/detect":
                language, confidence = self.app.engine.detect_language(str(body.get("text", "")))
                result = {"language": language, "confidence": confidence, "detector": DETECTOR_ID}
            elif self.path == "/arbitrate":
                result = self.app.engine.arbitrate_candidates(body.get("candidates", []))
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
            elif self.path == "/vision":
                output, mode = self.app.engine.analyze_image(
                    image_data_url=str(body.get("image_data_url", "")),
                    mode=str(body.get("mode", "explain")),
                    prompt=str(body.get("prompt", "")),
                    source_language=str(body.get("source_language", "auto")),
                    target_language=str(body.get("target_language", "English")),
                    max_tokens=_optional_int(body.get("max_tokens")),
                )
                result = {"output": output, "mode": mode, "model": VISION_ID}
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

    def __init__(
        self,
        address: tuple[str, int],
        engine: TranslationEngine,
        api_key: str,
        max_body_bytes: int,
        max_vision_body_bytes: int,
    ) -> None:
        super().__init__(address, ApiHandler)
        self.engine = engine
        self.api_key = api_key
        self.max_body_bytes = max_body_bytes
        self.max_vision_body_bytes = max_vision_body_bytes


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _vision_prompt(
    mode: str,
    prompt: str,
    source_language: str,
    target_language: str,
) -> str:
    normalized = mode.strip().lower()
    if normalized not in VISION_MODES:
        raise ValueError(
            f"mode must be one of: {', '.join(VISION_MODES)}"
        )
    if normalized == "custom":
        value = prompt.strip()
        if not value:
            raise ValueError("prompt is required when mode is custom")
        return value
    target_key = target_language.strip().lower()
    target = LANGUAGE_NAMES.get(target_key, target_language.strip())
    if not target:
        raise ValueError("target_language is empty")
    if normalized == "explain":
        return (
            "Describe and explain this image accurately and concisely. "
            "Include important visible text, labels, relationships, and context. "
            f"Respond in {target}. Do not invent details that are not visible."
        )

    source_key = source_language.strip().lower()
    source = LANGUAGE_NAMES.get(source_key, source_language.strip())
    source_hint = "Detect the source language." if source_key in {
        "", "auto", "detect"
    } else f"The visible text is in {source}."
    return (
        "Read every meaningful piece of visible text in the image. "
        f"{source_hint} Translate it into {target}. "
        "Preserve the document order, line breaks, names, numbers, URLs, code, "
        "and formatting as closely as possible. Return only the translation."
    )


def _decode_image_array(
    image_data_url: str,
    max_image_bytes: int,
    max_image_pixels: int,
) -> Any:
    value = image_data_url.strip()
    header, separator, payload = value.partition(",")
    if not separator or not header.lower().startswith("data:image/"):
        raise ValueError("image_data_url must be a base64 image data URL")
    if ";base64" not in header.lower():
        raise ValueError("image_data_url must use base64 encoding")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_data_url contains invalid base64") from exc
    if not raw:
        raise ValueError("image data is empty")
    if len(raw) > max_image_bytes:
        raise ValueError(
            f"decoded image exceeds the {max_image_bytes}-byte limit"
        )

    import numpy as np
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(raw)) as opened:
            width, height = opened.size
            if width <= 0 or height <= 0:
                raise ValueError("image dimensions are invalid")
            if width * height > max_image_pixels:
                raise ValueError(
                    f"image exceeds the {max_image_pixels}-pixel limit"
                )
            image = ImageOps.exif_transpose(opened).convert("RGB")
            return np.ascontiguousarray(np.array(image, dtype=np.uint8, copy=True))
    except Image.DecompressionBombError as exc:
        raise ValueError("image exceeds safe pixel limits") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("image data could not be decoded") from exc


def _normalize_candidates(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("candidates must be a non-empty array")
    normalized: list[dict[str, Any]] = []
    for position, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"candidate {position} must be an object")
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        normalized.append({
            "source_index": int(item.get("index", position)),
            "language_prompt": str(item.get("language_prompt", "unknown")),
            "acoustic_probability": float(item.get("acoustic_probability", 0.0)),
            "text": text,
            "terminal_punctuation": bool(item.get("terminal_punctuation", False)),
            "ended_with_eot": bool(item.get("ended_with_eot", False)),
            "hit_token_limit": bool(item.get("hit_token_limit", False)),
            "generation_tokens": int(item.get("generation_tokens", 0)),
            "punctuation_count": int(item.get("punctuation_count", 0)),
            "word_count": int(item.get("word_count", len(text.split()))),
            "character_count": int(item.get("character_count", len(text))),
        })
    if not normalized:
        raise ValueError("candidates contain no text")
    return normalized


def _fragment_arbiter_prompt(
    candidates: list[dict[str, Any]], maximum: int
) -> str:
    lines = [
        "Rank the complete speech-transcript rows.",
        "Reject fragments, clipped tails, gibberish, transliteration, echoes, and repetition.",
        "A valid row must be grammatical in its named language and form a natural utterance.",
        f"Return at most {maximum} zero-based row numbers, best first.",
        "Output comma-separated integers only. Do not translate.",
    ]
    for row, candidate in enumerate(candidates):
        lines.append(
            f"{row}: [prompt_language={LANGUAGE_NAMES.get(candidate['language_prompt'], candidate['language_prompt'])}; "
            f"terminal_punctuation={str(candidate['terminal_punctuation']).lower()}; "
            f"ended={str(candidate['ended_with_eot']).lower()}; "
            f"acoustic={candidate['acoustic_probability']:.4f}; "
            f"punctuation_count={candidate['punctuation_count']}; "
            f"words={candidate['word_count']}] {candidate['text']}"
        )
    return "\n".join(lines)


def _ambiguity_prompt(candidates: list[dict[str, Any]]) -> str:
    lines = [
        "Choose the most coherent translation of one spoken utterance.",
        "Each row names the decoder language, its transcript, and its English translation.",
        "Reject mistranscription, gibberish, fragments, and semantically broken translations.",
        "Return only the zero-based row number: one integer, no prose.",
    ]
    for row, candidate in enumerate(candidates):
        lines.append(
            f"{row}: [{LANGUAGE_NAMES.get(candidate['language_prompt'], candidate['language_prompt'])}] "
            f"source={candidate['text']} | english={candidate['comparison_translation']}"
        )
    return "\n".join(lines)


def _openvino_text(result: object) -> str:
    texts = getattr(result, "texts", None)
    if isinstance(texts, (list, tuple)) and texts:
        return str(texts[0]).strip()
    return str(result).strip()


def _parse_candidate_index(value: str, count: int) -> int | None:
    match = re.search(r"(?<!\d)(\d+)(?!\d)", value)
    if not match:
        return None
    index = int(match.group(1))
    return index if 0 <= index < count else None


def _parse_candidate_indexes(value: str, count: int, maximum: int) -> list[int]:
    indexes: list[int] = []
    for match in re.finditer(r"(?<!\d)(\d+)(?!\d)", value):
        index = int(match.group(1))
        if 0 <= index < count and index not in indexes:
            indexes.append(index)
        if len(indexes) >= maximum:
            break
    return indexes


def _language_code(value: str) -> str | None:
    normalized = value.strip().lower()
    if normalized in LANGUAGE_NAMES:
        return normalized
    for code, name in LANGUAGE_NAMES.items():
        if name.lower() == normalized:
            return code
    return None


def _comparable_text(value: str) -> str:
    return re.sub(r"\W+", "", value, flags=re.UNICODE).casefold()


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
        model_path=_env_path("TRANSLATE_MODEL_PATH", "~/multimedia/models/text-only/mradermacher--EraX-Translator-V1.0-GGUF/EraX-Translator-V1.0.Q8_0.gguf"),
        detector_path=_env_path("TRANSLATE_DETECTOR_PATH", "~/multimedia/models/text-only/papluca--xlm-roberta-base-language-detection"),
        arbiter_model_path=_env_path("TRANSLATE_ARBITER_MODEL_PATH", "~/multimedia/models/text-only/anhbn--EraX-VL-7B-V1.5-Openvino-INT4"),
        arbiter_runtime_path=_env_path("TRANSLATE_ARBITER_RUNTIME_PATH", "~/multimedia/translate/.runtime/erax-vl-openvino"),
        arbiter_device=os.environ.get("TRANSLATE_ARBITER_DEVICE", "CPU"),
        arbiter_max_tokens=int(os.environ.get("TRANSLATE_ARBITER_MAX_TOKENS", "16")),
        arbiter_finalists=int(os.environ.get("TRANSLATE_ARBITER_FINALISTS", "4")),
        vision_max_tokens=int(os.environ.get("TRANSLATE_VISION_MAX_TOKENS", "512")),
        max_image_bytes=int(os.environ.get("TRANSLATE_MAX_IMAGE_BYTES", "20971520")),
        max_image_pixels=int(os.environ.get("TRANSLATE_MAX_IMAGE_PIXELS", "50000000")),
        max_tokens=int(os.environ.get("TRANSLATE_MAX_TOKENS", "512")),
        n_ctx=int(os.environ.get("TRANSLATE_CONTEXT", "2048")),
        n_gpu_layers=int(os.environ.get("TRANSLATE_GPU_LAYERS", "-1")),
        n_threads=int(os.environ.get("TRANSLATE_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))),
    )
    server = TranslationServer(
        (args.host, args.port),
        engine,
        os.environ.get("TRANSLATE_API_KEY", ""),
        int(os.environ.get("TRANSLATE_MAX_BODY_BYTES", "1048576")),
        int(os.environ.get("TRANSLATE_MAX_VISION_BODY_BYTES", "33554432")),
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
