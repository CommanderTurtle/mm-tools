from __future__ import annotations

import gc
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer

import audiodit  # noqa: F401 - registers the local model with Transformers
from audiodit import AudioDiTModel


@dataclass(frozen=True)
class SynthesisOptions:
    steps: int = 16
    guidance_strength: float = 4.0
    guidance_method: str = "apg"
    seed: int = 1024
    duration_scale: float = 1.0

    def validate(self) -> None:
        if not 2 <= self.steps <= 64:
            raise ValueError("steps must be between 2 and 64")
        if not 0.0 <= self.guidance_strength <= 20.0:
            raise ValueError("guidance strength must be between 0 and 20")
        if self.guidance_method not in {"cfg", "apg"}:
            raise ValueError("guidance method must be cfg or apg")
        if not 0.5 <= self.duration_scale <= 2.0:
            raise ValueError("duration scale must be between 0.5 and 2.0")


@dataclass(frozen=True)
class SynthesisResult:
    waveform: np.ndarray
    sample_rate: int
    generation_seconds: float
    audio_seconds: float
    options: SynthesisOptions

    def metadata(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "generation_seconds": round(self.generation_seconds, 3),
            "audio_seconds": round(self.audio_seconds, 3),
            "options": asdict(self.options),
        }


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'["“”‘’]', " ", text)
    return re.sub(r"\s+", " ", text).strip()


def approximate_duration(text: str, maximum: float) -> float:
    compact = re.sub(r"\s+", "", text)
    chinese = sum("\u4e00" <= character <= "\u9fff" for character in compact)
    latin = sum(character.isalpha() and not ("\u4e00" <= character <= "\u9fff") for character in compact)
    other = max(0, len(compact) - chinese - latin)
    if chinese > latin:
        chinese += other
    else:
        latin += other
    return min(maximum, chinese * 0.21 + latin * 0.082)


def _decode_prompt(path: Path, sample_rate: int) -> torch.Tensor:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to normalize reference audio")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        converted = Path(handle.name)
    try:
        subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_f32le",
                str(converted),
            ],
            check=True,
        )
        audio, actual_rate = sf.read(converted, dtype="float32", always_2d=False)
    finally:
        converted.unlink(missing_ok=True)
    if actual_rate != sample_rate:
        raise RuntimeError(f"ffmpeg returned {actual_rate} Hz instead of {sample_rate} Hz")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if not audio.size:
        raise ValueError("reference audio is empty")
    return torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)


class LongCatEngine:
    def __init__(self) -> None:
        self.model_path = Path(os.environ.get("LONGCAT_MODEL_PATH", ""))
        self.tokenizer_path = Path(os.environ.get("LONGCAT_TOKENIZER_PATH", ""))
        self.device = os.environ.get("LONGCAT_DEVICE", "cuda")
        self.dtype_name = os.environ.get("LONGCAT_DTYPE", "bfloat16")
        self.model: AudioDiTModel | None = None
        self.tokenizer = None
        self._state_lock = threading.RLock()
        self._generation_lock = threading.Lock()

    def status(self) -> dict:
        loaded = self.model is not None
        payload = {
            "loaded": loaded,
            "model_path": str(self.model_path),
            "tokenizer_path": str(self.tokenizer_path),
            "device": self.device,
            "dtype": self.dtype_name,
            "busy": self._generation_lock.locked(),
        }
        if torch.cuda.is_available():
            payload["cuda"] = torch.cuda.get_device_name(0)
            payload["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2)
        return payload

    def load(self) -> None:
        with self._state_lock:
            if self.model is not None:
                return
            if not (self.model_path / "model.safetensors").is_file():
                raise FileNotFoundError(f"LongCat checkpoint missing: {self.model_path}")
            if not (self.tokenizer_path / "tokenizer_config.json").is_file():
                raise FileNotFoundError(f"UMT5 tokenizer missing: {self.tokenizer_path}")
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            if self.dtype_name not in {"float32", "float16", "bfloat16"}:
                raise ValueError("LONGCAT_DTYPE must be float32, float16, or bfloat16")
            dtype = getattr(torch, self.dtype_name)
            if self.device == "cpu" and dtype != torch.float32:
                raise RuntimeError("CPU inference requires LONGCAT_DTYPE=float32")

            torch.set_float32_matmul_precision("high")
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.tokenizer_path), local_files_only=True, use_fast=True
            )
            self.model = AudioDiTModel.from_pretrained(
                str(self.model_path), local_files_only=True, dtype=dtype
            ).to(self.device)
            # The authors explicitly run the waveform VAE in fp16 even when
            # the DiT uses another dtype.
            self.model.vae.to_half()
            self.model.eval()

    def unload(self) -> None:
        if not self._generation_lock.acquire(blocking=False):
            raise RuntimeError("Cannot unload while synthesis is active")
        try:
            with self._state_lock:
                self.model = None
                self.tokenizer = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            self._generation_lock.release()

    def synthesize(
        self,
        text: str,
        *,
        prompt_audio: Path | None = None,
        prompt_text: str | None = None,
        options: SynthesisOptions | None = None,
    ) -> SynthesisResult:
        options = options or SynthesisOptions()
        options.validate()
        text = normalize_text(text)
        if not text:
            raise ValueError("text is empty")
        if prompt_audio is not None and not normalize_text(prompt_text or ""):
            raise ValueError("prompt_text is required when prompt_audio is supplied")
        if prompt_text and prompt_audio is None:
            raise ValueError("prompt_audio is required when prompt_text is supplied")

        with self._generation_lock:
            self.load()
            assert self.model is not None and self.tokenizer is not None
            model = self.model
            sample_rate = int(model.config.sampling_rate)
            hop = int(model.config.latent_hop)
            maximum = float(model.config.max_wav_duration)

            reference = None
            prompt_frames = 0
            full_text = text
            prompt_seconds = 0.0
            if prompt_audio is not None:
                prompt_audio = prompt_audio.expanduser().resolve()
                if not prompt_audio.is_file():
                    raise FileNotFoundError(f"reference audio not found: {prompt_audio}")
                prompt_wave = _decode_prompt(prompt_audio, sample_rate)
                # Ask the model's native encoder for the exact padded VAE
                # duration; conditioning itself is [batch, channel, time].
                reference = prompt_wave.unsqueeze(0)
                with torch.inference_mode():
                    _, prompt_frames = model.encode_prompt_audio(reference)
                prompt_frames = int(prompt_frames)
                prompt_seconds = prompt_frames * hop / sample_rate
                if prompt_seconds >= maximum - 0.25:
                    raise ValueError("reference audio leaves no room for generated speech")
                clean_prompt = normalize_text(prompt_text or "")
                full_text = f"{clean_prompt} {text}"

            remaining = maximum - prompt_seconds
            generated_seconds = approximate_duration(text, remaining)
            if reference is not None:
                expected_prompt = max(approximate_duration(prompt_text or "", maximum), 0.1)
                generated_seconds *= float(np.clip(prompt_seconds / expected_prompt, 1.0, 1.5))
            generated_seconds = min(remaining, max(hop / sample_rate, generated_seconds * options.duration_scale))
            generated_frames = max(1, int(generated_seconds * sample_rate // hop))
            total_frames = min(prompt_frames + generated_frames, int(maximum * sample_rate // hop))

            tokenized = self.tokenizer([full_text], padding="longest", return_tensors="pt")
            torch.manual_seed(options.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(options.seed)

            started = time.perf_counter()
            with torch.inference_mode():
                output = model(
                    input_ids=tokenized.input_ids,
                    attention_mask=tokenized.attention_mask,
                    prompt_audio=reference,
                    duration=total_frames,
                    steps=options.steps,
                    cfg_strength=options.guidance_strength,
                    guidance_method=options.guidance_method,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            waveform = output.waveform.squeeze().detach().float().cpu().numpy()
            waveform = np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)
            waveform = np.clip(waveform, -1.0, 1.0)
            return SynthesisResult(
                waveform=waveform,
                sample_rate=sample_rate,
                generation_seconds=elapsed,
                audio_seconds=len(waveform) / sample_rate,
                options=options,
            )
