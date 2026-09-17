from __future__ import annotations

import gc
import ipaddress
import json
import math
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import torch
import torchaudio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MONOREPO_ROOT = PROJECT_ROOT.parent
for candidate in (MONOREPO_ROOT, PROJECT_ROOT / "src"):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from studio import StudioAdapter, StudioContext, StudioOutput  # noqa: E402


SUPPORTED_MODES = {
    "zero_shot_tts",
    "instruct_tts",
    "speech_content_edit",
    "lyric_edit",
    "pitch_edit",
    "speed_edit",
    "volume_edit",
    "emotion_edit",
    "timbre_edit",
    "deaccent",
    "nonverbal_edit",
    "whisper",
    "speech_enhance",
    "speech_separate",
    "vocal_extract",
    "custom_instruction",
}


def _number(value: Any, fallback: float | None = None) -> float | None:
    if value in (None, ""):
        return fallback
    return float(value)


def _signed_direction(value: float, positive: str, negative: str) -> str:
    if value > 0:
        return f"{positive} {abs(value):g}"
    if value < 0:
        return f"{negative} {abs(value):g}"
    return "keep unchanged at 0"


def _ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _language(text: str) -> str:
    return "zh" if re.search(r"[\u3400-\u9fff]", text) else "en"


def _private_llm_url(value: str) -> bool:
    try:
        hostname = urlsplit(value).hostname
    except ValueError:
        return False
    if not hostname:
        return False
    if hostname in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        return ipaddress.ip_address(hostname).is_private and os.getenv("AUK_ALLOW_PRIVATE_LAN_LLM") == "1"
    except ValueError:
        return hostname.endswith((".local", ".lan")) and os.getenv("AUK_ALLOW_PRIVATE_LAN_LLM") == "1"


class _TranscriptASR:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript.strip()

    def transcribe(self, _: str):
        from auk.infer.pe import ASRCall

        return ASRCall(
            model="user-supplied-transcript",
            text=self.transcript,
            language=_language(self.transcript),
            raw_response={"source": "visible studio transcript"},
            error=None,
        )


class Adapter(StudioAdapter):
    def __init__(self, project_root: Path, runtime_root: Path) -> None:
        super().__init__(project_root, runtime_root)
        self.model_root = project_root / "ckpts"
        self.qwen_path = self.model_root / "Qwen2.5-Omni-3B"
        self.variants = {
            "base": {
                "checkpoint": self.model_root / "AuK" / "auk_base.safetensors",
                "config": self.model_root / "AuK" / "config.yaml",
                "vae": self.model_root / "AuK" / "vae.safetensors",
            },
            "flash": {
                "checkpoint": self.model_root / "AuK-Flash" / "auk_flash.safetensors",
                "config": self.model_root / "AuK-Flash" / "config.yaml",
                "vae": self.model_root / "AuK-Flash" / "vae.safetensors",
            },
        }
        self._engine = None
        self._engine_key: tuple[str, str] | None = None

    def _required_paths(self) -> dict[str, Path]:
        return {
            "AuK Base": self.variants["base"]["checkpoint"],
            "AuK config": self.variants["base"]["config"],
            "AuK VAE": self.variants["base"]["vae"],
            "Qwen2.5-Omni": self.qwen_path / "config.json",
        }

    def health(self) -> dict[str, Any]:
        paths = self._required_paths()
        missing = [label for label, path in paths.items() if not path.is_file()]
        cuda = torch.cuda.is_available()
        details = [
            {"label": label, "ready": path.is_file(), "name": path.name}
            for label, path in paths.items()
        ]
        details.append(
            {
                "label": "AuK-Flash (optional)",
                "ready": self.variants["flash"]["checkpoint"].is_file(),
                "name": "auk_flash.safetensors",
            }
        )
        details.append(
            {
                "label": "Local Prompt Enhancer (optional)",
                "ready": bool(os.getenv("LLM_BASE_URL") and os.getenv("LLM_MODEL_NAME")),
                "name": os.getenv("LLM_MODEL_NAME", "not configured"),
            }
        )
        return {
            "ready": cuda and not missing,
            "loaded": self._engine is not None,
            "loaded_variant": self._engine_key[0] if self._engine_key else None,
            "device": torch.cuda.get_device_name(0) if cuda else "CUDA unavailable",
            "missing": missing,
            "details": details,
            "prompt_enhancer": bool(os.getenv("LLM_BASE_URL") and os.getenv("LLM_MODEL_NAME")),
        }

    def validate(self, request: dict[str, Any], resolve_asset) -> None:
        mode = str(request.get("mode", ""))
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported AuK workflow: {mode!r}")
        controls = request.get("controls")
        if not isinstance(controls, dict):
            raise ValueError("AuK controls must be an object.")
        source_id = controls.get("source_audio_asset")
        if source_id:
            resolve_asset(str(source_id))
        elif mode not in {"instruct_tts", "custom_instruction"}:
            raise ValueError("This AuK workflow requires source audio.")
        if mode == "instruct_tts" and not str(controls.get("target_text", "")).strip():
            raise ValueError("Directed TTS needs a script.")
        if mode == "zero_shot_tts" and not str(controls.get("target_text", "")).strip():
            raise ValueError("Voice-clone TTS needs target text.")
        if mode == "emotion_edit":
            mix = controls.get("emotion_mix") or {}
            expressive = any(float(value or 0) > 0 for value in mix.values())
            expressive = expressive or any(
                abs(float(controls.get(name) or 0)) > 0
                for name in ("warmth", "energy", "breathiness")
            )
            expressive = expressive or bool(str(controls.get("performance_note", "")).strip())
            if not expressive:
                raise ValueError("Set at least one emotion or performance direction.")
        variant = str(controls.get("model_variant", "base"))
        if variant not in self.variants:
            raise ValueError(f"Unknown AuK model variant: {variant}")
        for label, path in self.variants[variant].items():
            if not path.is_file():
                raise FileNotFoundError(f"{variant.title()} {label} is missing: {path.name}")
        if not self.qwen_path.joinpath("config.json").is_file():
            raise FileNotFoundError("The local Qwen2.5-Omni encoder snapshot is incomplete.")
        if controls.get("use_prompt_enhancer"):
            base_url = os.getenv("LLM_BASE_URL", "")
            if not base_url or not os.getenv("LLM_MODEL_NAME"):
                raise ValueError("Prompt Enhancer is enabled but LLM_BASE_URL / LLM_MODEL_NAME are not configured.")
            if not _private_llm_url(base_url):
                raise ValueError(
                    "Prompt Enhancer refuses a public endpoint. Use loopback, or explicitly allow a private-LAN endpoint with AUK_ALLOW_PRIVATE_LAN_LLM=1."
                )
            if source_id and not str(controls.get("source_transcript", "")).strip():
                raise ValueError("Audio-backed Prompt Enhancer requires the visible source transcript; no cloud/CPU ASR fallback is used.")

    def _load_engine(self, variant: str, dtype: str, context: StudioContext | None = None):
        key = (variant, dtype)
        if self._engine is not None and self._engine_key == key:
            return self._engine
        self.unload()
        if not torch.cuda.is_available():
            raise RuntimeError("AuK's mm-tools profile is CUDA-only; CPU execution and CPU offload are disabled.")
        if context:
            context.update("Loading AuK and Qwen on CUDA", 0.06, f"Loading {variant} in {dtype} without CPU offload.")
        from auk.infer.infer_auk import AukInfer

        selected = self.variants[variant]
        self._engine = AukInfer(
            config_path=str(selected["config"]),
            ckpt_path=str(selected["checkpoint"]),
            device="cuda:0",
            dtype=dtype,
            qwen_path=str(self.qwen_path),
            cpu_offload=False,
        )
        self._engine_key = key
        return self._engine

    def load(self) -> dict[str, Any]:
        self._load_engine("base", "bf16")
        return self.health()

    def unload(self) -> dict[str, Any]:
        if self._engine is not None:
            del self._engine
            self._engine = None
            self._engine_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        return self.health()

    def close(self) -> None:
        self.unload()

    @staticmethod
    def _expression(controls: dict[str, Any]) -> str:
        parts: list[str] = []
        mix = controls.get("emotion_mix") or {}
        active = sorted(
            ((str(name), float(strength)) for name, strength in mix.items() if float(strength or 0) > 0),
            key=lambda item: item[1],
            reverse=True,
        )
        intensity = float(controls.get("emotion_intensity", 65))
        if active:
            blend = ", ".join(f"{name} ({strength:g}%)" for name, strength in active)
            force = "subtle" if intensity < 35 else "clear" if intensity < 75 else "intense"
            parts.append(f"Use a {force} emotional blend of {blend}")
        warmth = float(controls.get("warmth") or 0)
        energy = float(controls.get("energy") or 0)
        breathiness = float(controls.get("breathiness") or 0)
        pace = float(controls.get("performance_pace") or 1)
        if warmth:
            parts.append(f"make the tone {'warmer and more intimate' if warmth > 0 else 'cooler and more restrained'} ({abs(warmth):g}%)")
        if energy:
            parts.append(f"make the delivery {'more energetic and forceful' if energy > 0 else 'more subdued and gentle'} ({abs(energy):g}%)")
        if breathiness:
            parts.append(f"make the voice {'airier and breathier' if breathiness > 0 else 'firmer and clearer'} ({abs(breathiness):g}%)")
        if not math.isclose(pace, 1.0):
            parts.append(f"use an expressive pace around {pace:g}x while keeping the intended duration")
        note = str(controls.get("performance_note", "")).strip()
        if note:
            parts.append(note.rstrip("."))
        return "; ".join(parts)

    def _instruction(self, mode: str, controls: dict[str, Any]) -> str:
        expression = self._expression(controls)
        if mode == "zero_shot_tts":
            instruction = f"Say the following with the same voice: '{str(controls['target_text']).strip()}'"
            return f"{instruction}. {expression}." if expression else instruction
        if mode == "instruct_tts":
            description = str(controls["voice_description"]).strip().rstrip(".")
            if expression:
                description = f"{description}. Performance direction: {expression}"
            return f"Based on the following description: \"{description}\", generate speech content \"{str(controls['target_text']).strip()}\"."
        if mode == "speech_content_edit":
            return f"Replace '{str(controls['original_text']).strip()}' with '{str(controls['replacement_text']).strip()}' while preserving the speaker and surrounding delivery."
        if mode == "lyric_edit":
            return f"Replace '{str(controls['original_text']).strip()}' with '{str(controls['replacement_text']).strip()}' in the lyrics while preserving melody, singer, and phrasing."
        if mode == "pitch_edit":
            value = float(controls["semitones"])
            return f"{_signed_direction(value, 'Raise the pitch by', 'Lower the pitch by')} semitones. Preserve the content, voice, and duration."
        if mode == "speed_edit":
            return f"Adjust the speech speed to {float(controls['speed_factor']):g}x while preserving the speaker and content."
        if mode == "volume_edit":
            value = float(controls["gain_db"])
            return f"{_signed_direction(value, 'Increase the volume by', 'Decrease the volume by')} dB. Preserve content, speaker, and duration."
        if mode == "emotion_edit":
            return f"Keep the words and speaker identity unchanged, then re-perform the line. {expression}."
        if mode == "timbre_edit":
            instruction = f"Keep the spoken content unchanged and change the timbre to: \"{str(controls['timbre_description']).strip()}\""
            return f"{instruction}. Performance direction: {expression}." if expression else f"{instruction}."
        if mode == "deaccent":
            return f"Remove the regional accent and use {str(controls['accent_target']).strip()}, while preserving the speaker's voice, content, and duration."
        if mode == "nonverbal_edit":
            instructions: list[str] = []
            for event in controls.get("nonverbal_events") or []:
                operation = event.get("operation", "add")
                sound = event.get("sound", "breath")
                placement = event.get("placement", "before")
                anchor = str(event.get("anchor", "")).strip()
                if operation == "remove":
                    if anchor and placement in {"before", "after"}:
                        instructions.append(f"Remove the {sound} {placement} '{anchor}'")
                    else:
                        instructions.append(f"Remove all {sound} from the audio")
                elif anchor and placement in {"before", "after"}:
                    instructions.append(f"Add a {sound} {placement} '{anchor}'")
                else:
                    where = "at the beginning" if placement == "beginning" else "at the end" if placement == "end" else f"at the {placement}"
                    instructions.append(f"Add a {sound} {where}")
            return ". ".join(instructions) + ". Preserve the spoken content and speaker identity."
        if mode == "whisper":
            if controls.get("whisper_direction") == "to_normal":
                return "Convert this whispered speech into a normal speaking voice while preserving the speaker, content, and duration."
            return "Convert this speech into a soft whisper while preserving the speaker, content, and duration."
        if mode == "speech_enhance":
            actions: list[str] = []
            if controls.get("denoise"):
                actions.append("remove background noise")
            if controls.get("dereverb"):
                actions.append("remove room reverberation")
            damage = [*controls.get("damage", [])]
            custom = str(controls.get("custom_damage", "")).strip()
            if custom:
                damage.append(custom)
            if damage:
                actions.append("repair " + ", ".join(damage))
            if not actions:
                actions.append("restore natural, clear audio quality")
            policy = "preserve every original speaker" if controls.get("preserve_all_speakers", True) else "preserve the principal speaker"
            return f"Please {', '.join(actions)} and {policy}; output clean speech of the same length as the input."
        if mode == "speech_separate":
            if controls.get("speaker_method") == "order":
                target = f"the {_ordinal(int(controls.get('speaker_order', 1)))} speaker to start talking"
            else:
                target = f"the speaker who says \"{str(controls.get('speaker_phrase', '')).strip()}\""
            cleanup = []
            if controls.get("denoise"):
                cleanup.append("denoise")
            if controls.get("dereverb"):
                cleanup.append("remove room reverberation from")
            clean = f"; {' and '.join(cleanup)} the retained voice" if cleanup else ""
            return f"Keep only {target} and remove every other speaker{clean}; output single-speaker audio of the same length."
        if mode == "vocal_extract":
            quality = " and improve its clarity" if controls.get("enhance_vocal") else ""
            return f"Extract the singing voice{quality}; remove the accompaniment and preserve the full duration."
        return str(controls.get("custom_instruction", "")).strip()

    @staticmethod
    def _duration(mode: str, controls: dict[str, Any], source: Path | None) -> float | None:
        requested = _number(controls.get("target_seconds"))
        if requested and requested > 0:
            return requested
        if mode == "instruct_tts":
            words = len(str(controls.get("target_text", "")).split())
            return max(0.8, words / 2.45)
        if mode == "speed_edit" and source:
            info = torchaudio.info(str(source))
            return (info.num_frames / info.sample_rate) / float(controls.get("speed_factor", 1))
        return None

    def _enhance(
        self,
        instruction: str,
        controls: dict[str, Any],
        source: Path | None,
        duration: float | None,
    ):
        if not controls.get("use_prompt_enhancer"):
            return nullcontext(None)
        from auk.infer.pe import PromptEnhancer

        transcript = str(controls.get("source_transcript", "")).strip()
        enhancer = PromptEnhancer(
            llm_api_key=os.getenv("LLM_API_KEY") or "local",
            llm_base_url=os.environ["LLM_BASE_URL"],
            llm_model=os.environ["LLM_MODEL_NAME"],
            asr_provider=_TranscriptASR(transcript) if source else None,
        )
        return enhancer.prepare(instruction, str(source) if source else None, target_duration=duration)

    def run(self, request: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        mode = str(request["mode"])
        controls = dict(request["controls"])
        source = context.asset(str(controls["source_audio_asset"])) if controls.get("source_audio_asset") else None
        instruction = self._instruction(mode, controls)
        duration = self._duration(mode, controls, source)
        variant = str(controls.get("model_variant", "base"))
        dtype = str(controls.get("dtype", "bf16"))
        engine = self._load_engine(variant, dtype, context)
        context.update("Compiling native AuK message", 0.2, f"Compiled task: {mode}")
        enhanced_metadata = None
        with self._enhance(instruction, controls, source, duration) as enhanced:
            if enhanced is not None:
                context.update("Prompt Enhancer normalized the request", 0.25)
                instruction = enhanced.instruction
                source = Path(enhanced.audio) if enhanced.audio else None
                duration = enhanced.gen_seconds
                enhanced_metadata = enhanced.debug_dict(include_raw_responses=False)
            content: list[dict[str, str]] = [{"type": "text", "text": instruction}]
            if source:
                content.append({"type": "audio", "audio": str(source)})
            messages = [{"role": "user", "content": content}]
            takes = int(controls.get("takes", 1))
            seed = int(controls.get("seed", 42))
            output_format = str(controls.get("output_format", "wav"))
            nfe = int(controls.get("nfe", 32))
            cfg = float(controls.get("cfg", 2.0))
            sway = float(controls.get("sway", -1.0))
            ceiling = float(controls.get("normalization_ceiling", 0.98))
            outputs: list[StudioOutput] = []
            for index in range(takes):
                context.check_cancelled()
                progress = 0.28 + (0.62 * index / max(takes, 1))
                context.update(f"Rendering take {index + 1} of {takes}", progress)
                audio, sample_rate = engine.generate(
                    messages,
                    audio=str(source) if source else None,
                    gen_seconds=duration,
                    nfe=nfe,
                    cfg_strength=cfg,
                    sway_sampling_coef=sway,
                    t_grid=None,
                    seed=(seed + index) & 0xFFFFFFFF,
                )
                peak = float(audio.abs().max().item()) if audio.numel() else 0.0
                attenuated = False
                if peak > ceiling > 0:
                    audio = audio * (ceiling / peak)
                    attenuated = True
                path = context.output_dir / f"{mode}-take-{index + 1:02d}.{output_format}"
                torchaudio.save(str(path), audio.to(torch.float32).cpu(), sample_rate)
                outputs.append(
                    StudioOutput(
                        path=path,
                        kind="audio",
                        label=f"Take {index + 1}",
                        media_type="audio/flac" if output_format == "flac" else "audio/wav",
                        metadata={
                            "sample_rate": sample_rate,
                            "duration_seconds": round(audio.shape[-1] / sample_rate, 4),
                            "seed": (seed + index) & 0xFFFFFFFF,
                            "variant": variant,
                            "dtype": dtype,
                            "nfe": 4 if variant == "flash" else nfe,
                            "cfg": 0.0 if variant == "flash" else cfg,
                            "peak_before_safety": peak,
                            "safety_attenuated": attenuated,
                        },
                    )
                )
        context.update("Writing reproducibility record", 0.94)
        direction = context.output_dir / "direction.txt"
        direction.write_text(instruction.rstrip() + "\n", encoding="utf-8")
        record = context.output_dir / "generation.json"
        record.write_text(
            json.dumps(
                {
                    "mode": mode,
                    "compiled_instruction": instruction,
                    "target_duration_seconds": duration,
                    "prompt_enhancer": enhanced_metadata,
                    "controls": controls,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        outputs.extend(
            [
                StudioOutput(direction, "text", "Compiled direction", "text/plain"),
                StudioOutput(record, "json", "Generation record", "application/json"),
            ]
        )
        return outputs

    def vox_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("input", "")).strip()
        if not text:
            raise ValueError("The OpenAI-compatible audio request needs a non-empty 'input'.")
        reference = str(payload.get("reference_asset", "")).strip()
        voice = str(payload.get("voice", "")).strip()
        mode = "zero_shot_tts" if reference else "instruct_tts"
        controls: dict[str, Any] = {
            "source_audio_asset": reference,
            "target_text": text,
            "voice_description": voice if voice and voice not in {"alloy", "default"} else "A natural, clear, balanced adult voice",
            "target_seconds": _number(payload.get("duration")) or max(0.8, len(text.split()) / 2.45),
            "emotion_mix": payload.get("emotions") or {},
            "emotion_intensity": float(payload.get("emotion_intensity", 65)),
            "warmth": float(payload.get("warmth", 0)),
            "energy": float(payload.get("energy", 0)),
            "breathiness": float(payload.get("breathiness", 0)),
            "performance_pace": float(payload.get("speed", 1.0)),
            "performance_note": str(payload.get("instructions", "")),
            "model_variant": str(payload.get("model", "base")).removeprefix("auk-"),
            "takes": 1,
            "seed": int(payload.get("seed", 42)),
            "nfe": int(payload.get("nfe", 32)),
            "cfg": float(payload.get("cfg", 2.0)),
            "sway": float(payload.get("sway", -1.0)),
            "dtype": "bf16",
            "output_format": "flac" if payload.get("response_format") == "flac" else "wav",
            "normalization_ceiling": 0.98,
            "use_prompt_enhancer": False,
            "source_transcript": str(payload.get("reference_transcript", "")),
        }
        return {"mode": mode, "controls": controls, "client": {"source": "openai-compatible-v1-audio", "schema": 1}}
