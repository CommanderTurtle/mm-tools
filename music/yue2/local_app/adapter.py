from __future__ import annotations

import gc
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
import torch

from local_app.runtime import StudioAdapter, StudioContext, StudioOutput


GENERATION_MODES = {
    "compose_full": "full",
    "compose_melody": "melody",
    "compose_direct": "off",
    "realize_score": "full",
    "edit_score": "full",
    "cover_audio": "melody",
    "resume_plan": "full",
}
PLAN_FILES = frozenset({"plan.json", "plan_manifest.json", "abc_tokens.npy", "prefix.npy", "score.abc"})
WINDOW = re.compile(r"Window\s+(\d+)/(\d+)")


def _safe_id(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip(".-")
    return (clean[:180] or "song")


def _media_kind(path: Path) -> tuple[str, str | None]:
    suffix = path.suffix.lower()
    if suffix in {".flac", ".wav", ".mp3", ".ogg", ".m4a"}:
        return "audio", {".flac": "audio/flac", ".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".m4a": "audio/mp4"}.get(suffix)
    if suffix in {".mid", ".midi"}:
        return "midi", "audio/midi"
    if suffix in {".abc", ".json", ".lab", ".txt", ".md"}:
        return "text", "text/plain" if suffix != ".json" else "application/json"
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".svg"}:
        return "image", "image/svg+xml" if suffix == ".svg" else f"image/{'jpeg' if suffix in {'.jpg', '.jpeg'} else suffix[1:]}"
    return "data", "application/octet-stream"


class Adapter(StudioAdapter):
    def __init__(self, project_root: Path, runtime_root: Path) -> None:
        super().__init__(project_root, runtime_root)
        self.model_dir = project_root / "models" / "YuE2-3B"
        self.vae_dir = project_root / "models" / "YuE2-Vae"
        self.sheetsage_dir = project_root / "models" / "SheetSage2"
        self.mert_dir = project_root / "models" / "MERT-v2-FullSong"
        self.sheetsage_python = project_root / ".venv-sheetsage2" / "bin" / "python"
        self.pipe: Any | None = None
        self.profile: tuple[str, str] | None = None
        self._abc_tools: Any | None = None

    def health(self) -> dict[str, Any]:
        checks = {
            "YuE2-3B": self.model_dir / "model.safetensors",
            "YuE2 VAE": self.vae_dir / "model.safetensors",
            "SheetSage2": self.sheetsage_dir / "model.safetensors",
            "MERT-v2-FullSong": self.mert_dir / "model.safetensors",
            "SheetSage environment": self.sheetsage_python,
        }
        details = [{"label": label, "ready": path.is_file(), "path": str(path)} for label, path in checks.items()]
        cuda = torch.cuda.is_available()
        if cuda:
            total = torch.cuda.get_device_properties(0).total_memory
            details.append({"label": "CUDA GPU >= 24 GiB", "ready": total >= 24 * 1024**3, "value": torch.cuda.get_device_name(0)})
        else:
            details.append({"label": "CUDA GPU >= 24 GiB", "ready": False, "value": "CUDA unavailable"})
        return {
            "ready": all(item["ready"] for item in details),
            "loaded": bool(self.pipe is not None and getattr(self.pipe, "_model", None) is not None),
            "profile": self.profile,
            "details": details,
        }

    def validate(self, request: dict[str, Any], resolve_asset: Callable[[str], Path]) -> None:
        mode = str(request.get("mode", ""))
        controls = request.get("controls")
        if mode not in {*GENERATION_MODES, "plan_only", "transcribe", "score_lab", "decode_latent", "krea_plan"}:
            raise ValueError(f"Unsupported YuE2 workflow: {mode}")
        if not isinstance(controls, dict):
            raise ValueError("YuE2 controls must be an object.")
        if mode in {*GENERATION_MODES, "plan_only"}:
            if mode != "resume_plan":
                if not str(controls.get("style", "")).strip():
                    raise ValueError("Describe the language, genre, instrumentation, vocal character, and tempo.")
                if not str(controls.get("lyrics", "")).strip():
                    raise ValueError("Lyrics with section markers are required.")
            takes = int(controls.get("takes", 1))
            if not 1 <= takes <= 8:
                raise ValueError("Generate between one and eight candidates.")
        for key in ("source_audio_asset", "abc_asset", "original_abc_asset", "latent_asset", "plan_bundle_asset"):
            value = controls.get(key)
            if value:
                resolve_asset(str(value))
        if mode in {"transcribe", "cover_audio"} and not controls.get("source_audio_asset"):
            raise ValueError("Upload a source recording.")
        if mode in {"realize_score", "edit_score"} and not (str(controls.get("abc_text", "")).strip() or controls.get("abc_asset")):
            raise ValueError("Paste or upload a native YuE2 ABC score.")
        if mode == "decode_latent" and not controls.get("latent_asset"):
            raise ValueError("Upload a latent.npy artifact.")
        if mode == "resume_plan" and not controls.get("plan_bundle_asset"):
            raise ValueError("Upload a plan-bundle.zip created by this studio.")
        if mode == "krea_plan":
            if not str(controls.get("krea_brief", "")).strip():
                raise ValueError("Describe the musical direction to plan.")
            lane = str(controls.get("krea_mode", "brief"))
            if lane not in {"brief", "keep_lyrics", "song", "ask"}:
                raise ValueError("Unknown Krea planning lane.")
            if lane == "keep_lyrics" and not str(controls.get("krea_lyrics", "")).strip():
                raise ValueError("Keep-lyrics planning requires the finished lyrics.")
        gpu = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 24 * 1024**3
        if mode != "score_lab" and not gpu:
            raise RuntimeError("YuE2 Studio requires a BF16-capable CUDA GPU with at least 24 GiB VRAM.")

    def load(self) -> dict[str, Any]:
        self._ensure_pipe({"backend": "cuda_graph", "precision": "native_bf16"})
        self.pipe._load_model()
        return self.health()

    def unload(self) -> dict[str, Any]:
        if self.pipe is not None:
            try:
                self.pipe.close()
            finally:
                self.pipe = None
                self.profile = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.health()

    def close(self) -> None:
        self.unload()

    def run(self, request: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        mode = request["mode"]
        controls = request["controls"]
        if mode == "transcribe":
            self._release_pipe()
            return self._transcribe(controls, context, context.output_dir)
        if mode == "score_lab":
            return self._score_lab(controls, context)
        if mode == "krea_plan":
            self._release_pipe()
            return self._run_krea(controls, context)
        if mode == "decode_latent":
            return self._decode_latent(controls, context)
        if mode == "cover_audio":
            self._release_pipe()
            score_dir = context.output_dir / "source-score"
            transcription = self._transcribe(controls, context, score_dir, force_melody=not bool(controls.get("keep_source_harmony")))
            controls = {**controls, "abc_text": (score_dir / "score.abc").read_text(encoding="utf-8")}
            context.update("Source score ready; loading YuE2", 0.20)
            return [*transcription, *self._generate(mode, controls, context)]
        return self._generate(mode, controls, context)

    def _ensure_pipe(self, controls: dict[str, Any]) -> Any:
        backend = "torch-eager" if controls.get("backend") == "eager" else "torch"
        quantization = "fp8" if controls.get("precision") == "fp8_ar" else "none"
        wanted = (backend, quantization)
        if self.pipe is not None and self.profile != wanted:
            self._release_pipe()
        if self.pipe is None:
            from yue2 import YuE2Pipeline

            self.pipe = YuE2Pipeline.from_pretrained(
                str(self.model_dir),
                vae=str(self.vae_dir),
                local_files_only=True,
                device="cuda",
                memory_budget_gib=30,
                backend=backend,
                quantization=quantization,
                offload_ar=False,
                progress=False,
            )
            self.profile = wanted
        return self.pipe

    def _release_pipe(self) -> None:
        if self.pipe is not None:
            self.pipe.close()
        self.pipe = None
        self.profile = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generation_config(self, controls: dict[str, Any]) -> Any:
        from yue2.protocol import GenerationConfig, Sampling

        abc = Sampling(
            float(controls.get("abc_temperature", 0.7)), float(controls.get("abc_top_p", 0.9)),
            int(controls.get("abc_top_k", 30)), float(controls.get("abc_repetition_penalty", 1.005)),
            int(controls.get("abc_penalty_window", 100)), int(controls.get("abc_min_tokens", 32)),
            int(controls.get("abc_max_tokens", 4096)),
        )
        semantic = Sampling(
            float(controls.get("semantic_temperature", 1.0)), float(controls.get("semantic_top_p", 0.95)),
            int(controls.get("semantic_top_k", 100)), float(controls.get("semantic_repetition_penalty", 1.2)),
            int(controls.get("semantic_penalty_window", 50)), int(controls.get("semantic_min_tokens", 200)),
            int(controls.get("semantic_max_tokens", 9000)),
        )
        return GenerationConfig(abc=abc, semantic=semantic, ode_steps=int(controls.get("ode_steps", 32)))

    def _song_request(self, mode: str, controls: dict[str, Any], seed: int, score: str | None) -> Any:
        from yue2.protocol import SongRequest

        cot = GENERATION_MODES.get(mode, str(controls.get("planning", "full")))
        if mode == "cover_audio":
            cot = "full" if controls.get("keep_source_harmony") else "melody"
        if mode in {"realize_score", "edit_score"}:
            cot = str(controls.get("score_mode", "full"))
        cfg = controls.get("cfg_scale")
        return SongRequest(
            style=self._compiled_style(controls),
            lyrics=str(controls.get("lyrics", "")).strip(),
            cot=cot,
            seed=seed,
            abc=score,
            cfg_scale=None if cfg in {None, ""} else float(cfg),
            id=_safe_id(str(controls.get("project_name", "song"))),
        )

    def _compiled_style(self, controls: dict[str, Any]) -> str:
        parts = [str(controls.get("style", "")).strip()]
        language = str(controls.get("language", "")).strip()
        if language and language != "auto":
            parts.append(f"Language: {language}")
        for key, label in (("genres", "Genres"), ("instruments", "Instrumentation"), ("moods", "Mood")):
            values = controls.get(key) or []
            if values:
                parts.append(f"{label}: {', '.join(str(value) for value in values)}")
        for key, label in (("vocal_direction", "Vocal direction"), ("key_hint", "Key"), ("meter_hint", "Meter"), ("production_notes", "Production")):
            value = str(controls.get(key, "")).strip()
            if value:
                parts.append(f"{label}: {value}")
        bpm = controls.get("bpm")
        if bpm not in {None, "", 0, 0.0}:
            parts.append(f"Tempo: {int(bpm)} BPM")
        avoid = str(controls.get("avoid", "")).strip()
        if avoid:
            parts.append(f"Avoid: {avoid}")
        return ". ".join(part.rstrip(". ") for part in parts if part).strip()

    def _score_text(self, controls: dict[str, Any], context: StudioContext) -> str | None:
        text = str(controls.get("abc_text", "")).strip()
        if text:
            return text + ("\n" if not text.endswith("\n") else "")
        asset = controls.get("abc_asset")
        if asset:
            return context.asset(str(asset)).read_text(encoding="utf-8")
        return None

    def _generate(self, mode: str, controls: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        from yue2 import SymbolicPlan

        pipe = self._ensure_pipe(controls)
        pipe.generation_config = self._generation_config(controls)
        pipe.vae_core_frames = int(controls.get("vae_core_frames", 1024))
        score = self._score_text(controls, context)
        if score and controls.get("strip_chords"):
            score = self._tools().strip_chords(score, str(controls.get("keep_voice", "both")))
        restored: Any | None = None
        if mode == "resume_plan":
            plan_dir = context.output_dir / "restored-plan"
            self._extract_plan(context.asset(str(controls["plan_bundle_asset"])), plan_dir)
            restored = SymbolicPlan.load(plan_dir)
        takes = 1 if mode in {"plan_only", "resume_plan"} else int(controls.get("takes", 1))
        seed = int(controls.get("seed", 831001))
        step = int(controls.get("seed_step", 1))
        outputs: list[StudioOutput] = []
        for index in range(takes):
            context.check_cancelled()
            take_seed = seed + index * step
            destination = context.output_dir / ("plan" if mode == "plan_only" else f"take-{index + 1:02d}")
            destination.mkdir(parents=True, exist_ok=True)
            base = index / max(takes, 1)
            span = 1 / max(takes, 1)
            context.update(f"Candidate {index + 1}/{takes}: preparing", 0.03 + base * 0.9)
            if restored is not None:
                request = restored.request
                plan = restored
            else:
                request = self._song_request(mode, controls, take_seed, score)
                token_progress = self._token_progress(context, base, span, pipe.generation_config)
                plan = pipe.plan(request=request, cancelled=context.cancel_event.is_set, on_token=token_progress)
            plan.save(destination)
            self._write_plan_bundle(destination)
            if mode == "plan_only":
                outputs.extend(self._collect(destination, controls, "Symbolic plan"))
                continue
            token_progress = self._token_progress(context, base, span, pipe.generation_config)
            context.update(f"Candidate {index + 1}/{takes}: generating semantic music", 0.12 + base * 0.86)
            semantic = pipe.generate_semantic(plan, cancelled=context.cancel_event.is_set, on_token=token_progress)
            context.update(f"Candidate {index + 1}/{takes}: flow-matching acoustics", 0.55 + base * 0.4)
            nar_start = time.perf_counter()
            latents = pipe.synthesize(semantic, cancelled=context.cancel_event.is_set)
            nar_seconds = time.perf_counter() - nar_start
            context.check_cancelled()
            context.update(f"Candidate {index + 1}/{takes}: decoding 48 kHz stereo", 0.78 + base * 0.18)
            vae_start = time.perf_counter()
            audio = self._decode_gpu_only(latents, pipe.vae_core_frames, context)
            timing = {
                "abc": plan.timing,
                "semantic": semantic.timing,
                "nar_seconds": nar_seconds,
                "vae_seconds": time.perf_counter() - vae_start,
                "load": dict(pipe.load_timing),
            }
            config = pipe.effective_config(request)
            config["gpu_residency"] = "AR/NAR remains on CUDA; VAE is created and destroyed on CUDA; no CPU model offload"
            self._save_song(destination, request, plan, semantic, latents, audio, config, pipe.weights, timing)
            if plan.truncated or semantic.truncated:
                context.log(f"Candidate {index + 1} reached a native token limit; artifacts were retained.", "warning")
            outputs.extend(self._collect(destination, controls, f"Candidate {index + 1}"))
        context.update("YuE2 artifacts ready", 0.99)
        return outputs

    def _run_krea(self, controls: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        from local_app.krea import CHECKPOINT_NAME, KreaPlanner

        planner = KreaPlanner(self.project_root)
        sampling = {key: controls.get(f"krea_{key}") for key in (
            "temperature", "top_k", "top_p", "min_p", "repetition_penalty",
            "presence_penalty", "seed", "max_length",
        ) if controls.get(f"krea_{key}") is not None}
        result = planner.plan(
            mode=str(controls.get("krea_mode", "brief")),
            direction=str(controls.get("krea_brief", "")),
            lyrics=str(controls.get("krea_lyrics", "")),
            constraints=str(controls.get("krea_constraints", "")),
            web_search=bool(controls.get("krea_research", False)),
            search_query=str(controls.get("krea_research_query", "")),
            sampling=sampling or None,
            context=context,
        )
        context.check_cancelled()
        destination = context.output_dir
        (destination / "direction.md").write_text(result["text"] + "\n", encoding="utf-8")
        manifest = {
            "planner": "krea2",
            "mode": result["mode"],
            "checkpoint": CHECKPOINT_NAME,
            "sections": result["sections"],
            "sources": result["sources"],
        }
        (destination / "krea_plan.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        caption_style = "\n\n".join(
            result["sections"][name] for name in ("Global Metadata", "Vocal Details", "Arrangement") if result["sections"].get(name)
        )
        outputs = [
            StudioOutput(
                destination / "direction.md",
                kind="text",
                label="Krea direction",
                metadata={
                    "krea": True,
                    "krea_mode": result["mode"],
                    "style": caption_style,
                    "lyrics": result["sections"].get("Lyrics", ""),
                },
            ),
            StudioOutput(destination / "krea_plan.json", kind="text", label="Krea plan manifest"),
        ]
        if result["sections"].get("Lyrics"):
            (destination / "lyrics.txt").write_text(result["sections"]["Lyrics"] + "\n", encoding="utf-8")
            outputs.append(StudioOutput(destination / "lyrics.txt", kind="text", label="Drafted lyrics"))
        context.update("Krea plan ready", 0.99)
        return outputs

    def _token_progress(self, context: StudioContext, base: float, span: float, config: Any) -> Callable[[str, int], None]:
        counts = {"abc": 0, "semantic": 0}
        last = [0.0]

        def callback(phase: str, _: int) -> None:
            counts[phase] = counts.get(phase, 0) + 1
            now = time.monotonic()
            if now - last[0] < 0.35:
                return
            last[0] = now
            if phase == "abc":
                fraction = min(counts[phase] / max(config.abc.max_tokens, 1), 1)
                progress = base * 0.9 + 0.04 + span * (0.18 * fraction)
                stage = f"Planning score · {counts[phase]} tokens"
            else:
                fraction = min(counts[phase] / max(config.semantic.max_tokens, 1), 1)
                progress = base * 0.9 + 0.22 + span * (0.38 * fraction)
                stage = f"Writing song · {counts[phase]} semantic tokens"
            context.update(stage, min(progress, 0.94))

        return callback

    def _decode_gpu_only(self, latents: np.ndarray, core_frames: int, context: StudioContext) -> np.ndarray:
        from yue2.modeling_vae import YuE2VAE

        vae = YuE2VAE.from_pretrained(self.vae_dir, decoder_only=True, device="cuda", local_files_only=True)
        z = torch.as_tensor(latents, dtype=torch.float32)
        if z.ndim == 2 and z.shape[1] == 64:
            z = z.T.unsqueeze(0)
        if z.ndim != 3 or z.shape[0] != 1 or z.shape[1] != 64:
            raise ValueError("Expected native acoustic latents shaped [T,64] or [1,64,T].")
        total = max(1, math.ceil(z.shape[-1] / core_frames))
        try:
            with torch.inference_mode():
                audio = vae.decode_tiled(
                    z,
                    core_frames=core_frames,
                    halo_frames=16,
                    output_device="cpu",
                    on_progress=lambda done, _: context.update(f"Decoding audio · {done}/{total} tiles", 0.80 + 0.16 * done / total),
                )
            if not torch.isfinite(audio).all():
                raise ValueError("YuE2 VAE produced non-finite audio.")
            return audio[0].float().clamp(-1, 1).T.contiguous().numpy()
        finally:
            del vae
            gc.collect()
            torch.cuda.empty_cache()

    def _save_song(
        self,
        destination: Path,
        request: Any,
        plan: Any,
        semantic: Any,
        latents: np.ndarray,
        audio: np.ndarray,
        config: dict[str, Any],
        weights: dict[str, Any],
        timing: dict[str, Any],
    ) -> None:
        from yue2.storage import collect_hashes, identity, write_json

        plan.save(destination)
        sf.write(destination / "audio.flac", audio, 48000, subtype="PCM_24")
        np.save(destination / "semantic.npy", np.asarray(semantic.tokens, dtype=np.int32))
        np.save(destination / "latent.npy", latents.astype(np.float32))
        write_json(destination / "request.json", request.to_dict())
        write_json(destination / "config.json", config)
        request_identity = identity({"request": request.to_dict(), "config": config, "weights": weights})
        result = {
            "status": "complete",
            "identity": request_identity,
            "truncated": {"abc": plan.truncated, "semantic": semantic.truncated},
            "sample_rate": 48000,
            "audio_seconds": len(audio) / 48000,
            "weights": weights,
            "timing": timing,
            "artifacts": collect_hashes(destination),
        }
        write_json(destination / "result.json", result)

    def _write_plan_bundle(self, directory: Path) -> None:
        bundle = directory / "plan-bundle.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in sorted(PLAN_FILES):
                path = directory / name
                if path.is_file():
                    archive.write(path, name)

    def _extract_plan(self, source: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            names = {Path(item.filename).name for item in members}
            if not {"plan.json", "plan_manifest.json", "abc_tokens.npy", "prefix.npy"} <= names:
                raise ValueError("Plan bundle is missing required exact-plan artifacts.")
            for item in members:
                name = Path(item.filename).name
                if name not in PLAN_FILES or Path(item.filename).is_absolute() or ".." in Path(item.filename).parts:
                    raise ValueError(f"Unexpected plan-bundle entry: {item.filename}")
                if item.external_attr >> 16 & 0o170000 == 0o120000:
                    raise ValueError("Plan bundles cannot contain symlinks.")
                (destination / name).write_bytes(archive.read(item))

    def _transcribe(
        self,
        controls: dict[str, Any],
        context: StudioContext,
        destination: Path,
        force_melody: bool | None = None,
    ) -> list[StudioOutput]:
        if not self.sheetsage_python.is_file():
            raise RuntimeError("Run ./setupwithuv.sh before using SheetSage2.")
        source = context.asset(str(controls["source_audio_asset"]))
        destination.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.sheetsage_python), str(self.project_root / "local_app" / "transcribe.py"),
            str(source), "--output", str(destination), "--model", str(self.sheetsage_dir),
            "--base-model", str(self.mert_dir), "--device", "cuda", "--dtype", str(controls.get("transcribe_dtype", "bf16")),
            "--preset", str(controls.get("transcribe_preset", "default")), "--local-files-only",
        ]
        melody = bool(controls.get("melody_only", False)) if force_melody is None else force_melody
        if melody:
            command.append("--melody-only")
        for flag, key in (("--export-logits", "export_logits"), ("--export-scores", "export_scores"), ("--export-embeddings", "export_embeddings"), ("--all-layers", "all_layers")):
            if controls.get(key):
                command.append(flag)
        for flag, key in (("--max-seconds", "max_seconds"), ("--overlap", "overlap_seconds"), ("--lookahead", "lookahead_seconds")):
            value = controls.get(key)
            if value not in {None, "", 0, 0.0}:
                command.extend([flag, str(value)])
        prompts = controls.get("transcription_prompts") or []
        if prompts:
            command.append("--prompts")
            command.extend(str(value) for value in prompts)

        def progress(line: str) -> tuple[str, float] | None:
            match = WINDOW.search(line)
            if not match:
                return None
            done, total = map(int, match.groups())
            return f"SheetSage2 window {done}/{total}", 0.04 + 0.14 * done / max(total, 1)

        context.update("Loading SheetSage2 + MERT-v2", 0.03)
        context.run_process(command, cwd=self.project_root, progress_parser=progress)
        if not (destination / "score.abc").is_file():
            raise RuntimeError("SheetSage2 did not produce a usable ABC score; inspect the job log.")
        return self._collect(destination, controls, "SheetSage2 transcription")

    def _score_lab(self, controls: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        tools = self._tools()
        score = self._score_text(controls, context)
        if not score:
            raise ValueError("Paste or upload the edited ABC score.")
        if controls.get("strip_chords"):
            score = tools.strip_chords(score, str(controls.get("keep_voice", "both")))
        destination = context.output_dir
        score_path = destination / "checked-score.abc"
        score_path.write_text(score, encoding="utf-8")
        report = tools.report(tools.parse_abc(score))
        if controls.get("original_abc_asset"):
            original = context.asset(str(controls["original_abc_asset"])).read_text(encoding="utf-8")
            names = tools.VOICES if controls.get("compare_voices", "both") == "both" else (controls["compare_voices"],)
            report["comparison"] = tools.compare(
                tools.parse_abc(original), tools.parse_abc(score), names, bool(controls.get("allow_tempo_change")),
            )
        (destination / "score-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        context.update("Score checked against the native YuE2 dialect", 0.98)
        return self._collect(destination, controls, "Score lab")

    def _decode_latent(self, controls: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        source = context.asset(str(controls["latent_asset"]))
        latents = np.load(source, allow_pickle=False)
        context.update("Decoding saved acoustic latents on CUDA", 0.15)
        audio = self._decode_gpu_only(latents, int(controls.get("vae_core_frames", 1024)), context)
        target = context.output_dir / "decoded.flac"
        sf.write(target, audio, 48000, subtype="PCM_24")
        metadata = {"sample_rate": 48000, "audio_seconds": len(audio) / 48000, "source": source.name}
        (context.output_dir / "decode.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return self._collect(context.output_dir, controls, "Decoded latents")

    def _tools(self) -> Any:
        if self._abc_tools is None:
            path = self.project_root / "skills" / "yue2-music" / "scripts" / "abc_tools.py"
            spec = importlib.util.spec_from_file_location("mmtools_yue2_abc_tools", path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Cannot load the pinned YuE2 ABC checker.")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self._abc_tools = module
        return self._abc_tools

    def _collect(self, directory: Path, controls: dict[str, Any], label: str) -> list[StudioOutput]:
        outputs: list[StudioOutput] = []
        lyrics = str(controls.get("lyrics", ""))
        style = str(controls.get("style", ""))
        scanned = [path for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "manifest.json"]
        timestamps = self._vocal_timestamps(directory, scanned, lyrics)
        for path in scanned:
            kind, media_type = _media_kind(path)
            relative = path.relative_to(directory).as_posix()
            metadata: dict[str, Any] = {"workflow": label, "relative": relative}
            if kind == "audio":
                metadata.update({"lyrics": lyrics, "style": style, "sample_rate": 48000})
                if timestamps:
                    metadata["lyric_timestamps"] = timestamps
            outputs.append(StudioOutput(path, kind, f"{label} · {relative}", media_type, metadata))
        return outputs

    def _vocal_timestamps(self, directory: Path, files: list[Path], lyrics: str) -> list[dict[str, Any]] | None:
        """StemKit post-pass: align lyric lines to real vocal activity in the
        rendered take. Any failure keeps the visualizer's uniform spread."""
        lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
        if len(lines) < 2:
            return None
        audio = next((path for path in files if path.suffix.lower() == ".wav"), None)
        if audio is None:
            source = next((path for path in files if path.suffix.lower() in {".flac", ".mp3"}), None)
            if source is None:
                return None
            audio = directory / "vocal-sync.wav"
            try:
                data, rate = sf.read(source, always_2d=True)
                sf.write(audio, data.T, int(rate), subtype="PCM_16")
            except Exception:
                return None
        try:
            from stemkit import studio_api
        except Exception:
            return None
        try:
            return studio_api.vocal_timestamps_for_lyrics(audio, lines, directory / "vocal-sync-work")
        except Exception:
            return None
