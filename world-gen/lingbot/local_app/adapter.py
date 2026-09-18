from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

from local_app.runtime import StudioAdapter, StudioContext, StudioOutput, gpu_snapshot

from local_app.camera_path import design, import_bundle, package


PROGRESS = re.compile(r"^MM_PROGRESS\s+([0-9.]+)\s+(.*)$")


class Adapter(StudioAdapter):
    def __init__(self, project_root: Path, runtime_root: Path) -> None:
        super().__init__(project_root, runtime_root)
        self.checkpoint = project_root / "lingbot-world-v2-1.3b-causal-fast"
        self.assets = project_root / "lingbot-world-v2-14b-causal-fast"

    def health(self) -> dict[str, Any]:
        gpu = gpu_snapshot()
        device = (gpu.get("devices") or [{}])[0]
        checks = [
            ("1.3B causal-fast DiT", self.checkpoint / "model.safetensors.index.json"),
            ("Shared UMT5 encoder", self.assets / "models_t5_umt5-xxl-enc-bf16.pth"),
            ("Shared Wan VAE", self.assets / "Wan2.1_VAE.pth"),
            ("Shared tokenizer", self.assets / "google" / "umt5-xxl" / "tokenizer.json"),
            ("Python environment", self.project_root / ".venv" / "bin" / "python"),
        ]
        details = [{"label": label, "ready": path.is_file(), "path": str(path)} for label, path in checks]
        gpu_ok = bool(device) and int(device.get("memory_total_mib", 0)) >= 30000
        details.append({"label": "Single CUDA GPU ≥ 30 GiB", "ready": gpu_ok, "value": device.get("name", "unavailable")})
        return {"ready": all(item[1].is_file() for item in checks) and gpu_ok, "loaded": False, "details": details}

    def validate(self, request: dict[str, Any], resolve_asset: Callable[[str], Path]) -> None:
        mode = str(request.get("mode", ""))
        controls = request.get("controls") or {}
        if mode not in {"generate_world", "camera_designer", "generate_from_bundle"}:
            raise ValueError(f"Unknown LingBot mode: {mode}")
        frames = int(controls.get("frame_num", 81))
        if frames < 5 or (frames - 1) % 4:
            raise ValueError("Frame count must be 4n+1 and at least five.")
        if mode != "camera_designer":
            image = resolve_asset(str(controls.get("source_image_asset", "")))
            if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                raise ValueError("World generation requires a PNG, JPEG, or WebP source image.")
            if not str(controls.get("prompt", "")).strip():
                raise ValueError("Describe the world and intended interaction.")
        if mode == "generate_from_bundle":
            bundle = resolve_asset(str(controls.get("action_bundle_asset", "")))
            if bundle.suffix.lower() not in {".zip", ".npz"}:
                raise ValueError("Imported camera actions must be a ZIP or NPZ bundle.")

    def run(self, request: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        mode = request["mode"]
        controls = request.get("controls") or {}
        action = context.output_dir / "camera"
        context.update("Preparing the camera timeline", 0.04)
        if mode == "generate_from_bundle":
            metadata = import_bundle(context.asset(str(controls["action_bundle_asset"])), action)
            controls = {**controls, "frame_num": int(metadata["frames"])}
        else:
            metadata = design(controls, action)
        bundle = package(action, context.output_dir / "lingbot-camera.zip")
        outputs = [
            StudioOutput(action / "camera-path.svg", "image", "Camera path preview", "image/svg+xml", metadata),
            StudioOutput(action / "camera.json", "document", "Camera path metadata", "application/json", metadata),
            StudioOutput(bundle, "file", "Reusable LingBot camera bundle", "application/zip", metadata),
        ]
        if mode == "camera_designer":
            context.update("Camera path ready", 0.99)
            return outputs

        image = context.asset(str(controls["source_image_asset"]))
        destination = context.output_dir / "lingbot-world.mp4"
        payload = {
            "project_root": str(self.project_root),
            "checkpoint": str(self.checkpoint),
            "assets": str(self.assets),
            "image": str(image),
            "action_path": str(action),
            "output": str(destination),
            "prompt": str(controls["prompt"]).strip(),
            "size": str(controls.get("size", "480*832")),
            "frame_num": int(controls.get("frame_num", metadata["frames"])),
            "chunk_size": int(controls.get("chunk_size", 4)),
            "seed": int(controls.get("seed", 42)),
            "sample_shift": float(controls.get("sample_shift", 5.0)),
            "local_attn_size": int(controls.get("local_attn_size", 18)),
            "sink_size": int(controls.get("sink_size", 6)),
            "max_attention_size": int(controls.get("max_attention_size", 0)),
            "convert_model_dtype": bool(controls.get("convert_model_dtype", False)),
        }
        receipt = context.output_dir / "request.json"
        receipt.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        def progress(line: str) -> tuple[str, float] | None:
            match = PROGRESS.match(line)
            return (match.group(2), float(match.group(1))) if match else None

        context.run_process(
            [sys.executable, str(self.project_root / "local_app" / "run.py"), str(receipt)],
            cwd=self.project_root,
            env={"PYTHONPATH": f"{self.project_root.parent.parent}:{self.project_root}"},
            progress_parser=progress,
        )
        outputs.insert(0, StudioOutput(destination, "video", "Interactive world capture", "video/mp4", {**metadata, "prompt": payload["prompt"], "seed": payload["seed"]}))
        outputs.append(StudioOutput(receipt, "document", "Reproducible generation request", "application/json"))
        return outputs
