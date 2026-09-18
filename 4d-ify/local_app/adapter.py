from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

from studio.runtime import StudioAdapter, StudioContext, StudioOutput, gpu_snapshot


PROGRESS = re.compile(r"^MM_PROGRESS\s+([0-9.]+)\s+(.*)$")


def _kind(path: Path) -> tuple[str, str | None]:
    suffix = path.suffix.lower()
    if suffix in {".mp4", ".webm", ".mov"}:
        return "video", "video/mp4" if suffix == ".mp4" else None
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image", f"image/{'jpeg' if suffix in {'.jpg', '.jpeg'} else suffix[1:]}"
    if suffix in {".json", ".txt"}:
        return "document", "application/json" if suffix == ".json" else "text/plain"
    if suffix == ".safetensors":
        return "data", "application/octet-stream"
    return "file", None


class Adapter(StudioAdapter):
    def __init__(self, project_root: Path, runtime_root: Path) -> None:
        super().__init__(project_root, runtime_root)
        self.model_dir = project_root / "models"
        self.gvhmr_root = project_root
        self.smplx = self.model_dir / "smplx" / "SMPLX_NEUTRAL.npz"

    def health(self) -> dict[str, Any]:
        gpu = gpu_snapshot()
        device = gpu.get("devices", [{}])[0] if gpu.get("available") else {}
        checks = [
            ("Integrated SMPL-X runtime", self.project_root / "smplx" / "__init__.py", True),
            ("4DAnyone checkpoint", self.model_dir / "4danyone" / "model.safetensors", True),
            ("Wan VAE", self.model_dir / "4danyone" / "Wan2.2_VAE.pth", True),
            ("GVHMR checkpoint", self.model_dir / "gvhmr" / "gvhmr_siga24_release.ckpt", True),
            ("DPVO moving-camera checkpoint", self.model_dir / "gvhmr" / "dpvo.pth", False),
            ("BiRefNet", self.model_dir / "birefnet" / "model.safetensors", True),
            ("SMPL-X neutral body", self.smplx, False),
            ("Python environment", self.project_root / ".venv" / "bin" / "python", True),
        ]
        details = [
            {"label": label, "ready": path.is_file(), "required": required, "path": str(path)}
            for label, path, required in checks
        ]
        gpu_ok = bool(device) and int(device.get("memory_total_mib", 0)) >= 22000
        details.append({"label": "CUDA GPU >= 22 GiB", "ready": gpu_ok, "value": device.get("name", "unavailable")})
        required_ready = all(item[1].is_file() for item in checks if item[2]) and gpu_ok
        return {
            "ready": required_ready,
            "generation_ready": required_ready and self.smplx.is_file(),
            "loaded": False,
            "details": details,
        }

    def validate(self, request: dict[str, Any], resolve_asset: Callable[[str], Path]) -> None:
        mode = str(request.get("mode", ""))
        controls = request.get("controls") or {}
        if mode == "install_smplx":
            source = resolve_asset(str(controls.get("smplx_asset", "")))
            if source.suffix.lower() not in {".zip", ".npz"}:
                raise ValueError("SMPL-X setup accepts models_smplx_v1_1.zip or SMPLX_NEUTRAL.npz.")
            return
        if mode != "generate_4d":
            raise ValueError(f"Unknown 4DAnyone mode: {mode}")
        resolve_asset(str(controls.get("source_video_asset", "")))
        views = int(controls.get("views_per_layer", 6))
        pitches = controls.get("layer_pitches") or [15]
        if views * len(pitches) % 6:
            raise ValueError("Total target views must be divisible by six.")
        if not 1 <= views <= 48:
            raise ValueError("Views per layer must be between 1 and 48.")
        if any(not -15 <= int(value) <= 45 for value in pitches):
            raise ValueError("Every pitch must be between -15° and 45°.")
        camera_motion = str(controls.get("camera_motion", "simple_vo"))
        if camera_motion not in {"static", "simple_vo", "dpvo"}:
            raise ValueError("Camera recovery must be Static, SimpleVO, or DPVO.")
        focal_length = float(controls.get("focal_length_mm", 0) or 0)
        if focal_length and not 1 <= focal_length <= 1000:
            raise ValueError("Lens focal length must be 0 (auto) or between 1 and 1000 mm.")
        if camera_motion == "dpvo" and not (self.model_dir / "gvhmr" / "dpvo.pth").is_file():
            raise ValueError("DPVO needs models/gvhmr/dpvo.pth. Run ../models/download_models.py 4d first.")
        if not self.smplx.is_file():
            raise ValueError("Install the licensed SMPL-X neutral body in the Setup mode before generation.")

    def run(self, request: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        controls = request.get("controls") or {}
        if request["mode"] == "install_smplx":
            from fdanyone.download import install_smplx

            source = context.asset(str(controls["smplx_asset"]))
            context.update("Installing the licensed SMPL-X body model", 0.20)
            installed = install_smplx(source, self.model_dir, self.gvhmr_root)
            receipt = context.output_dir / "smplx-installation.json"
            receipt.write_text(json.dumps({"installed": str(installed), "source": source.name}, indent=2) + "\n", encoding="utf-8")
            context.update("SMPL-X is ready", 0.99)
            return [StudioOutput(receipt, "document", "SMPL-X installation receipt", "application/json")]

        source = context.asset(str(controls["source_video_asset"]))
        destination = context.output_dir / "capture"
        pitches = [int(value) for value in (controls.get("layer_pitches") or [15])]
        focal_length = float(controls.get("focal_length_mm", 0) or 0)
        payload = {
            "video_path": str(source),
            "output_dir": str(destination),
            "views_per_layer": int(controls.get("views_per_layer", 6)),
            "layer_pitches": pitches,
            "start_yaw": int(controls.get("start_yaw", 0)),
            "yaw_span": int(controls.get("yaw_span", 360)),
            "enable_rcp": bool(controls.get("enable_rcp", True)),
            "enable_tcr": bool(controls.get("enable_tcr", True)),
            "enable_turbo": controls.get("model_variant", "turbo") == "turbo",
            "model_dir": str(self.model_dir),
            "gvhmr_root": str(self.gvhmr_root),
            "gpu_ids": [0],
            "attention_backend": str(controls.get("attention_backend", "auto")),
            "target_fps": controls.get("target_fps", "auto"),
            "start_time": float(controls.get("start_time", 0.0)),
            "camera_motion": str(controls.get("camera_motion", "simple_vo")),
            "focal_length_mm": focal_length or None,
            "seed": int(controls.get("seed", 42)),
        }
        request_file = context.output_dir / "native-request.json"
        request_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        def progress(line: str) -> tuple[str, float] | None:
            match = PROGRESS.match(line)
            return (match.group(2), float(match.group(1))) if match else None

        context.update("Launching the native 4DAnyone pipeline", 0.01)
        context.run_process(
            [sys.executable, str(self.project_root / "local_app" / "run.py"), str(request_file)],
            cwd=self.project_root,
            env={"PYTHONPATH": str(self.project_root.parent)},
            progress_parser=progress,
        )
        return self._collect(context.output_dir, len(pitches), int(payload["views_per_layer"]))

    def _collect(self, root: Path, layers: int, per_layer: int) -> list[StudioOutput]:
        outputs: list[StudioOutput] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == "manifest.json":
                continue
            kind, media = _kind(path)
            relative = path.relative_to(root).as_posix()
            metadata: dict[str, Any] = {"relative": relative, "views": layers * per_layer}
            if "/dense/" in f"/{relative}":
                metadata.update({"role": "target-view", "synchronized": True})
            elif "/sparse/" in f"/{relative}":
                metadata.update({"role": "proposal-view", "synchronized": True})
            elif "/skeletons/" in f"/{relative}":
                metadata.update({"role": "pose-conditioning", "synchronized": True})
            outputs.append(StudioOutput(path, kind, relative, media, metadata))
        return outputs
