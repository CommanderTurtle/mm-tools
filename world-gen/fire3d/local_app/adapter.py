from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

from studio.runtime import StudioAdapter, StudioContext, StudioOutput, gpu_snapshot

from local_app.preparation import (
    default_intrinsics,
    extract_bundle,
    image_dimensions,
    make_protocol,
    package_dataset,
    prepare_rgbd,
    stage_scene,
)


class Adapter(StudioAdapter):
    def __init__(self, project_root: Path, runtime_root: Path) -> None:
        super().__init__(project_root, runtime_root)
        self.models = project_root / "checkpoints" / "Fire3D"
        self.protocol = project_root / "configs" / "inference" / "fire3d_single_image_v1.json"

    def health(self) -> dict[str, Any]:
        gpu = gpu_snapshot()
        device = (gpu.get("devices") or [{}])[0]
        checks = [
            ("Perception checkpoint", self.models / "perception" / "model.pt"),
            ("SS reconstruction flow", self.models / "reconstruction" / "flows" / "ss" / "model.pt"),
            ("Shape reconstruction flow", self.models / "reconstruction" / "flows" / "shape" / "model.pt"),
            ("PBR reconstruction flow", self.models / "reconstruction" / "flows" / "pbr" / "model.pt"),
            ("DINOv3 runtime source", self.project_root / "third_party" / "dinov3" / "dinov3" / "__init__.py"),
            ("Python environment", self.project_root / ".venv" / "bin" / "python"),
        ]
        details = [{"label": label, "ready": path.is_file(), "path": str(path)} for label, path in checks]
        gpu_ok = bool(device) and int(device.get("memory_total_mib", 0)) >= 30000
        details.append(
            {
                "label": "Single CUDA GPU ≥ 30 GiB",
                "ready": gpu_ok,
                "value": device.get("name", "unavailable"),
            }
        )
        details.append(
            {
                "label": "32 GiB staged profile",
                "ready": True,
                "value": "BF16 flows · object batches 1 · no CPU offload",
            }
        )
        return {"ready": all(path.is_file() for _, path in checks) and gpu_ok, "loaded": False, "details": details}

    def validate(self, request: dict[str, Any], resolve_asset: Callable[[str], Path]) -> None:
        mode = str(request.get("mode", ""))
        controls = request.get("controls") or {}
        if mode not in {"reconstruct_files", "reconstruct_bundle", "reconstruct_release", "prepare_rgbd", "inspect_glb"}:
            raise ValueError(f"Unknown Fire3D mode: {mode}")
        if mode == "reconstruct_files":
            rgb = resolve_asset(str(controls.get("rgb_asset", "")))
            pcd = resolve_asset(str(controls.get("pointcloud_asset", "")))
            if rgb.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"} or pcd.suffix.lower() != ".ply":
                raise ValueError("Direct reconstruction requires an RGB image and aligned PLY point cloud.")
        elif mode == "reconstruct_bundle":
            bundle = resolve_asset(str(controls.get("bundle_asset", "")))
            if bundle.suffix.lower() != ".zip":
                raise ValueError("Prepared Fire3D inputs must be a ZIP bundle.")
        elif mode == "reconstruct_release":
            root = Path(str(controls.get("data_root", ""))).expanduser()
            if not root.is_dir():
                raise ValueError(f"Dataset root does not exist: {root}")
            if str(controls.get("dataset", "single_image")) not in {"single_image", "scannetpp", "ithor", "imaginarium"}:
                raise ValueError("Unsupported Fire3D release dataset.")
        elif mode == "prepare_rgbd":
            resolve_asset(str(controls.get("rgb_asset", "")))
            depth = resolve_asset(str(controls.get("depth_asset", "")))
            if depth.suffix.lower() not in {".npy", ".png", ".tif", ".tiff"}:
                raise ValueError("Depth must be a 2-D NPY, PNG, or TIFF file.")
        elif mode == "inspect_glb":
            model = resolve_asset(str(controls.get("glb_asset", "")))
            if model.suffix.lower() not in {".glb", ".gltf"}:
                raise ValueError("The scene inspector accepts GLB or glTF files.")

    def _stage_request(self, mode: str, controls: dict[str, Any], context: StudioContext) -> tuple[Path, str, Path]:
        scene_id = str(controls.get("scene_id", "local-scene"))
        data_root = context.output_dir / "input"
        if mode == "reconstruct_bundle":
            expanded = context.output_dir / "expanded-bundle"
            rgb, pcd, camera = extract_bundle(context.asset(str(controls["bundle_asset"])), expanded)
            metadata = stage_scene(rgb, pcd, data_root, scene_id, camera=camera)
        else:
            rgb = context.asset(str(controls["rgb_asset"]))
            pcd = context.asset(str(controls["pointcloud_asset"]))
            camera_asset = str(controls.get("camera_asset", "")).strip()
            camera = context.asset(camera_asset) if camera_asset else None
            width, height = image_dimensions(rgb)
            metadata = stage_scene(
                rgb,
                pcd,
                data_root,
                scene_id,
                camera=camera,
                intrinsics=default_intrinsics(controls, width, height),
            )
        receipt = context.output_dir / "input-metadata.json"
        receipt.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        package_dataset(data_root, context.output_dir / "fire3d-input.zip")
        return data_root, str(metadata["scene_id"]), receipt

    @staticmethod
    def _progress(line: str) -> tuple[str, float] | None:
        if "perception" in line.lower():
            return "Predicting scene objects", 0.2
        if line.startswith("[scene] predicting"):
            return "Generating object geometry", 0.5
        if "pbr" in line.lower() or "appearance" in line.lower():
            return "Baking PBR appearance", 0.72
        match = re.search(r"\[driver\]\s+\((\d+)/(\d+)\)", line)
        if match:
            return "Composing the reconstructed scene", 0.82 + 0.12 * int(match.group(1)) / max(int(match.group(2)), 1)
        return None

    def _collect(self, root: Path) -> list[StudioOutput]:
        outputs: list[StudioOutput] = []
        preferred = [
            ("predicted_textured_world_scene.glb", "World-space textured scene"),
            ("predicted_textured_training_scene.glb", "Normalized textured scene"),
        ]
        selected: set[Path] = set()
        for name, label in preferred:
            for path in root.rglob(name):
                outputs.append(StudioOutput(path, "model", label, "model/gltf-binary"))
                selected.add(path.resolve())
        for path in sorted(root.rglob("*.glb")):
            if path.resolve() not in selected:
                outputs.append(StudioOutput(path, "model", f"Object · {path.stem}", "model/gltf-binary"))
        for name, label in (
            ("appearance_summary.json", "Appearance report"),
            ("inference_summary.json", "Reconstruction report"),
            ("input_audit.json", "Input audit"),
            ("resolved_protocol.json", "Resolved native protocol"),
            ("run_summary.json", "Release run summary"),
        ):
            for path in root.rglob(name):
                outputs.append(StudioOutput(path, "document", label, "application/json"))
        if not outputs:
            raise RuntimeError("Fire3D exited without a GLB or reconstruction report.")
        return outputs

    def run(self, request: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        mode = str(request["mode"])
        controls = request.get("controls") or {}
        if mode == "inspect_glb":
            source = context.asset(str(controls["glb_asset"]))
            destination = context.output_dir / source.name
            shutil.copy2(source, destination)
            context.update("Scene ready in the native viewer", 0.99)
            return [StudioOutput(destination, "model", "Interactive scene", "model/gltf-binary")]

        if mode == "prepare_rgbd":
            rgb = context.asset(str(controls["rgb_asset"]))
            depth = context.asset(str(controls["depth_asset"]))
            width, height = image_dimensions(rgb)
            fx, fy, cx, cy = default_intrinsics(controls, width, height)
            context.update("Projecting metric depth into an organized point cloud", 0.25)
            metadata = prepare_rgbd(
                rgb,
                depth,
                context.output_dir,
                str(controls.get("scene_id", "local-scene")),
                depth_scale=float(controls.get("depth_scale", 0.001)),
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                minimum_depth=float(controls.get("minimum_depth", 0.05)),
                maximum_depth=float(controls.get("maximum_depth", 100.0)),
            )
            metadata_path = context.output_dir / "rgbd-metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            bundle = package_dataset(context.output_dir / "dataset", context.output_dir / "fire3d-input.zip")
            scene = context.output_dir / "dataset" / "single_image" / "data" / metadata["scene_id"]
            context.update("Reusable Fire3D input is ready", 0.99)
            return [
                StudioOutput(bundle, "file", "Prepared Fire3D input", "application/zip", metadata),
                StudioOutput(scene / "aligned_pcd.ply", "model", "Organized metric point cloud", "application/octet-stream", metadata),
                StudioOutput(scene / "camera.json", "document", "Native camera contract", "application/json"),
                StudioOutput(metadata_path, "document", "Preparation receipt", "application/json"),
            ]

        context.update("Resolving Fire3D input contract", 0.04)
        if mode in {"reconstruct_files", "reconstruct_bundle"}:
            data_root, scene_id, input_receipt = self._stage_request(mode, controls, context)
            dataset = "single_image"
        else:
            data_root = Path(str(controls["data_root"])).expanduser().resolve()
            scene_id = str(controls.get("scene_id", "")).strip()
            dataset = str(controls.get("dataset", "single_image"))
            input_receipt = None

        protocol = make_protocol(self.protocol, context.output_dir / "fire3d-5090-protocol.json", controls)
        result_root = context.output_dir / "reconstruction"
        command = [
            sys.executable,
            "-m",
            "fire3d",
            "infer",
            "--dataset",
            dataset,
            "--data-root",
            str(data_root),
            "--output-root",
            str(result_root),
            "--protocol",
            str(protocol),
            "--gpu",
            str(int(controls.get("gpu", 0))),
        ]
        if scene_id:
            command.extend(["--scene-id", scene_id])
        if not bool(controls.get("render_views", False)):
            command.append("--skip-render")
        if bool(controls.get("skip_existing", True)):
            command.append("--skip-existing")
        receipt = context.output_dir / "request.json"
        receipt.write_text(
            json.dumps(
                {
                    "mode": mode,
                    "dataset": dataset,
                    "scene_id": scene_id,
                    "data_root": str(data_root),
                    "command": command,
                    "environment": {"FF_FIRE3D_MODEL_DTYPE": "bfloat16"},
                    "controls": controls,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        context.update("Starting staged CUDA reconstruction", 0.08)
        context.run_process(
            command,
            cwd=self.project_root,
            env={
                "PYTHONPATH": f"{self.project_root.parent.parent}:{self.project_root}:{self.project_root / 'trellis2_x2'}",
                "FF_FIRE3D_MODEL_DTYPE": "bfloat16",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
            progress_parser=self._progress,
        )
        outputs = self._collect(result_root)
        outputs.extend(
            [
                StudioOutput(protocol, "document", "RTX 5090 protocol", "application/json"),
                StudioOutput(receipt, "document", "Reproducible request", "application/json"),
            ]
        )
        if input_receipt:
            outputs.extend(
                [
                    StudioOutput(context.output_dir / "fire3d-input.zip", "file", "Reusable Fire3D input", "application/zip"),
                    StudioOutput(input_receipt, "document", "Input contract", "application/json"),
                ]
            )
        return outputs
