from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

from PIL import Image, ImageDraw, ImageOps

from studio.runtime import StudioAdapter, StudioContext, StudioOutput, gpu_snapshot


ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
MAX_ARCHIVE_FILES = 250_000
MAX_EXPANDED_BYTES = 96 * 1024**3
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _archive_suffix(path: Path) -> str:
    name = path.name.lower()
    return next((suffix for suffix in ARCHIVE_SUFFIXES if name.endswith(suffix)), "")


def _safe_target(root: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"Unsafe archive path: {member_name!r}")
    target = (root / Path(*pure.parts)).resolve()
    resolved_root = root.resolve()
    if target != resolved_root and resolved_root not in target.parents:
        raise ValueError(f"Archive path escapes the work directory: {member_name!r}")
    return target


def _extract_archive(source: Path, destination: Path, context: StudioContext) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    suffix = _archive_suffix(source)
    expanded = 0
    count = 0
    if suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                context.check_cancelled()
                count += 1
                if count > MAX_ARCHIVE_FILES:
                    raise ValueError(f"Archive exceeds the {MAX_ARCHIVE_FILES:,}-file safety ceiling.")
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError(f"Archive symlinks are not accepted: {member.filename!r}")
                target = _safe_target(destination, member.filename)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                expanded += int(member.file_size)
                if expanded > MAX_EXPANDED_BYTES:
                    raise ValueError("Expanded scene package exceeds the 96 GiB safety ceiling.")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as incoming, target.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=8 * 1024**2)
    else:
        mode = "r:gz" if suffix in {".tar.gz", ".tgz"} else "r:"
        with tarfile.open(source, mode) as archive:
            for member in archive:
                context.check_cancelled()
                count += 1
                if count > MAX_ARCHIVE_FILES:
                    raise ValueError(f"Archive exceeds the {MAX_ARCHIVE_FILES:,}-file safety ceiling.")
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise ValueError(f"Archive links/devices are not accepted: {member.name!r}")
                target = _safe_target(destination, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    continue
                expanded += int(member.size)
                if expanded > MAX_EXPANDED_BYTES:
                    raise ValueError("Expanded scene package exceeds the 96 GiB safety ceiling.")
                incoming = archive.extractfile(member)
                if incoming is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with incoming, target.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=8 * 1024**2)
    context.log(f"Safely expanded {count:,} members ({expanded / 1024**3:.2f} GiB).")


def _scene_roots(root: Path) -> list[Path]:
    scenes: list[Path] = []
    for transforms in root.rglob("transforms.json"):
        relative = transforms.relative_to(root)
        if len(relative.parts) > 9 or any(part.startswith("_") for part in relative.parts[:-1]):
            continue
        try:
            meta = json.loads(transforms.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(meta, dict) and isinstance(meta.get("frames"), list) and meta.get("instances") is not None:
            scenes.append(transforms.parent)
    return sorted(set(scenes), key=lambda value: value.as_posix().lower())


def _slug(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-.")
    return clean[:80] or "scene"


def _scene_summary(scene: Path, package_root: Path) -> dict[str, Any]:
    meta = json.loads((scene / "transforms.json").read_text(encoding="utf-8"))
    frames = meta.get("frames") or []
    instances = meta.get("instances") or []
    instance_names = list(instances) if isinstance(instances, dict) else [str(value) for value in instances]
    frame_paths: list[Path] = []
    missing_frames: list[str] = []
    for frame in frames:
        raw = str(frame.get("file_path", ""))
        if not raw:
            continue
        path = (scene / raw).resolve()
        if path.is_file():
            frame_paths.append(path)
        else:
            missing_frames.append(raw)
    mask_root = scene / "masks"
    masks = sum(1 for path in mask_root.rglob("*") if path.is_file()) if mask_root.is_dir() else 0
    return {
        "scene": scene.relative_to(package_root).as_posix() or ".",
        "frames": len(frames),
        "available_frames": len(frame_paths),
        "missing_frames": missing_frames[:50],
        "instances": len(instance_names),
        "instance_names": instance_names,
        "masks": masks,
        "resolution": [meta.get("w"), meta.get("h")],
        "intrinsics": {key: meta.get(key) for key in ("fl_x", "fl_y", "cx", "cy")},
        "frame_paths": frame_paths,
    }


class Adapter(StudioAdapter):
    def __init__(self, project_root: Path, runtime_root: Path) -> None:
        super().__init__(project_root, runtime_root)
        self.sculpt_root = project_root.parent
        self.python = self.sculpt_root / ".venv" / "bin" / "python"
        self.pixal = project_root / "pretrained" / "Pixal3D"
        self.ss_stage = project_root / "pretrained" / "ss_ft64_mv_lora_ibr_texverse"
        self.shape_stage = project_root / "pretrained" / "shape_ft1024_mv_lora_ibr_texverse_fixedmem05"
        self.deps = self.sculpt_root / "pretrained" / "deps"
        self.vendor = self.sculpt_root / ".runtime" / "vendor"
        self._localize_pipeline()

    def _checks(self) -> list[tuple[str, Path]]:
        return [
            ("Shared sculpting Python", self.python),
            ("Pixal3D pipeline", self.pixal / "pipeline.json"),
            ("WorldSculpt sparse-stage config", self.ss_stage / "config.json"),
            ("WorldSculpt sparse-stage denoiser", self.ss_stage / "ckpts" / "denoiser_step0015000.pt"),
            ("WorldSculpt sparse-stage aggregator", self.ss_stage / "ckpts" / "mv_aggregator_step0015000.pt"),
            ("WorldSculpt shape-stage config", self.shape_stage / "config.json"),
            ("WorldSculpt shape-stage denoiser", self.shape_stage / "ckpts" / "denoiser_step0015000.pt"),
            ("WorldSculpt shape-stage aggregator", self.shape_stage / "ckpts" / "mv_aggregator_step0015000.pt"),
            ("DINOv3", self.deps / "camenduru--dinov3-vitl16-pretrain-lvd1689m" / "model.safetensors"),
            ("BiRefNet", self.deps / "ZhengPeng7--BiRefNet" / "model.safetensors"),
            ("MoGe-2", self.deps / "Ruicheng--moge-2-vitl" / "model.pt"),
            ("NAF weights", self.deps / "valeoai--NAF" / "naf_release.pth"),
            ("Pinned NAF source", self.vendor / "NAF" / "hubconf.py"),
            ("Pinned MoGe source", self.vendor / "MoGe" / "moge" / "model" / "v2.py"),
        ]

    def health(self) -> dict[str, Any]:
        details = [{"label": label, "ready": path.is_file(), "path": str(path)} for label, path in self._checks()]
        gpu = gpu_snapshot()
        device = gpu.get("devices", [{}])[0] if gpu.get("available") else {}
        gpu_ok = bool(device) and int(device.get("memory_total_mib", 0)) >= 30000
        details.append({"label": "CUDA GPU >= 30,000 MiB", "ready": gpu_ok, "value": device.get("name", "unavailable")})
        return {
            "ready": all(path.is_file() for _, path in self._checks()) and gpu_ok,
            "loaded": False,
            "profile": "subprocess-resident per scene",
            "details": details,
        }

    def validate(self, request: dict[str, Any], resolve_asset: Callable[[str], Path]) -> None:
        mode = str(request.get("mode", ""))
        if mode not in {"inspect_package", "reconstruct_scene"}:
            raise ValueError(f"Unknown WorldSculpt workflow: {mode}")
        controls = request.get("controls") or {}
        source = resolve_asset(str(controls.get("scene_package_asset", "")))
        if not _archive_suffix(source):
            raise ValueError("Upload a .zip, .tar, .tar.gz, or .tgz WorldSculpt scene package.")
        if mode == "reconstruct_scene":
            if int(controls.get("resolution", 1024)) not in {512, 1024}:
                raise ValueError("WorldSculpt reconstruction supports 512 or 1024 pixel crops.")
            tokens = int(controls.get("max_num_tokens", 49152))
            if not 4096 <= tokens <= 65536:
                raise ValueError("Sparse-token capacity must be between 4,096 and 65,536.")
            max_views = int(controls.get("max_views", 20))
            if not 0 <= max_views <= 128:
                raise ValueError("View cap must be between 0 (unlimited) and 128.")

    def load(self) -> dict[str, Any]:
        return self.health()

    def unload(self) -> dict[str, Any]:
        return self.health()

    def run(self, request: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        controls = request.get("controls") or {}
        source = context.asset(str(controls["scene_package_asset"]))
        work = self.runtime_root / "work" / context.job_id
        package_root = work / "package"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=False)
        try:
            context.update("Verifying and expanding the scene package", 0.02)
            _extract_archive(source, package_root, context)
            scenes = _scene_roots(package_root)
            if not scenes:
                raise ValueError(
                    "No scene root was found. A scene needs transforms.json with frames and instances plus its frame/mask files."
                )
            selector = str(controls.get("scene_path", "")).strip().replace("\\", "/").strip("/")
            if selector:
                scenes = [scene for scene in scenes if scene.relative_to(package_root).as_posix() == selector]
                if not scenes:
                    raise ValueError(f"Scene path {selector!r} was not found in this package.")
            if request["mode"] == "inspect_package":
                return self._inspect(scenes, package_root, context)
            if not bool(controls.get("run_all_scenes", False)):
                scenes = scenes[:1]
            return self._reconstruct(scenes, package_root, controls, work, context)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _inspect(self, scenes: list[Path], package_root: Path, context: StudioContext) -> list[StudioOutput]:
        summaries = [_scene_summary(scene, package_root) for scene in scenes]
        serializable = [{key: value for key, value in summary.items() if key != "frame_paths"} for summary in summaries]
        report = {
            "format": "mm-tools.worldsculpt.package-audit.v1",
            "scene_count": len(summaries),
            "scenes": serializable,
        }
        json_path = context.output_dir / "scene-package-audit.json"
        json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        markdown = ["# WorldSculpt scene package", "", f"Detected **{len(summaries)}** scene(s).", ""]
        for summary in serializable:
            markdown.extend([
                f"## `{summary['scene']}`",
                "",
                f"- Frames: {summary['available_frames']} available / {summary['frames']} declared",
                f"- Instances: {summary['instances']} ({', '.join(summary['instance_names'][:20]) or 'none'})",
                f"- Masks: {summary['masks']}",
                f"- Resolution: {summary['resolution']}",
                f"- Missing frame references: {len(summary['missing_frames'])}",
                "",
            ])
        markdown_path = context.output_dir / "scene-package-audit.md"
        markdown_path.write_text("\n".join(markdown), encoding="utf-8")
        outputs = [
            StudioOutput(markdown_path, "text", "Human-readable package audit", "text/markdown"),
            StudioOutput(json_path, "data", "Machine-readable package audit", "application/json"),
        ]
        contact = self._contact_sheet(summaries, context.output_dir / "scene-contact-sheet.jpg")
        if contact:
            outputs.insert(0, StudioOutput(contact, "image", "Scene contact sheet", "image/jpeg"))
        context.update("Package inspection complete", 0.99)
        return outputs

    @staticmethod
    def _contact_sheet(summaries: list[dict[str, Any]], destination: Path) -> Path | None:
        cards: list[tuple[Image.Image, str]] = []
        for summary in summaries[:12]:
            paths: list[Path] = summary["frame_paths"]
            if not paths:
                continue
            try:
                image = Image.open(paths[len(paths) // 2]).convert("RGB")
            except OSError:
                continue
            cards.append((ImageOps.fit(image, (480, 300), method=Image.Resampling.LANCZOS), str(summary["scene"])))
        if not cards:
            return None
        width = 960
        rows = (len(cards) + 1) // 2
        sheet = Image.new("RGB", (width, rows * 350), "#0b1017")
        draw = ImageDraw.Draw(sheet)
        for index, (image, label) in enumerate(cards):
            x = (index % 2) * 480
            y = (index // 2) * 350
            sheet.paste(image, (x, y))
            draw.rectangle((x, y + 300, x + 480, y + 350), fill="#15202b")
            draw.text((x + 16, y + 316), label, fill="#e7f5ef")
        sheet.save(destination, quality=92, optimize=True)
        return destination

    def _reconstruct(
        self,
        scenes: list[Path],
        package_root: Path,
        controls: dict[str, Any],
        work: Path,
        context: StudioContext,
    ) -> list[StudioOutput]:
        outputs: list[StudioOutput] = []
        summaries: list[dict[str, Any]] = []
        env = self._environment(work)
        count = len(scenes)
        for index, scene in enumerate(scenes):
            context.check_cancelled()
            relative = scene.relative_to(package_root).as_posix()
            label = _slug(relative if relative != "." else scene.name)
            case_root = work / "cases" / f"{index + 1:03d}-{label}"
            scene_output = case_root / "_scene"
            start = index / count
            span = 1 / count
            context.update(f"Scene {index + 1}/{count}: preparing calibrated crops", 0.04 + start * 0.90)
            self._prepare(scene, case_root, controls, env, context)
            context.update(f"Scene {index + 1}/{count}: reconstructing every selected object", 0.14 + start * 0.90)
            self._reconstruct_batch(case_root, controls, env, context)
            context.update(f"Scene {index + 1}/{count}: composing metric scene", 0.65 + start * 0.90)
            self._compose(case_root, scene_output, controls, env, context)
            if bool(controls.get("pointcloud_diagnostics", True)):
                context.update(f"Scene {index + 1}/{count}: plotting spatial diagnostics", 0.83 + start * 0.90)
                self._visualize(case_root, scene_output, controls, env, context)
            scene_outputs = self._collect_outputs(label, relative, case_root, scene_output, controls, context)
            outputs.extend(scene_outputs)
            summaries.append({
                "scene": relative,
                "case": label,
                "outputs": [Path(item.path).name for item in scene_outputs],
                "profile": {
                    "resolution": int(controls.get("resolution", 1024)),
                    "views": controls.get("views", "all"),
                    "max_views": int(controls.get("max_views", 20)),
                    "view_select": controls.get("view_select", "area"),
                    "sampler": controls.get("sampler", "train"),
                    "seed": int(controls.get("seed", 42)),
                    "geometry_only": True,
                },
            })
            context.update(f"Scene {index + 1}/{count} complete", min(0.94, 0.04 + (start + span) * 0.90))
        report = context.output_dir / "worldsculpt-run.json"
        report.write_text(json.dumps({"format": "mm-tools.worldsculpt.run.v1", "scenes": summaries}, indent=2) + "\n", encoding="utf-8")
        outputs.append(StudioOutput(report, "data", "WorldSculpt run manifest", "application/json"))
        context.update("WorldSculpt delivery ready", 0.99)
        return outputs

    def _prepare(self, scene: Path, case_root: Path, c: dict[str, Any], env: dict[str, str], context: StudioContext) -> None:
        command = [
            str(self.python), str(self.project_root / "prepare_crops_scene.py"),
            "--scene_dir", str(scene), "--case_root", str(case_root),
            "--pad", str(float(c.get("crop_pad", 0.005))),
            "--min_mask_pixels", str(int(c.get("min_mask_pixels", 500))),
            "--min_mask_ratio", str(float(c.get("min_mask_ratio", 0.0))),
            "--crop_resolution", str(int(c.get("crop_resolution", 1024))),
            "--max_crop_ratio", str(float(c.get("max_crop_ratio", 4.0))),
            "--alpha_erode_kernel", str(int(c.get("alpha_erode_kernel", 3))),
            "--alpha_erode_iters", str(int(c.get("alpha_erode_iters", 3))),
            "--mask_fit_pct", str(float(c.get("mask_fit_pct", 95.0))),
            "--anchor_frame", str(c.get("anchor_frame", "anchor")),
        ]
        if bool(c.get("save_alignments", True)):
            command.append("--save_alignments")
        if bool(c.get("mask_fit_scale", True)):
            command.append("--mask_fit_scale")
        context.run_process(command, cwd=self.project_root, env=env)

    def _reconstruct_batch(self, case_root: Path, c: dict[str, Any], env: dict[str, str], context: StudioContext) -> None:
        command = [
            str(self.python), str(self.project_root / "reconstruct_batch.py"),
            "--case_root", str(case_root), "--recon_subdir", "_recon",
            "--views", str(c.get("views", "all")),
            "--max_views", str(int(c.get("max_views", 20))),
            "--view_select", str(c.get("view_select", "area")),
            "--anchor", str(int(c.get("anchor", -1))),
            "--resolution", str(int(c.get("resolution", 1024))),
            "--seed", str(int(c.get("seed", 42))),
            "--sampler", str(c.get("sampler", "train")),
            "--max_num_tokens", str(int(c.get("max_num_tokens", 49152))),
            "--glb_faces", str(int(c.get("per_object_glb_faces", 100000))),
            "--ss_config", str(self.ss_stage / "config.json"),
            "--ss_ckpt_dir", str(self.ss_stage / "ckpts"), "--ss_step", "15000",
            "--shape_config", str(self.shape_stage / "config.json"),
            "--shape_ckpt_dir", str(self.shape_stage / "ckpts"), "--shape_step", "15000",
            "--no-ema", "--no_tex", "--no_glb",
        ]
        instances = str(c.get("instances", "")).replace(" ", "")
        if instances:
            command.extend(["--instances", instances])
        if bool(c.get("overwrite", False)):
            command.append("--overwrite")
        if bool(c.get("visualize_sparse_structure", True)):
            command.append("--vis_ss")
        context.run_process(command, cwd=self.project_root, env=env, progress_parser=self._batch_progress)

    @staticmethod
    def _batch_progress(line: str) -> tuple[str, float] | None:
        match = re.search(r"\[batch\].*?\[(\d+)/(\d+)\]", line)
        if not match:
            return None
        current, total = map(int, match.groups())
        return f"Reconstructing object {current}/{total}", 0.14 + 0.48 * min(1.0, current / max(total, 1))

    def _compose(self, case_root: Path, output: Path, c: dict[str, Any], env: dict[str, str], context: StudioContext) -> None:
        command = [
            str(self.python), str(self.project_root / "compose_scene.py"),
            "--case_root", str(case_root), "--recon_dir", str(case_root / "_recon"),
            "--output_dir", str(output), "--normal",
            "--render_frames", str(c.get("render_frames", "all")),
            "--preview_long", str(int(c.get("preview_long", 1280))),
            "--ssaa", str(int(c.get("ssaa", 2))),
            "--peel_layers", str(int(c.get("peel_layers", 4))),
            "--face_budget", str(int(c.get("render_face_budget", 3000000))),
            "--glb_decimation", str(int(c.get("glb_faces_per_object", 1000000))),
            "--texture_size", str(int(c.get("texture_size", 4096))),
            "--video_fps", str(int(c.get("video_fps", 10))),
            "--video_crf", str(int(c.get("video_crf", 28))),
        ]
        instances = str(c.get("instances", "")).replace(" ", "")
        if instances:
            command.extend(["--instances", instances])
        if bool(c.get("full_resolution_renders", False)):
            command.append("--full_res")
        if bool(c.get("white_background", False)):
            command.append("--white_bg")
        if bool(c.get("raw_geometry", False)):
            command.append("--raw_geometry")
        if not bool(c.get("render_previews", True)):
            command.append("--no_render")
        if not bool(c.get("make_video", True)):
            command.append("--no_video")
        context.run_process(command, cwd=self.project_root, env=env)

    def _visualize(self, case_root: Path, output: Path, c: dict[str, Any], env: dict[str, str], context: StudioContext) -> None:
        command = [
            str(self.python), str(self.project_root / "visualize_pointcloud.py"),
            "--case_root", str(case_root), "--recon_dir", str(case_root / "_recon"),
            "--output_dir", str(output), "--max_pts", str(int(c.get("diagnostic_points", 8000))),
        ]
        instances = str(c.get("instances", "")).replace(" ", "")
        if instances:
            command.extend(["--instances", instances])
        context.run_process(command, cwd=self.project_root, env=env)

    def _collect_outputs(
        self,
        label: str,
        relative: str,
        case_root: Path,
        scene_output: Path,
        controls: dict[str, Any],
        context: StudioContext,
    ) -> list[StudioOutput]:
        outputs: list[StudioOutput] = []
        candidates = [
            (scene_output / "scene.glb", "model", "Interactive metric scene", "model/gltf-binary"),
            (scene_output / "scene_mesh.glb", "model", "Merged metric mesh", "model/gltf-binary"),
            (scene_output / "scene.mp4", "video", "Calibrated scene camera pass", "video/mp4"),
            (scene_output / "mesh_scene_3d.png", "image", "Mesh cloud · 3D", "image/png"),
            (scene_output / "mesh_scene_topdown.png", "image", "Mesh cloud · top-down", "image/png"),
            (scene_output / "cloud_scene_3d.png", "image", "Backprojection cloud · 3D", "image/png"),
            (scene_output / "cloud_scene_topdown.png", "image", "Backprojection cloud · top-down", "image/png"),
        ]
        for source, kind, title, media_type in candidates:
            if not source.is_file():
                continue
            destination = context.output_dir / f"{label}-{source.name}"
            shutil.move(str(source), destination)
            outputs.append(StudioOutput(destination, kind, f"{relative} · {title}", media_type, {"scene": relative}))
        render_dir = scene_output / "renders"
        if render_dir.is_dir() and any(render_dir.iterdir()):
            archive_base = context.output_dir / f"{label}-calibrated-renders"
            archive = Path(shutil.make_archive(str(archive_base), "zip", root_dir=render_dir))
            outputs.append(StudioOutput(archive, "archive", f"{relative} · calibrated render frames", "application/zip"))
        if bool(controls.get("bundle_reconstruction", False)):
            archive_base = context.output_dir / f"{label}-worldsculpt-project"
            archive = Path(shutil.make_archive(str(archive_base), "gztar", root_dir=case_root))
            outputs.append(StudioOutput(archive, "archive", f"{relative} · resumable WorldSculpt project", "application/gzip"))
        return outputs

    def _environment(self, work: Path) -> dict[str, str]:
        (work / "tmp").mkdir(parents=True, exist_ok=True)
        python_path = [
            str(self.sculpt_root.parent), str(self.sculpt_root), str(self.project_root),
            str(self.vendor / "NAF"), str(self.vendor / "MoGe"),
        ]
        existing = os.environ.get("PYTHONPATH")
        if existing:
            python_path.append(existing)
        return {
            "PYTHONPATH": os.pathsep.join(python_path),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DO_NOT_TRACK": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "ATTN_BACKEND": "sdpa",
            "SPARSE_ATTN_BACKEND": "sdpa",
            "SPARSE_CONV_BACKEND": "flex_gemm",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PIXAL_PIPELINE_CONFIG": "pipeline.mmtools.json",
            "PIXAL_NAF_SOURCE": str(self.vendor / "NAF"),
            "PIXAL_NAF_CHECKPOINT": str(self.deps / "valeoai--NAF" / "naf_release.pth"),
            "FLEX_GEMM_AUTOTUNE_CACHE_PATH": str(self.runtime_root / "flex_gemm_autotune.json"),
            "TMPDIR": str(work / "tmp"),
        }

    def _localize_pipeline(self) -> None:
        source = self.pixal / "pipeline.json"
        if not source.is_file():
            return
        config = json.loads(source.read_text(encoding="utf-8"))
        args = config.get("args", {})
        image_cond = args.get("image_cond_model")
        if isinstance(image_cond, dict):
            image_cond.setdefault("args", {})["model_name"] = str(
                self.deps / "camenduru--dinov3-vitl16-pretrain-lvd1689m"
            )
        rembg = args.get("rembg_model")
        if isinstance(rembg, dict):
            rembg.setdefault("args", {})["model_name"] = str(self.deps / "ZhengPeng7--BiRefNet")
        args["low_vram"] = False
        destination = self.pixal / "pipeline.mmtools.json"
        rendered = json.dumps(config, indent=2) + "\n"
        if not destination.is_file() or destination.read_text(encoding="utf-8") != rendered:
            destination.write_text(rendered, encoding="utf-8")
