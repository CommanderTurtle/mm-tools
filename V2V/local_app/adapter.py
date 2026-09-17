from __future__ import annotations

import gc
import json
import math
import shutil
from pathlib import Path
from typing import Any, Callable

from studio.comfy_runtime import ComfyRuntime
from studio.runtime import StudioAdapter, StudioContext, StudioOutput, gpu_snapshot


PIXEL_MODEL = "wan_2.1_idv2v_int8_convrot.safetensors"
GEOMETRY_MODEL = "wan_2.1_idv2v_with_normal_depth_int8_convrot.safetensors"
TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
CLIP_VISION = "clip_vision_h.safetensors"
VAE = "Wan2_1_VAE_bf16.safetensors"
NEGATIVE = (
    "oversaturated, overexposed, static, blurred details, subtitles, watermark, low quality, "
    "jpeg artifacts, deformed face, malformed hands, fused fingers, duplicate limbs, identity drift, "
    "frozen expression, temporal flicker, abrupt lighting changes, inconsistent background"
)


def _node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


def _save_video(video_ref: list[Any], prefix: str, container: str, codec: str, crf: int) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "video": video_ref,
        "filename_prefix": prefix,
        "format": container,
        "format.codec": codec,
    }
    if codec in {"h264", "av1"}:
        inputs["format.codec.encoding"] = "re-encode"
        inputs["format.codec.encoding.crf"] = crf
    return _node("SaveVideo", **inputs)


class Adapter(StudioAdapter):
    def __init__(self, project_root: Path, runtime_root: Path) -> None:
        super().__init__(project_root, runtime_root)
        self.repo_root = project_root.parent
        self.comfy = project_root / "ComfyUI"
        self.python = project_root / ".venv" / "bin" / "python"
        self.models = self.comfy / "models"
        self.birefnet = self.repo_root / "sculpting" / "pretrained" / "deps" / "ZhengPeng7--BiRefNet"

    def health(self) -> dict[str, Any]:
        required = [
            ("ID-V2V INT8", self.models / "diffusion_models" / PIXEL_MODEL),
            ("ID-V2V normal + depth INT8", self.models / "diffusion_models" / GEOMETRY_MODEL),
            ("UMT5 XXL FP8", self.models / "text_encoders" / TEXT_ENCODER),
            ("CLIP Vision H", self.models / "clip_vision" / CLIP_VISION),
            ("Wan VAE BF16", self.models / "vae" / VAE),
            ("Shared Python environment", self.python),
        ]
        details = [
            {"label": label, "ready": path.is_file(), "required": True, "path": str(path)}
            for label, path in required
        ]
        matte_ready = (self.birefnet / "model.safetensors").is_file()
        details.append({
            "label": "Local BiRefNet subject matte",
            "ready": matte_ready,
            "required": False,
            "path": str(self.birefnet),
            "note": "Required only for automatic foreground-on-gray preprocessing.",
        })
        gpu = gpu_snapshot()
        device = gpu.get("devices", [{}])[0] if gpu.get("available") else {}
        gpu_ok = bool(device) and int(device.get("memory_total_mib", 0)) >= 30000
        details.append({
            "label": "CUDA GPU >= 30 GiB", "ready": gpu_ok, "required": True,
            "value": device.get("name", "unavailable"),
        })
        return {
            "ready": all(path.is_file() for _, path in required) and gpu_ok,
            "loaded": False,
            "details": details,
        }

    def validate(self, request: dict[str, Any], resolve_asset: Callable[[str], Path]) -> None:
        mode = str(request.get("mode", ""))
        controls = request.get("controls") or {}
        if mode not in {"restyle", "relight", "normal_depth", "control_lab"}:
            raise ValueError(f"Unknown ID-V2V mode: {mode}")
        resolve_asset(str(controls.get("source_video_asset", "")))
        if mode != "control_lab":
            resolve_asset(str(controls.get("style_frame_asset", "")))
        strategy = "source" if mode == "relight" else str(controls.get("control_strategy", "auto_matte"))
        if strategy == "precomputed":
            resolve_asset(str(controls.get("pixel_control_asset", "")))
        if strategy == "auto_matte" and not (self.birefnet / "model.safetensors").is_file():
            raise ValueError("Automatic subject matting needs the shared sculpting BiRefNet model.")
        if mode == "normal_depth":
            resolve_asset(str(controls.get("normal_control_asset", "")))
            resolve_asset(str(controls.get("depth_control_asset", "")))
        width, height = int(controls.get("width", 832)), int(controls.get("height", 480))
        if width % 16 or height % 16 or not 256 <= width <= 1280 or not 256 <= height <= 1280:
            raise ValueError("Width and height must be multiples of 16 between 256 and 1280.")
        clip = int(controls.get("clip_frames", 81))
        if clip < 17 or clip > 161 or (clip - 1) % 4:
            raise ValueError("Frames per clip must be 4n+1, from 17 through 161.")
        maximum = int(controls.get("max_frames", 81))
        if maximum < 1 or maximum > 10000:
            raise ValueError("Maximum output frames must be from 1 through 10,000.")
        assets = list(controls.get("keyframe_assets") or [])
        indices = list(controls.get("keyframe_indices") or [])
        if len(assets) != len(indices):
            raise ValueError("Extra keyframe images and frame indices must have the same item count and order.")
        seen: set[int] = set()
        for asset_id, raw in zip(assets, indices):
            resolve_asset(str(asset_id))
            try:
                index = int(str(raw).strip())
            except ValueError as exc:
                raise ValueError(f"Keyframe index {raw!r} is not an integer.") from exc
            if index <= 0 or index % 4:
                raise ValueError("Extra keyframes must use positive 0-based frame indices divisible by four.")
            if index in seen:
                raise ValueError(f"Keyframe index {index} is duplicated.")
            seen.add(index)

    @staticmethod
    def _video_info(path: Path) -> tuple[int, float]:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path.name}")
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
        capture.release()
        if frames <= 0:
            raise ValueError(f"Video contains no decodable frames: {path.name}")
        return frames, fps

    @classmethod
    def _require_synchronized_control(cls, path: Path, *, frames: int, label: str) -> None:
        control_frames, _ = cls._video_info(path)
        if control_frames < frames:
            raise ValueError(
                f"{label} has {control_frames} frames, but this run needs {frames}. "
                "Provide a frame-aligned control video at least as long as the source selection."
            )

    def _foreground_control(
        self,
        source: Path,
        destination: Path,
        *,
        width: int,
        height: int,
        max_frames: int,
        threshold: float,
        feather: int,
        gray: int,
        context: StudioContext,
    ) -> Path:
        """Build the official foreground-on-gray signal with a local matte.

        ID-V2V's native release uses gated SAM3. BiRefNet is already an
        allowlisted, local-only dependency in this monorepo and produces the
        same control contract without a token or hidden network fetch.
        """
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        destination.parent.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Could not decode {source.name}")
        source_frames = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        count = min(source_frames, max_frames)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
        writer = cv2.VideoWriter(
            str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("OpenCV could not initialize the local matte video encoder.")
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        model = AutoModelForImageSegmentation.from_pretrained(
            str(self.birefnet), trust_remote_code=True, local_files_only=True
        ).eval().to("cuda")
        try:
            for index in range(count):
                context.check_cancelled()
                ok, frame = capture.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to("cuda")
                with torch.inference_mode():
                    mask = model(tensor)[-1].sigmoid()[0, 0].float().cpu().numpy()
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
                denominator = max(0.01, 1.0 - threshold)
                mask = np.clip((mask - threshold) / denominator, 0.0, 1.0)
                if feather > 0:
                    kernel = max(3, feather * 2 + 1)
                    mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
                alpha = mask[:, :, None]
                gray_frame = np.full_like(frame, max(0, min(gray, 255)))
                composite = (frame.astype(np.float32) * alpha + gray_frame * (1.0 - alpha)).astype(np.uint8)
                writer.write(composite)
                if index % 4 == 0 or index + 1 == count:
                    context.update(
                        f"Extracting the subject matte · frame {index + 1}/{count}",
                        0.03 + 0.13 * ((index + 1) / count),
                    )
        finally:
            capture.release()
            writer.release()
            del model
            gc.collect()
            torch.cuda.empty_cache()
        return destination

    @staticmethod
    def _round_clip_length(value: int) -> int:
        return max(17, int(math.ceil((value - 1) / 4.0) * 4 + 1))

    @staticmethod
    def _segments(total: int, clip_frames: int, keyframes: set[int]) -> list[tuple[int, int]]:
        starts = [0]
        cursor = 0
        stride = clip_frames - 1
        while cursor < total - 1:
            boundary = min(cursor + stride, total - 1)
            anchors = [index for index in keyframes if cursor < index <= boundary]
            if anchors:
                next_start = min(anchors)
            elif boundary >= total - 1:
                break
            else:
                next_start = boundary
            if next_start <= cursor:
                break
            starts.append(next_start)
            cursor = next_start
        result: list[tuple[int, int]] = []
        for position, start in enumerate(starts):
            if start >= total:
                continue
            next_start = starts[position + 1] if position + 1 < len(starts) else None
            requested = (next_start - start + 1) if next_start is not None else min(clip_frames, total - start)
            result.append((start, Adapter._round_clip_length(requested)))
        return result

    def _graph(
        self,
        *,
        control_name: str,
        normal_name: str | None,
        depth_name: str | None,
        style_name: str,
        model_name: str,
        offset: int,
        length: int,
        width: int,
        height: int,
        fps: float,
        controls: dict[str, Any],
        prefix: str,
        seed: int,
    ) -> dict[str, Any]:
        graph: dict[str, Any] = {}

        def add(class_type: str, **inputs: Any) -> str:
            identifier = str(len(graph) + 1)
            graph[identifier] = _node(class_type, **inputs)
            return identifier

        style_load = add("LoadImage", image=style_name)
        style = add(
            "ResizeImageMaskNode", input=[style_load, 0], resize_type="scale dimensions",
            **{
                "resize_type.width": width, "resize_type.height": height,
                "resize_type.crop": str(controls.get("crop", "center")),
                "scale_method": str(controls.get("scale_method", "lanczos")),
            },
        )
        control_load = add("LoadVideo", file=control_name)
        control_parts = add("GetVideoComponents", video=[control_load, 0])
        control_slice = add("ImageFromBatch", image=[control_parts, 0], batch_index=offset, length=length)
        control = add(
            "ResizeImageMaskNode", input=[control_slice, 0], resize_type="scale dimensions",
            **{
                "resize_type.width": width, "resize_type.height": height,
                "resize_type.crop": str(controls.get("crop", "center")),
                "scale_method": str(controls.get("scale_method", "lanczos")),
            },
        )
        model = add("UNETLoader", unet_name=model_name, weight_dtype="default")
        shifted = add("ModelSamplingSD3", model=[model, 0], shift=float(controls.get("shift", 5)))
        clip = add("CLIPLoader", clip_name=TEXT_ENCODER, type="wan", device="default")
        positive = add("CLIPTextEncode", clip=[clip, 0], text=str(controls.get("prompt", "")))
        negative = add("CLIPTextEncode", clip=[clip, 0], text=str(controls.get("negative_prompt", NEGATIVE)))
        clip_vision = add("CLIPVisionLoader", clip_name=CLIP_VISION)
        style_vision = add("CLIPVisionEncode", clip_vision=[clip_vision, 0], image=[style, 0], crop="none")
        vae = add("VAELoader", vae_name=VAE)
        image_condition = add(
            "WanImageToVideo", positive=[positive, 0], negative=[negative, 0], vae=[vae, 0],
            width=width, height=height, length=length, batch_size=1,
            clip_vision_output=[style_vision, 0], start_image=[style, 0], ref_pad_image=[style, 0],
        )
        conditioned = add(
            "WanVaceToVideo", positive=[image_condition, 0], negative=[image_condition, 1], vae=[vae, 0],
            width=width, height=height, length=length, batch_size=1,
            strength=float(controls.get("pixel_strength", 1.0)), control_video=[control, 0],
        )
        if normal_name and depth_name:
            for name, strength_key in ((normal_name, "normal_strength"), (depth_name, "depth_strength")):
                loader = add("LoadVideo", file=name)
                parts = add("GetVideoComponents", video=[loader, 0])
                sliced = add("ImageFromBatch", image=[parts, 0], batch_index=offset, length=length)
                resized = add(
                    "ResizeImageMaskNode", input=[sliced, 0], resize_type="scale dimensions",
                    **{
                        "resize_type.width": width, "resize_type.height": height,
                        "resize_type.crop": str(controls.get("crop", "center")),
                        "scale_method": str(controls.get("scale_method", "lanczos")),
                    },
                )
                conditioned = add(
                    "WanVaceToVideo", positive=[conditioned, 0], negative=[conditioned, 1], vae=[vae, 0],
                    width=width, height=height, length=length, batch_size=1,
                    strength=float(controls.get(strength_key, 1.0)), control_video=[resized, 0],
                )
        steps = int(controls.get("steps", 24))
        scheduler = add(
            "BasicScheduler", model=[shifted, 0], scheduler=str(controls.get("scheduler", "simple")),
            steps=steps, denoise=float(controls.get("denoise", 1.0)),
        )
        sampler = add("KSamplerSelect", sampler_name=str(controls.get("sampler", "er_sde")))
        sample = add(
            "SamplerCustom", model=[shifted, 0], positive=[conditioned, 0], negative=[conditioned, 1],
            sampler=[sampler, 0], sigmas=[scheduler, 0], latent_image=[conditioned, 2],
            add_noise=bool(controls.get("add_noise", True)), noise_seed=seed,
            cfg=float(controls.get("guidance", 1.0)),
        )
        decoded = add("VAEDecode", samples=[sample, 0], vae=[vae, 0])
        video = add(
            "CreateVideo", images=[decoded, 0], fps=fps,
            bit_depth=int(controls.get("bit_depth", 8)),
            color_space=str(controls.get("color_space", "sRGB")), codec="none",
        )
        graph[str(len(graph) + 1)] = _save_video(
            [video, 0], prefix, "mp4", "h264", int(controls.get("crf", 18))
        )
        return graph

    @staticmethod
    def _copy_video(source: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    @staticmethod
    def _codec(controls: dict[str, Any]) -> list[str]:
        if str(controls.get("codec", "h264")) == "av1":
            return ["-c:v", "libsvtav1", "-preset", "8"]
        return ["-c:v", "libx264", "-preset", "medium"]

    def _stitch(
        self,
        clips: list[Path],
        destination: Path,
        *,
        segments: list[tuple[int, int]],
        total_frames: int,
        controls: dict[str, Any],
        context: StudioContext,
    ) -> Path:
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        for clip in clips:
            command.extend(["-i", str(clip)])
        filters: list[str] = []
        streams: list[str] = []
        for index in range(len(clips)):
            # Each clip overlaps the next by one frame. Keep the next clip's
            # first frame (it may be an explicit user keyframe) and discard the
            # duplicate final frame from the preceding clip.
            trim = "" if index + 1 == len(clips) else f"trim=end_frame={segments[index][1] - 1},"
            filters.append(f"[{index}:v]{trim}setpts=PTS-STARTPTS[v{index}]")
            streams.append(f"[v{index}]")
        filters.append(f"{''.join(streams)}concat=n={len(clips)}:v=1:a=0[outv]")
        command.extend([
            "-filter_complex", ";".join(filters), "-map", "[outv]", "-frames:v", str(total_frames),
            *self._codec(controls), "-crf", str(int(controls.get("crf", 18))),
            "-pix_fmt", "yuv420p", str(destination),
        ])
        context.update("Stitching overlap-safe ID-V2V clips", 0.94)
        context.run_process(command)
        return destination

    def _derived_videos(
        self,
        generated: Path,
        source: Path,
        *,
        width: int,
        height: int,
        total_frames: int,
        controls: dict[str, Any],
        context: StudioContext,
    ) -> tuple[Path, Path | None, Path | None]:
        original = context.output_dir / "original_video.mp4"
        context.run_process([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}",
            "-frames:v", str(total_frames), "-an", "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p", str(original),
        ])
        comparison: Path | None = None
        if bool(controls.get("save_side_by_side", True)):
            comparison = context.output_dir / "generated_video_with_source.mp4"
            context.run_process([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(generated), "-i", str(original),
                "-filter_complex", "[0:v][1:v]hstack=inputs=2[outv]", "-map", "[outv]",
                "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(comparison),
            ])
        flip: Path | None = None
        if bool(controls.get("save_flip_test", True)):
            flip = context.output_dir / "flip_test.mp4"
            interval = max(0.5, float(controls.get("flip_interval", 2.0)))
            expression = f"if(lt(mod(T\\,{interval * 2:g})\\,{interval:g})\\,A\\,B)"
            context.run_process([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(generated), "-i", str(original),
                "-filter_complex", f"[0:v][1:v]blend=all_expr={expression}[outv]", "-map", "[outv]",
                "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(flip),
            ])
        return original, comparison, flip

    @staticmethod
    def _output(path: Path, label: str, metadata: dict[str, Any] | None = None) -> StudioOutput:
        media = {".mp4": "video/mp4", ".mkv": "video/x-matroska", ".webm": "video/webm"}.get(path.suffix.lower(), "video/mp4")
        return StudioOutput(path, "video", label, media, metadata or {})

    def run(self, request: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        mode = str(request["mode"])
        controls = request.get("controls") or {}
        source = context.asset(str(controls["source_video_asset"]))
        width, height = int(controls.get("width", 832)), int(controls.get("height", 480))
        source_frames, source_fps = self._video_info(source)
        total_frames = min(source_frames, int(controls.get("max_frames", source_frames)))
        if total_frames <= 0:
            raise ValueError("The selected source range contains no frames.")
        output_fps = float(controls.get("fps", source_fps))
        strategy = "source" if mode == "relight" else str(controls.get("control_strategy", "auto_matte"))

        if strategy == "auto_matte":
            pixel_control = self._foreground_control(
                source, context.output_dir / "condition_foreground_gray.mp4",
                width=width, height=height, max_frames=total_frames,
                threshold=float(controls.get("matte_threshold", 0.35)),
                feather=int(controls.get("matte_feather", 5)),
                gray=int(controls.get("background_gray", 127)), context=context,
            )
        elif strategy == "precomputed":
            pixel_control = context.asset(str(controls["pixel_control_asset"]))
        else:
            pixel_control = source

        if mode == "control_lab":
            destination = context.output_dir / f"foreground_on_gray{pixel_control.suffix.lower()}"
            if pixel_control.resolve() != destination.resolve():
                self._copy_video(pixel_control, destination)
            settings = context.output_dir / "control-settings.json"
            settings.write_text(json.dumps(request, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            context.update("Foreground control ready", 0.99)
            return [self._output(destination, "Foreground-on-gray control"), StudioOutput(settings, "document", "Control settings", "application/json")]

        style = context.asset(str(controls["style_frame_asset"]))
        extra_assets = [context.asset(str(item)) for item in controls.get("keyframe_assets") or []]
        extra_indices = [int(str(item).strip()) for item in controls.get("keyframe_indices") or []]
        outside = [index for index in extra_indices if index >= total_frames]
        if outside:
            raise ValueError(
                f"Keyframe indices {outside} fall outside the selected {total_frames}-frame source range."
            )
        keyframes = {index: asset for index, asset in zip(extra_indices, extra_assets) if index < total_frames}
        normal = context.asset(str(controls["normal_control_asset"])) if mode == "normal_depth" else None
        depth = context.asset(str(controls["depth_control_asset"])) if mode == "normal_depth" else None
        if strategy == "precomputed":
            self._require_synchronized_control(
                pixel_control, frames=total_frames, label="Prepared pixel control"
            )
        if normal and depth:
            self._require_synchronized_control(normal, frames=total_frames, label="Surface-normal control")
            self._require_synchronized_control(depth, frames=total_frames, label="Depth control")
        model_name = GEOMETRY_MODEL if mode == "normal_depth" else PIXEL_MODEL
        segments = self._segments(total_frames, int(controls.get("clip_frames", 81)), set(keyframes))
        context.log(f"Resolved {total_frames} frames into {len(segments)} overlap-safe clip(s).")

        clips_dir = context.output_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        generated_clips: list[Path] = []
        with ComfyRuntime(
            python=self.python, comfy_root=self.comfy, runtime_root=self.runtime_root, context=context,
        ) as runtime:
            control_name = runtime.add_input(pixel_control, f"pixel-control-{pixel_control.name}")
            normal_name = runtime.add_input(normal, f"normal-control-{normal.name}") if normal else None
            depth_name = runtime.add_input(depth, f"depth-control-{depth.name}") if depth else None
            continuity: Path | None = None
            for clip_index, (offset, length) in enumerate(segments):
                context.check_cancelled()
                if offset in keyframes:
                    clip_style = keyframes[offset]
                    context.log(f"Clip {clip_index + 1} pins user keyframe {offset}.")
                elif continuity is not None:
                    clip_style = continuity
                else:
                    clip_style = style
                style_name = runtime.add_input(clip_style, f"style-{clip_index:03d}-{clip_style.name}")
                before = set(runtime.output_files())
                seed = int(controls.get("seed", 42))
                if str(controls.get("seed_policy", "increment")) == "increment":
                    seed += clip_index
                graph = self._graph(
                    control_name=control_name, normal_name=normal_name, depth_name=depth_name,
                    style_name=style_name, model_name=model_name, offset=offset, length=length,
                    width=width, height=height, fps=output_fps, controls=controls,
                    prefix=f"v2v/{context.job_id}/clip-{clip_index:03d}", seed=seed,
                )
                runtime.execute(
                    graph,
                    stage=f"Generating clip {clip_index + 1}/{len(segments)} · source frame {offset}",
                )
                created = [path for path in runtime.output_files() if path not in before and path.suffix.lower() in {".mp4", ".mkv", ".webm"}]
                if not created:
                    raise RuntimeError(f"Comfy finished clip {clip_index + 1} without a video artifact.")
                clip_path = clips_dir / f"clip-{clip_index:03d}.mp4"
                self._copy_video(created[-1], clip_path)
                generated_clips.append(clip_path)
                continuity = context.output_dir / f".continuity-{clip_index:03d}.png"
                context.run_process([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-sseof", "-0.2",
                    "-i", str(clip_path), "-frames:v", "1", str(continuity),
                ])

        container = str(controls.get("container", "mp4"))
        generated = self._stitch(
            generated_clips, context.output_dir / f"generated_video.{container}",
            segments=segments, total_frames=total_frames, controls=controls, context=context,
        )
        original, comparison, flip = self._derived_videos(
            generated, source, width=width, height=height, total_frames=total_frames,
            controls=controls, context=context,
        )
        for hidden in context.output_dir.glob(".continuity-*.png"):
            hidden.unlink(missing_ok=True)
        outputs: list[StudioOutput] = [
            self._output(generated, "Generated video", {"model": model_name, "frames": total_frames, "clips": len(segments)}),
            self._output(original, "Original video · matched framing"),
        ]
        if flip:
            outputs.append(self._output(flip, "Flip test · generated ↔ source"))
        if comparison:
            outputs.append(self._output(comparison, "Generated | source comparison"))
        if strategy == "auto_matte":
            outputs.append(self._output(pixel_control, "Foreground-on-gray control"))
        elif strategy == "precomputed":
            copied = self._copy_video(pixel_control, context.output_dir / f"condition_pixel{pixel_control.suffix.lower()}")
            outputs.append(self._output(copied, "Pixel control"))
        if normal and depth:
            normal_copy = self._copy_video(normal, context.output_dir / f"condition_normal{normal.suffix.lower()}")
            depth_copy = self._copy_video(depth, context.output_dir / f"condition_depth{depth.suffix.lower()}")
            outputs.extend([self._output(normal_copy, "Surface-normal control"), self._output(depth_copy, "Depth control")])
        if bool(controls.get("save_verbose", True)):
            outputs.extend(self._output(path, f"Generated clip {index + 1}", {"offset": segments[index][0]}) for index, path in enumerate(generated_clips))
        else:
            shutil.rmtree(clips_dir, ignore_errors=True)
        settings = context.output_dir / "run_config.json"
        settings.write_text(
            json.dumps({"request": request, "segments": segments, "source_fps": source_fps, "output_fps": output_fps}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        outputs.append(StudioOutput(settings, "document", "Resolved run configuration", "application/json"))
        context.update("Identity-preserving video package ready", 0.99)
        return outputs
