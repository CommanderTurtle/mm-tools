from __future__ import annotations

import gc
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from local_app.comfy_runtime import ComfyRuntime
from local_app.runtime import StudioAdapter, StudioContext, StudioOutput, gpu_snapshot


DISTILLED_MODEL = "wan_animate_2_distill_int8_convrot.safetensors"
BASE_MODEL = "wan_animate_2_int8_convrot.safetensors"
LIGHTX2V_LORA = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
CLIP_VISION = "clip_vision_h.safetensors"
VAE = "Wan2_1_VAE_bf16.safetensors"
REPLACEMENT_MODEL = "wan2.2_animate_14B_int8_convrot.safetensors"
MAX_MOTION_FRAMES = 1921
PIXEL_BUDGET_480P = 480 * 832
PIXEL_BUDGET_720P = 1280 * 720
PIXEL_BUDGET_1080P = 1920 * 1080
DELIVERY_MAX_DIMENSION = 2160
PROFILES: dict[str, dict[str, Any]] = {
    "distilled": {
        "label": "native distilled INT8",
        "model": DISTILLED_MODEL,
        "lora": None,
        "steps": 10,
        "sampler": "euler",
        "scheduler": "simple",
        "guidance": 1.0,
    },
    "lightx2v": {
        "label": "LightX2V official Comfy recipe",
        "model": BASE_MODEL,
        "lora": LIGHTX2V_LORA,
        "steps": 6,
        "sampler": "lcm",
        "scheduler": "simple",
        "guidance": 1.0,
    },
    "lightx2v_4step": {
        "label": "LightX2V four-step speed",
        "model": BASE_MODEL,
        "lora": LIGHTX2V_LORA,
        "steps": 4,
        "sampler": "lcm",
        "scheduler": "simple",
        "guidance": 1.0,
    },
}
NEGATIVE = (
    "oversaturated, overexposed, static, blurred details, subtitles, painting, still frame, "
    "gray cast, worst quality, low quality, jpeg artifacts, ugly, malformed limbs, fused fingers, "
    "bad hands, bad face, frozen motion, cluttered background, duplicate limbs, backwards walking"
)


def _node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


def _wan_length(value: Any) -> int:
    """Clamp and round upward to Wan's required 4n+1 frame count."""

    requested = max(17, min(MAX_MOTION_FRAMES, int(value)))
    return min(MAX_MOTION_FRAMES, requested + ((1 - requested) % 4))


def _latent_size(value: int) -> int:
    return (value + 15) // 16 * 16


def _ratio(value: Any) -> float:
    text = str(value or "").strip()
    if not text or text in {"0/0", "N/A"}:
        return 0.0
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    return float(text)


def _video_probe(path: Path) -> dict[str, Any]:
    """Read the source timeline without decoding its frames into tensors."""

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:stream_tags=rotate:stream_side_data=rotation:format=duration",
                "-of", "json", str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(result.stdout)
        stream = (payload.get("streams") or [])[0]
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError, IndexError) as exc:
        raise ValueError(f"Could not inspect driving video metadata: {path.name}") from exc

    fps = _ratio(stream.get("avg_frame_rate")) or _ratio(stream.get("r_frame_rate"))
    duration = float(stream.get("duration") or (payload.get("format") or {}).get("duration") or 0)
    raw_frames = str(stream.get("nb_frames") or "").strip()
    frames = int(raw_frames) if raw_frames.isdigit() else int(round(duration * fps))
    width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
    side_data = stream.get("side_data_list") or []
    rotation = next((item.get("rotation") for item in side_data if item.get("rotation") is not None), None)
    if rotation is None:
        rotation = (stream.get("tags") or {}).get("rotate", 0)
    if abs(int(float(rotation or 0))) % 180 == 90:
        width, height = height, width
    if fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Driving video does not expose usable frame metadata: {path.name}")
    return {"fps": fps, "frames": frames, "width": width, "height": height}
def _inference_canvas(width: int, height: int, budget: int = PIXEL_BUDGET_720P) -> tuple[int, int]:
    """Fit the generation canvas to the selected Animate pixel budget."""

    scale = min(1.0, (budget / max(1, width * height)) ** 0.5)
    canvas_width = max(256, int(width * scale) // 16 * 16)
    canvas_height = max(256, int(height * scale) // 16 * 16)
    while canvas_width * canvas_height > budget:
        if canvas_width >= canvas_height and canvas_width > 256:
            canvas_width -= 16
        elif canvas_height > 256:
            canvas_height -= 16
        else:
            break
    return canvas_width, canvas_height


def _delivery_size(width: int, height: int) -> tuple[int, int]:
    """Keep the delivered video at the requested size up to the 2160 ceiling."""

    scale = min(1.0, DELIVERY_MAX_DIMENSION / width, DELIVERY_MAX_DIMENSION / height)
    if min(width * scale, height * scale) < 256:
        scale = max(scale, 256 / min(width, height))
    return (
        max(256, min(DELIVERY_MAX_DIMENSION, int(round(width * scale / 8)) * 8)),
        max(256, min(DELIVERY_MAX_DIMENSION, int(round(height * scale / 8)) * 8)),
    )


def _profile_budget(controls: dict[str, Any], profile: dict[str, Any]) -> int:
    """Resolve the inference budget from the selected canvas class."""

    pinned = profile.get("pixel_budget")
    if pinned:
        return int(pinned)
    resolution = str(controls.get("inference_resolution", "720p"))
    if resolution == "1080p":
        return PIXEL_BUDGET_1080P
    if resolution == "480p":
        return PIXEL_BUDGET_480P
    return PIXEL_BUDGET_720P

def _profile(controls: dict[str, Any]) -> dict[str, Any]:
    profile_id = str(controls.get("inference_profile", "distilled"))
    try:
        return PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown Animate inference profile: {profile_id}") from exc


def _sampling(controls: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    if not bool(controls.get("manual_sampling", False)):
        return profile
    return {
        **profile,
        "steps": int(controls.get("steps", profile["steps"])),
        "sampler": str(controls.get("sampler", profile["sampler"])),
        "scheduler": str(controls.get("scheduler", profile["scheduler"])),
        "guidance": float(controls.get("guidance", profile["guidance"])),
    }


def _kind(path: Path) -> tuple[str, str | None]:
    suffix = path.suffix.lower()
    if suffix in {".mp4", ".webm", ".mkv", ".mov"}:
        return "video", {".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska"}.get(suffix)
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image", f"image/{'jpeg' if suffix in {'.jpg', '.jpeg'} else suffix[1:]}"
    if suffix in {".json", ".txt"}:
        return "document", "application/json" if suffix == ".json" else "text/plain"
    return "file", None


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
        self.v2v = self.repo_root / "V2V"
        self.comfy = self.v2v / "ComfyUI"
        self.python = self.project_root / ".venv" / "bin" / "python"
        self.models = self.comfy / "models"
        self.birefnet = self.repo_root / "sculpting" / "pretrained" / "deps" / "ZhengPeng7--BiRefNet"

    def health(self) -> dict[str, Any]:
        checks = [
            ("Animate 2 distilled INT8", self.models / "diffusion_models" / DISTILLED_MODEL),
            ("UMT5 FP8", self.models / "text_encoders" / TEXT_ENCODER),
            ("CLIP Vision H", self.models / "clip_vision" / CLIP_VISION),
            ("Wan VAE BF16", self.models / "vae" / VAE),
            ("ViTPose ONNX", self.models / "detection" / "vitpose_h_wholebody_model.onnx"),
            ("ViTPose external data", self.models / "detection" / "vitpose_h_wholebody_data.bin"),
            ("YOLOv10m ONNX", self.models / "detection" / "yolov10m.onnx"),
            ("Animate Python environment", self.python),
        ]
        details = [{"label": label, "ready": path.is_file(), "required": True, "path": str(path)} for label, path in checks]
        optional = [
            ("Animate 2 base INT8 · LightX2V", self.models / "diffusion_models" / BASE_MODEL),
            ("LightX2V acceleration LoRA", self.models / "loras" / LIGHTX2V_LORA),
            ("Animate 2 scene-preserving replacement", self.models / "diffusion_models" / REPLACEMENT_MODEL),
            ("Local BiRefNet character matte", self.birefnet / "model.safetensors"),
        ]
        details.extend(
            {"label": label, "ready": path.is_file(), "required": False, "path": str(path)}
            for label, path in optional
        )
        gpu = gpu_snapshot()
        device = gpu.get("devices", [{}])[0] if gpu.get("available") else {}
        gpu_ok = bool(device) and int(device.get("memory_total_mib", 0)) >= 30000
        details.append({"label": "CUDA GPU >= 30 GiB", "ready": gpu_ok, "required": True, "value": device.get("name", "unavailable")})
        return {"ready": all(item[1].is_file() for item in checks) and gpu_ok, "loaded": False, "details": details}

    def validate(self, request: dict[str, Any], resolve_asset: Callable[[str], Path]) -> None:
        mode = str(request.get("mode", ""))
        controls = request.get("controls") or {}
        if mode not in {"motion_transfer", "character_replace", "pose_lab"}:
            raise ValueError(f"Unknown Animate mode: {mode}")
        resolve_asset(str(controls.get("driving_video_asset", "")))
        width, height = int(controls.get("width", 480)), int(controls.get("height", 832))
        if width % 8 or height % 8 or not 256 <= width <= 2160 or not 256 <= height <= 2160:
            raise ValueError("Width and height must be multiples of 8 between 256 and 2160 pixels.")
        if mode in {"motion_transfer", "character_replace"}:
            resolve_asset(str(controls.get("reference_image_asset", "")))
            length = int(controls.get("length", 81 if mode == "motion_transfer" else 77))
            maximum = MAX_MOTION_FRAMES if mode == "motion_transfer" else 77
            if length < 17 or length > maximum:
                raise ValueError(
                    f"Frame count must be from 17 through {maximum}; it is rounded to 4n+1 automatically."
                )
        if mode == "motion_transfer":
            profile = _profile(controls)
            profile_files = [self.models / "diffusion_models" / str(profile["model"])]
            if profile["lora"]:
                profile_files.append(self.models / "loras" / str(profile["lora"]))
            missing = [path.name for path in profile_files if not path.is_file()]
            if missing:
                raise ValueError(
                    f"{profile['label']} is not installed ({', '.join(missing)}). "
                    "Run the model downloader's animate bundle, then retry."
                )
            start, end = float(controls.get("pose_start_percent", 0)), float(controls.get("pose_end_percent", 1))
            if not 0 <= start <= end <= 1:
                raise ValueError("Pose influence start must be no later than its end, both in the 0–1 range.")
            optional = str(controls.get("continue_motion_asset", "")).strip()
            if optional:
                resolve_asset(optional)
        elif mode == "character_replace":
            required = [
                self.models / "diffusion_models" / REPLACEMENT_MODEL,
                self.models / "loras" / LIGHTX2V_LORA,
            ]
            missing = [path.name for path in required if not path.is_file()]
            if missing:
                raise ValueError(
                    f"Animate 2 replacement mode is not installed ({', '.join(missing)}). "
                    "Run the model downloader's animate bundle, then retry."
                )
            strategy = str(controls.get("mask_strategy", "automatic"))
            if strategy == "automatic":
                if not (self.birefnet / "model.safetensors").is_file():
                    raise ValueError("Automatic character replacement needs the local BiRefNet matte model.")
            elif strategy == "prepared":
                resolve_asset(str(controls.get("character_mask_asset", "")))
            else:
                raise ValueError(f"Unknown character mask source: {strategy}")
        else:
            optional = str(controls.get("retarget_image_asset", "")).strip()
            if optional:
                resolve_asset(optional)

    def _character_mask(
        self,
        source: Path,
        destination: Path,
        *,
        source_start: int,
        length: int,
        width: int,
        height: int,
        threshold: float,
        expand: int,
        feather: int,
    ) -> Path:
        """Extract the source performer one frame at a time with local BiRefNet."""

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
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_start)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
        writer = cv2.VideoWriter(
            str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("OpenCV could not initialize the local character-mask encoder.")

        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        model = AutoModelForImageSegmentation.from_pretrained(
            str(self.birefnet), trust_remote_code=True, local_files_only=True
        ).eval().to("cuda")
        written = 0
        try:
            for index in range(length):
                self._active_context.check_cancelled()
                ok, frame = capture.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to("cuda")
                with torch.inference_mode():
                    matte = model(tensor)[-1].sigmoid()[0, 0].float().cpu().numpy()
                matte = cv2.resize(matte, (width, height), interpolation=cv2.INTER_LINEAR)
                matte = np.clip((matte - threshold) / max(0.01, 1.0 - threshold), 0.0, 1.0)
                if expand > 0:
                    kernel = np.ones((expand * 2 + 1, expand * 2 + 1), dtype=np.uint8)
                    matte = cv2.dilate(matte, kernel, iterations=1)
                if feather > 0:
                    size = feather * 2 + 1
                    matte = cv2.GaussianBlur(matte, (size, size), 0)
                mono = np.rint(matte * 255.0).astype(np.uint8)
                writer.write(cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR))
                written += 1
                if index % 4 == 0 or index + 1 == length:
                    self._active_context.update(
                        f"Extracting the source character matte · frame {index + 1}/{length}",
                        0.03 + 0.12 * ((index + 1) / length),
                    )
        finally:
            capture.release()
            writer.release()
            del model
            gc.collect()
            torch.cuda.empty_cache()
        if written != length:
            destination.unlink(missing_ok=True)
            raise ValueError(f"The driving video ended after {written} of {length} requested frames.")
        return destination

    def _motion_graph(self, runtime: ComfyRuntime, c: dict[str, Any], job_id: str) -> dict[str, Any]:
        reference = runtime.add_input(context_asset := self._asset(c, "reference_image_asset"), f"reference-{context_asset.name}")
        driving_path = self._asset(c, "driving_video_asset")
        driving = runtime.add_input(driving_path, f"driving-{driving_path.name}")
        width, height = int(c.get("width", 480)), int(c.get("height", 832))
        profile = _profile(c)
        latent_width, latent_height = _inference_canvas(width, height, _profile_budget(c, profile))
        delivery_width, delivery_height = _delivery_size(width, height)
        length = _wan_length(c.get("length", 81))
        source = _video_probe(driving_path)
        continuation = str(c.get("continue_motion_asset", "")).strip()
        requested_offset = max(0, int(c.get("video_frame_offset", 0)))
        # Animate 2 consumes one prior output frame when continuing. Seek that
        # overlap in the lazy VIDEO object before GetVideoComponents creates a
        # float tensor, then give Wan a local, zero-based window.
        source_start = max(0, requested_offset - (1 if continuation else 0))
        available = max(1, int(source["frames"]) - source_start)
        source_length = min(length, available)
        source_fps = float(source["fps"])
        self._active_context.log(
            f"Input window: decoding {source_length} frame{'s' if source_length != 1 else ''} "
            f"from frame {source_start} of {source['frames']} before tensor materialization."
        )
        if (latent_width, latent_height) != (width, height) or (delivery_width, delivery_height) != (width, height):
            self._active_context.log(
                f"Inference canvas {latent_width}x{latent_height}; "
                f"delivery {delivery_width}x{delivery_height} from the requested {width}x{height}."
            )
        sampling = _sampling(c, profile)
        graph: dict[str, Any] = {
            "1": _node("LoadImage", image=reference),
            "2": _node("LoadVideo", file=driving),
            "33": _node(
                "Video Slice", video=["2", 0], start_time=source_start / source_fps,
                duration=source_length / source_fps, strict_duration=False,
            ),
            "4": _node("CLIPLoader", clip_name=TEXT_ENCODER, type="wan", device="default"),
            "5": _node("CLIPTextEncode", clip=["4", 0], text=str(c.get("prompt", ""))),
            "6": _node("CLIPTextEncode", clip=["4", 0], text=str(c.get("negative_prompt", NEGATIVE))),
            "7": _node("CLIPTextEncode", clip=["4", 0], text=str(c.get("pose_prompt", ""))),
            "8": _node("CLIPVisionLoader", clip_name=CLIP_VISION),
            "9": _node("VAELoader", vae_name=VAE),
            "10": _node("ResizeImageMaskNode", input=["1", 0], resize_type="scale dimensions", **{
                "resize_type.width": latent_width, "resize_type.height": latent_height, "resize_type.crop": str(c.get("crop", "center")), "scale_method": str(c.get("scale_method", "area"))}),
            "11": _node("CLIPVisionEncode", clip_vision=["8", 0], image=["10", 0], crop="none"),
            "12": _node("GetVideoComponents", video=["33", 0]),
            "13": _node("ResizeImageMaskNode", input=["12", 0], resize_type="scale dimensions", **{
                "resize_type.width": latent_width, "resize_type.height": latent_height, "resize_type.crop": str(c.get("crop", "center")), "scale_method": str(c.get("scale_method", "area"))}),
            "14": _node("ImageFromBatch", image=["13", 0], batch_index=0, length=1),
            "15": _node("CLIPVisionEncode", clip_vision=["8", 0], image=["14", 0], crop="none"),
        }
        graph["32"] = _node(
            "MMToolsGpuOnlyWanLoader",
            unet_name=str(profile["model"]),
            positive=["5", 0], negative=["6", 0], positive_pose=["7", 0],
            clip_vision_output=["11", 0], clip_vision_output_pose=["15", 0],
            text_encoder=["4", 0], vision_encoder=["8", 0],
        )
        model_ref: list[Any] = ["32", 0]
        if profile["lora"]:
            graph["31"] = _node(
                "LoraLoaderModelOnly", model=model_ref,
                lora_name=str(profile["lora"]), strength_model=1.0,
            )
            model_ref = ["31", 0]
        graph["16"] = _node(
            "ContextWindowsManual", model=model_ref,
            context_length=21, context_overlap=8, context_schedule="standard_static",
            context_stride=1, closed_loop=False, fuse_method="pyramid", dim=2,
            freenoise=True, cond_retain_index_list="0", split_conds_to_windows=False,
            latent_retain_index_list="", causal_window_fix=True,
        )
        model_ref = ["16", 0]
        cache = str(c.get("cache_precision", "disabled"))
        if cache != "disabled":
            graph["17"] = _node("WanAnimate2Cache", model=model_ref, device="gpu", dtype=cache)
            model_ref = ["17", 0]
        graph["18"] = _node(
            "BasicScheduler", model=model_ref, scheduler=str(sampling["scheduler"]),
            steps=int(sampling["steps"]), denoise=float(c.get("denoise", 1)),
        )
        graph["19"] = _node("ModelSamplingSD3", model=model_ref, shift=float(c.get("shift", 5)))
        graph["20"] = _node("KSamplerSelect", sampler_name=str(sampling["sampler"]))
        conditioning: dict[str, Any] = {
            "positive": ["32", 1], "negative": ["32", 2], "vae": ["9", 0],
            "width": latent_width, "height": latent_height, "length": length, "batch_size": 1,
            "reference_image": ["10", 0], "pose_video": ["13", 0],
            "clip_vision_output": ["32", 4], "positive_pose": ["32", 3],
            "clip_vision_output_pose": ["32", 5],
            "video_frame_offset": 0,
            "pose_strength": float(c.get("pose_strength", 1)),
            "pose_start_percent": float(c.get("pose_start_percent", 0)),
            "pose_end_percent": float(c.get("pose_end_percent", 1)),
            "reference_image_strength": float(c.get("reference_strength", 1)),
        }
        if continuation:
            continuation_path = self._asset(c, "continue_motion_asset")
            continuation_name = runtime.add_input(continuation_path, f"continue-{continuation_path.name}")
            continuation_source = _video_probe(continuation_path)
            continuation_fps = float(continuation_source["fps"])
            continuation_start = max(0, int(continuation_source["frames"]) - 1)
            graph["21"] = _node("LoadVideo", file=continuation_name)
            graph["34"] = _node(
                "Video Slice", video=["21", 0], start_time=continuation_start / continuation_fps,
                duration=1 / continuation_fps, strict_duration=False,
            )
            graph["22"] = _node("GetVideoComponents", video=["34", 0])
            conditioning["continue_motion"] = ["22", 0]
        graph["23"] = _node("WanAnimate2ToVideo", **conditioning)
        graph["24"] = _node(
            "SamplerCustom", model=["19", 0], positive=["23", 0], negative=["23", 1], sampler=["20", 0],
            sigmas=["18", 0], latent_image=["23", 2], add_noise=bool(c.get("add_noise", True)),
            noise_seed=int(c.get("seed", 42)), cfg=float(sampling["guidance"]),
        )
        graph["25"] = _node("TrimVideoLatent", samples=["24", 0], trim_amount=["23", 3])
        graph["26"] = _node("VAEDecode", samples=["25", 0], vae=["9", 0])
        image_ref: list[Any] = ["26", 0]
        if bool(c.get("trim_duplicate", False)):
            graph["27"] = _node("ImageFromBatch", image=image_ref, batch_index=1, length=4096)
            image_ref = ["27", 0]
        if (latent_width, latent_height) != (delivery_width, delivery_height):
            graph["30"] = _node(
                "ImageScale", image=image_ref, upscale_method="lanczos",
                width=delivery_width, height=delivery_height, crop="disabled",
            )
            image_ref = ["30", 0]
        graph["28"] = _node("CreateVideo", images=image_ref, fps=float(c.get("fps", 24)), bit_depth=int(c.get("bit_depth", 8)), color_space=str(c.get("color_space", "sRGB")), codec="none")
        graph["29"] = _save_video(["28", 0], f"animate/{job_id}/motion-transfer", str(c.get("container", "mp4")), str(c.get("codec", "h264")), int(c.get("crf", 18)))
        return graph

    def _replacement_graph(
        self,
        runtime: ComfyRuntime,
        c: dict[str, Any],
        job_id: str,
        mask_path: Path,
        *,
        mask_windowed: bool,
    ) -> dict[str, Any]:
        reference_path = self._asset(c, "reference_image_asset")
        driving_path = self._asset(c, "driving_video_asset")
        reference = runtime.add_input(reference_path, f"replacement-{reference_path.name}")
        driving = runtime.add_input(driving_path, f"source-{driving_path.name}")
        mask = runtime.add_input(mask_path, f"mask-{mask_path.name}")
        width, height = int(c.get("width", 480)), int(c.get("height", 832))
        latent_width, latent_height = _inference_canvas(width, height, PIXEL_BUDGET_480P)
        delivery_width, delivery_height = _delivery_size(width, height)
        length = min(77, _wan_length(c.get("length", 77)))
        source = _video_probe(driving_path)
        source_start = max(0, int(c.get("video_frame_offset", 0)))
        if source_start + length > int(source["frames"]):
            raise ValueError(
                f"The selected {length}-frame replacement window exceeds the "
                f"{source['frames']}-frame driving video at offset {source_start}."
            )
        source_fps = float(source["fps"])
        self._active_context.log(
            f"Replacement window: decoding {length} frames from frame {source_start} "
            f"of {source['frames']} before tensor materialization."
        )
        if (latent_width, latent_height) != (width, height) or (delivery_width, delivery_height) != (width, height):
            self._active_context.log(
                f"Inference canvas {latent_width}x{latent_height}; "
                f"delivery {delivery_width}x{delivery_height} from the requested {width}x{height}."
            )

        graph: dict[str, Any] = {
            "1": _node("LoadImage", image=reference),
            "2": _node("LoadVideo", file=driving),
            "3": _node(
                "Video Slice", video=["2", 0], start_time=source_start / source_fps,
                duration=length / source_fps, strict_duration=False,
            ),
            "4": _node("GetVideoComponents", video=["3", 0]),
            "5": _node("ResizeImageMaskNode", input=["4", 0], resize_type="scale dimensions", **{
                "resize_type.width": latent_width, "resize_type.height": latent_height,
                "resize_type.crop": str(c.get("crop", "center")),
                "scale_method": str(c.get("scale_method", "area")),
            }),
            "6": _node(
                "OnnxDetectionModelLoader",
                vitpose_model="vitpose_h_wholebody_model.onnx",
                yolo_model="yolov10m.onnx",
                onnx_device="CUDAExecutionProvider",
            ),
            "7": _node(
                "PoseAndFaceDetection", model=["6", 0], images=["5", 0],
                width=latent_width, height=latent_height,
                face_padding=int(c.get("face_padding", 24)),
            ),
            "8": _node(
                "DrawViTPose", pose_data=["7", 0], width=latent_width, height=latent_height,
                retarget_padding=int(c.get("retarget_padding", 16)),
                body_stick_width=-1, hand_stick_width=-1, draw_head=True,
            ),
            "9": _node("LoadVideo", file=mask),
            "14": _node("CLIPLoader", clip_name=TEXT_ENCODER, type="wan", device="default"),
            "15": _node("CLIPTextEncode", clip=["14", 0], text=str(c.get("prompt", ""))),
            "16": _node("ConditioningZeroOut", conditioning=["15", 0]),
            "17": _node("VAELoader", vae_name=VAE),
            "18": _node(
                "MMToolsGpuOnlyWan22Loader", unet_name=REPLACEMENT_MODEL,
                positive=["15", 0], negative=["16", 0], text_encoder=["14", 0],
            ),
            "19": _node(
                "LoraLoaderModelOnly", model=["18", 0], lora_name=LIGHTX2V_LORA,
                strength_model=float(c.get("lora_strength", 1.0)),
            ),
            "20": _node(
                "BasicScheduler", model=["19", 0], scheduler="simple",
                steps=4, denoise=float(c.get("denoise", 1)),
            ),
            "21": _node("KSamplerSelect", sampler_name="lcm"),
        }
        mask_video: list[Any] = ["9", 0]
        if not mask_windowed:
            graph["10"] = _node(
                "Video Slice", video=mask_video, start_time=source_start / source_fps,
                duration=length / source_fps, strict_duration=False,
            )
            mask_video = ["10", 0]
        graph["11"] = _node("GetVideoComponents", video=mask_video)
        graph["12"] = _node("ResizeImageMaskNode", input=["11", 0], resize_type="scale dimensions", **{
            "resize_type.width": latent_width, "resize_type.height": latent_height,
            "resize_type.crop": str(c.get("crop", "center")),
            "scale_method": "bilinear",
        })
        graph["13"] = _node("ImageToMask", image=["12", 0], channel="red")
        graph["22"] = _node(
            "WanAnimateToVideo",
            positive=["18", 1], negative=["18", 2], vae=["17", 0],
            width=latent_width, height=latent_height, length=length, batch_size=1,
            reference_image=["1", 0], face_video=["7", 1], pose_video=["8", 0],
            background_video=["5", 0], character_mask=["13", 0],
            continue_motion_max_frames=5, video_frame_offset=0,
        )
        graph["23"] = _node(
            "CFGGuider", model=["19", 0], positive=["22", 0], negative=["22", 1], cfg=1.0,
        )
        graph["24"] = _node("RandomNoise", noise_seed=int(c.get("seed", 42)))
        graph["25"] = _node(
            "SamplerCustomAdvanced", noise=["24", 0], guider=["23", 0], sampler=["21", 0],
            sigmas=["20", 0], latent_image=["22", 2],
        )
        graph["26"] = _node("TrimVideoLatent", samples=["25", 0], trim_amount=["22", 3])
        graph["27"] = _node("VAEDecode", samples=["26", 0], vae=["17", 0])
        image_ref: list[Any] = ["27", 0]
        if (latent_width, latent_height) != (delivery_width, delivery_height):
            graph["28"] = _node(
                "ImageScale", image=image_ref, upscale_method="lanczos",
                width=delivery_width, height=delivery_height, crop="disabled",
            )
            image_ref = ["28", 0]
        graph["29"] = _node(
            "CreateVideo", images=image_ref, fps=float(c.get("fps", source_fps)),
            bit_depth=int(c.get("bit_depth", 8)), color_space=str(c.get("color_space", "sRGB")),
            codec="none",
        )
        graph["30"] = _save_video(
            ["29", 0], f"animate/{job_id}/character-replacement",
            str(c.get("container", "mp4")), str(c.get("codec", "h264")), int(c.get("crf", 18)),
        )
        return graph

    def _pose_graph(self, runtime: ComfyRuntime, c: dict[str, Any], job_id: str) -> dict[str, Any]:
        driving_path = self._asset(c, "driving_video_asset")
        driving = runtime.add_input(driving_path, f"pose-lab-{driving_path.name}")
        width, height = int(c.get("width", 480)), int(c.get("height", 832))
        graph: dict[str, Any] = {
            "1": _node("LoadVideo", file=driving),
            "2": _node("GetVideoComponents", video=["1", 0]),
            "3": _node("OnnxDetectionModelLoader", vitpose_model="vitpose_h_wholebody_model.onnx", yolo_model="yolov10m.onnx", onnx_device="CUDAExecutionProvider"),
        }
        detect: dict[str, Any] = {
            "model": ["3", 0], "images": ["2", 0], "width": width, "height": height,
            "face_padding": int(c.get("face_padding", 24)),
        }
        retarget = str(c.get("retarget_image_asset", "")).strip()
        if retarget:
            retarget_path = self._asset(c, "retarget_image_asset")
            retarget_name = runtime.add_input(retarget_path, f"retarget-{retarget_path.name}")
            graph["4"] = _node("LoadImage", image=retarget_name)
            detect["retarget_image"] = ["4", 0]
        graph["5"] = _node("PoseAndFaceDetection", **detect)
        graph["6"] = _node(
            "DrawViTPose", pose_data=["5", 0], width=width, height=height,
            retarget_padding=int(c.get("retarget_padding", 64)),
            body_stick_width=int(c.get("body_stick_width", -1)),
            hand_stick_width=int(c.get("hand_stick_width", -1)), draw_head=bool(c.get("draw_head", True)),
        )
        graph["7"] = _node("CreateVideo", images=["6", 0], fps=["2", 2], bit_depth=8, color_space="sRGB", codec="none")
        graph["8"] = _save_video(["7", 0], f"animate/{job_id}/pose-diagnostic", "mp4", "h264", int(c.get("crf", 18)))
        graph["9"] = _node("CreateVideo", images=["5", 1], fps=["2", 2], bit_depth=8, color_space="sRGB", codec="none")
        graph["10"] = _save_video(["9", 0], f"animate/{job_id}/face-crops", "mp4", "h264", int(c.get("crf", 18)))
        graph["11"] = _node("SaveText", text=["5", 2], filename_prefix=f"animate/{job_id}/key-body-points", format="json")
        return graph

    def _asset(self, controls: dict[str, Any], key: str) -> Path:
        # Studio validation has already established the identifier. Resolving
        # here through the active context is intentionally deferred by storing
        # the callback for the current run.
        return self._active_context.asset(str(controls[key]))

    def _collect(self, runtime: ComfyRuntime, context: StudioContext, request: dict[str, Any]) -> list[StudioOutput]:
        outputs: list[StudioOutput] = []
        for index, source in enumerate(runtime.output_files()):
            relative = source.relative_to(runtime.output_dir)
            destination = context.output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            kind, media = _kind(destination)
            outputs.append(StudioOutput(destination, kind, relative.as_posix(), media, {"native_comfy": True}))
        if not outputs:
            raise RuntimeError("Comfy completed without producing a saved artifact.")
        receipt = context.output_dir / "run-settings.json"
        receipt.write_text(json.dumps(request, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        outputs.append(StudioOutput(receipt, "document", "Run settings", "application/json"))
        return outputs

    @property
    def _active_context(self) -> StudioContext:
        if not hasattr(self, "_context"):
            raise RuntimeError("No active Studio context")
        return self._context

    @_active_context.setter
    def _active_context(self, value: StudioContext) -> None:
        self._context = value

    def run(self, request: dict[str, Any], context: StudioContext) -> list[StudioOutput]:  # type: ignore[override]
        self._active_context = context
        try:
            return self._run_active(request, context)
        finally:
            del self._context

    def _run_active(self, request: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        controls = request.get("controls") or {}
        mode = str(request["mode"])
        generated_mask: Path | None = None
        mask_path: Path | None = None
        mask_windowed = False
        try:
            if mode == "character_replace":
                strategy = str(controls.get("mask_strategy", "automatic"))
                if strategy == "prepared":
                    mask_path = self._asset(controls, "character_mask_asset")
                else:
                    source = self._asset(controls, "driving_video_asset")
                    source_info = _video_probe(source)
                    source_start = max(0, int(controls.get("video_frame_offset", 0)))
                    length = min(77, _wan_length(controls.get("length", 77)))
                    if source_start + length > int(source_info["frames"]):
                        raise ValueError(
                            f"The selected {length}-frame replacement window exceeds the "
                            f"{source_info['frames']}-frame driving video at offset {source_start}."
                        )
                    generated_mask = self.runtime_root / "character-masks" / f"{context.job_id}.mp4"
                    mask_path = self._character_mask(
                        source,
                        generated_mask,
                        source_start=source_start,
                        length=length,
                        width=_latent_size(int(controls.get("width", 480))),
                        height=_latent_size(int(controls.get("height", 832))),
                        threshold=float(controls.get("matte_threshold", 0.35)),
                        expand=int(controls.get("matte_expand", 6)),
                        feather=int(controls.get("matte_feather", 5)),
                    )
                    mask_windowed = True

            if mode == "pose_lab":
                custom = ["mmtools_wan_animate_preprocess"]
            elif mode == "character_replace":
                custom = ["mmtools_animate", "mmtools_wan_animate_preprocess"]
            else:
                custom = ["mmtools_animate"]
            with ComfyRuntime(
                python=self.python, comfy_root=self.comfy, runtime_root=self.runtime_root,
                context=context, custom_node_allowlist=custom,
            ) as runtime:
                if mode == "pose_lab":
                    graph = self._pose_graph(runtime, controls, context.job_id)
                    stage = "Detecting body, hands, face, and retargeted pose"
                elif mode == "character_replace":
                    assert mask_path is not None
                    graph = self._replacement_graph(
                        runtime, controls, context.job_id, mask_path, mask_windowed=mask_windowed
                    )
                    stage = "Replacing the source character while preserving the scene"
                else:
                    graph = self._motion_graph(runtime, controls, context.job_id)
                    stage = "Transferring motion with Wan Animate 2"
                runtime.execute(graph, stage=stage)
                outputs = self._collect(runtime, context, request)
        finally:
            if generated_mask is not None:
                generated_mask.unlink(missing_ok=True)
        context.update("Animation artifacts are ready", 0.99)
        return outputs
