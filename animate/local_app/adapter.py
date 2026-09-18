from __future__ import annotations

import json
import shutil
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
CANONICAL_CONTEXT_TOKENS = 21 * (480 // 16) * (832 // 16)
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

    requested = max(17, min(241, int(value)))
    return min(241, requested + ((1 - requested) % 4))


def _latent_size(value: int) -> int:
    return (value + 15) // 16 * 16


def _context_window_plan(
    width: int,
    height: int,
    requested_length: int,
    requested_overlap: int,
) -> tuple[int, int]:
    """Fit a temporal window to Animate 2's official 480p token budget."""

    spatial_tokens = max(1, (width // 16) * (height // 16))
    safe_length = max(3, CANONICAL_CONTEXT_TOKENS // spatial_tokens)
    length = max(3, min(requested_length, safe_length))
    if requested_overlap <= 0:
        return length, 0
    overlap = max(1, requested_overlap * length // max(1, requested_length))
    return length, min(length - 1, overlap)


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
        if mode not in {"motion_transfer", "pose_lab"}:
            raise ValueError(f"Unknown Animate mode: {mode}")
        resolve_asset(str(controls.get("driving_video_asset", "")))
        width, height = int(controls.get("width", 480)), int(controls.get("height", 832))
        if width % 8 or height % 8 or not 256 <= width <= 2160 or not 256 <= height <= 2160:
            raise ValueError("Width and height must be multiples of 8 between 256 and 2160 pixels.")
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
            resolve_asset(str(controls.get("reference_image_asset", "")))
            length = int(controls.get("length", 81))
            if length < 17 or length > 241:
                raise ValueError("Frame count must be from 17 through 241; it is rounded up to 4n+1 automatically.")
            start, end = float(controls.get("pose_start_percent", 0)), float(controls.get("pose_end_percent", 1))
            if not 0 <= start <= end <= 1:
                raise ValueError("Pose influence start must be no later than its end, both in the 0–1 range.")
            if bool(controls.get("enable_context", True)):
                context_length = int(controls.get("context_length", 21))
                overlap = int(controls.get("context_overlap", 8))
                if context_length < 5 or overlap < 0 or overlap >= context_length:
                    raise ValueError("Context overlap must be non-negative and smaller than context length.")
            optional = str(controls.get("continue_motion_asset", "")).strip()
            if optional:
                resolve_asset(optional)
        else:
            optional = str(controls.get("retarget_image_asset", "")).strip()
            if optional:
                resolve_asset(optional)

    def _motion_graph(self, runtime: ComfyRuntime, c: dict[str, Any], job_id: str) -> dict[str, Any]:
        reference = runtime.add_input(context_asset := self._asset(c, "reference_image_asset"), f"reference-{context_asset.name}")
        driving_path = self._asset(c, "driving_video_asset")
        driving = runtime.add_input(driving_path, f"driving-{driving_path.name}")
        width, height = int(c.get("width", 480)), int(c.get("height", 832))
        latent_width, latent_height = _latent_size(width), _latent_size(height)
        length = _wan_length(c.get("length", 81))
        profile = _profile(c)
        sampling = _sampling(c, profile)
        graph: dict[str, Any] = {
            "1": _node("LoadImage", image=reference),
            "2": _node("LoadVideo", file=driving),
            "4": _node("CLIPLoader", clip_name=TEXT_ENCODER, type="wan", device="default"),
            "5": _node("CLIPTextEncode", clip=["4", 0], text=str(c.get("prompt", ""))),
            "6": _node("CLIPTextEncode", clip=["4", 0], text=str(c.get("negative_prompt", NEGATIVE))),
            "7": _node("CLIPTextEncode", clip=["4", 0], text=str(c.get("pose_prompt", ""))),
            "8": _node("CLIPVisionLoader", clip_name=CLIP_VISION),
            "9": _node("VAELoader", vae_name=VAE),
            "10": _node("ResizeImageMaskNode", input=["1", 0], resize_type="scale dimensions", **{
                "resize_type.width": latent_width, "resize_type.height": latent_height, "resize_type.crop": str(c.get("crop", "center")), "scale_method": str(c.get("scale_method", "area"))}),
            "11": _node("CLIPVisionEncode", clip_vision=["8", 0], image=["10", 0], crop="none"),
            "12": _node("GetVideoComponents", video=["2", 0]),
            "13": _node("ResizeImageMaskNode", input=["12", 0], resize_type="scale dimensions", **{
                "resize_type.width": latent_width, "resize_type.height": latent_height, "resize_type.crop": str(c.get("crop", "center")), "scale_method": str(c.get("scale_method", "area"))}),
            "14": _node("ImageFromBatch", image=["13", 0], batch_index=int(c.get("video_frame_offset", 0)), length=1),
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
        if bool(c.get("enable_context", True)):
            requested_context = int(c.get("context_length", 21))
            requested_overlap = int(c.get("context_overlap", 8))
            context_length, context_overlap = _context_window_plan(
                latent_width,
                latent_height,
                requested_context,
                requested_overlap,
            )
            if (context_length, context_overlap) != (requested_context, requested_overlap):
                self._active_context.log(
                    "GPU window planner: "
                    f"{requested_context}/{requested_overlap} -> {context_length}/{context_overlap} "
                    f"latent frames at {width}x{height}; full {length}-frame output is preserved."
                )
            graph["16"] = _node(
                "ContextWindowsManual", model=model_ref,
                context_length=context_length,
                context_overlap=context_overlap,
                context_schedule=str(c.get("context_schedule", "standard_static")),
                context_stride=int(c.get("context_stride", 1)), closed_loop=bool(c.get("closed_loop", False)),
                fuse_method=str(c.get("fuse_method", "pyramid")), dim=2,
                freenoise=bool(c.get("freenoise", True)), cond_retain_index_list="0",
                split_conds_to_windows=False, latent_retain_index_list="", causal_window_fix=True,
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
            "video_frame_offset": int(c.get("video_frame_offset", 0)),
            "pose_strength": float(c.get("pose_strength", 1)),
            "pose_start_percent": float(c.get("pose_start_percent", 0)),
            "pose_end_percent": float(c.get("pose_end_percent", 1)),
            "reference_image_strength": float(c.get("reference_strength", 1)),
        }
        continuation = str(c.get("continue_motion_asset", "")).strip()
        if continuation:
            continuation_path = self._asset(c, "continue_motion_asset")
            continuation_name = runtime.add_input(continuation_path, f"continue-{continuation_path.name}")
            graph["21"] = _node("LoadVideo", file=continuation_name)
            graph["22"] = _node("GetVideoComponents", video=["21", 0])
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
        if (latent_width, latent_height) != (width, height):
            graph["30"] = _node(
                "ImageCrop", image=image_ref, width=width, height=height,
                x=(latent_width - width) // 2, y=(latent_height - height) // 2,
            )
            image_ref = ["30", 0]
        graph["28"] = _node("CreateVideo", images=image_ref, fps=float(c.get("fps", 24)), bit_depth=int(c.get("bit_depth", 8)), color_space=str(c.get("color_space", "sRGB")), codec="none")
        graph["29"] = _save_video(["28", 0], f"animate/{job_id}/motion-transfer", str(c.get("container", "mp4")), str(c.get("codec", "h264")), int(c.get("crf", 18)))
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
        custom = ["mmtools_wan_animate_preprocess"] if request["mode"] == "pose_lab" else ["mmtools_animate"]
        with ComfyRuntime(
            python=self.python, comfy_root=self.comfy, runtime_root=self.runtime_root,
            context=context, custom_node_allowlist=custom,
        ) as runtime:
            if request["mode"] == "pose_lab":
                graph, stage = self._pose_graph(runtime, controls, context.job_id), "Detecting body, hands, face, and retargeted pose"
            else:
                graph, stage = self._motion_graph(runtime, controls, context.job_id), "Transferring motion with Wan Animate 2"
            runtime.execute(graph, stage=stage)
            outputs = self._collect(runtime, context, request)
        context.update("Animation artifacts are ready", 0.99)
        return outputs
