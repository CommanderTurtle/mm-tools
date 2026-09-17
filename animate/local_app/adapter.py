from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

from studio.comfy_runtime import ComfyRuntime
from studio.runtime import StudioAdapter, StudioContext, StudioOutput, gpu_snapshot


MODEL = "wan_animate_2_distill_int8_convrot.safetensors"
TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
CLIP_VISION = "clip_vision_h.safetensors"
VAE = "Wan2_1_VAE_bf16.safetensors"
NEGATIVE = (
    "oversaturated, overexposed, static, blurred details, subtitles, painting, still frame, "
    "gray cast, worst quality, low quality, jpeg artifacts, ugly, malformed limbs, fused fingers, "
    "bad hands, bad face, frozen motion, cluttered background, duplicate limbs, backwards walking"
)


def _node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


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
        self.python = self.v2v / ".venv" / "bin" / "python"
        self.models = self.comfy / "models"

    def health(self) -> dict[str, Any]:
        checks = [
            ("Animate 2 distilled INT8", self.models / "diffusion_models" / MODEL),
            ("UMT5 FP8", self.models / "text_encoders" / TEXT_ENCODER),
            ("CLIP Vision H", self.models / "clip_vision" / CLIP_VISION),
            ("Wan VAE BF16", self.models / "vae" / VAE),
            ("ViTPose ONNX", self.models / "detection" / "vitpose_h_wholebody_model.onnx"),
            ("ViTPose external data", self.models / "detection" / "vitpose_h_wholebody_data.bin"),
            ("YOLOv10m ONNX", self.models / "detection" / "yolov10m.onnx"),
            ("Shared Python environment", self.python),
        ]
        details = [{"label": label, "ready": path.is_file(), "required": True, "path": str(path)} for label, path in checks]
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
        if width % 16 or height % 16 or not 256 <= width <= 1280 or not 256 <= height <= 1280:
            raise ValueError("Width and height must be multiples of 16 between 256 and 1280 pixels.")
        if mode == "motion_transfer":
            resolve_asset(str(controls.get("reference_image_asset", "")))
            length = int(controls.get("length", 81))
            if length < 17 or length > 241 or (length - 1) % 4:
                raise ValueError("Frame count must be 4n+1, from 17 through 241.")
            start, end = float(controls.get("pose_start_percent", 0)), float(controls.get("pose_end_percent", 1))
            if not 0 <= start <= end <= 1:
                raise ValueError("Pose influence start must be no later than its end, both in the 0–1 range.")
            if bool(controls.get("enable_context", False)):
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
        width, height, length = int(c.get("width", 480)), int(c.get("height", 832)), int(c.get("length", 81))
        graph: dict[str, Any] = {
            "1": _node("LoadImage", image=reference),
            "2": _node("LoadVideo", file=driving),
            "3": _node("UNETLoader", unet_name=MODEL, weight_dtype="default"),
            "4": _node("CLIPLoader", clip_name=TEXT_ENCODER, type="wan", device="default"),
            "5": _node("CLIPTextEncode", clip=["4", 0], text=str(c.get("prompt", ""))),
            "6": _node("CLIPTextEncode", clip=["4", 0], text=str(c.get("negative_prompt", NEGATIVE))),
            "7": _node("CLIPTextEncode", clip=["4", 0], text=str(c.get("pose_prompt", ""))),
            "8": _node("CLIPVisionLoader", clip_name=CLIP_VISION),
            "9": _node("VAELoader", vae_name=VAE),
            "10": _node("ResizeImageMaskNode", input=["1", 0], resize_type="scale dimensions", **{
                "resize_type.width": width, "resize_type.height": height, "resize_type.crop": str(c.get("crop", "center")), "scale_method": str(c.get("scale_method", "area"))}),
            "11": _node("CLIPVisionEncode", clip_vision=["8", 0], image=["10", 0], crop="none"),
            "12": _node("GetVideoComponents", video=["2", 0]),
            "13": _node("ResizeImageMaskNode", input=["12", 0], resize_type="scale dimensions", **{
                "resize_type.width": width, "resize_type.height": height, "resize_type.crop": str(c.get("crop", "center")), "scale_method": str(c.get("scale_method", "area"))}),
            "14": _node("ImageFromBatch", image=["13", 0], batch_index=int(c.get("video_frame_offset", 0)), length=1),
            "15": _node("CLIPVisionEncode", clip_vision=["8", 0], image=["14", 0], crop="none"),
        }
        model_ref: list[Any] = ["3", 0]
        if bool(c.get("enable_context", False)):
            graph["16"] = _node(
                "ContextWindowsManual", model=model_ref,
                context_length=int(c.get("context_length", 21)),
                context_overlap=int(c.get("context_overlap", 8)),
                context_schedule=str(c.get("context_schedule", "standard_static")),
                context_stride=int(c.get("context_stride", 1)), closed_loop=bool(c.get("closed_loop", False)),
                fuse_method=str(c.get("fuse_method", "pyramid")), dim=2,
                freenoise=bool(c.get("freenoise", True)), cond_retain_index_list="0",
                split_conds_to_windows=False, latent_retain_index_list="", causal_window_fix=True,
            )
            model_ref = ["16", 0]
        cache = str(c.get("cache_precision", "int8"))
        if cache != "disabled":
            graph["17"] = _node("WanAnimate2Cache", model=model_ref, device="gpu", dtype=cache)
            model_ref = ["17", 0]
        graph["18"] = _node("BasicScheduler", model=model_ref, scheduler=str(c.get("scheduler", "simple")), steps=int(c.get("steps", 10)), denoise=float(c.get("denoise", 1)))
        graph["19"] = _node("ModelSamplingSD3", model=model_ref, shift=float(c.get("shift", 5)))
        graph["20"] = _node("KSamplerSelect", sampler_name=str(c.get("sampler", "lcm")))
        conditioning: dict[str, Any] = {
            "positive": ["5", 0], "negative": ["6", 0], "vae": ["9", 0],
            "width": width, "height": height, "length": length, "batch_size": 1,
            "reference_image": ["10", 0], "pose_video": ["13", 0],
            "clip_vision_output": ["11", 0], "positive_pose": ["7", 0],
            "clip_vision_output_pose": ["15", 0],
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
            noise_seed=int(c.get("seed", 42)), cfg=float(c.get("guidance", 1)),
        )
        graph["25"] = _node("TrimVideoLatent", samples=["24", 0], trim_amount=["23", 3])
        graph["26"] = _node("VAEDecode", samples=["25", 0], vae=["9", 0])
        image_ref: list[Any] = ["26", 0]
        if bool(c.get("trim_duplicate", False)):
            graph["27"] = _node("ImageFromBatch", image=image_ref, batch_index=1, length=4096)
            image_ref = ["27", 0]
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
        custom = ["mmtools_wan_animate_preprocess"] if request["mode"] == "pose_lab" else []
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
