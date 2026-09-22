from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

from local_app.comfy_runtime import ComfyRuntime
from local_app.runtime import StudioAdapter, StudioContext, StudioOutput, gpu_snapshot


MODEL_INT8 = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
MODEL_NVFP4 = "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
TEXT_ENCODER = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
VIDEO_VAE = "ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
ENHANCER_ENCODER = "gemma4_e2b_it_int8_convrot.safetensors"
SPATIAL_UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
TEMPORAL_UPSCALER = "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"
DURATION_HEAD = "ltx-2.5-duration-head-bf16.safetensors"
WEIGHTS_VARIANTS = ("int8", "nvfp4")
MODES = ("t2v", "i2v", "flf2v")
MAX_FRAMES = 1000  # LTXVEmptyLatentAudio.frames_number schema ceiling

# Official two-stage sigma schedules from the ComfyUI LTX-2.5 T2V/I2V templates.
SIGMAS_STAGE1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
SIGMAS_STAGE2 = "0.85, 0.7250, 0.4219, 0.0"

NEGATIVE_T2V = "pc game, console game, video game, cartoon, childish, ugly"
NEGATIVE_FLF2V = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, unreadable text on shirt or hat, incorrect lettering on cap (\u201cPNTR\u201d), incorrect t-shirt slogan (\u201cJUST DO IT\u201d), missing microphone, misplaced microphone, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, smiling, laughing, exaggerated sadness, wrong gaze direction, eyes looking at camera, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, missing sniff sounds, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, missing door or shelves, missing shallow depth of field, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
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
        self.repo_root = project_root.parent.parent
        self.comfy = project_root.parent / "ComfyUI"
        self.python = project_root / ".venv" / "bin" / "python"
        self.models = self.comfy / "models"

    def health(self) -> dict[str, Any]:
        required = [
            ("LTX-2.5 22B distilled DiT · INT8 ConvRot", self.models / "diffusion_models" / MODEL_INT8),
            ("Gemma 4 12B projection encoder", self.models / "text_encoders" / TEXT_ENCODER),
            ("LTX video VAE BF16", self.models / "vae" / VIDEO_VAE),
            ("LTX audio VAE BF16", self.models / "vae" / AUDIO_VAE),
            ("x2 latent spatial upscaler", self.models / "latent_upscale_models" / SPATIAL_UPSCALER),
            ("Shared Python environment", self.python),
        ]
        details = [
            {"label": label, "ready": path.is_file(), "required": True, "path": str(path)}
            for label, path in required
        ]
        optional = [
            ("LTX-2.5 22B distilled DiT · NVFP4", self.models / "diffusion_models" / MODEL_NVFP4,
             "Needed only for the NVFP4 weights variant."),
            ("E2B prompt enhancer encoder", self.models / "text_encoders" / ENHANCER_ENCODER,
             "Needed only when prompt enhancement is enabled."),
            ("x2 latent temporal upscaler", self.models / "latent_upscale_models" / TEMPORAL_UPSCALER,
             "Bundled for the long-form lane; unused by the distilled studio graph."),
            ("Duration head patch", self.models / "model_patches" / DURATION_HEAD,
             "Bundled for the long-form lane; unused by the distilled studio graph."),
        ]
        for label, path, note in optional:
            details.append({
                "label": label, "ready": path.is_file(), "required": False,
                "path": str(path), "note": note,
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
        if mode not in MODES:
            raise ValueError(f"Unknown LTX-2.5 mode: {mode}")
        if str(controls.get("weights_variant", "int8")) not in WEIGHTS_VARIANTS:
            raise ValueError("Weights variant must be int8 or nvfp4.")
        if str(controls.get("weights_variant", "int8")) == "nvfp4" and not (
            self.models / "diffusion_models" / MODEL_NVFP4
        ).is_file():
            raise ValueError("The NVFP4 DiT is not installed. Run workflows/download_models.py for the ltx bundle, then retry.")
        width = int(controls.get("width", 1280))
        height = int(controls.get("height", 720))
        if width % 8 or height % 8 or not 256 <= width <= 2160 or not 256 <= height <= 2160:
            raise ValueError("Width and height must be multiples of 8 between 256 and 2160.")
        fps = int(controls.get("fps", 24))
        if not 1 <= fps <= 60:
            raise ValueError("Output FPS must be an integer from 1 through 60.")
        duration = int(controls.get("duration", 5))
        if not 1 <= duration <= 40:
            raise ValueError("Duration must be a whole second from 1 through 40.")
        frames = duration * fps + 1
        if frames > MAX_FRAMES:
            raise ValueError(
                f"{duration}s @ {fps} FPS is {frames} frames, above the LTX-2.5 audio latent "
                f"ceiling of {MAX_FRAMES}; keep it at or below {(MAX_FRAMES - 1) // fps} seconds "
                f"at this frame rate."
            )
        seed = int(controls.get("seed", 42))
        if not 0 <= seed <= 2**64 - 1:
            raise ValueError("Seed must be between 0 and 18446744073709551615.")
        container = str(controls.get("container", "mp4"))
        codec = str(controls.get("codec", "h264"))
        if container not in {"mp4", "mkv", "webm"}:
            raise ValueError("Container must be mp4, mkv, or webm.")
        if codec not in {"h264", "av1", "auto"}:
            raise ValueError("Codec must be h264, av1, or auto.")
        if container == "webm" and codec == "h264":
            raise ValueError("WebM delivery needs AV1 or auto codec selection.")
        if bool(controls.get("prompt_enhance", False)) and not (
            self.models / "text_encoders" / ENHANCER_ENCODER
        ).is_file():
            raise ValueError("Prompt enhancement needs the E2B enhancer encoder from the workflows ltx bundle.")
        if mode in {"t2v", "i2v"} and not (
            self.models / "latent_upscale_models" / SPATIAL_UPSCALER
        ).is_file():
            raise ValueError("Text/image generation needs the x2 spatial upscaler from the workflows ltx bundle.")
        if mode == "i2v":
            resolve_asset(str(controls.get("first_frame_asset", "")))
        elif mode == "flf2v":
            resolve_asset(str(controls.get("first_frame_asset", "")))
            resolve_asset(str(controls.get("last_frame_asset", "")))
        prompt = str(controls.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Describe the clip before generating.")

    @staticmethod
    def _delivery_size(mode: str, width: int, height: int) -> tuple[int, int]:
        """Official canvas rule: T2V/I2V generate at half resolution and run the
        x2 latent spatial upscaler; FLF2V generates at full resolution directly."""
        if mode == "flf2v":
            return (width // 32) * 32, (height // 32) * 32
        return ((width // 2) // 32) * 32 * 2, ((height // 2) // 32) * 32 * 2

    def _loaders(self, add: Callable[..., str], variant: str) -> dict[str, list[Any]]:
        model_name = MODEL_NVFP4 if variant == "nvfp4" else MODEL_INT8
        return {
            "model": add("UNETLoader", unet_name=model_name, weight_dtype="default"),
            "clip": add("CLIPLoader", clip_name=TEXT_ENCODER, type="ltxv"),
            "vae": add("VAELoader", vae_name=VIDEO_VAE),
            "audio_vae": add("VAELoader", vae_name=AUDIO_VAE),
        }

    def _prompt_conditioning(
        self,
        add: Callable[..., str],
        loaders: dict[str, list[Any]],
        prompt: str,
        negative: str,
        image_ref: list[Any] | None,
        controls: dict[str, Any],
    ) -> tuple[list[Any], list[Any]]:
        enhance = bool(controls.get("prompt_enhance", False))
        if enhance:
            enhancer_clip = add("CLIPLoader", clip_name=ENHANCER_ENCODER, type="ltxv")
            kwargs: dict[str, Any] = {
                "clip": [enhancer_clip, 0],
                "prompt": prompt,
                "max_length": 600,
                "sampling_mode": "on",
                "sampling_mode.temperature": 0.7,
                "sampling_mode.top_k": 64,
                "sampling_mode.top_p": 0.95,
                "sampling_mode.min_p": 0.05,
                "sampling_mode.repetition_penalty": 1.15,
                "sampling_mode.seed": 0,
                "sampling_mode.presence_penalty": 0,
                "thinking": False,
                "use_default_template": True,
            }
            if image_ref is not None:
                kwargs["image"] = image_ref
            text_ref: Any = [add("TextGenerateLTX2Prompt", **kwargs), 0]
        else:
            text_ref = prompt
        positive = add("CLIPTextEncode", clip=[loaders["clip"], 0], text=text_ref)
        negative = add("CLIPTextEncode", clip=[loaders["clip"], 0], text=negative)
        return positive, negative

    def _graph_t2v(
        self,
        *,
        add: Callable[..., str],
        loaders: dict[str, list[Any]],
        width: int,
        height: int,
        frames: int,
        fps: int,
        seed: int,
        controls: dict[str, Any],
        prefix: str,
    ) -> dict[str, Any]:
        graph: dict[str, Any] = {}
        positive, negative = self._prompt_conditioning(
            add, loaders, str(controls.get("prompt", "")), NEGATIVE_T2V, None, controls,
        )
        conditioning = add("LTXVConditioning", positive=[positive, 0], negative=[negative, 0], frame_rate=float(fps))
        empty = add(
            "EmptyLTXVLatentVideo",
            width=width // 2, height=height // 2, length=frames, batch_size=1,
        )
        audio = add(
            "LTXVEmptyLatentAudio",
            frames_number=frames, frame_rate=float(fps), batch_size=1, audio_vae=[loaders["audio_vae"], 0],
        )
        base = add("LTXVConcatAVLatent", video_latent=[empty, 0], audio_latent=[audio, 0])
        guider = add(
            "LTXVDualCFGGuider",
            model=[loaders["model"], 0], positive=[conditioning, 0], negative=[conditioning, 1],
            video_cfg=1, audio_cfg=1,
        )
        sampler = add("KSamplerSelect", sampler_name="euler_ancestral")
        sigmas1 = add("ManualSigmas", sigmas=SIGMAS_STAGE1)
        stage1 = add(
            "SamplerCustomAdvanced",
            noise=[add("RandomNoise", seed=seed), 0],
            guider=[guider, 0], sampler=[sampler, 0], sigmas=[sigmas1, 0],
            latent_image=[base, 0],
        )
        separated = add("LTXVSeparateAVLatent", av_latent=[stage1, 0])
        ups_model = add("LatentUpscaleModelLoader", model_name=SPATIAL_UPSCALER)
        upsampled = add(
            "LTXVLatentUpsampler",
            samples=[separated, 0], upscale_model=[ups_model, 0], vae=[loaders["vae"], 0],
        )
        refined = add("LTXVConcatAVLatent", video_latent=[upsampled, 0], audio_latent=[separated, 1])
        sigmas2 = add("ManualSigmas", sigmas=SIGMAS_STAGE2)
        stage2 = add(
            "SamplerCustomAdvanced",
            noise=[add("RandomNoise", seed=42), 0],
            guider=[guider, 0], sampler=[sampler, 0], sigmas=[sigmas2, 0],
            latent_image=[refined, 0],
        )
        final = add("LTXVSeparateAVLatent", av_latent=[stage2, 0])
        images = add(
            "VAEDecodeTiled",
            samples=[final, 0], vae=[loaders["vae"], 0],
            tile_size=512, overlap=64, temporal_size=64, temporal_overlap=16,
        )
        sound = add("LTXVAudioVAEDecode", samples=[final, 1], audio_vae=[loaders["audio_vae"], 0])
        video = add(
            "CreateVideo",
            images=[images, 0], fps=float(fps), audio=[sound, 0],
            bit_depth=int(controls.get("bit_depth", 8)), color_space=str(controls.get("color_space", "sRGB")),
        )
        graph["save"] = _save_video(
            [video, 0], prefix, str(controls.get("container", "mp4")),
            str(controls.get("codec", "h264")), int(controls.get("crf", 18)),
        )
        return graph

    def _graph_i2v(
        self,
        *,
        add: Callable[..., str],
        loaders: dict[str, list[Any]],
        first_frame_name: str,
        width: int,
        height: int,
        frames: int,
        fps: int,
        seed: int,
        controls: dict[str, Any],
        prefix: str,
    ) -> dict[str, Any]:
        graph: dict[str, Any] = {}
        loaded = add("LoadImage", image=first_frame_name)
        resized = add(
            "ResizeImageMaskNode",
            input=[loaded, 0], resize_type="scale longer dimension",
            **{"resize_type.longer_size": 1536, "scale_method": "lanczos"},
        )
        prepared = add("LTXVPreprocess", image=[resized, 0], img_compression=18)
        positive, negative = self._prompt_conditioning(
            add, loaders, str(controls.get("prompt", "")), NEGATIVE_T2V, [prepared, 0], controls,
        )
        conditioning = add("LTXVConditioning", positive=[positive, 0], negative=[negative, 0], frame_rate=float(fps))
        empty = add(
            "EmptyLTXVLatentVideo",
            width=width // 2, height=height // 2, length=frames, batch_size=1,
        )
        audio = add(
            "LTXVEmptyLatentAudio",
            frames_number=frames, frame_rate=float(fps), batch_size=1, audio_vae=[loaders["audio_vae"], 0],
        )
        guider = add(
            "LTXVDualCFGGuider",
            model=[loaders["model"], 0], positive=[conditioning, 0], negative=[conditioning, 1],
            video_cfg=1, audio_cfg=1,
        )
        sampler = add("KSamplerSelect", sampler_name="euler_ancestral")
        base = add(
            "LTXVImgToVideoInplace",
            vae=[loaders["vae"], 0], image=[prepared, 0], latent=[empty, 0], strength=0.7, bypass=False,
        )
        base_av = add("LTXVConcatAVLatent", video_latent=[base, 0], audio_latent=[audio, 0])
        sigmas1 = add("ManualSigmas", sigmas=SIGMAS_STAGE1)
        stage1 = add(
            "SamplerCustomAdvanced",
            noise=[add("RandomNoise", seed=seed), 0],
            guider=[guider, 0], sampler=[sampler, 0], sigmas=[sigmas1, 0],
            latent_image=[base_av, 0],
        )
        separated = add("LTXVSeparateAVLatent", av_latent=[stage1, 0])
        ups_model = add("LatentUpscaleModelLoader", model_name=SPATIAL_UPSCALER)
        upsampled = add(
            "LTXVLatentUpsampler",
            samples=[separated, 0], upscale_model=[ups_model, 0], vae=[loaders["vae"], 0],
        )
        refined = add(
            "LTXVImgToVideoInplace",
            vae=[loaders["vae"], 0], image=[prepared, 0], latent=[upsampled, 0], strength=1.0, bypass=False,
        )
        refined_av = add("LTXVConcatAVLatent", video_latent=[refined, 0], audio_latent=[separated, 1])
        sigmas2 = add("ManualSigmas", sigmas=SIGMAS_STAGE2)
        stage2 = add(
            "SamplerCustomAdvanced",
            noise=[add("RandomNoise", seed=42), 0],
            guider=[guider, 0], sampler=[sampler, 0], sigmas=[sigmas2, 0],
            latent_image=[refined_av, 0],
        )
        final = add("LTXVSeparateAVLatent", av_latent=[stage2, 0])
        images = add(
            "VAEDecodeTiled",
            samples=[final, 0], vae=[loaders["vae"], 0],
            tile_size=512, overlap=64, temporal_size=64, temporal_overlap=16,
        )
        sound = add("LTXVAudioVAEDecode", samples=[final, 1], audio_vae=[loaders["audio_vae"], 0])
        video = add(
            "CreateVideo",
            images=[images, 0], fps=float(fps), audio=[sound, 0],
            bit_depth=int(controls.get("bit_depth", 8)), color_space=str(controls.get("color_space", "sRGB")),
        )
        graph["save"] = _save_video(
            [video, 0], prefix, str(controls.get("container", "mp4")),
            str(controls.get("codec", "h264")), int(controls.get("crf", 18)),
        )
        return graph

    def _graph_flf2v(
        self,
        *,
        add: Callable[..., str],
        loaders: dict[str, list[Any]],
        first_frame_name: str,
        last_frame_name: str,
        width: int,
        height: int,
        frames: int,
        fps: int,
        seed: int,
        controls: dict[str, Any],
        prefix: str,
    ) -> dict[str, Any]:
        graph: dict[str, Any] = {}
        first = add(
            "ResizeImageMaskNode",
            input=[add("LoadImage", image=first_frame_name), 0],
            resize_type="scale dimensions",
            **{
                "resize_type.width": width, "resize_type.height": height,
                "resize_type.crop": "center", "scale_method": "nearest-exact",
            },
        )
        last = add(
            "ResizeImageMaskNode",
            input=[add("LoadImage", image=last_frame_name), 0],
            resize_type="scale dimensions",
            **{
                "resize_type.width": width, "resize_type.height": height,
                "resize_type.crop": "center", "scale_method": "nearest-exact",
            },
        )
        first_prep = add("LTXVPreprocess", image=[first, 0], img_compression=18)
        last_prep = add("LTXVPreprocess", image=[last, 0], img_compression=18)
        positive, negative = self._prompt_conditioning(
            add, loaders, str(controls.get("prompt", "")), NEGATIVE_FLF2V, [first_prep, 0], controls,
        )
        conditioning = add("LTXVConditioning", positive=[positive, 0], negative=[negative, 0], frame_rate=float(fps))
        empty = add(
            "EmptyLTXVLatentVideo",
            width=width, height=height, length=frames, batch_size=1,
        )
        guide_first = add(
            "LTXVAddGuide",
            positive=[conditioning, 0], negative=[conditioning, 1], vae=[loaders["vae"], 0],
            latent=[empty, 0], image=[first_prep, 0], frame_idx=0, strength=0.7,
        )
        guide_last = add(
            "LTXVAddGuide",
            positive=[guide_first, 0], negative=[guide_first, 1], vae=[loaders["vae"], 0],
            latent=[guide_first, 2], image=[last_prep, 0], frame_idx=-1, strength=0.7,
        )
        audio = add(
            "LTXVEmptyLatentAudio",
            frames_number=frames, frame_rate=float(fps), batch_size=1, audio_vae=[loaders["audio_vae"], 0],
        )
        base = add("LTXVConcatAVLatent", video_latent=[guide_last, 2], audio_latent=[audio, 0])
        guider = add(
            "LTXVDualCFGGuider",
            model=[loaders["model"], 0], positive=[guide_last, 0], negative=[guide_last, 1],
            video_cfg=1, audio_cfg=1,
        )
        sampler = add("SamplerEulerAncestral", eta=0, s_noise=1)
        sigmas1 = add("ManualSigmas", sigmas=SIGMAS_STAGE1)
        sampled = add(
            "SamplerCustomAdvanced",
            noise=[add("RandomNoise", seed=seed), 0],
            guider=[guider, 0], sampler=[sampler, 0], sigmas=[sigmas1, 0],
            latent_image=[base, 0],
        )
        separated = add("LTXVSeparateAVLatent", av_latent=[sampled, 1])
        cropped = add(
            "LTXVCropGuides",
            positive=[guide_last, 0], negative=[guide_last, 1], latent=[separated, 0],
        )
        images = add(
            "VAEDecodeTiled",
            samples=[cropped, 2], vae=[loaders["vae"], 0],
            tile_size=512, overlap=64, temporal_size=64, temporal_overlap=16,
        )
        sound = add("LTXVAudioVAEDecode", samples=[separated, 1], audio_vae=[loaders["audio_vae"], 0])
        video = add(
            "CreateVideo",
            images=[images, 0], fps=float(fps), audio=[sound, 0],
            bit_depth=int(controls.get("bit_depth", 8)), color_space=str(controls.get("color_space", "sRGB")),
        )
        graph["save"] = _save_video(
            [video, 0], prefix, str(controls.get("container", "mp4")),
            str(controls.get("codec", "h264")), int(controls.get("crf", 18)),
        )
        return graph

    @staticmethod
    def _output(path: Path, label: str, metadata: dict[str, Any] | None = None) -> StudioOutput:
        media = {".mp4": "video/mp4", ".mkv": "video/x-matroska", ".webm": "video/webm"}.get(
            path.suffix.lower(), "video/mp4"
        )
        return StudioOutput(path, "video", label, media, metadata or {})

    @staticmethod
    def _copy_video(source: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    def run(self, request: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        mode = str(request["mode"])
        controls = request.get("controls") or {}
        width = int(controls.get("width", 1280))
        height = int(controls.get("height", 720))
        fps = int(controls.get("fps", 24))
        duration = int(controls.get("duration", 5))
        frames = duration * fps + 1
        seed = int(controls.get("seed", 42))
        variant = str(controls.get("weights_variant", "int8"))
        delivery_w, delivery_h = self._delivery_size(mode, width, height)
        context.log(
            f"{mode.upper()} · {width}x{height} canvas · {duration}s @ {fps} FPS = {frames} frames · "
            f"{variant} weights · delivery {delivery_w}x{delivery_h}"
        )
        with ComfyRuntime(
            python=self.python, comfy_root=self.comfy, runtime_root=self.runtime_root, context=context,
        ) as runtime:
            first_name = last_name = ""
            if mode in {"i2v", "flf2v"}:
                first_name = runtime.add_input(
                    context.asset(str(controls["first_frame_asset"])),
                    f"first-frame-{context.job_id}",
                )
            if mode == "flf2v":
                last_name = runtime.add_input(
                    context.asset(str(controls["last_frame_asset"])),
                    f"last-frame-{context.job_id}",
                )
            graph: dict[str, Any] = {}
            counter = {"next": 0}

            def add(class_type: str, **inputs: Any) -> str:
                counter["next"] += 1
                node_id = str(counter["next"])
                graph[node_id] = _node(class_type, **inputs)
                return node_id

            loaders = self._loaders(add, variant)
            shared = dict(width=width, height=height, frames=frames, fps=fps, seed=seed, controls=controls, prefix=f"ltx/{context.job_id}")
            if mode == "t2v":
                graph.update(self._graph_t2v(add=add, loaders=loaders, **shared))
            elif mode == "i2v":
                graph.update(self._graph_i2v(add=add, loaders=loaders, first_frame_name=first_name, **shared))
            else:
                graph.update(self._graph_flf2v(add=add, loaders=loaders, first_frame_name=first_name, last_frame_name=last_name, **shared))
            before = set(runtime.output_files())
            context.update(f"Running the LTX-2.5 distilled recipe ({mode})", 0.35)
            runtime.execute(graph, stage="Sampling the two-stage distilled recipe")
            created = [
                path for path in runtime.output_files()
                if path not in before and path.suffix.lower() in {".mp4", ".mkv", ".webm"}
            ]
            if not created:
                raise RuntimeError("Comfy finished without an LTX-2.5 video artifact.")
        generated = self._copy_video(created[-1], context.output_dir / f"generated_video{created[-1].suffix.lower()}")
        settings = context.output_dir / "run_config.json"
        settings.write_text(
            json.dumps({
                "request": request,
                "frames": frames,
                "canvas": [width, height],
                "delivery": [delivery_w, delivery_h],
                "weights_variant": variant,
            }, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        outputs = [
            self._output(generated, "Generated clip with synchronized audio", {
                "frames": frames,
                "delivery": f"{delivery_w}x{delivery_h}",
                "model": MODEL_NVFP4 if variant == "nvfp4" else MODEL_INT8,
            }),
        ]
        outputs.append(StudioOutput(settings, "document", "Resolved run configuration", "application/json"))
        context.update("LTX-2.5 clip ready", 0.99)
        return outputs
