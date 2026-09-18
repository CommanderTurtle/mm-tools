from __future__ import annotations

import gc
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from studio.runtime import StudioAdapter, StudioContext, StudioOutput, gpu_snapshot


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MESH_SUFFIXES = {".glb", ".gltf", ".obj", ".ply", ".stl", ".off"}


def _load_mesh(path: Path) -> Any:
    import trimesh

    loaded = trimesh.load(path, force=None, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("The uploaded scene contains no triangle mesh.")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError("A non-empty triangle mesh is required.")
    return loaded


class Adapter(StudioAdapter):
    def __init__(self, project_root: Path, runtime_root: Path) -> None:
        super().__init__(project_root, runtime_root)
        self.trellis_root = project_root / "pretrained" / "TRELLIS.2-4B"
        self.pixal_root = project_root / "world" / "pretrained" / "Pixal3D"
        self.deps = project_root / "pretrained" / "deps"
        self.dino = self.deps / "camenduru--dinov3-vitl16-pretrain-lvd1689m"
        self.biref = self.deps / "ZhengPeng7--BiRefNet"
        self.moge = self.deps / "Ruicheng--moge-2-vitl" / "model.pt"
        self.naf = self.deps / "valeoai--NAF" / "naf_release.pth"
        self.native = project_root / "native"
        self.pipeline: Any | None = None
        self.profile: str | None = None
        self._localize_configs()

    def health(self) -> dict[str, Any]:
        checks = [
            ("TRELLIS.2-4B", self.trellis_root / "pipeline.json", True),
            ("Pixal3D", self.pixal_root / "pipeline.json", True),
            ("DINOv3 ViT-L/16", self.dino / "model.safetensors", True),
            ("BiRefNet matte", self.biref / "model.safetensors", True),
            ("MoGe-2 camera calibration", self.moge, True),
            ("NAF feature upsampler", self.naf, True),
            ("NAF source", self.native / "NAF" / "hubconf.py", True),
            ("MoGe source", self.native / "MoGe" / "moge" / "model" / "v2.py", True),
            ("Python environment", self.project_root / ".venv" / "bin" / "python", True),
        ]
        details = [
            {"label": label, "ready": path.is_file(), "required": required, "path": str(path)}
            for label, path, required in checks
        ]
        gpu = gpu_snapshot()
        device = gpu.get("devices", [{}])[0] if gpu.get("available") else {}
        gpu_ok = bool(device) and int(device.get("memory_total_mib", 0)) >= 30000
        details.append({"label": "CUDA GPU >= 30,000 MiB", "ready": gpu_ok, "value": device.get("name", "unavailable")})
        return {
            "ready": all(item[1].is_file() for item in checks if item[2]) and gpu_ok,
            "loaded": self.pipeline is not None,
            "profile": self.profile,
            "details": details,
        }

    def validate(self, request: dict[str, Any], resolve_asset: Callable[[str], Path]) -> None:
        mode = str(request.get("mode", ""))
        controls = request.get("controls") or {}
        if mode not in {"trellis_generate", "pixal_generate", "retexture", "mesh_lab"}:
            raise ValueError(f"Unknown sculpting workflow: {mode}")
        if mode in {"trellis_generate", "pixal_generate", "retexture"}:
            image = resolve_asset(str(controls.get("image_asset", "")))
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError("Upload a PNG, JPEG, WebP, BMP, or TIFF reference image.")
        if mode in {"retexture", "mesh_lab"}:
            mesh = resolve_asset(str(controls.get("mesh_asset", "")))
            if mesh.suffix.lower() not in MESH_SUFFIXES:
                raise ValueError("Upload a GLB, GLTF, OBJ, PLY, STL, or OFF triangle mesh.")
        if mode in {"trellis_generate", "pixal_generate"}:
            takes = int(controls.get("takes", 1))
            if not 1 <= takes <= 4:
                raise ValueError("Generate between one and four sequential candidates.")
            pipeline_type = str(controls.get("pipeline_type", "1024_cascade"))
            if mode == "pixal_generate" and pipeline_type not in {"1024_cascade", "1536_cascade"}:
                raise ValueError("Pixal3D supports only 1024 and 1536 cascade profiles.")
            if pipeline_type == "1536_cascade":
                total = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
                if total < 40 * 1024**3:
                    raise ValueError(
                        "1536 cascade cannot remain fully GPU-resident on a 32 GiB card. "
                        "Use 1024 cascade; the prohibited CPU-shuttling workaround is intentionally absent."
                    )
            if int(controls.get("max_num_tokens", 49152)) > 65536:
                raise ValueError("The verified sparse-token ceiling is 65,536.")
        if mode == "pixal_generate":
            camera_mode = str(controls.get("camera_mode", "auto"))
            if camera_mode == "manual":
                fov = float(controls.get("fov_degrees", 49.1))
                if not 5 <= fov <= 120:
                    raise ValueError("Manual horizontal FOV must be between 5° and 120°.")

    def load(self) -> dict[str, Any]:
        self._ensure_trellis("generate")
        return self.health()

    def unload(self) -> dict[str, Any]:
        self._release()
        return self.health()

    def close(self) -> None:
        self._release()

    def run(self, request: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        mode = request["mode"]
        controls = request.get("controls") or {}
        if mode == "mesh_lab":
            return self._mesh_lab(controls, context)
        if mode == "retexture":
            return self._retexture(controls, context)
        if mode == "pixal_generate":
            return self._generate_pixal(controls, context)
        return self._generate_trellis(controls, context)

    def _localize_configs(self) -> None:
        for root, names in (
            (self.trellis_root, ("pipeline.json", "texturing_pipeline.json")),
            (self.pixal_root, ("pipeline.json",)),
        ):
            for name in names:
                source = root / name
                if not source.is_file():
                    continue
                config = json.loads(source.read_text(encoding="utf-8"))
                args = config["args"]
                models = args.get("models", {})
                if "sparse_structure_decoder" in models:
                    models["sparse_structure_decoder"] = "ckpts/ss_dec_conv3d_16l8_fp16"
                image_cond = args.get("image_cond_model")
                if image_cond:
                    image_cond.setdefault("args", {})["model_name"] = str(self.dino)
                rembg = args.get("rembg_model")
                if rembg:
                    rembg.setdefault("args", {})["model_name"] = str(self.biref)
                args["low_vram"] = False
                destination = root / name.replace(".json", ".local.json")
                destination.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    def _release(self) -> None:
        self.pipeline = None
        self.profile = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _ensure_trellis(self, purpose: str) -> Any:
        wanted = f"trellis:{purpose}"
        if self.pipeline is not None and self.profile != wanted:
            self._release()
        if self.pipeline is None:
            if purpose == "texture":
                from trellis2.pipelines import Trellis2TexturingPipeline

                pipe = Trellis2TexturingPipeline.from_pretrained(
                    str(self.trellis_root), config_file="texturing_pipeline.local.json"
                )
            else:
                from trellis2.pipelines import Trellis2ImageTo3DPipeline

                pipe = Trellis2ImageTo3DPipeline.from_pretrained(
                    str(self.trellis_root), config_file="pipeline.local.json"
                )
            pipe.low_vram = False
            pipe.cuda()
            self.pipeline = pipe
            self.profile = wanted
        return self.pipeline

    def _load_naf(self) -> Any:
        root = self.native / "NAF"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("mm_naf_hub", root / "hubconf.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Pinned NAF source could not be imported.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.naf(pretrained=False, device="cuda")
        model.load_state_dict(torch.load(self.naf, map_location="cuda", weights_only=True))
        model.eval().requires_grad_(False)
        return model

    def _ensure_pixal(self) -> Any:
        if self.pipeline is not None and self.profile != "pixal":
            self._release()
        if self.pipeline is not None:
            return self.pipeline
        world_root = self.project_root / "world"
        if str(world_root) not in sys.path:
            sys.path.insert(0, str(world_root))
        from pixal3d.pipelines import Pixal3DImageTo3DPipeline
        from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor

        pipe = Pixal3DImageTo3DPipeline.from_pretrained(str(self.pixal_root), config_file="pipeline.local.json")
        configs = {
            "image_cond_model_ss": dict(model_name=str(self.dino), image_size=512, grid_resolution=16),
            "image_cond_model_shape_512": dict(model_name=str(self.dino), image_size=512, grid_resolution=32, use_naf_upsample=True, naf_target_size=512),
            "image_cond_model_shape_1024": dict(model_name=str(self.dino), image_size=1024, grid_resolution=64, use_naf_upsample=True, naf_target_size=512),
            "image_cond_model_tex_1024": dict(model_name=str(self.dino), image_size=1024, grid_resolution=64, use_naf_upsample=True, naf_target_size=1024),
        }
        for attribute, kwargs in configs.items():
            context_model = DinoV3ProjFeatureExtractor(**kwargs).eval().requires_grad_(False)
            if kwargs.get("use_naf_upsample"):
                context_model.naf_model = self._load_naf()
            setattr(pipe, attribute, context_model)
        pipe.low_vram = False
        pipe.cuda()
        for attribute in configs:
            getattr(pipe, attribute).cuda()
        self.pipeline = pipe
        self.profile = "pixal"
        return pipe

    @staticmethod
    def _samplers(controls: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        common = {"guidance_interval": [float(controls.get("guide_start", 0.6)), float(controls.get("guide_end", 1.0))]}
        sparse = {
            **common,
            "steps": int(controls.get("ss_steps", 12)),
            "guidance_strength": float(controls.get("ss_guidance", 7.5)),
            "guidance_rescale": float(controls.get("ss_rescale", 0.7)),
            "rescale_t": float(controls.get("ss_rescale_t", 5.0)),
        }
        shape = {
            **common,
            "steps": int(controls.get("shape_steps", 12)),
            "guidance_strength": float(controls.get("shape_guidance", 7.5)),
            "guidance_rescale": float(controls.get("shape_rescale", 0.5)),
            "rescale_t": float(controls.get("shape_rescale_t", 3.0)),
        }
        texture = {
            "steps": int(controls.get("texture_steps", 12)),
            "guidance_strength": float(controls.get("texture_guidance", 1.0)),
            "guidance_rescale": float(controls.get("texture_rescale", 0.0)),
            "guidance_interval": [float(controls.get("texture_guide_start", 0.6)), float(controls.get("texture_guide_end", 0.9))],
            "rescale_t": float(controls.get("texture_rescale_t", 3.0)),
        }
        return sparse, shape, texture

    def _generate_trellis(self, controls: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        source = context.asset(str(controls["image_asset"]))
        pipe = self._ensure_trellis("generate")
        image = Image.open(source).convert("RGBA" if source.suffix.lower() == ".png" else "RGB")
        context.update("Preparing the object matte", 0.04)
        prepared = pipe.preprocess_image(image) if bool(controls.get("remove_background", True)) else image.convert("RGB")
        prepared_path = context.output_dir / "prepared-reference.png"
        prepared.save(prepared_path)
        sparse, shape, texture = self._samplers(controls)
        takes = int(controls.get("takes", 1))
        seed = int(controls.get("seed", 42))
        seed_step = int(controls.get("seed_step", 1))
        outputs: list[StudioOutput] = [StudioOutput(prepared_path, "image", "Prepared reference", "image/png")]
        for index in range(takes):
            context.check_cancelled()
            context.update(f"Candidate {index + 1}/{takes}: generating sparse shape and PBR volume", 0.08 + index / takes * 0.62)
            meshes = pipe.run(
                prepared,
                num_samples=1,
                seed=seed + index * seed_step,
                sparse_structure_sampler_params=sparse,
                shape_slat_sampler_params=shape,
                tex_slat_sampler_params=texture,
                preprocess_image=False,
                pipeline_type=str(controls.get("pipeline_type", "1024_cascade")),
                max_num_tokens=int(controls.get("max_num_tokens", 49152)),
            )
            outputs.extend(self._export_mesh(meshes[0], controls, context, f"candidate-{index + 1:02d}", pipe))
        context.update("PBR assets ready", 0.99)
        return outputs

    def _matte_for_camera(self, source: Path, destination: Path) -> Image.Image:
        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms

        image = Image.open(source).convert("RGB")
        model = AutoModelForImageSegmentation.from_pretrained(
            str(self.biref), trust_remote_code=True, local_files_only=True
        ).eval().to("cuda")
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        with torch.inference_mode():
            mask = model(transform(image).unsqueeze(0).to("cuda"))[-1].sigmoid()[0].squeeze().float().cpu()
        alpha = transforms.ToPILImage()(mask).resize(image.size)
        rgba = image.copy()
        rgba.putalpha(alpha)
        box = alpha.getbbox()
        if box:
            left, top, right, bottom = box
            side = max(right - left, bottom - top)
            cx, cy = (left + right) / 2, (top + bottom) / 2
            pad = int(side * 0.08)
            rgba = rgba.crop((int(cx - side / 2 - pad), int(cy - side / 2 - pad), int(cx + side / 2 + pad), int(cy + side / 2 + pad)))
        background = Image.new("RGB", rgba.size, (0, 0, 0))
        background.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
        background.save(destination)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return background

    def _camera(self, prepared: Path, controls: dict[str, Any], context: StudioContext) -> dict[str, float]:
        mesh_scale = float(controls.get("mesh_scale", 1.0))
        resolution = int(controls.get("camera_resolution", 512))
        extend = int(controls.get("frame_extension", 0))
        if controls.get("camera_mode", "auto") == "manual":
            angle = math.radians(float(controls.get("fov_degrees", 49.1)))
        else:
            context.update("Estimating camera intrinsics with MoGe-2", 0.08)
            moge_root = self.native / "MoGe"
            if str(moge_root) not in sys.path:
                sys.path.insert(0, str(moge_root))
            from moge.model.v2 import MoGeModel

            model = MoGeModel.from_pretrained(self.moge).eval().to("cuda")
            array = np.asarray(Image.open(prepared).convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).to("cuda")
            with torch.inference_mode():
                result = model.infer(tensor)
            fx_normalized = float(result["intrinsics"].squeeze()[0, 0].detach().cpu())
            angle = 2 * math.atan(1 / max(2 * fx_normalized, 1e-6))
            del model, result, tensor
            gc.collect()
            torch.cuda.empty_cache()
        f_pixels = (16.0 / math.tan(angle / 2.0)) * resolution / 32.0
        xw, yw = -0.5 / mesh_scale, 0.0
        x_ndc = -resolution / 2.0 - extend
        distance = f_pixels * xw / x_ndc - yw
        manual_distance = controls.get("camera_distance")
        if controls.get("camera_mode") == "manual" and manual_distance not in {None, "", 0, 0.0}:
            distance = float(manual_distance)
        return {"camera_angle_x": angle, "distance": distance, "mesh_scale": mesh_scale}

    def _generate_pixal(self, controls: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        self._release()
        source = context.asset(str(controls["image_asset"]))
        prepared_path = context.output_dir / "camera-normalized-reference.png"
        context.update("Extracting the local object matte", 0.02)
        prepared = self._matte_for_camera(source, prepared_path)
        camera = self._camera(prepared_path, controls, context)
        camera_file = context.output_dir / "camera.json"
        camera_file.write_text(json.dumps({**camera, "fov_degrees": math.degrees(camera["camera_angle_x"])}, indent=2) + "\n", encoding="utf-8")
        context.update("Loading the camera-aware Pixal3D stack", 0.13)
        pipe = self._ensure_pixal()
        sparse, shape, texture = self._samplers(controls)
        takes = int(controls.get("takes", 1))
        seed = int(controls.get("seed", 42))
        step = int(controls.get("seed_step", 1))
        outputs: list[StudioOutput] = [
            StudioOutput(prepared_path, "image", "Camera-normalized reference", "image/png"),
            StudioOutput(camera_file, "document", "Recovered camera", "application/json"),
        ]
        for index in range(takes):
            context.check_cancelled()
            context.update(f"Candidate {index + 1}/{takes}: view-aligned sparse reconstruction", 0.18 + index / takes * 0.58)
            meshes = pipe.run(
                prepared, camera_params=camera, num_samples=1, seed=seed + index * step,
                sparse_structure_sampler_params=sparse, shape_slat_sampler_params=shape,
                tex_slat_sampler_params=texture, preprocess_image=False,
                pipeline_type=str(controls.get("pipeline_type", "1024_cascade")),
                max_num_tokens=int(controls.get("max_num_tokens", 49152)),
            )
            outputs.extend(self._export_mesh(meshes[0], controls, context, f"pixal-{index + 1:02d}", pipe))
        context.update("Camera-faithful PBR assets ready", 0.99)
        return outputs

    def _export_mesh(self, mesh: Any, controls: dict[str, Any], context: StudioContext, stem: str, pipe: Any) -> list[StudioOutput]:
        import imageio
        import o_voxel

        context.update(f"Baking {stem} PBR textures and mesh", min(0.94, 0.72 + 0.04))
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
            coords=mesh.coords, attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=int(controls.get("decimation_target", 300000)),
            texture_size=int(controls.get("texture_size", 2048)),
            remesh=bool(controls.get("remesh", True)),
            remesh_band=float(controls.get("remesh_band", 1.0)),
            remesh_project=float(controls.get("remesh_project", 0.0)),
            mesh_cluster_refine_iterations=int(controls.get("uv_refine_iterations", 0)),
            mesh_cluster_global_iterations=int(controls.get("uv_global_iterations", 1)),
            mesh_cluster_smooth_strength=float(controls.get("uv_smooth_strength", 1.0)),
            use_tqdm=False, verbose=False,
        )
        model_path = context.output_dir / f"{stem}.glb"
        glb.export(model_path, extension_webp=bool(controls.get("webp_textures", True)))
        outputs = [StudioOutput(model_path, "model", f"{stem} · PBR GLB", "model/gltf-binary", {"role": "interactive-3d"})]
        if bool(controls.get("export_ply", False)):
            ply = context.output_dir / f"{stem}.ply"
            glb.export(ply)
            outputs.append(StudioOutput(ply, "model", f"{stem} · PLY", "application/octet-stream"))
        if bool(controls.get("render_turntable", True)):
            from trellis2.renderers import EnvMap
            from trellis2.utils import render_utils

            env = EnvMap(self._environment(str(controls.get("environment", "studio"))))
            rendered = render_utils.render_video(
                mesh,
                resolution=int(controls.get("preview_resolution", 512)),
                num_frames=int(controls.get("turntable_frames", 72)),
                r=float(controls.get("orbit_radius", 2.0)),
                fov=float(controls.get("preview_fov", 40)),
                envmap=env,
                use_envmap_bg=bool(controls.get("environment_background", False)),
            )
            frames = render_utils.make_pbr_vis_frames(rendered) if controls.get("preview_pass", "beauty") == "pbr_board" else rendered["shaded"]
            video = context.output_dir / f"{stem}-turntable.mp4"
            imageio.mimsave(video, frames, fps=int(controls.get("preview_fps", 18)), macro_block_size=1)
            outputs.append(StudioOutput(video, "video", f"{stem} · turntable", "video/mp4"))
            for view in range(min(int(controls.get("snapshot_count", 4)), len(rendered["shaded"]))):
                index = round(view * (len(rendered["shaded"]) - 1) / max(int(controls.get("snapshot_count", 4)) - 1, 1))
                shot = context.output_dir / f"{stem}-view-{view + 1:02d}.png"
                Image.fromarray(rendered["shaded"][index]).save(shot)
                outputs.append(StudioOutput(shot, "image", f"{stem} · view {view + 1}", "image/png"))
        receipt = context.output_dir / f"{stem}-settings.json"
        receipt.write_text(json.dumps({"controls": controls, "pipeline": self.profile}, indent=2, default=str) + "\n", encoding="utf-8")
        outputs.append(StudioOutput(receipt, "document", f"{stem} · settings", "application/json"))
        return outputs

    @staticmethod
    def _environment(name: str) -> torch.Tensor:
        height, width = 256, 512
        y, x = torch.meshgrid(torch.linspace(0, 1, height, device="cuda"), torch.linspace(0, 1, width, device="cuda"), indexing="ij")
        palettes = {
            "studio": ((0.05, 0.06, 0.09), (1.25, 1.10, 0.92), 0.13, 0.27),
            "overcast": ((0.20, 0.24, 0.30), (0.75, 0.82, 0.90), 0.45, 0.50),
            "sunset": ((0.06, 0.04, 0.12), (1.90, 0.52, 0.18), 0.77, 0.38),
            "gallery": ((0.12, 0.10, 0.08), (1.15, 1.05, 0.90), 0.50, 0.22),
        }
        low, high, sun_x, sun_y = palettes.get(name, palettes["studio"])
        low_t = torch.tensor(low, device="cuda")[None, None, :]
        high_t = torch.tensor(high, device="cuda")[None, None, :]
        gradient = low_t * y[..., None] + high_t * (1 - y[..., None])
        dx = torch.minimum((x - sun_x).abs(), 1 - (x - sun_x).abs())
        sun = torch.exp(-((dx / 0.035) ** 2 + ((y - sun_y) / 0.055) ** 2))[..., None]
        return (gradient + sun * torch.tensor((4.0, 3.2, 2.4), device="cuda")).float()

    def _retexture(self, controls: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        mesh = _load_mesh(context.asset(str(controls["mesh_asset"])))
        image = Image.open(context.asset(str(controls["image_asset"]))).convert("RGB")
        pipe = self._ensure_trellis("texture")
        context.update("Encoding source geometry", 0.12)
        output = pipe.run(
            mesh, image, seed=int(controls.get("seed", 42)),
            tex_slat_sampler_params={
                "steps": int(controls.get("texture_steps", 12)),
                "guidance_strength": float(controls.get("texture_guidance", 1.0)),
                "guidance_rescale": float(controls.get("texture_rescale", 0.0)),
                "guidance_interval": [float(controls.get("texture_guide_start", 0.6)), float(controls.get("texture_guide_end", 0.9))],
                "rescale_t": float(controls.get("texture_rescale_t", 3.0)),
            },
            preprocess_image=bool(controls.get("remove_background", True)),
            resolution=int(controls.get("texture_resolution", 1024)),
            texture_size=int(controls.get("texture_size", 2048)),
        )
        destination = context.output_dir / "retextured.glb"
        output.export(destination, extension_webp=bool(controls.get("webp_textures", True)))
        context.update("New PBR material ready", 0.99)
        return [StudioOutput(destination, "model", "Retextured PBR asset", "model/gltf-binary", {"role": "interactive-3d"})]

    def _mesh_lab(self, controls: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        mesh = _load_mesh(context.asset(str(controls["mesh_asset"])))
        before = {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)), "watertight": bool(mesh.is_watertight)}
        if controls.get("merge_vertices", True):
            mesh.merge_vertices()
        if controls.get("remove_degenerate", True):
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.remove_unreferenced_vertices()
        if controls.get("fix_normals", True):
            mesh.fix_normals(multibody=True)
        if controls.get("center_origin", True):
            mesh.apply_translation(-mesh.bounding_box.centroid)
        scale = float(controls.get("uniform_scale", 1.0))
        if scale != 1:
            mesh.apply_scale(scale)
        target = int(controls.get("face_target", 0))
        if target and len(mesh.faces) > target:
            mesh = mesh.simplify_quadric_decimation(face_count=target)
        extension = str(controls.get("output_format", "glb"))
        destination = context.output_dir / f"prepared-mesh.{extension}"
        mesh.export(destination)
        report = context.output_dir / "mesh-report.json"
        report.write_text(json.dumps({"before": before, "after": {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)), "watertight": bool(mesh.is_watertight), "bounds": mesh.bounds.tolist()}}, indent=2) + "\n", encoding="utf-8")
        return [
            StudioOutput(destination, "model", "Prepared mesh", "model/gltf-binary" if extension == "glb" else "application/octet-stream", {"role": "interactive-3d"}),
            StudioOutput(report, "document", "Mesh integrity report", "application/json"),
        ]
