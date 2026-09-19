from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

from local_app.runtime import StudioAdapter, StudioContext, StudioOutput, gpu_snapshot


MODEL_NAMES = {
    "core_long": "ARDY-Core-RP-20FPS-Horizon40",
    "core_fast": "ARDY-Core-RP-20FPS-Horizon8",
    "g1_long": "ARDY-G1-RP-25FPS-Horizon52",
    "g1_fast": "ARDY-G1-RP-25FPS-Horizon8",
}


def _floats(value: Any, limit: int) -> list[float]:
    if isinstance(value, list):
        raw = value
    else:
        raw = str(value or "").replace(";", ",").split(",")
    result: list[float] = []
    for item in raw:
        if str(item).strip():
            result.append(float(item))
    if len(result) > limit:
        raise ValueError(f"At most {limit} coefficients are accepted; got {len(result)}.")
    return result


def _motion_payload(source: Path, destination: Path) -> dict[str, Any]:
    from ardy.skeleton.registry import build_skeleton

    with np.load(source, allow_pickle=False) as archive:
        if "posed_joints" not in archive:
            raise ValueError("Motion NPZ has no posed_joints array.")
        joints = np.asarray(archive["posed_joints"], dtype=np.float32)
        while joints.ndim > 3 and joints.shape[0] == 1:
            joints = joints[0]
        if joints.ndim != 3 or joints.shape[-1] != 3:
            raise ValueError(f"Expected posed_joints as [frames,joints,3], found {joints.shape}.")
        skeleton = build_skeleton(int(joints.shape[1]))
        fps_value = archive["fps"] if "fps" in archive.files else 20.0
        fps = float(np.asarray(fps_value).reshape(-1)[0])
        prompt_value = archive["text"] if "text" in archive.files else ""
        prompt = str(np.asarray(prompt_value).reshape(-1)[0]) if np.asarray(prompt_value).size else ""
        contacts_value = archive["foot_contacts"] if "foot_contacts" in archive.files else []
        contacts = np.asarray(contacts_value, dtype=np.float32)
        if contacts.ndim > 2 and contacts.shape[0] == 1:
            contacts = contacts[0]

    # Keep every authored frame for ordinary clips; bounded decimation prevents
    # a pathological upload from producing an enormous browser document.
    stride = max(1, int(np.ceil(joints.shape[0] / 2400)))
    sampled = joints[::stride]
    parent_ids = skeleton.joint_parents.detach().cpu().tolist()
    payload = {
        "format": "mm-tools-motion-v1",
        "fps": fps / stride,
        "source_fps": fps,
        "source_frames": int(joints.shape[0]),
        "stride": stride,
        "names": list(skeleton.bone_order_names),
        "parents": [int(value) for value in parent_ids],
        "frames": np.round(sampled, 5).tolist(),
        "contacts": np.round(contacts[::stride], 4).tolist() if contacts.size else [],
        "prompt": prompt,
    }
    destination.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return payload


def _copy_output(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


class Adapter(StudioAdapter):
    def __init__(self, project_root: Path, runtime_root: Path) -> None:
        super().__init__(project_root, runtime_root)
        self.python = project_root / ".venv" / "bin" / "python"
        self.checkpoints = project_root / "checkpoints"
        self.encoder = project_root / "text-encoders" / "voxta" / "Llama-3-8B-LLM2Vec-ARDY-INT8"
        self.soma_assets = project_root / "SOMA-X" / "assets"

    def health(self) -> dict[str, Any]:
        required: list[tuple[str, Path]] = [
            (name, self.checkpoints / folder / "denoiser.safetensors")
            for name, folder in MODEL_NAMES.items()
        ]
        required += [
            ("Merged ARDY LLM2Vec INT8", self.encoder / "model.safetensors"),
            ("SOMA body", self.soma_assets / "SOMA_neutral.npz"),
            ("SOMA template rig", self.soma_assets / "SOMA_template_rig.usda"),
            ("SOMA hand", self.soma_assets / "SOMAHand.npz"),
            ("SOMA correctives", self.soma_assets / "correctives_model.pt"),
            ("Python environment", self.python),
        ]
        details = [
            {"label": label, "ready": path.is_file(), "required": True, "path": str(path)}
            for label, path in required
        ]
        gpu = gpu_snapshot()
        device = gpu.get("devices", [{}])[0] if gpu.get("available") else {}
        gpu_ok = bool(device) and int(device.get("memory_total_mib", 0)) >= 30000
        details.append({
            "label": "CUDA GPU >= 30 GiB", "ready": gpu_ok, "required": True,
            "value": device.get("name", "unavailable"),
        })
        return {"ready": all(path.is_file() for _, path in required) and gpu_ok, "loaded": False, "details": details}

    def validate(self, request: dict[str, Any], resolve_asset: Callable[[str], Path]) -> None:
        mode = str(request.get("mode", ""))
        controls = request.get("controls") or {}
        if mode not in {"text_motion", "inspect_motion", "soma_body", "soma_hand", "mesh_motion"}:
            raise ValueError(f"Unknown NVIDIA studio mode: {mode}")
        if mode == "text_motion":
            if not str(controls.get("prompt", "")).strip():
                raise ValueError("Describe the motion to generate.")
            if str(controls.get("model", "")) not in MODEL_NAMES:
                raise ValueError("Select one of the four installed ARDY variants.")
            if not 0.5 <= float(controls.get("duration", 5)) <= 120:
                raise ValueError("Duration must be from 0.5 through 120 seconds.")
            if not 1 <= int(controls.get("samples", 1)) <= 8:
                raise ValueError("Samples must be from 1 through 8.")
            constraint = str(controls.get("constraint_asset", ""))
            if constraint:
                resolve_asset(constraint)
        elif mode == "inspect_motion":
            resolve_asset(str(controls.get("motion_asset", "")))
        elif mode == "soma_body":
            _floats(controls.get("identity_coefficients", []), 300)
            if str(controls.get("lod", "mid")) not in {"mid", "low", "xlo"}:
                raise ValueError("Body LOD must be mid, low, or xlo.")
        elif mode == "soma_hand":
            _floats(controls.get("identity_coefficients", []), 128)
            if str(controls.get("hand", "right")) not in {"left", "right"}:
                raise ValueError("Hand side must be left or right.")
        elif mode == "mesh_motion":
            resolve_asset(str(controls.get("motion_asset", "")))
            if str(controls.get("lod", "mid")) not in {"mid", "low", "xlo"}:
                raise ValueError("Body LOD must be mid, low, or xlo.")

    @staticmethod
    def _progress(line: str) -> tuple[str, float] | None:
        if "Loaded model:" in line:
            return "ARDY resident on the GPU", 0.18
        if "Will generate" in line:
            return "Diffusing autoregressive motion", 0.28
        if "Saving the npz" in line:
            return "Packaging native motion", 0.86
        return None

    def _text_motion(self, controls: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        model_key = str(controls.get("model", "core_long"))
        stem = context.output_dir / "motion"
        command = [
            str(self.python), "scripts/generate.py", str(controls.get("prompt", "")).strip(),
            "--model", MODEL_NAMES[model_key],
            "--duration", str(float(controls.get("duration", 5))),
            "--num_samples", str(int(controls.get("samples", 1))),
            "--output", str(stem),
            "--seed", str(int(controls.get("seed", 42))),
            "--cfg_weight", str(float(controls.get("text_guidance", 2.0))),
            str(float(controls.get("constraint_guidance", 2.0))),
            "--checkpoints_dir", str(self.checkpoints),
        ]
        steps = int(controls.get("diffusion_steps", 0))
        if steps:
            command += ["--diffusion_steps", str(steps)]
        history = int(controls.get("history_frames", 0))
        if history:
            command += ["--history_frames", str(history)]
        if not bool(controls.get("postprocess", True)):
            command.append("--no-postprocess")
        constraint_id = str(controls.get("constraint_asset", ""))
        if constraint_id:
            command += ["--constraints", str(context.asset(constraint_id))]

        environment = {
            "CHECKPOINTS_DIR": str(self.checkpoints),
            "TEXT_ENCODER_MODE": "local",
            "TEXT_ENCODER_MERGED_PATH": str(self.encoder),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        }
        context.update("Loading ARDY + merged INT8 semantic encoder", 0.05)
        context.run_process(command, cwd=self.project_root, env=environment, progress_parser=self._progress)
        npz_files = sorted(context.output_dir.glob("motion*.npz"))
        if not npz_files:
            npz_files = sorted((context.output_dir / "motion").glob("*.npz"))
        if not npz_files:
            raise RuntimeError("ARDY completed without a native NPZ output.")
        outputs: list[StudioOutput] = []
        for index, npz in enumerate(npz_files):
            motion_json = npz.with_suffix(".motion.json")
            payload = _motion_payload(npz, motion_json)
            label = f"Motion {index + 1}" if len(npz_files) > 1 else "Interactive motion"
            outputs += [
                StudioOutput(motion_json, "motion", label, "application/json", {
                    "fps": payload["source_fps"], "frames": payload["source_frames"],
                    "skeleton": len(payload["names"]), "model": model_key,
                }),
                StudioOutput(npz, "data", f"{label} · native NPZ", "application/octet-stream"),
            ]
            csv = npz.with_suffix(".csv")
            if csv.is_file():
                outputs.append(StudioOutput(csv, "data", f"{label} · MuJoCo qpos", "text/csv"))
        context.update("Motion ready", 1.0)
        return outputs

    def _inspect(self, controls: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        source = context.asset(str(controls.get("motion_asset", "")))
        native = _copy_output(source, context.output_dir / source.name)
        motion_json = context.output_dir / f"{source.stem}.motion.json"
        payload = _motion_payload(native, motion_json)
        context.update("Interactive skeleton prepared", 1.0)
        return [
            StudioOutput(motion_json, "motion", "Interactive skeleton", "application/json", payload),
            StudioOutput(native, "data", "Native motion", "application/octet-stream"),
        ]

    @staticmethod
    def _pose_body(layer: Any, controls: dict[str, Any], torch: Any) -> Any:
        pose = torch.zeros(1, 77, 3, device="cuda")
        names = list(layer.public_joint_names)[1:]
        preset = str(controls.get("pose", "neutral"))
        rotations: dict[str, tuple[float, float, float]] = {}
        if preset == "a_pose":
            rotations = {"LeftArm": (0, 0, -0.75), "RightArm": (0, 0, 0.75)}
        elif preset == "t_pose":
            rotations = {"LeftArm": (0, 0, -1.35), "RightArm": (0, 0, 1.35)}
        elif preset == "stride":
            rotations = {"LeftUpLeg": (0.55, 0, 0), "RightUpLeg": (-0.4, 0, 0), "LeftArm": (-0.4, 0, -0.35), "RightArm": (0.5, 0, 0.35)}
        elif preset == "hero":
            rotations = {"LeftArm": (-0.25, 0, -0.65), "RightArm": (-0.25, 0, 0.65), "Spine2": (0, 0.12, 0)}
        for wanted, values in rotations.items():
            match = next((index for index, name in enumerate(names) if wanted.lower() in name.lower()), None)
            if match is not None:
                pose[0, match] = torch.tensor(values, device="cuda")
        pose[0, 0, 1] = float(controls.get("body_yaw", 0)) * np.pi / 180.0
        return pose

    def _body_layer(self, controls: dict[str, Any]):
        """Build a SOMALayer + identity tensor from shared body controls."""
        import torch
        from soma.body import SOMALayer

        layer = SOMALayer(
            self.soma_assets, identity_model_type="soma", device="cuda",
            lod=str(controls.get("lod", "mid")), mode="warp",
            enable_procedural_transforms=True,
            correctives_model_path=(self.soma_assets / "correctives_model.pt") if bool(controls.get("correctives", True)) else None,
        )
        values = _floats(controls.get("identity_coefficients", []), int(layer.num_shape_components))
        identity = torch.zeros(1, int(layer.num_shape_components), device="cuda")
        if values:
            identity[0, : len(values)] = torch.tensor(values, device="cuda")
        return layer, identity, values

    def _body(self, controls: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        import torch
        import trimesh

        context.update("Building native SOMA identity", 0.12)
        layer, identity, values = self._body_layer(controls)
        pose = self._pose_body(layer, controls, torch)
        with torch.inference_mode():
            result = layer(
                pose, identity, apply_correctives=bool(controls.get("correctives", True)),
                global_scale=float(controls.get("global_scale", 1.0)),
            )
        vertices = result["vertices"][0].float().cpu().numpy()
        joints = result["joints"][0].float().cpu().numpy()
        faces = layer.faces.detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        glb = context.output_dir / "soma_body.glb"
        mesh.export(glb)
        npz = context.output_dir / "soma_body.npz"
        np.savez_compressed(
            npz, vertices=vertices, faces=faces, joints=joints,
            poses=pose.detach().cpu().numpy(), identity_coefficients=identity.detach().cpu().numpy(),
            joint_names=np.asarray(layer.public_joint_names),
            joint_parent_ids=layer.output_joint_parent_ids.detach().cpu().numpy(),
        )
        metadata = context.output_dir / "soma_body.json"
        metadata.write_text(json.dumps({
            "backend": "soma", "lod": str(controls.get("lod", "mid")),
            "vertices": int(vertices.shape[0]), "faces": int(faces.shape[0]),
            "pose": str(controls.get("pose", "neutral")), "correctives": bool(controls.get("correctives", True)),
            "identity_coefficients": values,
        }, indent=2), encoding="utf-8")
        del layer, result
        gc.collect(); torch.cuda.empty_cache()
        context.update("SOMA body ready", 1.0)
        return [
            StudioOutput(glb, "model", "SOMA body · interactive GLB", "model/gltf-binary"),
            StudioOutput(npz, "data", "SOMA body · native arrays", "application/octet-stream"),
            StudioOutput(metadata, "text", "Build receipt", "application/json"),
        ]

    def _hand(self, controls: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        import torch
        import trimesh
        from soma.hand import SOMAHandLayer

        hand = str(controls.get("hand", "right"))
        context.update(f"Building native SOMA {hand} hand", 0.12)
        layer = SOMAHandLayer(
            self.soma_assets, hand_type=hand, identity_model_type="soma", device="cuda",
            lod=str(controls.get("lod", "mid")), mode="warp", correctives_model_path=None,
        )
        values = _floats(controls.get("identity_coefficients", []), int(layer.num_shape_components))
        identity = torch.zeros(1, int(layer.num_shape_components), device="cuda")
        if values:
            identity[0, : len(values)] = torch.tensor(values, device="cuda")
        pose = torch.zeros(1, 25, 3, device="cuda")
        curl = float(controls.get("curl", 0))
        spread = float(controls.get("spread", 0))
        thumb = float(controls.get("thumb", 0))
        pose[:, 1:, 0] = curl
        pose[:, 1:, 2] = spread * torch.linspace(-1, 1, 24, device="cuda")
        pose[:, 1:5, 1] = thumb
        pose[:, 0, 1] = float(controls.get("wrist_yaw", 0)) * np.pi / 180.0
        with torch.inference_mode():
            result = layer(pose, identity, apply_correctives=False, global_scale=float(controls.get("global_scale", 1.0)))
        vertices = result["vertices"][0].float().cpu().numpy()
        joints = result["joints"][0].float().cpu().numpy()
        faces = layer.faces.detach().cpu().numpy()
        glb = context.output_dir / f"soma_{hand}_hand.glb"
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(glb)
        npz = context.output_dir / f"soma_{hand}_hand.npz"
        np.savez_compressed(npz, vertices=vertices, faces=faces, joints=joints, poses=pose.detach().cpu().numpy(), identity_coefficients=identity.detach().cpu().numpy())
        del layer, result
        gc.collect(); torch.cuda.empty_cache()
        context.update("SOMA hand ready", 1.0)
        return [
            StudioOutput(glb, "model", f"SOMA {hand} hand · interactive GLB", "model/gltf-binary"),
            StudioOutput(npz, "data", f"SOMA {hand} hand · native arrays", "application/octet-stream"),
        ]

    def _mesh_motion(self, controls: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        import torch
        import trimesh
        from ardy.skeleton.registry import build_skeleton
        from soma.body import SOMALayer

        from local_app.mesh_motion import (
            HUMANOID_JOINT_COUNTS,
            build_animated_glb,
            decompose_trs_batch,
            solve_soma_pose,
            soma_correspondence,
        )

        source = context.asset(str(controls.get("motion_asset", "")))
        with np.load(source, allow_pickle=False) as archive:
            joints_src = np.asarray(archive["posed_joints"], dtype=np.float32)
            while joints_src.ndim > 3 and joints_src.shape[0] == 1:
                joints_src = joints_src[0]
            fps = float(np.asarray(archive["fps"]).reshape(-1)[0]) if "fps" in archive.files else 20.0
        if joints_src.ndim != 3 or joints_src.shape[-1] != 3:
            raise ValueError(f"Expected posed_joints as [frames,joints,3], found {joints_src.shape}.")
        joint_count = int(joints_src.shape[1])
        if joint_count not in HUMANOID_JOINT_COUNTS:
            raise ValueError("Only humanoid ARDY motions (27, 30, or 77 joints) can drive the SOMA body.")
        skeleton = build_skeleton(joint_count)
        source_names = list(skeleton.bone_order_names)
        stride = max(1, int(np.ceil(joints_src.shape[0] / 2400)))
        joints_src = joints_src[::stride].astype(np.float32)

        context.update("Building native SOMA identity", 0.1)
        layer, identity, _values = self._body_layer(controls)
        layer.prepare_identity(identity, global_scale=float(controls.get("global_scale", 1.0)))
        rig_view = layer.public_rig_view()
        bind_world = rig_view.bind_transforms_world[0]
        soma_names = [str(name) for name in rig_view.joint_names]
        correspondence = soma_correspondence(source_names, soma_names, source_rig=str(skeleton.name))
        mapped_source = [index for index, value in enumerate(correspondence) if value is not None]
        if len(mapped_source) < 15 or 1 not in correspondence:
            raise ValueError("Motion skeleton is not a supported humanoid rig.")

        # Absorb any unit/scale mismatch (centimeter ARDY clips vs meter SOMA rest)
        # through the hips-to-head span so translation stays in scene units.
        rest_pos = bind_world[:, :3, 3].detach().cpu().numpy()
        head_target = soma_names.index("Head") if "Head" in soma_names else 8
        head_source = next((index for index, name in enumerate(source_names) if str(name) in {"Head", "Skull"}), None)
        if head_source is None:
            raise ValueError("Motion skeleton has no head joint for scale normalization.")
        src_span = float(np.linalg.norm(
            np.asarray(joints_src[:, 1]).mean(axis=0) - np.asarray(joints_src[:, head_source]).mean(axis=0)))
        soma_span = float(np.linalg.norm(rest_pos[head_target] - rest_pos[1]))
        scale = soma_span / src_span if src_span > 1e-6 else 1.0
        targets = joints_src * scale

        context.update("Solving the native SOMA pose chain", 0.3)
        parent_ids = [int(value) for value in rig_view.joint_parent_ids.detach().cpu().tolist()]
        solved = solve_soma_pose(targets, correspondence,
                                 bind_world.detach().cpu().numpy(), parent_ids)
        poses_aa = solved["poses_aa"]
        transl = solved["transl"]
        frames = int(poses_aa.shape[0])

        context.update("Running native forward kinematics", 0.5)
        world = []
        chunk = 96
        for start in range(0, frames, chunk):
            slice_ = slice(start, min(start + chunk, frames))
            with torch.inference_mode():
                result = layer.pose(
                    torch.from_numpy(poses_aa[slice_]).to("cuda"),
                    torch.from_numpy(transl[slice_]).to("cuda"),
                    apply_correctives=bool(controls.get("correctives", True)),
                )
            world.append(result.transforms.detach().float().cpu().numpy())
        world = np.concatenate(world, axis=0)  # [F, 78, 4, 4]

        # Local node TRS per frame: Root carries its world transform; every other
        # joint is parent-relative. Static node TRS is therefore omitted and the
        # animation channels fully define the hierarchy.
        inv_parent = np.linalg.inv(world[:, parent_ids, :])
        inv_parent[:, 0] = np.eye(4)[None]
        local = np.einsum("fpab,fpbc->fpac", inv_parent, world)
        anim_trans, anim_quat = decompose_trs_batch(local)

        context.update("Assembling the skinned GLB", 0.8)
        with torch.inference_mode():
            rest = layer.pose(torch.zeros(1, 77, 3, device="cuda"), torch.zeros(1, 3, device="cuda"))
        vertices = rest.vertices[0].float().cpu().numpy()
        faces = layer.faces.detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

        weights = rig_view.skinning_weights.to(torch.float32)
        top_indices = torch.topk(weights, 4, dim=-1).indices.to(torch.uint8).cpu().numpy()
        top_weights = torch.gather(weights, 1, torch.topk(weights, 4, dim=-1).indices)
        top_weights = (top_weights / top_weights.sum(dim=-1, keepdim=True)).cpu().numpy()

        times = (np.arange(frames, dtype=np.float32)) / (fps / stride)
        glb = context.output_dir / "mesh_motion.glb"
        glb.write_bytes(build_animated_glb(
            vertices=vertices, normals=normals, faces=faces,
            joint_indices=top_indices, joint_weights=top_weights,
            joint_rest_world=bind_world.detach().cpu().numpy().astype(np.float32),
            anim_trans=anim_trans.astype(np.float32), anim_quat=anim_quat.astype(np.float32),
            times=times, parent_ids=parent_ids,
        ))
        npz = context.output_dir / "mesh_motion.npz"
        np.savez_compressed(
            npz, poses_aa=poses_aa, transl=transl, times=times,
            joint_names=np.asarray(rig_view.joint_names),
            joint_parent_ids=np.asarray(parent_ids),
            source_names=np.asarray(source_names),
            mapped_source_indices=np.asarray(mapped_source, dtype=np.int64),
            unit_scale=np.float32(scale),
        )
        metadata = context.output_dir / "mesh_motion.json"
        metadata.write_text(json.dumps({
            "backend": "soma", "lod": str(controls.get("lod", "mid")),
            "source": source.name, "source_fps": fps, "stride": stride,
            "frames": frames, "mapped_joints": len(mapped_source),
            "unit_scale": scale,
            "vertices": int(vertices.shape[0]), "faces": int(faces.shape[0]),
            "correctives": bool(controls.get("correctives", True)),
        }, indent=2), encoding="utf-8")
        del layer, rest, world, local
        gc.collect(); torch.cuda.empty_cache()
        context.update("Skinned motion ready", 1.0)
        return [
            StudioOutput(glb, "model", "Skinned motion · animated GLB", "model/gltf-binary"),
            StudioOutput(npz, "data", "Native SOMA pose arrays", "application/octet-stream"),
            StudioOutput(metadata, "text", "Build receipt", "application/json"),
        ]

    def run(self, request: dict[str, Any], context: StudioContext) -> list[StudioOutput]:
        mode = str(request.get("mode", ""))
        controls = request.get("controls") or {}
        if mode == "text_motion":
            return self._text_motion(controls, context)
        if mode == "inspect_motion":
            return self._inspect(controls, context)
        if mode == "soma_body":
            return self._body(controls, context)
        if mode == "soma_hand":
            return self._hand(controls, context)
        if mode == "mesh_motion":
            return self._mesh_motion(controls, context)
        raise ValueError(f"Unknown NVIDIA studio mode: {mode}")
