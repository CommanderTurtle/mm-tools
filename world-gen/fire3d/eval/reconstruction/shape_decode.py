"""Shape decoder utilities used by the public Fire3D inference runner."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _resolve_vae_root(root: str | os.PathLike[str]) -> Path:
    path = Path(root)
    if (path / "config.json").is_file():
        return path
    if (path / "sc" / "config.json").is_file():
        return path / "sc"
    raise FileNotFoundError(f"Missing Shape VAE X2 config.json under {path}")


def load_shape_vae_x2_decoder(
    *,
    ckpt_root: str | os.PathLike[str],
    checkpoint: str | os.PathLike[str],
    use_ema: bool,
    device: torch.device,
) -> tuple[torch.nn.Module, str, torch.Tensor, torch.Tensor]:
    from trellis2 import models as trellis_models

    root = _resolve_vae_root(ckpt_root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    ckpt_dir = root / "ckpts"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Missing Shape VAE X2 ckpt dir: {ckpt_dir}")

    decoder_path = Path(checkpoint)
    if not decoder_path.is_absolute():
        decoder_path = ckpt_dir / decoder_path
    if not decoder_path.is_file():
        raise FileNotFoundError(decoder_path)

    decoder_config = config["models"]["decoder"]
    decoder = getattr(trellis_models, decoder_config["name"])(
        **decoder_config.get("args", {})
    )
    decoder.load_state_dict(torch.load(decoder_path, map_location="cpu", weights_only=True))
    decoder.to(device).eval()
    for parameter in decoder.parameters():
        parameter.requires_grad = False

    normalization = config.get("dataset", {}).get("args", {}).get("normalization")
    if normalization is None:
        raise ValueError("Shape VAE X2 config is missing dataset.args.normalization")
    mean = torch.tensor(normalization["mean"], dtype=torch.float32, device=device).reshape(1, -1)
    std = torch.clamp(
        torch.tensor(normalization["std"], dtype=torch.float32, device=device).reshape(1, -1),
        min=1e-6,
    )
    return decoder, str(decoder_path.resolve()), mean, std


def load_trellis_shape_decoder(pretrained: str, device: torch.device) -> torch.nn.Module:
    from trellis2 import models as trellis_models

    decoder = trellis_models.from_pretrained(pretrained).eval().to(device)
    for parameter in decoder.parameters():
        parameter.requires_grad = False
    return decoder


def _mesh_field_to_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def decoded_mesh_to_trimesh(
    mesh_obj: Any,
    *,
    resolution: int,
    decimation_target: int,
    remesh: bool,
    remesh_band: float,
    remesh_project: float,
) -> Any:
    import o_voxel
    import trimesh

    raw_vertices = mesh_obj.vertices
    raw_faces = mesh_obj.faces
    vertices = _mesh_field_to_numpy(raw_vertices)
    faces = _mesh_field_to_numpy(raw_faces)
    if vertices.size == 0 or faces.size == 0:
        return trimesh.Trimesh(
            vertices=np.zeros((0, 3)),
            faces=np.zeros((0, 3), dtype=np.int64),
            process=False,
        )
    try:
        return o_voxel.postprocess.to_ply_no_texture(
            vertices=raw_vertices,
            faces=raw_faces,
            voxel_size=1 / resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            remesh=remesh,
            remesh_band=remesh_band,
            remesh_project=remesh_project,
            verbose=False,
        )
    except Exception as exc:
        print(f"[warn] mesh postprocess failed; exporting raw decoded mesh: {exc}")
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def remap_sparse_batches(
    feats: torch.Tensor,
    coords: torch.Tensor,
    keep_batch_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, dict[int, int]]:
    if not keep_batch_ids:
        return feats[:0], coords[:0], {}
    keep = torch.tensor(keep_batch_ids, dtype=coords.dtype, device=coords.device)
    mask = (coords[:, 0:1] == keep[None]).any(dim=1)
    feats_out = feats[mask]
    coords_out = coords[mask].clone()
    remap = {int(old): new for new, old in enumerate(keep_batch_ids)}
    for old, new in remap.items():
        coords_out[coords_out[:, 0] == old, 0] = new
    return feats_out, coords_out, remap


@torch.no_grad()
def decode_x2_object_meshes(
    *,
    x2_decoder: torch.nn.Module,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    shape_decoder: torch.nn.Module,
    feats_raw: torch.Tensor,
    coords: torch.Tensor,
    object_count: int,
    mesh_decode_batch_size: int,
    shape_decode_resolution: int,
    decimation_target: int,
    remesh: bool,
    remesh_band: float,
    remesh_project: float,
    timing_accumulator: dict[str, float] | None = None,
) -> list[Any]:
    import trimesh
    from trellis2.modules.sparse import SparseTensor

    device = next(x2_decoder.parameters()).device
    feats_raw = feats_raw.to(device=device, dtype=torch.float32)
    coords = coords.to(device=device, dtype=torch.int32)
    meshes = [
        trimesh.Trimesh(
            vertices=np.zeros((0, 3)),
            faces=np.zeros((0, 3), dtype=np.int64),
            process=False,
        )
        for _ in range(object_count)
    ]

    shape_decoder.set_resolution(shape_decode_resolution)
    chunk_size = max(int(mesh_decode_batch_size), 1)
    for chunk_start in range(0, object_count, chunk_size):
        batch_ids = list(range(chunk_start, min(object_count, chunk_start + chunk_size)))
        feats_chunk, coords_chunk, _ = remap_sparse_batches(feats_raw, coords, batch_ids)
        if feats_chunk.shape[0] == 0:
            continue

        sparse = SparseTensor(feats=feats_chunk, coords=coords_chunk)
        started = time.perf_counter() if timing_accumulator is not None else None
        if timing_accumulator is not None and device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        decoded_norm = x2_decoder(sparse)
        if timing_accumulator is not None:
            if device.type == "cuda":
                end.record()
                end.synchronize()
                seconds = float(start.elapsed_time(end) / 1000.0)
            else:
                seconds = float(time.perf_counter() - started)
            timing_accumulator["shape_x2_decoder"] = (
                timing_accumulator.get("shape_x2_decoder", 0.0) + seconds
            )

        decoded = SparseTensor(
            feats=decoded_norm.feats.float() * input_std + input_mean,
            coords=decoded_norm.coords.int(),
        )
        started = time.perf_counter() if timing_accumulator is not None else None
        if timing_accumulator is not None and device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        decoded_meshes = shape_decoder(decoded)
        if timing_accumulator is not None:
            if device.type == "cuda":
                end.record()
                end.synchronize()
                seconds = float(start.elapsed_time(end) / 1000.0)
            else:
                seconds = float(time.perf_counter() - started)
            timing_accumulator["trellis_shape_decoder_gpu"] = (
                timing_accumulator.get("trellis_shape_decoder_gpu", 0.0) + seconds
            )

        for local_index, batch_id in enumerate(batch_ids):
            if local_index < len(decoded_meshes):
                meshes[batch_id] = decoded_mesh_to_trimesh(
                    decoded_meshes[local_index],
                    resolution=shape_decode_resolution,
                    decimation_target=decimation_target,
                    remesh=remesh,
                    remesh_band=remesh_band,
                    remesh_project=remesh_project,
                )
    return meshes
