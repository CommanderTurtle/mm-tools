#!/usr/bin/env python3
"""Decode generated LC64 Shape/PBR-X2 pairs and bake textured object meshes.

This module separates neural decoding from CuMesh/texture postprocessing so a
large scene can cache raw decoded ovoxels, unload the neural decoders, then
postprocess several meshes together with ``cumesh_batch``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from eval.reconstruction.batched_ovoxel_postprocess import (
    MeshInput,
    PostprocessConfig,
    postprocess_meshes,
)


AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
PBR_ATTR_LAYOUT = {
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}


@dataclass(frozen=True)
class DecoderPaths:
    shape_x2_root: Path
    pbr_x2_root: Path
    shape_x2_checkpoint: str = "decoder.pt"
    pbr_x2_checkpoint: str = "decoder.pt"
    shape_x2_use_ema: bool = True
    pbr_x2_use_ema: bool = False
    shape_decoder: str = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"
    pbr_decoder: str = "microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def load_sparse_pair(
    shape_path: Path, pbr_path: Path
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with np.load(shape_path, allow_pickle=False) as archive:
        shape = torch.from_numpy(np.asarray(archive["feats"], dtype=np.float32))
        shape_coords = torch.from_numpy(np.asarray(archive["coords"], dtype=np.int32))
    with np.load(pbr_path, allow_pickle=False) as archive:
        pbr = torch.from_numpy(np.asarray(archive["feats"], dtype=np.float32))
        pbr_coords = torch.from_numpy(np.asarray(archive["coords"], dtype=np.int32))
    if shape.ndim != 2 or shape.shape[1] <= 0:
        raise ValueError(f"Expected shape features [N,C], got {tuple(shape.shape)}")
    if pbr.ndim != 2 or pbr.shape[0] != shape.shape[0] or pbr.shape[1] <= 0:
        raise ValueError(
            f"Expected PBR features [N,C] aligned to shape; "
            f"pbr={tuple(pbr.shape)} shape={tuple(shape.shape)}"
        )
    if shape_coords.ndim != 2 or shape_coords.shape != (shape.shape[0], 4):
        raise ValueError(f"Expected batched coords [N,4], got {tuple(shape_coords.shape)}")
    if not torch.equal(shape_coords, pbr_coords):
        raise ValueError("Generated Shape-X2 and PBR-X2 coordinates differ")
    if not torch.isfinite(shape).all() or not torch.isfinite(pbr).all():
        raise ValueError("Generated Shape/PBR features contain non-finite values")
    return shape, pbr, shape_coords


def remap_sparse_chunk(
    shape: torch.Tensor,
    pbr: torch.Tensor,
    coords: torch.Tensor,
    object_positions: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape_parts: list[torch.Tensor] = []
    pbr_parts: list[torch.Tensor] = []
    coord_parts: list[torch.Tensor] = []
    for local_id, object_position in enumerate(object_positions):
        mask = coords[:, 0] == int(object_position)
        if not mask.any():
            raise ValueError(f"Object position {object_position} has no sparse tokens")
        local_coords = coords[mask].clone()
        local_coords[:, 0] = int(local_id)
        shape_parts.append(shape[mask])
        pbr_parts.append(pbr[mask])
        coord_parts.append(local_coords)
    return torch.cat(shape_parts), torch.cat(pbr_parts), torch.cat(coord_parts)


def load_decoder_bundle(paths: DecoderPaths, device: torch.device):
    from fire3d.runtime.model_loader import load_x2_decoder
    from trellis2 import models

    shape_x2, shape_mean, shape_std, shape_info = load_x2_decoder(
        paths.shape_x2_root,
        kind="shape",
        use_ema=paths.shape_x2_use_ema,
        checkpoint=paths.shape_x2_checkpoint,
        device=device,
    )
    pbr_x2, pbr_mean, pbr_std, pbr_info = load_x2_decoder(
        paths.pbr_x2_root,
        kind="pbr",
        use_ema=paths.pbr_x2_use_ema,
        checkpoint=paths.pbr_x2_checkpoint,
        device=device,
    )
    shape_decoder = models.from_pretrained(paths.shape_decoder).eval().to(device)
    pbr_decoder = models.from_pretrained(paths.pbr_decoder).eval().to(device)
    for module in (shape_x2, pbr_x2, shape_decoder, pbr_decoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return {
        "shape_x2": shape_x2,
        "shape_mean": shape_mean,
        "shape_std": shape_std,
        "shape_info": shape_info,
        "pbr_x2": pbr_x2,
        "pbr_mean": pbr_mean,
        "pbr_std": pbr_std,
        "pbr_info": pbr_info,
        "shape_decoder": shape_decoder,
        "pbr_decoder": pbr_decoder,
    }


def _sparse(feats: torch.Tensor, coords: torch.Tensor):
    from trellis2.modules.sparse import SparseTensor

    return SparseTensor(feats=feats, coords=coords)


@torch.no_grad()
def _decode_x2(
    decoder: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    feats: torch.Tensor,
    coords: torch.Tensor,
    *,
    guide_subs: Any = None,
    return_subs: bool = False,
):
    decoded = decoder(
        _sparse(feats, coords), guide_subs=guide_subs, return_subs=return_subs
    )
    if return_subs:
        normalized, subdivisions = decoded
    else:
        normalized, subdivisions = decoded, None
    raw = _sparse(
        normalized.feats.float() * std + mean,
        normalized.coords.int(),
    )
    return (raw, subdivisions) if return_subs else raw


def _tensor_numpy(value: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    return value.detach().cpu().numpy().astype(dtype, copy=False)


@torch.no_grad()
@torch.no_grad()
def decode_shape_vertices(
    *,
    models: dict[str, Any],
    shape_feats: torch.Tensor,
    coords: torch.Tensor,
    num_objects: int,
    resolution: int = 512,
) -> list[torch.Tensor]:
    """Shape-only decode to canonical surface vertices, one per active voxel.

    The PBR flow is conditioned on perception points that, at inference, are not
    necessarily aligned with the *generated* shape (in training they were, by
    construction). This decodes only the shape branch -- no PBR decoder, no
    texture -- so the 512-resolution shape surface exists before the PBR flow
    runs and the condition points can be snapped onto it. The returned vertices
    are the dual-contour vertices in the object's canonical [-0.5, 0.5] cube,
    exactly the positions the later joint decode will carry per voxel.
    """

    if shape_feats.shape[0] != coords.shape[0]:
        raise ValueError(
            f"shape/coords mismatch: {tuple(shape_feats.shape)} vs {tuple(coords.shape)}"
        )
    models["shape_decoder"].set_resolution(int(resolution))
    shape_slat, _ = _decode_x2(
        models["shape_x2"],
        models["shape_mean"],
        models["shape_std"],
        shape_feats,
        coords,
        return_subs=True,
    )
    meshes, _ = models["shape_decoder"](shape_slat, return_subs=True)
    if len(meshes) != num_objects:
        raise RuntimeError(
            f"shape decode returned {len(meshes)} meshes for {num_objects} objects"
        )
    return [mesh.vertices.detach().float().cpu() for mesh in meshes]


def decode_scene_to_raw_ovoxels(
    *,
    shape_path: Path,
    pbr_path: Path,
    output_dir: Path,
    object_count: int,
    decoder_paths: DecoderPaths,
    device: torch.device,
    resolution: int = 512,
    chunk_size: int = 2,
    overwrite: bool = False,
    models: dict[str, Any] | None = None,
    persist_raw: bool = True,
    return_raw: bool = False,
    detailed_profile: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], list[dict[str, torch.Tensor]]]:
    """Decode sparse pairs, optionally retaining raw objects only in memory."""

    summary_path = output_dir / "decode_summary.json"
    if persist_raw and summary_path.is_file() and not overwrite:
        summary = json.loads(summary_path.read_text())
        if all(Path(record["raw_npz"]).is_file() for record in summary["objects"]):
            if not return_raw:
                return summary
            return summary, [
                _load_raw(Path(record["raw_npz"])) for record in summary["objects"]
            ]

    profile_started = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    chunk_profiles: list[dict[str, Any]] = []

    def timed(name: str, function: Callable[[], Any], chunk: dict | None = None):
        if not detailed_profile:
            return function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = float(time.perf_counter() - started)
        stage_seconds[name] = stage_seconds.get(name, 0.0) + elapsed
        if chunk is not None:
            chunk_stages = chunk["stage_seconds"]
            chunk_stages[name] = chunk_stages.get(name, 0.0) + elapsed
        return result

    shape, pbr, coords = timed(
        "load_sparse_pair", lambda: load_sparse_pair(shape_path, pbr_path)
    )
    present = sorted({int(value) for value in coords[:, 0].tolist()})
    expected = list(range(object_count))
    if present != expected:
        raise ValueError(f"Sparse object batches {present} != expected {expected}")

    if models is None:
        models = timed(
            "load_decoder_bundle", lambda: load_decoder_bundle(decoder_paths, device)
        )
    models["shape_decoder"].set_resolution(int(resolution))
    if persist_raw:
        output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    raw_objects: list[dict[str, torch.Tensor]] = []

    for start in range(0, object_count, max(int(chunk_size), 1)):
        if detailed_profile and device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        chunk_started = time.perf_counter() if detailed_profile else None
        positions = list(range(start, min(object_count, start + max(int(chunk_size), 1))))
        chunk_profile = {
            "object_positions": positions,
            "object_count": len(positions),
            "stage_seconds": {},
        }
        shape_i, pbr_i, coords_i = timed(
            "sparse_pair_remap",
            lambda: remap_sparse_chunk(shape, pbr, coords, positions),
            chunk_profile,
        )

        def move_chunk_to_device(
            shape_chunk=shape_i,
            pbr_chunk=pbr_i,
            coords_chunk=coords_i,
        ):
            return (
                shape_chunk.to(device=device, dtype=torch.float32),
                pbr_chunk.to(device=device, dtype=torch.float32),
                coords_chunk.to(device=device, dtype=torch.int32),
            )

        shape_i, pbr_i, coords_i = timed(
            "host_to_device", move_chunk_to_device, chunk_profile
        )
        # flex_gemm sparse kernels do not autocast SparseTensor feature buffers.
        # This intentionally matches decode_pbr_x2_flow_pair.py: X2 decoders
        # remain fp32, while pretrained TRELLIS modules apply their own dtype.
        shape_slat, x2_subs = timed(
            "shape_x2_decode",
            lambda shape_chunk=shape_i, coords_chunk=coords_i: _decode_x2(
                models["shape_x2"],
                models["shape_mean"],
                models["shape_std"],
                shape_chunk,
                coords_chunk,
                return_subs=True,
            ),
            chunk_profile,
        )
        pbr_slat = timed(
            "pbr_x2_decode",
            lambda pbr_chunk=pbr_i, coords_chunk=coords_i: _decode_x2(
                models["pbr_x2"],
                models["pbr_mean"],
                models["pbr_std"],
                pbr_chunk,
                coords_chunk,
                guide_subs=x2_subs,
            ),
            chunk_profile,
        )
        meshes, trellis_subs = timed(
            "shape_ovoxel_decode",
            lambda shape_latent=shape_slat: models["shape_decoder"](
                shape_latent, return_subs=True
            ),
            chunk_profile,
        )
        tex_batch = timed(
            "pbr_ovoxel_decode",
            lambda pbr_latent=pbr_slat: (
                models["pbr_decoder"](pbr_latent, guide_subs=trellis_subs) * 0.5
                + 0.5
            ),
            chunk_profile,
        )
        if len(meshes) != len(positions) or len(tex_batch) != len(positions):
            raise RuntimeError(
                f"Decoder batch mismatch: objects={len(positions)} meshes={len(meshes)} "
                f"textures={len(tex_batch)}"
            )
        def transfer_raw_objects(
            chunk_meshes=meshes,
            chunk_textures=tex_batch,
            object_positions=positions,
        ):
            transferred = []
            for local_id, object_position in enumerate(object_positions):
                mesh = chunk_meshes[local_id]
                tex = chunk_textures[local_id]
                tex_coords = (
                    tex.coords[:, 1:] if tex.coords.shape[1] == 4 else tex.coords
                )
                transferred.append(
                    (
                        object_position,
                        mesh,
                        tex,
                        {
                            "vertices": mesh.vertices.detach().float().cpu(),
                            "faces": mesh.faces.detach().int().cpu(),
                            "attrs": tex.feats.detach().float().clamp(0, 1).cpu(),
                            "coords": tex_coords.detach().int().cpu(),
                        },
                    )
                )
            return transferred

        transferred = timed(
            "device_to_host", transfer_raw_objects, chunk_profile
        )
        for object_position, mesh, tex, raw_object in transferred:
            raw_objects.append(raw_object)
            raw_path = output_dir / f"object_{object_position:04d}.npz"
            if persist_raw:
                timed(
                    "raw_npz_export",
                    lambda raw_path=raw_path, raw_object=raw_object: atomic_npz(
                        raw_path,
                        vertices=raw_object["vertices"].numpy().astype(
                            np.float32, copy=False
                        ),
                        faces=raw_object["faces"].numpy().astype(
                            np.int32, copy=False
                        ),
                        attrs=raw_object["attrs"].numpy().astype(
                            np.float16, copy=False
                        ),
                        coords=raw_object["coords"].numpy().astype(
                            np.int32, copy=False
                        ),
                    ),
                    chunk_profile,
                )
            records.append(
                {
                    "object_position": object_position,
                    "raw_npz": str(raw_path) if persist_raw else None,
                    "vertices": int(mesh.vertices.shape[0]),
                    "faces": int(mesh.faces.shape[0]),
                    "attribute_voxels": int(tex.feats.shape[0]),
                    "attribute_channels": int(tex.feats.shape[1]),
                }
            )
        del shape_i, pbr_i, coords_i, shape_slat, pbr_slat, meshes, tex_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if detailed_profile:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                chunk_profile["peak_allocated_bytes"] = int(
                    torch.cuda.max_memory_allocated(device)
                )
                chunk_profile["peak_reserved_bytes"] = int(
                    torch.cuda.max_memory_reserved(device)
                )
            chunk_wall_seconds = float(time.perf_counter() - chunk_started)
            chunk_stage_seconds = float(
                sum(chunk_profile["stage_seconds"].values())
            )
            chunk_profile["stage_sum_seconds"] = chunk_stage_seconds
            chunk_profile["wall_seconds"] = chunk_wall_seconds
            chunk_profile["timer_accounting_gap_seconds"] = float(
                chunk_wall_seconds - chunk_stage_seconds
            )
            chunk_profiles.append(chunk_profile)

    summary = {
        "schema": "ff_holoscene.lc64_shape_pbr_raw_ovoxels.v1",
        "shape_x2": str(shape_path),
        "pbr_x2": str(pbr_path),
        "object_count": object_count,
        "resolution": resolution,
        "shape_x2_decoder": models["shape_info"],
        "pbr_x2_decoder": models["pbr_info"],
        "shape_decoder": decoder_paths.shape_decoder,
        "pbr_decoder": decoder_paths.pbr_decoder,
        "raw_persistence": "npz" if persist_raw else "memory_only",
        "objects": sorted(records, key=lambda row: row["object_position"]),
    }
    if detailed_profile:
        profile_wall_seconds = float(time.perf_counter() - profile_started)
        summary["detailed_profile"] = {
            "schema": "ff_holoscene.raw_ovoxel_decode_profile.v1",
            "object_count": int(object_count),
            "chunk_size": int(chunk_size),
            "chunk_count": len(chunk_profiles),
            "stage_seconds": stage_seconds,
            "stage_sum_seconds": float(sum(stage_seconds.values())),
            "wall_seconds": profile_wall_seconds,
            "timer_accounting_gap_seconds": float(
                profile_wall_seconds - sum(stage_seconds.values())
            ),
            "chunks": chunk_profiles,
        }
    if persist_raw:
        atomic_json(summary_path, summary)
    return (summary, raw_objects) if return_raw else summary


@torch.no_grad()
def decode_native_scene_to_raw_ovoxels(
    *,
    shape_path: Path,
    pbr_path: Path,
    output_dir: Path,
    object_count: int,
    device: torch.device,
    shape_decoder_pretrained: str,
    pbr_decoder_pretrained: str = "microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16",
    resolution: int = 512,
    chunk_size: int = 2,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Decode native TRELLIS.2 shape/PBR sparse latents without X2 VAEs."""
    from trellis2 import models as trellis_models

    summary_path = output_dir / "decode_summary.json"
    if summary_path.is_file() and not overwrite:
        summary = json.loads(summary_path.read_text())
        if all(Path(record["raw_npz"]).is_file() for record in summary["objects"]):
            return summary
    shape, pbr, coords = load_sparse_pair(shape_path, pbr_path)
    present = sorted({int(value) for value in coords[:, 0].tolist()})
    if present != list(range(object_count)):
        raise ValueError(f"Sparse object batches {present} != expected {list(range(object_count))}")
    shape_decoder = trellis_models.from_pretrained(shape_decoder_pretrained).eval().to(device)
    pbr_decoder = trellis_models.from_pretrained(pbr_decoder_pretrained).eval().to(device)
    shape_decoder.set_resolution(int(resolution))
    for module in (shape_decoder, pbr_decoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for start in range(0, object_count, max(int(chunk_size), 1)):
        positions = list(range(start, min(object_count, start + max(int(chunk_size), 1))))
        shape_i, pbr_i, coords_i = remap_sparse_chunk(shape, pbr, coords, positions)
        shape_slat = _sparse(shape_i.to(device).float(), coords_i.to(device).int())
        pbr_slat = _sparse(pbr_i.to(device).float(), coords_i.to(device).int())
        meshes, subdivisions = shape_decoder(shape_slat, return_subs=True)
        tex_batch = pbr_decoder(pbr_slat, guide_subs=subdivisions) * 0.5 + 0.5
        for local_id, object_position in enumerate(positions):
            mesh, tex = meshes[local_id], tex_batch[local_id]
            tex_coords = tex.coords[:, 1:] if tex.coords.shape[1] == 4 else tex.coords
            raw_path = output_dir / f"object_{object_position:04d}.npz"
            atomic_npz(
                raw_path,
                vertices=_tensor_numpy(mesh.vertices.float(), np.float32),
                faces=_tensor_numpy(mesh.faces.int(), np.int32),
                attrs=_tensor_numpy(tex.feats.float().clamp(0, 1), np.float16),
                coords=_tensor_numpy(tex_coords.int(), np.int32),
            )
            records.append({
                "object_position": object_position,
                "raw_npz": str(raw_path),
                "vertices": int(mesh.vertices.shape[0]),
                "faces": int(mesh.faces.shape[0]),
                "attribute_voxels": int(tex.feats.shape[0]),
                "attribute_channels": int(tex.feats.shape[1]),
            })
        del shape_i, pbr_i, coords_i, shape_slat, pbr_slat, meshes, tex_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = {
        "schema": "ff_holoscene.trellis2_native_shape_pbr_raw_ovoxels.v1",
        "shape_native": str(shape_path),
        "pbr_native": str(pbr_path),
        "object_count": object_count,
        "resolution": resolution,
        "shape_decoder": shape_decoder_pretrained,
        "pbr_decoder": pbr_decoder_pretrained,
        "objects": sorted(records, key=lambda row: row["object_position"]),
    }
    atomic_json(summary_path, summary)
    return summary


def _load_raw(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "vertices": torch.from_numpy(np.asarray(archive["vertices"], dtype=np.float32)),
            "faces": torch.from_numpy(np.asarray(archive["faces"], dtype=np.int32)),
            "attrs": torch.from_numpy(np.asarray(archive["attrs"], dtype=np.float32)),
            "coords": torch.from_numpy(np.asarray(archive["coords"], dtype=np.int32)),
        }


def _timed_texture_bake_stage(
    name: str,
    function: Callable[[], Any],
    *,
    device: torch.device,
    stage_seconds: dict[str, float] | None,
) -> Any:
    if stage_seconds is None:
        return function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stage_seconds[name] = float(time.perf_counter() - started)
    return result


def _pad_sparse_sample_queries(
    query_points: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, int, int]:
    """Pad query points to a reusable kernel shape without changing outputs."""

    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError(
            f"Expected query points [N,3], got {tuple(query_points.shape)}"
        )
    query_count = int(query_points.shape[0])
    if query_count == 0:
        raise ValueError("Cannot sample an empty set of texture queries")
    if mode == "exact":
        return query_points, query_count, query_count
    if mode != "power2":
        raise ValueError(f"Unsupported sparse query mode: {mode}")
    padded_count = 1 << (query_count - 1).bit_length()
    if padded_count == query_count:
        return query_points, query_count, padded_count
    padding = query_points[:1].expand(padded_count - query_count, -1)
    return torch.cat((query_points, padding), dim=0), query_count, padded_count


def _dilate_texture_padding(
    texture: torch.Tensor,
    valid_mask: torch.Tensor,
    pixels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate valid texels into a bounded border using GPU max-pool indices."""

    if texture.ndim != 3:
        raise ValueError(f"Expected texture [H,W,C], got {tuple(texture.shape)}")
    if valid_mask.shape != texture.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean and match texture height/width")
    if pixels < 0:
        raise ValueError("Texture dilation pixels must be non-negative")
    if pixels == 0:
        return texture, valid_mask

    values = texture.permute(2, 0, 1).unsqueeze(0).contiguous()
    valid = valid_mask[None, None]
    channels = int(values.shape[1])
    for _ in range(pixels):
        expanded, indices = F.max_pool2d(
            valid.float(),
            kernel_size=3,
            stride=1,
            padding=1,
            return_indices=True,
        )
        nearest = torch.gather(
            values.flatten(2),
            2,
            indices.flatten(2).expand(-1, channels, -1),
        ).view_as(values)
        newly_valid = (~valid) & (expanded > 0)
        values = torch.where(newly_valid.expand_as(values), nearest, values)
        valid = expanded > 0
    return values[0].permute(1, 2, 0).contiguous(), valid[0, 0]


def _prepare_uv_raster_vertices(
    uvs: torch.Tensor,
    uv_raster_depths: torch.Tensor | None,
) -> torch.Tensor:
    """Build clip-space UV vertices, preserving projection depth when present."""

    if uv_raster_depths is None:
        depths = torch.zeros_like(uvs[:, :1])
    else:
        if uv_raster_depths.ndim != 1 or uv_raster_depths.shape[0] != uvs.shape[0]:
            raise ValueError("uv_raster_depths must have shape [num_uv_vertices]")
        if not torch.isfinite(uv_raster_depths).all().item():
            raise ValueError("uv_raster_depths contains non-finite values")
        depths = uv_raster_depths[:, None].to(device=uvs.device, dtype=uvs.dtype)
    return torch.cat(
        [
            uvs * 2 - 1,
            depths,
            torch.ones_like(uvs[:, :1]),
        ],
        dim=-1,
    ).unsqueeze(0)


def _rasterize_uv_atlas(
    context: Any,
    raster_vertices: torch.Tensor,
    faces: torch.Tensor,
    texture_size: int,
    *,
    depth_aware: bool,
) -> torch.Tensor:
    """Rasterize UV triangles and preserve camera depth across face chunks."""

    import nvdiffrast.torch as dr

    output = torch.zeros(
        (1, texture_size, texture_size, 4),
        device=raster_vertices.device,
        dtype=torch.float32,
    )
    for start in range(0, faces.shape[0], 100000):
        chunk, _ = dr.rasterize(
            context,
            raster_vertices,
            faces[start : start + 100000],
            resolution=[texture_size, texture_size],
        )
        chunk_mask = chunk[..., 3:4] > 0
        chunk[..., 3:4] += start
        if not depth_aware:
            output = torch.where(chunk_mask, chunk, output)
            continue
        output_mask = output[..., 3:4] > 0
        replace = chunk_mask & ((~output_mask) | (chunk[..., 2:3] < output[..., 2:3]))
        output = torch.where(replace, chunk, output)
    return output


_RASTER_CONTEXTS: dict[str, Any] = {}


def _shared_raster_context(device: torch.device):
    """One nvdiffrast context per device for the process lifetime.

    Contexts hold CUDA resources that are never released; creating one per
    postprocess call leaked across scenes in --batch-scenes mode until the
    second scene's remesh ran out of memory.
    """
    import nvdiffrast.torch as dr

    key = str(device)
    if key not in _RASTER_CONTEXTS:
        _RASTER_CONTEXTS[key] = dr.RasterizeCudaContext(device=device)
    return _RASTER_CONTEXTS[key]


def _jump_flood_texture_fill(
    texture: torch.Tensor,
    valid_mask: torch.Tensor,
    pixels: int,
    erode_pixels: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-valid-texel fill via jump flooding, entirely on the GPU.

    Same semantics as the Video2Game nearest_push (every empty texel copies all
    channels from its Euclidean-nearest valid texel) but ~log2(N) strided
    gather passes instead of a CPU kd-tree, and unlike iterative gpu_dilate it
    has no directional tie bias. The nearest-seed distance field also gives the
    fill ring exactly: only texels within ``pixels`` of a valid texel are
    filled, matching the dilate-ring contract of the other fill modes.
    """
    if texture.ndim != 3:
        raise ValueError(f"Expected texture [H,W,C], got {tuple(texture.shape)}")
    if valid_mask.shape != texture.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean and match texture height/width")
    if pixels < 0:
        raise ValueError("Texture dilation pixels must be non-negative")
    if pixels == 0 or not valid_mask.any() or valid_mask.all():
        return texture, valid_mask

    device = texture.device
    height, width = valid_mask.shape

    # Optional rim refresh: erode the valid mask so the outermost texels of
    # every chart (partial raster coverage, boundary snap-back artifacts) are
    # refilled from clean chart interiors. Guarded below so a thin chart that
    # erodes away keeps its original texels instead of borrowing a neighbour
    # chart's colors.
    seed_mask = valid_mask
    if erode_pixels > 0:
        eroded = valid_mask[None, None].float()
        for _ in range(int(erode_pixels)):
            eroded = -F.max_pool2d(-eroded, kernel_size=3, stride=1, padding=1)
        eroded_mask = eroded[0, 0] > 0.5
        if eroded_mask.any():
            seed_mask = eroded_mask
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    coords = torch.stack((ys, xs), dim=0).float()  # [2, H, W]
    # Seeds carry their own coordinates; empty texels start unreachable.
    seed = torch.where(seed_mask[None], coords, torch.full_like(coords, -1e6))
    best = seed.clone()

    def distance_sq(candidate: torch.Tensor) -> torch.Tensor:
        return (candidate[0] - coords[0]) ** 2 + (candidate[1] - coords[1]) ** 2

    best_d = distance_sq(best)
    stride = 1
    while stride < max(height, width):
        stride <<= 1
    stride >>= 1
    strides = []
    while stride >= 1:
        strides.append(stride)
        stride >>= 1
    strides.append(1)  # extra unit pass (1+JFA) removes most residual errors
    for step in strides:
        for dy in (-step, 0, step):
            for dx in (-step, 0, step):
                if dy == 0 and dx == 0:
                    continue
                candidate = torch.roll(best, shifts=(dy, dx), dims=(1, 2))
                # torch.roll wraps; invalidate wrapped rows/cols
                if dy > 0:
                    candidate[:, :dy, :] = -1e6
                elif dy < 0:
                    candidate[:, dy:, :] = -1e6
                if dx > 0:
                    candidate[:, :, :dx] = -1e6
                elif dx < 0:
                    candidate[:, :, dx:] = -1e6
                candidate_d = distance_sq(candidate)
                take = candidate_d < best_d
                best = torch.where(take[None], candidate, best)
                best_d = torch.where(take, candidate_d, best_d)

    within_ring = best_d <= float(pixels) ** 2
    fill_region = within_ring & ~valid_mask
    if erode_pixels > 0 and not seed_mask.equal(valid_mask):
        # Rim refresh: replace eroded-away valid texels too, but only when the
        # nearest surviving seed is close enough to belong to the same chart
        # interior; distant seeds would be another chart's colors.
        rim = valid_mask & ~seed_mask
        same_chart = best_d <= float(erode_pixels + 2) ** 2
        fill_region = fill_region | (rim & same_chart)
    source_rows = best[0].round().long().clamp(0, height - 1)
    source_cols = best[1].round().long().clamp(0, width - 1)
    filled = texture.clone()
    gathered = texture[source_rows[fill_region], source_cols[fill_region]]
    filled[fill_region] = gathered
    return filled, valid_mask | fill_region


def _nearest_push_texture_fill(
    sampled: torch.Tensor,
    mask: torch.Tensor,
    dilate_iterations: int,
    erode_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill uncovered texels by copying the nearest valid boundary texel.

    Faithful port of the Video2Game atlas completion
    (``baking_pretrain_export.py``): dilate the coverage mask to obtain the
    ring of empty texels to fill, erode it to obtain the thin band of valid
    texels just inside the charts to search against, then copy each target
    texel's value from its nearest source texel.

    Unlike Telea inpainting this invents no new values -- every filled texel is
    an exact copy of a predicted one, and all PBR channels are copied together
    from the same source texel, so they cannot disagree. Searching only the
    eroded boundary shell keeps the neighbour query small without changing the
    nearest-neighbour answer for texels outside the charts.
    """
    from scipy.ndimage import binary_dilation, binary_erosion
    from sklearn.neighbors import NearestNeighbors

    mask_np = mask.detach().cpu().numpy().astype(bool)
    if not mask_np.any():
        return sampled, mask

    inpaint_region = binary_dilation(mask_np, iterations=int(dilate_iterations))
    inpaint_region[mask_np] = 0

    search_region = mask_np.copy()
    if erode_iterations > 0:
        not_search_region = binary_erosion(search_region, iterations=int(erode_iterations))
        search_region[not_search_region] = 0

    inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)
    search_coords = np.stack(np.nonzero(search_region), axis=-1)
    if inpaint_coords.shape[0] == 0 or search_coords.shape[0] == 0:
        return sampled, mask

    knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(search_coords)
    _, indices = knn.kneighbors(inpaint_coords)
    source_coords = search_coords[indices[:, 0]]

    device = sampled.device
    target_rows = torch.from_numpy(inpaint_coords[:, 0]).to(device=device, dtype=torch.long)
    target_cols = torch.from_numpy(inpaint_coords[:, 1]).to(device=device, dtype=torch.long)
    source_rows = torch.from_numpy(source_coords[:, 0]).to(device=device, dtype=torch.long)
    source_cols = torch.from_numpy(source_coords[:, 1]).to(device=device, dtype=torch.long)

    filled = sampled.clone()
    filled[target_rows, target_cols] = sampled[source_rows, source_cols]
    filled_mask = torch.from_numpy(mask_np | inpaint_region).to(device=mask.device)
    return filled, filled_mask


def bake_preprocessed_texture(
    *,
    source: dict[str, torch.Tensor],
    processed: Any,
    device: torch.device,
    resolution: int,
    texture_size: int,
    raster_context: Any = None,
    surface_mapping: str = "source_bvh",
    sparse_query_mode: str = "exact",
    texture_fill_mode: str = "gpu_dilate",
    texture_dilation_pixels: int = 32,
    texture_erode_iterations: int = 2,
    profile: dict[str, Any] | None = None,
):
    """Bake sparse PBR attributes onto one already postprocessed UV mesh."""

    import nvdiffrast.torch as dr
    import trimesh
    import trimesh.visual
    from flex_gemm.ops.grid_sample import grid_sample_3d

    if surface_mapping not in {"source_bvh", "processed"}:
        raise ValueError(f"Unsupported surface mapping: {surface_mapping}")
    if sparse_query_mode not in {"exact", "power2"}:
        raise ValueError(f"Unsupported sparse query mode: {sparse_query_mode}")
    if texture_fill_mode not in {"telea", "gpu_dilate", "nearest_push", "jfa"}:
        raise ValueError(f"Unsupported texture fill mode: {texture_fill_mode}")
    if texture_dilation_pixels < 0:
        raise ValueError("Texture dilation pixels must be non-negative")

    stage_seconds: dict[str, float] | None = {} if profile is not None else None
    if profile is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    profile_started = time.perf_counter() if profile is not None else None

    def prepare_inputs():
        uv_raster_depths = getattr(processed, "uv_raster_depths", None)
        return (
            (
                source["vertices"].to(device=device, dtype=torch.float32)
                if surface_mapping == "source_bvh"
                else None
            ),
            (
                source["faces"].to(device=device, dtype=torch.int32)
                if surface_mapping == "source_bvh"
                else None
            ),
            source["attrs"].to(device=device, dtype=torch.float32),
            source["coords"].to(device=device, dtype=torch.int32),
            processed.vertices.to(device=device, dtype=torch.float32),
            processed.faces.to(device=device, dtype=torch.int32),
            processed.uvs.to(device=device, dtype=torch.float32),
            (
                uv_raster_depths.to(device=device, dtype=torch.float32)
                if uv_raster_depths is not None
                else None
            ),
            processed.normals.detach().cpu().numpy().astype(np.float32, copy=False),
        )

    (
        source_vertices,
        source_faces,
        attrs_volume,
        coords,
        vertices,
        faces,
        uvs,
        uv_raster_depths,
        normals,
    ) = _timed_texture_bake_stage(
        "prepare_inputs",
        prepare_inputs,
        device=device,
        stage_seconds=stage_seconds,
    )

    bvh = None
    if surface_mapping == "source_bvh":
        import cumesh

        bvh = _timed_texture_bake_stage(
            "build_source_bvh",
            lambda: cumesh.cuBVH(source_vertices, source_faces),
            device=device,
            stage_seconds=stage_seconds,
        )
    context = raster_context or dr.RasterizeCudaContext(device=device)

    def prepare_uv_raster():
        return _prepare_uv_raster_vertices(uvs, uv_raster_depths)

    uvs_rast = _timed_texture_bake_stage(
        "prepare_uv_raster",
        prepare_uv_raster,
        device=device,
        stage_seconds=stage_seconds,
    )

    def rasterize_uv():
        return _rasterize_uv_atlas(
            context,
            uvs_rast,
            faces,
            texture_size,
            depth_aware=uv_raster_depths is not None,
        )

    rast = _timed_texture_bake_stage(
        "rasterize_uv",
        rasterize_uv,
        device=device,
        stage_seconds=stage_seconds,
    )

    def interpolate_surface_positions():
        valid_mask = rast[0, ..., 3] > 0
        all_positions = dr.interpolate(vertices.unsqueeze(0), rast, faces)[0][0]
        return valid_mask, all_positions[valid_mask]

    mask, valid_positions = _timed_texture_bake_stage(
        "interpolate_surface_positions",
        interpolate_surface_positions,
        device=device,
        stage_seconds=stage_seconds,
    )
    if surface_mapping == "source_bvh":
        face_id, barycentric = _timed_texture_bake_stage(
            "query_source_bvh",
            lambda: bvh.unsigned_distance(valid_positions, return_uvw=True)[1:],
            device=device,
            stage_seconds=stage_seconds,
        )

        def snap_to_source_surface():
            source_triangles = source_vertices[source_faces[face_id.long()]]
            return (source_triangles * barycentric.unsqueeze(-1)).sum(dim=1)

        valid_positions = _timed_texture_bake_stage(
            "snap_to_source_surface",
            snap_to_source_surface,
            device=device,
            stage_seconds=stage_seconds,
        )

    def sample_sparse_pbr():
        aabb = torch.tensor(AABB, device=device, dtype=torch.float32)
        grid_size = torch.tensor([resolution] * 3, device=device, dtype=torch.int32)
        voxel_size = (aabb[1] - aabb[0]) / grid_size
        texture = torch.zeros(
            texture_size,
            texture_size,
            attrs_volume.shape[1],
            device=device,
        )
        query_points = (valid_positions - aabb[0]) / voxel_size
        padded_points, query_count, padded_query_count = _pad_sparse_sample_queries(
            query_points, sparse_query_mode
        )
        sampled_values = grid_sample_3d(
            attrs_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=torch.Size(
                [1, attrs_volume.shape[1], resolution, resolution, resolution]
            ),
            grid=padded_points.reshape(1, -1, 3),
            mode="trilinear",
        )
        texture[mask] = sampled_values[:, :query_count]
        return texture, query_count, padded_query_count

    sampled, query_count, padded_query_count = _timed_texture_bake_stage(
        "sample_sparse_pbr",
        sample_sparse_pbr,
        device=device,
        stage_seconds=stage_seconds,
    )

    if sampled.shape[-1] < 6:
        raise ValueError(f"Expected at least six decoded PBR channels, got {sampled.shape[-1]}")

    filled_mask = mask
    if texture_fill_mode == "nearest_push":
        sampled, filled_mask = _timed_texture_bake_stage(
            "nearest_push_texture_fill",
            lambda: _nearest_push_texture_fill(
                sampled, mask, texture_dilation_pixels, texture_erode_iterations
            ),
            device=device,
            stage_seconds=stage_seconds,
        )
    if texture_fill_mode == "jfa":
        sampled, mask = _timed_texture_bake_stage(
            "jump_flood_fill",
            lambda: _jump_flood_texture_fill(
                sampled, mask, texture_dilation_pixels,
                erode_pixels=texture_erode_iterations,
            ),
            device=device,
            stage_seconds=stage_seconds,
        )
    if texture_fill_mode == "gpu_dilate":
        sampled, filled_mask = _timed_texture_bake_stage(
            "dilate_texture_padding",
            lambda: _dilate_texture_padding(
                sampled, mask, texture_dilation_pixels
            ),
            device=device,
            stage_seconds=stage_seconds,
        )

    def transfer_and_quantize_textures():
        return (
            mask.cpu().numpy(),
            np.clip(
                sampled[..., PBR_ATTR_LAYOUT["base_color"]].cpu().numpy() * 255,
                0,
                255,
            ).astype(np.uint8),
            np.clip(
                sampled[..., PBR_ATTR_LAYOUT["metallic"]].cpu().numpy() * 255,
                0,
                255,
            ).astype(np.uint8),
            np.clip(
                sampled[..., PBR_ATTR_LAYOUT["roughness"]].cpu().numpy() * 255,
                0,
                255,
            ).astype(np.uint8),
            np.clip(
                sampled[..., PBR_ATTR_LAYOUT["alpha"]].cpu().numpy() * 255,
                0,
                255,
            ).astype(np.uint8),
        )

    mask_np, base_color, metallic, roughness, alpha = _timed_texture_bake_stage(
        "transfer_quantize_textures",
        transfer_and_quantize_textures,
        device=device,
        stage_seconds=stage_seconds,
    )
    if texture_fill_mode == "telea":
        invalid = (~mask_np).astype(np.uint8)
        base_color = _timed_texture_bake_stage(
            "inpaint_base_color",
            lambda: cv2.inpaint(base_color, invalid, 3, cv2.INPAINT_TELEA),
            device=device,
            stage_seconds=stage_seconds,
        )
        metallic = _timed_texture_bake_stage(
            "inpaint_metallic",
            lambda: cv2.inpaint(metallic, invalid, 1, cv2.INPAINT_TELEA)[..., None],
            device=device,
            stage_seconds=stage_seconds,
        )
        roughness = _timed_texture_bake_stage(
            "inpaint_roughness",
            lambda: cv2.inpaint(roughness, invalid, 1, cv2.INPAINT_TELEA)[..., None],
            device=device,
            stage_seconds=stage_seconds,
        )
        alpha = _timed_texture_bake_stage(
            "inpaint_alpha",
            lambda: cv2.inpaint(alpha, invalid, 1, cv2.INPAINT_TELEA)[..., None],
            device=device,
            stage_seconds=stage_seconds,
        )

    def construct_material():
        return trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(
                np.concatenate([base_color, alpha], axis=-1)
            ),
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=Image.fromarray(
                np.concatenate(
                    [np.zeros_like(metallic), roughness, metallic], axis=-1
                )
            ),
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode="OPAQUE",
            doubleSided=True,
        )

    material = _timed_texture_bake_stage(
        "construct_material",
        construct_material,
        device=device,
        stage_seconds=stage_seconds,
    )

    def construct_trimesh():
        vertices_np = processed.vertices.numpy().astype(np.float32, copy=False)
        faces_np = processed.faces.numpy().astype(np.int64, copy=False)
        uvs_np = processed.uvs.numpy().astype(np.float32, copy=True)
        uvs_np[:, 1] = 1 - uvs_np[:, 1]
        return trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            vertex_normals=normals,
            process=False,
            visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material),
        )

    textured = _timed_texture_bake_stage(
        "construct_trimesh",
        construct_trimesh,
        device=device,
        stage_seconds=stage_seconds,
    )
    if profile is not None:
        wall_seconds = float(time.perf_counter() - profile_started)
        stage_sum_seconds = float(sum(stage_seconds.values()))
        valid_texels = int(mask.sum().item())
        total_texels = int(mask.numel())
        profile.update(
            {
                "schema": "ff_holoscene.texture_bake_profile.v1",
                "stage_seconds": stage_seconds,
                "stage_sum_seconds": stage_sum_seconds,
                "wall_seconds": wall_seconds,
                "timer_accounting_gap_seconds": wall_seconds - stage_sum_seconds,
                "texture_size": int(texture_size),
                "texture_texels": total_texels,
                "valid_texels": valid_texels,
                "valid_texel_fraction": valid_texels / total_texels,
                "filled_texels": (
                    total_texels
                    if texture_fill_mode == "telea"
                    else int(filled_mask.sum().item())
                ),
                "texture_erode_iterations": int(texture_erode_iterations),
                "surface_mapping": surface_mapping,
                "sparse_query_mode": sparse_query_mode,
                "query_count": query_count,
                "padded_query_count": padded_query_count,
                "query_padding_fraction": (
                    padded_query_count - query_count
                ) / padded_query_count,
                "texture_fill_mode": texture_fill_mode,
                "texture_dilation_pixels": int(texture_dilation_pixels),
                "uv_raster_depth_mode": (
                    "projection_camera_depth"
                    if uv_raster_depths is not None
                    else "flat_draw_order"
                ),
                "source_vertices": int(source["vertices"].shape[0]),
                "source_faces": int(source["faces"].shape[0]),
                "processed_vertices": int(vertices.shape[0]),
                "processed_faces": int(faces.shape[0]),
                "attribute_voxels": int(attrs_volume.shape[0]),
                "attribute_channels": int(attrs_volume.shape[1]),
            }
        )
    return textured


def background_aware_decimation_target(
    position: int,
    decimation_target: int,
    background_object_position: int | None,
    multiplier: float,
) -> int:
    """Decimation target for one object, boosting the reconstructed room.

    The room spans the whole scene while furniture occupies a fraction of it,
    so one shared target leaves the background with far lower triangle density
    per unit area. ``background_object_position`` must come from the room-box
    prior's ``background_local_instance_id``; composition labels position 0
    "background" but that is frequently an ordinary object, so position 0 is
    not a safe proxy.
    """
    if background_object_position is None:
        return int(decimation_target)
    if int(position) != int(background_object_position):
        return int(decimation_target)
    return max(1, int(round(decimation_target * float(multiplier))))


def postprocess_and_bake_scene(
    *,
    decode_summary: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    resolution: int = 512,
    decimation_target: int = 100000,
    texture_size: int = 512,
    mesh_batch_size: int = 4,
    backend: str = "batch",
    hybrid_foreground_fill_mode: str = "gpu_dilate",
    hybrid_foreground_dilation_pixels: int = 4,
    hybrid_foreground_surface_mapping: str = "processed",
    hybrid_foreground_decimation_target: int = 20000,
    hybrid_atlas_size: int = 2048,
    projection_num_views: int = 32,
    projection_assignment_resolution: int = 512,
    projection_padding_pixels: int = 4,
    projection_view_assignment_mode: str = "visible_pixels",
    projection_preferred_visibility_ratio: float = 0.9,
    projection_raster_instance_batch_size: int = 8,
    projection_allow_scalar_fallback: bool = True,
    topology_config: PostprocessConfig | None = None,
    surface_mapping: str = "source_bvh",
    sparse_query_mode: str = "exact",
    texture_fill_mode: str = "gpu_dilate",
    texture_dilation_pixels: int = 32,
    texture_erode_iterations: int = 2,
    background_object_position: int | None = None,
    background_decimation_multiplier: float = 1.0,
    overwrite: bool = False,
    raw_objects: Sequence[dict[str, torch.Tensor]] | None = None,
    detailed_profile: bool = False,
) -> dict[str, Any]:
    """Run topology/UV processing, then bake and export canonical GLBs."""

    topology_config = topology_config or PostprocessConfig()
    topology_config.validate()
    topology_settings = asdict(topology_config)
    # Worker count affects execution only, not mesh/UV output compatibility.
    topology_settings.pop("uv_cpu_workers", None)
    summary_path = output_dir / "mesh_summary.json"
    if summary_path.is_file() and not overwrite:
        summary = json.loads(summary_path.read_text())
        compatible = (
            int(summary.get("resolution", -1)) == int(resolution)
            and int(summary.get("decimation_target", -1)) == int(decimation_target)
            and summary.get("background_object_position", None)
            == background_object_position
            and float(summary.get("background_decimation_multiplier", 1.0))
            == float(background_decimation_multiplier)
            and int(summary.get("texture_size", -1)) == int(texture_size)
            and summary.get("mesh_backend") == backend
            and summary.get("topology_config", asdict(PostprocessConfig()))
            == topology_settings
            and summary.get("surface_mapping", "source_bvh") == surface_mapping
            and summary.get("sparse_query_mode", "exact") == sparse_query_mode
            and summary.get("texture_fill_mode", "telea") == texture_fill_mode
            and int(summary.get("texture_dilation_pixels", 4))
            == int(texture_dilation_pixels)
            and int(summary.get("texture_erode_iterations", 2))
            == int(texture_erode_iterations)
            and (
                backend != "hybrid"
                or (
                    summary.get("hybrid_foreground_fill_mode")
                    == hybrid_foreground_fill_mode
                    and int(summary.get("hybrid_foreground_dilation_pixels", -1))
                    == int(hybrid_foreground_dilation_pixels)
                    and summary.get("hybrid_foreground_surface_mapping")
                    == hybrid_foreground_surface_mapping
                    and int(summary.get("hybrid_atlas_size", -1))
                    == int(hybrid_atlas_size)
                    and int(
                        summary.get("hybrid_foreground_decimation_target", -1)
                    )
                    == int(hybrid_foreground_decimation_target)
                )
            )
            and (
                backend not in ("projection", "hybrid")
                or (
                    int(summary.get("projection_num_views", -1))
                    == int(projection_num_views)
                    and int(summary.get("projection_assignment_resolution", -1))
                    == int(projection_assignment_resolution)
                    and int(summary.get("projection_padding_pixels", -1))
                    == int(projection_padding_pixels)
                    and summary.get(
                        "projection_view_assignment_mode", "visible_pixels"
                    )
                    == projection_view_assignment_mode
                    and float(
                        summary.get(
                            "projection_preferred_visibility_ratio", 0.9
                        )
                    )
                    == float(projection_preferred_visibility_ratio)
                    and int(
                        summary.get("projection_raster_instance_batch_size", 8)
                    )
                    == int(projection_raster_instance_batch_size)
                    and bool(
                        summary.get("projection_allow_scalar_fallback", True)
                    )
                    == bool(projection_allow_scalar_fallback)
                )
            )
        )
        if compatible and all(
            Path(record["canonical_glb"]).is_file()
            for record in summary.get("objects", [])
        ):
            return summary
        if not compatible:
            raise RuntimeError(
                f"Existing mesh settings do not match the requested backend: {summary_path}. "
                "Use a new output directory or enable overwrite."
            )
    if surface_mapping == "processed" and topology_config.remesh and backend != "hybrid":
        # 2026-08-27 recipe context: `processed` samples the PBR o-voxels
        # directly at the postprocessed surface and predates the narrow-band
        # remesh. The remesh moves vertices off the DC surface, so direct
        # sampling misses the sparse voxels -- measured as catastrophic black
        # regions (clean ablation D, 2026-09-01). The hybrid backend is exempt
        # because its foreground path disables remesh internally.
        raise ValueError(
            "surface_mapping='processed' is incompatible with the narrow-band "
            "remesh (black textures): pass --appearance-surface-mapping "
            "source_bvh or --no-appearance-topology-remesh"
        )
    raw_records = sorted(
        decode_summary["objects"], key=lambda row: int(row["object_position"])
    )
    scene_started = time.perf_counter()
    raw = (
        list(raw_objects)
        if raw_objects is not None
        else [_load_raw(Path(record["raw_npz"])) for record in raw_records]
    )
    if len(raw) != len(raw_records):
        raise ValueError(
            f"Raw object count {len(raw)} does not match records {len(raw_records)}"
        )
    def _target_for(position: int) -> int:
        return background_aware_decimation_target(
            position,
            decimation_target,
            background_object_position,
            background_decimation_multiplier,
        )

    mesh_inputs = [
        MeshInput(
            name=f"object_{int(record['object_position']):04d}",
            vertices=data["vertices"],
            faces=data["faces"],
            decimation_target=_target_for(int(record["object_position"])),
            metadata={"raw_npz": record.get("raw_npz")},
        )
        for record, data in zip(raw_records, raw)
    ]
    raster_context = _shared_raster_context(device)

    def _projection_postprocess(inputs, atlas_size, topology=None):
        from eval.rendering.nvdiffrast_projection_uv import (
            ProjectionAtlasConfig,
            postprocess_meshes_projection_uv,
        )

        atlas_config = ProjectionAtlasConfig(
            num_views=projection_num_views,
            assignment_resolution=projection_assignment_resolution,
            atlas_size=atlas_size,
            padding_pixels=projection_padding_pixels,
            view_assignment_mode=projection_view_assignment_mode,
            preferred_visibility_ratio=(
                projection_preferred_visibility_ratio
            ),
            raster_instance_batch_size=projection_raster_instance_batch_size,
        )
        atlas_config.validate()
        return postprocess_meshes_projection_uv(
            inputs,
            device=device,
            batch_size=mesh_batch_size,
            topology_config=topology if topology is not None else topology_config,
            atlas_config=atlas_config,
            raster_context=raster_context,
            allow_scalar_fallback=projection_allow_scalar_fallback,
        )

    # Per-object bake settings; the hybrid backend overrides foreground rows.
    bake_texture_sizes = [int(texture_size)] * len(mesh_inputs)
    bake_fill_modes = [texture_fill_mode] * len(mesh_inputs)
    bake_dilation_pixels = [int(texture_dilation_pixels)] * len(mesh_inputs)
    bake_surface_mappings = [surface_mapping] * len(mesh_inputs)

    if backend == "projection":
        processed, postprocess_summary = _projection_postprocess(
            mesh_inputs, texture_size
        )  # pure projection backend keeps the caller's topology config
    elif backend == "hybrid":
        # 2026-08-29 gate: projection texturing is validated for ordinary
        # objects but produces faceted texture on the room-scale background
        # (out of the PBR training distribution). Route the background
        # instance through the reference CuMesh-UV path and everything else
        # through the fast projection path.
        background_rows = [
            row
            for row, record in enumerate(raw_records)
            if background_object_position is not None
            and int(record["object_position"]) == int(background_object_position)
        ]
        foreground_rows = [
            row for row in range(len(mesh_inputs)) if row not in background_rows
        ]
        processed = [None] * len(mesh_inputs)
        postprocess_summary = {"backend": "hybrid"}
        if foreground_rows:
            # Foreground reproduces the 2026-08-29 gated projection recipe
            # exactly: no narrow-band remesh (it creates interior surfaces no
            # projection view can see -> black texels), 20k decimation,
            # multiplier 6, threshold 1e-6.
            from dataclasses import replace as dataclass_replace

            fg_topology = dataclass_replace(
                topology_config,
                remesh=False,
                initial_target_multiplier=6,
                simplify_threshold=1e-6,
            )
            fg_inputs = [
                dataclass_replace(
                    mesh_inputs[row],
                    decimation_target=int(hybrid_foreground_decimation_target),
                )
                for row in foreground_rows
            ]
            fg_processed, fg_summary = _projection_postprocess(
                fg_inputs, hybrid_atlas_size, fg_topology
            )
            for row, mesh in zip(foreground_rows, fg_processed):
                processed[row] = mesh
                bake_texture_sizes[row] = int(hybrid_atlas_size)
                bake_fill_modes[row] = hybrid_foreground_fill_mode
                bake_dilation_pixels[row] = int(hybrid_foreground_dilation_pixels)
                bake_surface_mappings[row] = hybrid_foreground_surface_mapping
            postprocess_summary["foreground"] = fg_summary
        if background_rows:
            bg_processed, bg_summary = postprocess_meshes(
                [mesh_inputs[row] for row in background_rows],
                backend="scalar",
                batch_size=1,
                device=device,
                allow_scalar_fallback=True,
                config=topology_config,
                detailed_profile=detailed_profile,
            )
            for row, mesh in zip(background_rows, bg_processed):
                processed[row] = mesh
            postprocess_summary["background"] = bg_summary
        if any(mesh is None for mesh in processed):
            raise RuntimeError("Hybrid postprocess left unassigned objects")
    else:
        processed, postprocess_summary = postprocess_meshes(
            mesh_inputs,
            backend=backend,
            batch_size=mesh_batch_size,
            device=device,
            allow_scalar_fallback=True,
            config=topology_config,
            detailed_profile=detailed_profile,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    objects = []
    for row, (record, source, mesh) in enumerate(zip(raw_records, raw, processed)):
        position = int(record["object_position"])
        object_dir = output_dir / f"object_{position:04d}"
        object_dir.mkdir(parents=True, exist_ok=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        bake_started = time.perf_counter()
        texture_profile = {} if detailed_profile else None
        textured = bake_preprocessed_texture(
            source=source,
            processed=mesh,
            device=device,
            resolution=resolution,
            texture_size=bake_texture_sizes[row],
            raster_context=raster_context,
            surface_mapping=bake_surface_mappings[row],
            sparse_query_mode=sparse_query_mode,
            texture_fill_mode=bake_fill_modes[row],
            texture_dilation_pixels=bake_dilation_pixels[row],
            texture_erode_iterations=texture_erode_iterations,
            profile=texture_profile,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        texture_bake_seconds = float(time.perf_counter() - bake_started)
        if texture_profile is not None:
            texture_profile["wall_seconds"] = texture_bake_seconds
        glb_path = object_dir / "canonical.glb"
        ply_path = object_dir / "canonical_geometry.ply"
        export_started = time.perf_counter()
        textured.export(glb_path)
        textured.copy().export(ply_path)
        export_seconds = float(time.perf_counter() - export_started)
        objects.append(
            {
                "object_position": position,
                "canonical_glb": str(glb_path),
                "canonical_ply": str(ply_path),
                "vertices": int(len(textured.vertices)),
                "faces": int(len(textured.faces)),
                "texture_bake_seconds": texture_bake_seconds,
                "texture_bake_profile": texture_profile,
                "mesh_export_seconds": export_seconds,
                "postprocess": mesh.record,
            }
        )
        del textured
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = {
        "schema": "ff_holoscene.lc64_shape_pbr_textured_objects.v1",
        "resolution": resolution,
        "decimation_target": decimation_target,
        "background_object_position": background_object_position,
        "background_decimation_multiplier": background_decimation_multiplier,
        "texture_size": texture_size,
        "mesh_backend": backend,
        "hybrid_foreground_fill_mode": (
            hybrid_foreground_fill_mode if backend == "hybrid" else None
        ),
        "hybrid_foreground_dilation_pixels": (
            int(hybrid_foreground_dilation_pixels) if backend == "hybrid" else None
        ),
        "hybrid_foreground_surface_mapping": (
            hybrid_foreground_surface_mapping if backend == "hybrid" else None
        ),
        "hybrid_atlas_size": (
            int(hybrid_atlas_size) if backend == "hybrid" else None
        ),
        "hybrid_foreground_decimation_target": (
            int(hybrid_foreground_decimation_target)
            if backend == "hybrid"
            else None
        ),
        "mesh_batch_size": mesh_batch_size,
        "uv_cpu_workers": topology_config.uv_cpu_workers,
        "detailed_profile": bool(detailed_profile),
        "topology_config": topology_settings,
        "surface_mapping": surface_mapping,
        "sparse_query_mode": sparse_query_mode,
        "texture_fill_mode": texture_fill_mode,
        "texture_dilation_pixels": texture_dilation_pixels,
        "texture_erode_iterations": texture_erode_iterations,
        "projection_num_views": projection_num_views if backend == "projection" else None,
        "projection_assignment_resolution": (
            projection_assignment_resolution if backend == "projection" else None
        ),
        "projection_padding_pixels": (
            projection_padding_pixels if backend == "projection" else None
        ),
        "projection_view_assignment_mode": (
            projection_view_assignment_mode if backend == "projection" else None
        ),
        "projection_preferred_visibility_ratio": (
            projection_preferred_visibility_ratio
            if backend == "projection"
            else None
        ),
        "projection_raster_instance_batch_size": (
            projection_raster_instance_batch_size
            if backend == "projection"
            else None
        ),
        "projection_allow_scalar_fallback": (
            bool(projection_allow_scalar_fallback)
            if backend == "projection"
            else None
        ),
        "postprocess_summary": postprocess_summary,
        "texture_bake_seconds": float(
            sum(record["texture_bake_seconds"] for record in objects)
        ),
        "mesh_export_seconds": float(
            sum(record["mesh_export_seconds"] for record in objects)
        ),
        "wall_seconds": float(time.perf_counter() - scene_started),
        "objects": objects,
    }
    atomic_json(summary_path, summary)
    return summary


def compose_textured_scene(
    *,
    mesh_summary: dict[str, Any],
    object_transforms: np.ndarray,
    selected_instance_ids: Sequence[int],
    output_path: Path,
    source_instance_ids: Sequence[int] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Place canonical textured objects with authoritative dataset transforms."""

    import trimesh

    if output_path.is_file() and not overwrite:
        return {"output_glb": str(output_path), "status": "existing"}
    transforms = np.asarray(object_transforms, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(f"Expected transforms [K,4,4], got {transforms.shape}")
    if len(mesh_summary["objects"]) != len(selected_instance_ids):
        raise ValueError("Textured object count and selected instance count differ")
    output_ids = (
        selected_instance_ids
        if source_instance_ids is None
        else source_instance_ids
    )
    if len(mesh_summary["objects"]) != len(output_ids):
        raise ValueError("Textured object count and source instance count differ")
    scene = trimesh.Scene()
    records = []
    for record, transform_index, instance_id in zip(
        mesh_summary["objects"], selected_instance_ids, output_ids
    ):
        position = int(record["object_position"])
        transform_index = int(transform_index)
        instance_id = int(instance_id)
        if transform_index < 0 or transform_index >= len(transforms):
            raise IndexError(
                f"Transform index {transform_index} is outside transform array"
            )
        mesh = trimesh.load(record["canonical_glb"], force="mesh", process=False)
        local_to_world = np.linalg.inv(transforms[transform_index])
        name = "background" if instance_id == 0 else f"instance_{instance_id:04d}"
        node_name = f"{name}_position_{position:04d}"
        scene.add_geometry(
            mesh,
            geom_name=node_name,
            node_name=node_name,
            transform=local_to_world,
        )
        records.append(
            {
                "object_position": position,
                "instance_id": instance_id,
                "node_name": node_name,
                "canonical_glb": record["canonical_glb"],
                "object_to_world": local_to_world.tolist(),
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "background_pbr_out_of_training_distribution": instance_id == 0,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output_path)
    return {
        "schema": "ff_holoscene.lc64_shape_pbr_composed_scene.v1",
        "output_glb": str(output_path),
        "num_objects": len(records),
        "objects": records,
        "status": "written",
    }
