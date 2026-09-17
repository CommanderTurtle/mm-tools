#!/usr/bin/env python3
"""Batch CuMesh cleanup, simplification, and UV unwrap for decoded ovoxel meshes.

The module is both an importable adapter for the LC64 reconstruction pipeline
and a standalone diagnostic CLI.  It deliberately stops before PBR attribute
sampling/material construction: the returned UV-ready meshes are the input to
the pipeline's factorized per-object texture-baking stage, without repeating
the topology work.

The batched implementation mirrors the ``remesh=False`` topology path in
``o_voxel.postprocess.to_glb``.  Raw flexible-dual-grid extraction is outside
this adapter and remains per object.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PostprocessConfig:
    """Options matching the scalar o_voxel ``to_glb(remesh=False)`` path."""

    hole_perimeter: float = 3e-2
    small_component_area: float = 1e-5
    initial_target_multiplier: int = 3
    simplify_threshold: float = 1e-8
    chart_cone_half_angle_rad: float = math.radians(90.0)
    chart_refine_iterations: int = 0
    chart_global_iterations: int = 1
    chart_smooth_strength: float = 1.0
    chart_area_penalty_weight: float = 0.1
    chart_perimeter_area_ratio_weight: float = 0.0001
    topology_schedule: str = "full"
    # CuMesh narrow-band dual-contouring remesh, mirroring
    # o_voxel.postprocess.to_glb(remesh=True). On by default at band 1; pass
    # remesh=False to restore the older cleanup-only schedule.
    remesh: bool = True
    remesh_band: float = 1.0
    remesh_project: float = 0.0
    remesh_aabb_extent: float = 1.0
    remesh_resolution: int = 512
    # xatlas PackOptions.block_align: aligning charts to 4x4 blocks trades a
    # little packing tightness for a large reported packing speedup.
    xatlas_block_align: bool = False
    # None uses one worker per mesh, capped by the host CPU count. This only
    # controls the per-mesh xatlas stage; CuMesh chart extraction stays batched.
    uv_cpu_workers: int | None = None

    def validate(self) -> None:
        if self.hole_perimeter < 0:
            raise ValueError("hole_perimeter must be non-negative")
        if self.small_component_area < 0:
            raise ValueError("small_component_area must be non-negative")
        if self.initial_target_multiplier < 1:
            raise ValueError("initial_target_multiplier must be at least one")
        if self.simplify_threshold <= 0:
            raise ValueError("simplify_threshold must be positive")
        if self.chart_area_penalty_weight < 0:
            raise ValueError("chart_area_penalty_weight must be non-negative")
        if self.chart_perimeter_area_ratio_weight < 0:
            raise ValueError(
                "chart_perimeter_area_ratio_weight must be non-negative"
            )
        if self.remesh_band <= 0:
            raise ValueError("remesh_band must be positive")
        if not 0.0 <= self.remesh_project <= 1.0:
            raise ValueError("remesh_project must be within [0, 1]")
        if self.remesh_resolution <= 0:
            raise ValueError("remesh_resolution must be positive")
        if self.uv_cpu_workers is not None and self.uv_cpu_workers <= 0:
            raise ValueError("uv_cpu_workers must be positive")
        if self.topology_schedule not in {
            "full",
            "single_cleanup",
            "simplify_orient",
        }:
            raise ValueError(
                f"Unsupported topology schedule: {self.topology_schedule}"
            )


@dataclass
class MeshInput:
    name: str
    vertices: torch.Tensor
    faces: torch.Tensor
    decimation_target: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PostprocessedMesh:
    name: str
    vertices: torch.Tensor
    faces: torch.Tensor
    uvs: torch.Tensor
    normals: torch.Tensor
    vmaps: torch.Tensor
    record: dict[str, Any]
    # Projection atlases use camera-space depth to resolve overlapping UV
    # triangles during texture baking. XAtlas meshes leave this unset.
    uv_raster_depths: torch.Tensor | None = None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _start_batch_profile(
    device: torch.device, enabled: bool
) -> dict[str, float | int] | None:
    if not enabled:
        return None
    _sync(device)
    profile: dict[str, float | int] = {"started": time.perf_counter()}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        profile["start_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
        profile["start_reserved_bytes"] = int(torch.cuda.memory_reserved(device))
    return profile


def _finish_batch_profile(
    device: torch.device, state: dict[str, float | int] | None
) -> dict[str, float | int] | None:
    if state is None:
        return None
    _sync(device)
    profile = {key: value for key, value in state.items() if key != "started"}
    profile["wall_seconds"] = float(time.perf_counter() - state["started"])
    if device.type == "cuda":
        profile["peak_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        profile["peak_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device)
        )
    return profile


def _load_cumesh_modules():
    """Load compiled cumesh first, then the repo's pure-Python batch wrapper.

    The compiled ``cumesh`` extension comes from the environment. The batch
    wrapper is pure Python and is loaded from this checkout first so committed
    worker controls and profiling stay in sync with the unified runner.
    """

    try:
        import cumesh
    except ImportError as error:  # pragma: no cover - cluster dependency
        raise RuntimeError(
            "The compiled cumesh package is unavailable in this environment"
        ) from error

    source_root = REPO_ROOT / "trellis2_x2" / "CuMesh"
    if (source_root / "cumesh_batch" / "__init__.py").is_file():
        source_text = str(source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
    try:
        import cumesh_batch
    except ImportError:
        if not (source_root / "cumesh_batch" / "__init__.py").is_file():
            raise RuntimeError(
                "cumesh_batch is neither installed nor present at "
                f"{source_root / 'cumesh_batch'}"
            )
        try:
            import cumesh_batch
        except ImportError as error:  # pragma: no cover - cluster dependency
            raise RuntimeError(f"Could not import cumesh_batch from {source_root}") from error
    return cumesh, cumesh_batch


def _as_mesh_list(value: Any, expected: int, name: str) -> list[torch.Tensor]:
    if isinstance(value, (list, tuple)):
        result = list(value)
    elif torch.is_tensor(value) and expected == 1:
        result = [value]
    else:
        raise TypeError(f"{name} must be a tensor for one mesh or a tensor sequence")
    if len(result) != expected:
        raise RuntimeError(f"{name} returned {len(result)} meshes, expected {expected}")
    return result


def _validate_input(mesh: MeshInput) -> None:
    if not mesh.name:
        raise ValueError("Mesh name cannot be empty")
    if mesh.decimation_target <= 0:
        raise ValueError(f"{mesh.name}: decimation_target must be positive")
    if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3:
        raise ValueError(f"{mesh.name}: vertices must have shape [V,3]")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError(f"{mesh.name}: faces must have shape [F,3]")
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise ValueError(f"{mesh.name}: empty meshes cannot be postprocessed")
    if not torch.isfinite(mesh.vertices).all().item():
        raise ValueError(f"{mesh.name}: vertices contain non-finite values")
    faces_long = mesh.faces.long()
    if int(faces_long.min().item()) < 0:
        raise ValueError(f"{mesh.name}: faces contain negative indices")
    if int(faces_long.max().item()) >= mesh.vertices.shape[0]:
        raise ValueError(f"{mesh.name}: face index exceeds vertex count")


def _prepare_inputs(meshes: Sequence[MeshInput], device: torch.device) -> list[MeshInput]:
    if device.type != "cuda":
        raise ValueError("CuMesh postprocessing requires a CUDA device")
    if not meshes:
        raise ValueError("At least one mesh is required")
    prepared: list[MeshInput] = []
    names: set[str] = set()
    for mesh in meshes:
        _validate_input(mesh)
        if mesh.name in names:
            raise ValueError(f"Duplicate mesh name: {mesh.name}")
        names.add(mesh.name)
        prepared.append(
            MeshInput(
                name=mesh.name,
                vertices=mesh.vertices.detach().to(device=device, dtype=torch.float32).contiguous(),
                faces=mesh.faces.detach().to(device=device, dtype=torch.int32).contiguous(),
                decimation_target=int(mesh.decimation_target),
                metadata=dict(mesh.metadata),
            )
        )
    return prepared


def _run_remesh_topology(
    handler: Any,
    targets: int | list[int],
    config: PostprocessConfig,
    *,
    device: torch.device | None = None,
    stage_timings: dict[str, float] | None = None,
) -> None:
    """Mirror o_voxel.postprocess.to_glb's remesh=True branch.

    fill holes, build a BVH over the current mesh, rebuild topology with
    CuMesh narrow-band dual contouring, then simplify once to the target. The
    cleanup loop and unify_face_orientations of the non-remesh branch are
    deliberately absent there, so they are absent here too.

    A list of targets means ``handler`` is a ``cumesh_batch.CuMesh``: the
    batched ``cumesh_batch.remeshing.remesh_narrow_band_dc`` re-contours every
    mesh in one pass. ``read()``/``init()`` on the batch handler speak per-mesh
    lists in the original mesh frames, so the two paths share the exact
    parameterisation (center 0, padded scale, one resolution).
    """
    cumesh, cumesh_batch = _load_cumesh_modules()
    batched = isinstance(targets, list)

    def run(name: str, function):
        if stage_timings is None:
            return function()
        _sync(device)
        started = time.perf_counter()
        result = function()
        _sync(device)
        stage_timings[name] = float(time.perf_counter() - started)
        return result

    run("remesh_fill_holes",
        lambda: handler.fill_holes(max_hole_perimeter=config.hole_perimeter))
    vertices, faces = handler.read()

    resolution = int(config.remesh_resolution)
    scale = float(config.remesh_aabb_extent)
    padded_scale = (resolution + 3 * config.remesh_band) / resolution * scale

    if batched:
        bvh = run("remesh_build_bvh", lambda: cumesh_batch.cuBVH(vertices, faces))
        center = torch.zeros(3, device=vertices[0].device, dtype=vertices[0].dtype)
        remeshed = run(
            "remesh_narrow_band_dc",
            lambda: cumesh_batch.remeshing.remesh_narrow_band_dc(
                vertices,
                faces,
                center=center,
                scale=padded_scale,
                resolution=resolution,
                band=config.remesh_band,
                project_back=config.remesh_project,
                verbose=False,
                bvh=bvh,
            ),
        )
        out_vertices, out_faces = remeshed
        empty = [
            index
            for index, mesh_faces in enumerate(out_faces)
            if int(mesh_faces.shape[0]) == 0
        ]
        if empty:
            raise RuntimeError(
                f"Batched narrow-band remesh returned empty meshes at positions {empty}"
            )
        handler.init(out_vertices, out_faces)
    else:
        bvh = run("remesh_build_bvh", lambda: cumesh.cuBVH(vertices, faces))
        center = torch.zeros(3, device=vertices.device, dtype=vertices.dtype)
        remeshed = run(
            "remesh_narrow_band_dc",
            lambda: cumesh.remeshing.remesh_narrow_band_dc(
                vertices,
                faces,
                center=center,
                scale=padded_scale,
                resolution=resolution,
                band=config.remesh_band,
                project_back=config.remesh_project,
                verbose=False,
                bvh=bvh,
            ),
        )
        handler.init(*remeshed)

    simplify_kwargs: dict[str, Any] = {"verbose": False}
    if config.simplify_threshold != PostprocessConfig.simplify_threshold:
        simplify_kwargs["options"] = {"thresh": config.simplify_threshold}
    run("remesh_simplify", lambda: handler.simplify(targets, **simplify_kwargs))


def _run_topology(
    handler: Any,
    targets: int | list[int],
    config: PostprocessConfig,
    *,
    device: torch.device | None = None,
    stage_timings: dict[str, float] | None = None,
) -> None:
    """Mirror o_voxel.postprocess.to_glb's non-remeshing topology sequence."""

    if config.remesh:
        _run_remesh_topology(
            handler, targets, config, device=device, stage_timings=stage_timings
        )
        return

    if stage_timings is not None and device is None:
        raise ValueError("device is required when collecting topology stage timings")

    def run(name: str, function) -> None:
        if stage_timings is None:
            function()
            return
        _sync(device)
        started = time.perf_counter()
        function()
        _sync(device)
        stage_timings[name] = float(time.perf_counter() - started)

    if isinstance(targets, list):
        first_targets: int | list[int] = [
            target * config.initial_target_multiplier for target in targets
        ]
    else:
        first_targets = targets * config.initial_target_multiplier

    simplify_kwargs: dict[str, Any] = {"verbose": False}
    if config.simplify_threshold != PostprocessConfig.simplify_threshold:
        simplify_kwargs["options"] = {"thresh": config.simplify_threshold}

    if config.topology_schedule == "full":
        run(
            "initial_fill_holes",
            lambda: handler.fill_holes(
                max_hole_perimeter=config.hole_perimeter
            ),
        )
        run(
            "initial_simplify",
            lambda: handler.simplify(first_targets, **simplify_kwargs),
        )
        run("initial_remove_duplicate_faces", handler.remove_duplicate_faces)
        run("initial_repair_non_manifold_edges", handler.repair_non_manifold_edges)
        run(
            "initial_remove_small_components",
            lambda: handler.remove_small_connected_components(
                config.small_component_area
            ),
        )
        run(
            "middle_fill_holes",
            lambda: handler.fill_holes(
                max_hole_perimeter=config.hole_perimeter
            ),
        )

    run("final_simplify", lambda: handler.simplify(targets, **simplify_kwargs))

    if config.topology_schedule in {"full", "single_cleanup"}:
        run("final_remove_duplicate_faces", handler.remove_duplicate_faces)
        run("final_repair_non_manifold_edges", handler.repair_non_manifold_edges)
        run(
            "final_remove_small_components",
            lambda: handler.remove_small_connected_components(
                config.small_component_area
            ),
        )
        run(
            "final_fill_holes",
            lambda: handler.fill_holes(
                max_hole_perimeter=config.hole_perimeter
            ),
        )

    run("unify_face_orientations", handler.unify_face_orientations)


def _uv_kwargs(config: PostprocessConfig) -> dict[str, Any]:
    return {
        "compute_charts_kwargs": {
            "threshold_cone_half_angle_rad": config.chart_cone_half_angle_rad,
            "refine_iterations": config.chart_refine_iterations,
            "global_iterations": config.chart_global_iterations,
            "smooth_strength": config.chart_smooth_strength,
            "area_penalty_weight": config.chart_area_penalty_weight,
            "perimeter_area_ratio_weight": (
                config.chart_perimeter_area_ratio_weight
            ),
        },
        "xatlas_pack_charts_kwargs": {
            "block_align": config.xatlas_block_align,
        },
        "return_vmaps": True,
        "verbose": False,
    }


def _local_batch_vmaps(handler: Any, global_vmaps: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Convert cumesh_batch's packed global vmaps into per-object local maps."""

    _, _, _, batch_vertex_ids, _ = handler._read_partition()  # batch wrapper contract
    packed_vertices, _ = handler.mesh.read()
    local_vmaps: list[torch.Tensor] = []
    for global_ids, vmap in zip(batch_vertex_ids, global_vmaps):
        remap = torch.full(
            (packed_vertices.shape[0],),
            -1,
            dtype=torch.long,
            device=packed_vertices.device,
        )
        remap[global_ids.long()] = torch.arange(
            global_ids.shape[0], device=packed_vertices.device, dtype=torch.long
        )
        local = remap[vmap.to(device=packed_vertices.device, dtype=torch.long)]
        if (local < 0).any().item():
            raise RuntimeError("UV vmap crossed a packed-mesh batch boundary")
        local_vmaps.append(local)
    return local_vmaps


def _bounds(vertices: torch.Tensor) -> list[list[float]]:
    return torch.stack((vertices.min(dim=0).values, vertices.max(dim=0).values)).tolist()


def _validate_output(
    name: str,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    uvs: torch.Tensor,
    normals: torch.Tensor,
    vmaps: torch.Tensor,
) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise RuntimeError(f"{name}: invalid output vertices {tuple(vertices.shape)}")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise RuntimeError(f"{name}: invalid output faces {tuple(faces.shape)}")
    if uvs.shape != (vertices.shape[0], 2):
        raise RuntimeError(f"{name}: UV/vertex shape mismatch")
    if normals.shape != vertices.shape:
        raise RuntimeError(f"{name}: normal/vertex shape mismatch")
    if vmaps.shape != (vertices.shape[0],):
        raise RuntimeError(f"{name}: vmap/vertex shape mismatch")
    if not all(torch.isfinite(value).all().item() for value in (vertices, uvs, normals)):
        raise RuntimeError(f"{name}: non-finite output values")
    faces_long = faces.long()
    if int(faces_long.min().item()) < 0 or int(faces_long.max().item()) >= vertices.shape[0]:
        raise RuntimeError(f"{name}: output face indices are not local to this mesh")
    normal_lengths = torch.linalg.vector_norm(normals.float(), dim=1)
    if not (normal_lengths > 0).any().item():
        raise RuntimeError(f"{name}: every output normal is zero")


def _result(
    mesh: MeshInput,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    uvs: torch.Tensor,
    normals: torch.Tensor,
    vmaps: torch.Tensor,
    *,
    backend: str,
    batch_id: int,
    batch_size: int,
    topology_seconds: float,
    uv_seconds: float,
    fallback_reason: str | None,
    uv_stage_timings: dict[str, float] | None = None,
    topology_stage_timings: dict[str, float] | None = None,
) -> PostprocessedMesh:
    values = [vertices, faces, uvs, normals, vmaps]
    vertices_cpu, faces_cpu, uvs_cpu, normals_cpu, vmaps_cpu = [
        value.detach().cpu().contiguous() for value in values
    ]
    _validate_output(
        mesh.name, vertices_cpu, faces_cpu, uvs_cpu, normals_cpu, vmaps_cpu
    )
    record = {
        "name": mesh.name,
        "backend": backend,
        "batch_id": batch_id,
        "batch_size": batch_size,
        "fallback_reason": fallback_reason,
        "decimation_target": mesh.decimation_target,
        "input_vertices": int(mesh.vertices.shape[0]),
        "input_faces": int(mesh.faces.shape[0]),
        "output_vertices_uv_expanded": int(vertices_cpu.shape[0]),
        "output_faces": int(faces_cpu.shape[0]),
        "input_bounds": _bounds(mesh.vertices.detach().cpu()),
        "output_bounds": _bounds(vertices_cpu),
        "topology_seconds_shared": topology_seconds,
        "uv_seconds_shared": uv_seconds,
        **(
            {f"uv_{key}_shared": value for key, value in uv_stage_timings.items()}
            if uv_stage_timings
            else {}
        ),
        **(
            {
                f"topology_{key}_shared": value
                for key, value in topology_stage_timings.items()
            }
            if topology_stage_timings
            else {}
        ),
        "total_seconds_shared": topology_seconds + uv_seconds,
        "metadata": mesh.metadata,
    }
    return PostprocessedMesh(
        name=mesh.name,
        vertices=vertices_cpu,
        faces=faces_cpu,
        uvs=uvs_cpu,
        normals=normals_cpu,
        vmaps=vmaps_cpu,
        record=record,
    )


def _postprocess_scalar(
    mesh: MeshInput,
    *,
    config: PostprocessConfig,
    device: torch.device,
    batch_id: int,
    fallback_reason: str | None = None,
    detailed_profile: bool = False,
) -> PostprocessedMesh:
    cumesh, _ = _load_cumesh_modules()
    handler = cumesh.CuMesh()
    handler.init(mesh.vertices, mesh.faces)
    _sync(device)
    start = time.perf_counter()
    topology_stage_timings = {} if detailed_profile else None
    _run_topology(
        handler,
        mesh.decimation_target,
        config,
        device=device,
        stage_timings=topology_stage_timings,
    )
    _sync(device)
    topology_done = time.perf_counter()
    vertices, faces, uvs, vmaps = handler.uv_unwrap(**_uv_kwargs(config))
    handler.compute_vertex_normals()
    source_normals = handler.read_vertex_normals()
    normals = source_normals[vmaps.to(source_normals.device, dtype=torch.long)]
    _sync(device)
    uv_done = time.perf_counter()
    return _result(
        mesh,
        vertices,
        faces,
        uvs,
        normals,
        vmaps.long(),
        backend="scalar" if fallback_reason is None else "scalar_fallback",
        batch_id=batch_id,
        batch_size=1,
        topology_seconds=topology_done - start,
        uv_seconds=uv_done - topology_done,
        fallback_reason=fallback_reason,
        topology_stage_timings=topology_stage_timings,
    )


def _postprocess_batch_chunk(
    meshes: Sequence[MeshInput],
    *,
    config: PostprocessConfig,
    device: torch.device,
    batch_id: int,
    detailed_profile: bool = False,
) -> list[PostprocessedMesh]:
    _, cumesh_batch = _load_cumesh_modules()
    handler = cumesh_batch.CuMesh()
    handler.init([mesh.vertices for mesh in meshes], [mesh.faces for mesh in meshes])
    _sync(device)
    start = time.perf_counter()
    topology_stage_timings = {} if detailed_profile else None
    _run_topology(
        handler,
        [mesh.decimation_target for mesh in meshes],
        config,
        device=device,
        stage_timings=topology_stage_timings,
    )
    _sync(device)
    topology_done = time.perf_counter()
    uv_kwargs = _uv_kwargs(config)
    if config.uv_cpu_workers is not None:
        uv_kwargs["parallel_workers"] = int(config.uv_cpu_workers)
    vertices, faces, uvs, global_vmaps = handler.uv_unwrap(**uv_kwargs)
    uv_stage_timings = getattr(handler, "last_uv_timings", None)
    handler.compute_vertex_normals()
    source_normals = _as_mesh_list(handler.read_vertex_normals(), len(meshes), "normals")
    vertices_list = _as_mesh_list(vertices, len(meshes), "vertices")
    faces_list = _as_mesh_list(faces, len(meshes), "faces")
    uvs_list = _as_mesh_list(uvs, len(meshes), "uvs")
    global_vmaps_list = _as_mesh_list(global_vmaps, len(meshes), "vmaps")
    local_vmaps = _local_batch_vmaps(handler, global_vmaps_list)
    normals = [
        source_normal[vmap.to(source_normal.device, dtype=torch.long)]
        for source_normal, vmap in zip(source_normals, local_vmaps)
    ]
    _sync(device)
    uv_done = time.perf_counter()
    return [
        _result(
            mesh,
            vertex,
            face,
            uv,
            normal,
            vmap,
            backend="batch",
            batch_id=batch_id,
            batch_size=len(meshes),
            topology_seconds=topology_done - start,
            uv_seconds=uv_done - topology_done,
            fallback_reason=None,
            uv_stage_timings=uv_stage_timings,
            topology_stage_timings=topology_stage_timings,
        )
        for mesh, vertex, face, uv, normal, vmap in zip(
            meshes, vertices_list, faces_list, uvs_list, normals, local_vmaps
        )
    ]


def _chunks(values: Sequence[MeshInput], size: int) -> Iterable[Sequence[MeshInput]]:
    if size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def postprocess_meshes(
    meshes: Sequence[MeshInput],
    *,
    backend: str = "batch",
    batch_size: int = 4,
    device: str | torch.device = "cuda",
    allow_scalar_fallback: bool = True,
    config: PostprocessConfig | None = None,
    detailed_profile: bool = False,
) -> tuple[list[PostprocessedMesh], dict[str, Any]]:
    """Postprocess decoded meshes and return CPU UV-ready tensors plus summary."""

    if backend not in {"batch", "scalar"}:
        raise ValueError("backend must be 'batch' or 'scalar'")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    config = config or PostprocessConfig()
    config.validate()
    device_obj = torch.device(device)
    prepared = _prepare_inputs(meshes, device_obj)
    started = time.perf_counter()
    results: list[PostprocessedMesh] = []
    batch_records: list[dict[str, Any]] = []

    if backend == "scalar":
        for batch_id, mesh in enumerate(prepared):
            batch_profile = _start_batch_profile(device_obj, detailed_profile)
            result = _postprocess_scalar(
                mesh,
                config=config,
                device=device_obj,
                batch_id=batch_id,
                detailed_profile=detailed_profile,
            )
            results.append(result)
            batch_records.append(
                {
                    "batch_id": batch_id,
                    "requested_backend": "scalar",
                    "actual_backend": "scalar",
                    "names": [mesh.name],
                    "fallback_reason": None,
                    "profile": _finish_batch_profile(device_obj, batch_profile),
                }
            )
    else:
        for batch_id, chunk in enumerate(_chunks(prepared, batch_size)):
            names = [mesh.name for mesh in chunk]
            batch_profile = _start_batch_profile(device_obj, detailed_profile)
            if len(chunk) == 1:
                result = _postprocess_scalar(
                    chunk[0],
                    config=config,
                    device=device_obj,
                    batch_id=batch_id,
                    detailed_profile=detailed_profile,
                )
                result.record["backend"] = "scalar_singleton"
                results.append(result)
                batch_records.append(
                    {
                        "batch_id": batch_id,
                        "requested_backend": "batch",
                        "actual_backend": "scalar_singleton",
                        "names": names,
                        "fallback_reason": None,
                        "profile": _finish_batch_profile(
                            device_obj, batch_profile
                        ),
                    }
                )
                continue
            try:
                chunk_results = _postprocess_batch_chunk(
                    chunk,
                    config=config,
                    device=device_obj,
                    batch_id=batch_id,
                    detailed_profile=detailed_profile,
                )
                results.extend(chunk_results)
                batch_records.append(
                    {
                        "batch_id": batch_id,
                        "requested_backend": "batch",
                        "actual_backend": "batch",
                        "names": names,
                        "fallback_reason": None,
                        "profile": _finish_batch_profile(
                            device_obj, batch_profile
                        ),
                    }
                )
            except Exception as error:
                if not allow_scalar_fallback:
                    raise
                reason = f"{type(error).__name__}: {error}"
                if "out of memory" in str(error).lower():
                    # A remesh batch of many 512^3 narrow-band grids can
                    # exhaust the device; free the failed batch's memory
                    # before the scalar retries or they OOM too (observed on
                    # scannetpp 13285009a4, 83 objects at batch 16).
                    gc.collect()
                    if device_obj.type == "cuda":
                        torch.cuda.empty_cache()
                for mesh in chunk:
                    results.append(
                        _postprocess_scalar(
                            mesh,
                            config=config,
                            device=device_obj,
                            batch_id=batch_id,
                            fallback_reason=reason,
                            detailed_profile=detailed_profile,
                        )
                    )
                    if device_obj.type == "cuda" and "out of memory" in str(
                        error
                    ).lower():
                        torch.cuda.empty_cache()
                batch_records.append(
                    {
                        "batch_id": batch_id,
                        "requested_backend": "batch",
                        "actual_backend": "scalar_fallback",
                        "names": names,
                        "fallback_reason": reason,
                        "profile": _finish_batch_profile(
                            device_obj, batch_profile
                        ),
                    }
                )

    elapsed = time.perf_counter() - started
    summary = {
        "requested_backend": backend,
        "requested_batch_size": batch_size,
        "allow_scalar_fallback": allow_scalar_fallback,
        "device": str(device_obj),
        "num_meshes": len(results),
        "wall_seconds": elapsed,
        "config": asdict(config),
        "detailed_profile": bool(detailed_profile),
        "batches": batch_records,
        "meshes": [result.record for result in results],
    }
    return results, summary


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "mesh"


def _load_mesh_file(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            if "vertices" not in archive or "faces" not in archive:
                raise ValueError(f"{path}: NPZ must contain vertices and faces")
            vertices = np.asarray(archive["vertices"], dtype=np.float32)
            faces = np.asarray(archive["faces"], dtype=np.int32)
    else:
        import trimesh

        loaded = trimesh.load(path, force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.dump(concatenate=True)
        vertices = np.asarray(loaded.vertices, dtype=np.float32)
        faces = np.asarray(loaded.faces, dtype=np.int32)
    return torch.from_numpy(vertices), torch.from_numpy(faces)


def _manifest_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("meshes")
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("Manifest must be a list or an object containing a 'meshes' list")
    return payload


def _cli_inputs(args: argparse.Namespace) -> list[MeshInput]:
    entries: list[dict[str, Any]] = []
    base = Path.cwd()
    if args.manifest is not None:
        base = args.manifest.resolve().parent
        entries.extend(_manifest_entries(args.manifest))
    else:
        entries.extend({"path": str(path)} for path in args.mesh)

    meshes: list[MeshInput] = []
    for index, entry in enumerate(entries):
        raw_path = entry.get("path") or entry.get("mesh")
        if raw_path is None:
            raise ValueError(f"Manifest entry {index} has no path")
        source = Path(raw_path).expanduser()
        if not source.is_absolute():
            source = base / source
        if not source.is_file():
            raise FileNotFoundError(source)
        name = str(entry.get("name") or source.stem)
        target = int(entry.get("decimation_target", args.decimation_target))
        vertices, faces = _load_mesh_file(source)
        metadata = dict(entry.get("metadata") or {})
        metadata["source"] = str(source)
        meshes.append(
            MeshInput(
                name=name,
                vertices=vertices,
                faces=faces,
                decimation_target=target,
                metadata=metadata,
            )
        )
    return meshes


def _save_results(
    results: Sequence[PostprocessedMesh],
    summary: dict[str, Any],
    output: Path,
    *,
    export_ply: bool,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    artifact_records: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        stem = f"{index:04d}_{_safe_name(result.name)}"
        while stem in used:
            stem += "_dup"
        used.add(stem)
        npz_path = output / f"{stem}.npz"
        np.savez_compressed(
            npz_path,
            vertices=result.vertices.numpy(),
            faces=result.faces.numpy(),
            uvs=result.uvs.numpy(),
            normals=result.normals.numpy(),
            vmaps=result.vmaps.numpy(),
        )
        artifact = {"name": result.name, "npz": str(npz_path)}
        if export_ply:
            import trimesh

            ply_path = output / f"{stem}.ply"
            trimesh.Trimesh(
                vertices=result.vertices.numpy(),
                faces=result.faces.numpy(),
                vertex_normals=result.normals.numpy(),
                process=False,
            ).export(ply_path)
            artifact["ply"] = str(ply_path)
        artifact_records.append(artifact)
    summary = dict(summary)
    summary["artifacts"] = artifact_records
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--mesh", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("batch", "scalar"), default="batch")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--decimation-target", type=int, default=100000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-scalar-fallback", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--export-ply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meshes = _cli_inputs(args)
    results, summary = postprocess_meshes(
        meshes,
        backend=args.backend,
        batch_size=args.batch_size,
        device=args.device,
        allow_scalar_fallback=args.allow_scalar_fallback,
    )
    summary_path = _save_results(
        results, summary, args.output, export_ply=args.export_ply
    )
    print(summary_path)


if __name__ == "__main__":
    main()
