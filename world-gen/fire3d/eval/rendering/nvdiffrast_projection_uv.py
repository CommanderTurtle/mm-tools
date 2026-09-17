#!/usr/bin/env python3
"""Fast multi-view projection UVs for CuMesh-cleaned object meshes.

This module is an experimental alternative to XAtlas. It keeps the production
CuMesh topology sequence, rasterizes the cleaned mesh from several orthographic
views, assigns each face to its best visible view, and packs the projections
into fixed atlas tiles. The result has the same ``PostprocessedMesh`` contract
used by the texture baker.

Projection atlases are not guaranteed to be overlap-free. Callers must inspect
the returned visibility and overdraw diagnostics before using this backend for
production assets.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from eval.reconstruction.batched_ovoxel_postprocess import (
    MeshInput,
    PostprocessConfig,
    PostprocessedMesh,
    _bounds,
    _load_cumesh_modules,
    _prepare_inputs,
    _run_topology,
    _sync,
    _validate_input,
    _validate_output,
)


@dataclass(frozen=True)
class ProjectionAtlasConfig:
    """Configuration for raster visibility assignment and fixed-grid packing."""

    num_views: int = 8
    assignment_resolution: int = 512
    atlas_size: int = 512
    padding_pixels: int = 4
    projection_margin: float = 0.04
    front_facing_mode: str = "two_sided"
    front_facing_epsilon: float = 1e-5
    view_assignment_mode: str = "visible_pixels"
    preferred_visibility_ratio: float = 0.9
    low_visibility_ratio: float = 0.25
    raster_instance_batch_size: int = 8

    def validate(self) -> None:
        if self.num_views < 1:
            raise ValueError("num_views must be positive")
        if self.assignment_resolution < 8:
            raise ValueError("assignment_resolution must be at least 8")
        if self.atlas_size < 8:
            raise ValueError("atlas_size must be at least 8")
        if self.padding_pixels < 0:
            raise ValueError("padding_pixels must be non-negative")
        if not 0 <= self.projection_margin < 0.5:
            raise ValueError("projection_margin must be in [0, 0.5)")
        if self.front_facing_mode not in {"two_sided", "normal_gate"}:
            raise ValueError(
                "front_facing_mode must be 'two_sided' or 'normal_gate'"
            )
        if self.view_assignment_mode not in {"visible_pixels", "coverage_first"}:
            raise ValueError(
                "view_assignment_mode must be 'visible_pixels' or "
                "'coverage_first'"
            )
        if not 0 <= self.preferred_visibility_ratio <= 1:
            raise ValueError("preferred_visibility_ratio must be in [0, 1]")
        if not 0 <= self.low_visibility_ratio <= 1:
            raise ValueError("low_visibility_ratio must be in [0, 1]")
        if self.raster_instance_batch_size < 1:
            raise ValueError("raster_instance_batch_size must be positive")
        columns, rows = atlas_grid(self.num_views)
        if 2 * self.padding_pixels >= min(
            self.atlas_size / columns, self.atlas_size / rows
        ):
            raise ValueError("padding consumes an entire atlas tile")


@dataclass
class ProjectionUVResult:
    """GPU-resident UV-expanded mesh and projection quality measurements."""

    vertices: torch.Tensor
    faces: torch.Tensor
    uvs: torch.Tensor
    normals: torch.Tensor
    vmaps: torch.Tensor
    uv_raster_depths: torch.Tensor
    face_view_ids: torch.Tensor
    face_seen: torch.Tensor
    diagnostics: dict[str, Any]
    timings: dict[str, float]


def atlas_grid(num_views: int) -> tuple[int, int]:
    """Return a near-square fixed tile grid for ``num_views`` cameras."""

    if num_views < 1:
        raise ValueError("num_views must be positive")
    columns = int(math.ceil(math.sqrt(num_views)))
    rows = int(math.ceil(num_views / columns))
    return columns, rows


def view_directions(
    num_views: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create approximately uniform camera directions on a sphere."""

    if num_views < 1:
        raise ValueError("num_views must be positive")
    if num_views == 1:
        return torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype)
    index = torch.arange(num_views, device=device, dtype=dtype)
    z = 1.0 - 2.0 * (index + 0.5) / num_views
    radius = torch.sqrt(torch.clamp(1.0 - z.square(), min=0.0))
    azimuth = index * (math.pi * (3.0 - math.sqrt(5.0)))
    return torch.stack(
        (radius * torch.cos(azimuth), radius * torch.sin(azimuth), z),
        dim=1,
    )


def _camera_bases(directions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    world_up = torch.tensor(
        [0.0, 0.0, 1.0], device=directions.device, dtype=directions.dtype
    ).expand_as(directions)
    alternate_up = torch.tensor(
        [0.0, 1.0, 0.0], device=directions.device, dtype=directions.dtype
    ).expand_as(directions)
    near_pole = torch.abs((directions * world_up).sum(dim=1, keepdim=True)) > 0.95
    up_hint = torch.where(near_pole, alternate_up, world_up)
    right = F.normalize(torch.cross(up_hint, directions, dim=1), dim=1)
    up = F.normalize(torch.cross(directions, right, dim=1), dim=1)
    return right, up


def _project_vertices(
    vertices: torch.Tensor,
    directions: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    center = (vertices.amin(dim=0) + vertices.amax(dim=0)) * 0.5
    radius = torch.linalg.vector_norm(vertices - center, dim=1).amax()
    radius = torch.clamp(radius / (1.0 - margin), min=1e-8)
    relative = vertices - center
    right, up = _camera_bases(directions)
    x = torch.einsum("vc,nc->nv", relative, right) / radius
    y = torch.einsum("vc,nc->nv", relative, up) / radius
    # OpenGL NDC uses smaller z for points nearer the camera.
    z = -torch.einsum("vc,nc->nv", relative, directions) / radius
    ones = torch.ones_like(x)
    clip_positions = torch.stack((x, y, z, ones), dim=-1)
    return clip_positions, radius


def _face_geometry(
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    triangles = vertices[faces.long()]
    cross = torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=1)
    double_area = torch.linalg.vector_norm(cross, dim=1)
    normals = cross / torch.clamp(double_area[:, None], min=1e-12)
    return normals, double_area * 0.5


def _projected_face_areas(
    clip_positions: torch.Tensor,
    faces: torch.Tensor,
    resolution: int,
) -> torch.Tensor:
    triangles = clip_positions[:, faces.long(), :2]
    edge_a = triangles[:, :, 1] - triangles[:, :, 0]
    edge_b = triangles[:, :, 2] - triangles[:, :, 0]
    double_area_ndc = torch.abs(edge_a[..., 0] * edge_b[..., 1] - edge_a[..., 1] * edge_b[..., 0])
    return double_area_ndc * (float(resolution) ** 2) / 8.0


def _visibility_counts(
    context: Any,
    clip_positions: torch.Tensor,
    faces: torch.Tensor,
    resolution: int,
    instance_batch_size: int,
) -> torch.Tensor:
    return _visibility_counts_batch(
        context,
        [clip_positions],
        [faces],
        resolution,
        instance_batch_size,
    )[0]


def _visibility_counts_batch(
    context: Any,
    clip_positions: Sequence[torch.Tensor],
    faces: Sequence[torch.Tensor],
    resolution: int,
    instance_batch_size: int,
) -> list[torch.Tensor]:
    """Rasterize object/view instances together with nvdiffrast range mode."""

    import nvdiffrast.torch as dr

    if len(clip_positions) != len(faces) or not clip_positions:
        raise ValueError("clip_positions and faces must be nonempty equal-size batches")
    if instance_batch_size < 1:
        raise ValueError("instance_batch_size must be positive")
    num_views = int(clip_positions[0].shape[0])
    if any(int(value.shape[0]) != num_views for value in clip_positions):
        raise ValueError("Every object must use the same number of projection views")
    outputs = [
        torch.zeros(
            (num_views, int(object_faces.shape[0])),
            device=object_faces.device,
            dtype=torch.float32,
        )
        for object_faces in faces
    ]
    # View-major order ensures every raster launch spans multiple objects when
    # object batching is enabled.
    instances = [
        (object_id, view_id)
        for view_id in range(num_views)
        for object_id in range(len(faces))
    ]
    for start in range(0, len(instances), instance_batch_size):
        chunk = instances[start : start + instance_batch_size]
        position_parts = []
        triangle_parts = []
        ranges = []
        metadata = []
        vertex_offset = 0
        triangle_offset = 0
        for object_id, view_id in chunk:
            positions = clip_positions[object_id][view_id]
            object_faces = faces[object_id]
            position_parts.append(positions)
            triangle_parts.append(object_faces + vertex_offset)
            ranges.append((triangle_offset, int(object_faces.shape[0])))
            metadata.append((object_id, view_id, triangle_offset))
            vertex_offset += int(positions.shape[0])
            triangle_offset += int(object_faces.shape[0])
        packed_positions = torch.cat(position_parts, dim=0).contiguous()
        packed_triangles = torch.cat(triangle_parts, dim=0).to(
            dtype=torch.int32
        ).contiguous()
        range_tensor = torch.tensor(ranges, dtype=torch.int32, device="cpu")
        raster, _ = dr.rasterize(
            context,
            packed_positions,
            packed_triangles,
            resolution=[resolution, resolution],
            ranges=range_tensor,
        )
        for raster_id, (object_id, view_id, triangle_start) in enumerate(metadata):
            local_ids = raster[raster_id, ..., 3].long() - 1 - triangle_start
            visible = local_ids >= 0
            outputs[object_id][view_id] = torch.bincount(
                local_ids[visible], minlength=int(faces[object_id].shape[0])
            ).float()
    return outputs


def _tile_uvs(
    projected_xy: torch.Tensor,
    unique_vertex_ids: torch.Tensor,
    unique_view_ids: torch.Tensor,
    config: ProjectionAtlasConfig,
) -> torch.Tensor:
    columns, rows = atlas_grid(config.num_views)
    local = projected_xy[unique_view_ids, unique_vertex_ids] * 0.5 + 0.5
    local = local.clamp(0.0, 1.0)
    tile_x = torch.remainder(unique_view_ids, columns).to(local.dtype)
    tile_y = torch.div(unique_view_ids, columns, rounding_mode="floor").to(local.dtype)
    tile_width = float(config.atlas_size) / columns
    tile_height = float(config.atlas_size) / rows
    inner_width = tile_width - 2 * config.padding_pixels
    inner_height = tile_height - 2 * config.padding_pixels
    u = (tile_x * tile_width + config.padding_pixels + local[:, 0] * inner_width) / config.atlas_size
    v = (tile_y * tile_height + config.padding_pixels + local[:, 1] * inner_height) / config.atlas_size
    return torch.stack((u, v), dim=1)


def _atlas_occupancy(
    context: Any,
    uvs: torch.Tensor,
    faces: torch.Tensor,
    atlas_size: int,
) -> int:
    import nvdiffrast.torch as dr

    clip = torch.cat(
        (
            uvs * 2 - 1,
            torch.zeros_like(uvs[:, :1]),
            torch.ones_like(uvs[:, :1]),
        ),
        dim=1,
    )
    raster, _ = dr.rasterize(
        context,
        clip.unsqueeze(0),
        faces,
        resolution=[atlas_size, atlas_size],
    )
    return int((raster[0, ..., 3] > 0).sum().item())


def _weighted_fraction(mask: torch.Tensor, weights: torch.Tensor) -> float:
    denominator = float(weights.sum().item())
    if denominator <= 0:
        return 0.0
    return float(weights[mask].sum().item() / denominator)


def _select_face_views(
    visible_counts: torch.Tensor,
    projected_areas: torch.Tensor,
    front_alignment: torch.Tensor,
    config: ProjectionAtlasConfig,
) -> tuple[torch.Tensor, ...]:
    """Select one projection per face, balancing coverage and texel density."""

    eligible_counts = visible_counts
    if config.front_facing_mode == "normal_gate":
        eligible_counts = visible_counts * (
            front_alignment > config.front_facing_epsilon
        )
    visibility_ratios = torch.clamp(
        eligible_counts / torch.clamp(projected_areas, min=1.0),
        max=1.0,
    )
    if config.view_assignment_mode == "visible_pixels":
        selected_counts, selected_views = eligible_counts.max(dim=0)
    else:
        preferred = (
            (eligible_counts > 0)
            & (visibility_ratios >= config.preferred_visibility_ratio)
        )
        preferred_scores = torch.where(
            preferred,
            projected_areas,
            torch.full_like(projected_areas, -1.0),
        )
        _, preferred_views = preferred_scores.max(dim=0)
        fallback_scores = eligible_counts * visibility_ratios
        _, fallback_views = fallback_scores.max(dim=0)
        has_preferred = preferred.any(dim=0)
        selected_views = torch.where(
            has_preferred,
            preferred_views,
            fallback_views,
        )
        face_ids = torch.arange(
            visible_counts.shape[1], device=visible_counts.device
        )
        selected_counts = eligible_counts[selected_views, face_ids]
    fallback = selected_counts <= 0
    fallback_views = front_alignment.argmax(dim=0)
    selected_views = torch.where(fallback, fallback_views, selected_views)
    face_ids = torch.arange(visible_counts.shape[1], device=visible_counts.device)
    selected_counts = eligible_counts[selected_views, face_ids]
    selected_projected = projected_areas[selected_views, face_ids]
    selected_ratio = visibility_ratios[selected_views, face_ids]
    selected_alignment = front_alignment[selected_views, face_ids]
    selected_preferred = selected_ratio >= config.preferred_visibility_ratio
    return (
        selected_views,
        selected_counts,
        selected_projected,
        selected_ratio,
        selected_alignment,
        selected_preferred,
        fallback,
    )


def _quality_assessment(diagnostics: dict[str, Any]) -> tuple[bool, list[str]]:
    """Apply conservative gates for using a projection atlas as an unwrap."""

    issues = []
    unseen_area = float(diagnostics["fallback_unseen_surface_area_fraction"])
    low_visibility_area = float(diagnostics["low_visibility_surface_area_fraction"])
    overdraw = float(diagnostics["projected_overdraw_ratio"])
    if unseen_area > 0.01:
        issues.append(f"unseen surface area is {unseen_area:.3%}, above 1%")
    if low_visibility_area > 0.05:
        issues.append(
            f"low-visibility surface area is {low_visibility_area:.3%}, above 5%"
        )
    if overdraw > 1.05:
        issues.append(f"projected overdraw is {overdraw:.3f}x, above 1.05x")
    return not issues, issues


@torch.inference_mode()
def projection_uv_unwrap(
    *,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vertex_normals: torch.Tensor,
    device: torch.device,
    config: ProjectionAtlasConfig = ProjectionAtlasConfig(),
    raster_context: Any = None,
    _visible_counts_override: torch.Tensor | None = None,
) -> ProjectionUVResult:
    """Generate a tiled projective UV atlas for one cleaned CUDA mesh."""

    import nvdiffrast.torch as dr

    config.validate()
    if device.type != "cuda":
        raise ValueError("nvdiffrast projection UV generation requires CUDA")
    vertices = vertices.to(device=device, dtype=torch.float32).contiguous()
    faces = faces.to(device=device, dtype=torch.int32).contiguous()
    vertex_normals = vertex_normals.to(device=device, dtype=torch.float32).contiguous()
    context = raster_context or dr.RasterizeCudaContext(device=device)
    timings: dict[str, float] = {}

    def timed(name: str, function):
        _sync(device)
        started = time.perf_counter()
        value = function()
        _sync(device)
        timings[name] = float(time.perf_counter() - started)
        return value

    directions = view_directions(config.num_views, device=device)
    clip_positions, radius = timed(
        "camera_projection",
        lambda: _project_vertices(vertices, directions, config.projection_margin),
    )
    face_normals, face_areas, projected_areas = timed(
        "projected_face_geometry",
        lambda: (
            *_face_geometry(vertices, faces),
            _projected_face_areas(
                clip_positions, faces, config.assignment_resolution
            ),
        ),
    )
    if _visible_counts_override is None:
        visible_counts = timed(
            "multiview_rasterization",
            lambda: _visibility_counts(
                context,
                clip_positions,
                faces,
                config.assignment_resolution,
                config.raster_instance_batch_size,
            ),
        )
    else:
        visible_counts = _visible_counts_override.to(
            device=device, dtype=torch.float32
        )
        timings["multiview_rasterization"] = 0.0

    def assign_faces() -> tuple[torch.Tensor, ...]:
        return _select_face_views(
            visible_counts,
            projected_areas,
            directions @ face_normals.T,
            config,
        )

    (
        selected_views,
        selected_counts,
        selected_projected,
        selected_ratio,
        selected_alignment,
        selected_preferred,
        fallback,
    ) = timed("face_view_assignment", assign_faces)

    def build_atlas() -> tuple[torch.Tensor, ...]:
        corner_vertices = faces.long().reshape(-1)
        corner_views = selected_views[:, None].expand(-1, 3).reshape(-1)
        keys = corner_views * vertices.shape[0] + corner_vertices
        unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        unique_views = torch.div(unique_keys, vertices.shape[0], rounding_mode="floor")
        unique_vertices = torch.remainder(unique_keys, vertices.shape[0])
        output_uvs = _tile_uvs(
            clip_positions[..., :2], unique_vertices, unique_views, config
        )
        return (
            vertices[unique_vertices],
            inverse.reshape(-1, 3).to(torch.int32),
            output_uvs,
            vertex_normals[unique_vertices],
            unique_vertices.long(),
            clip_positions[unique_views, unique_vertices, 2].contiguous(),
        )

    (
        output_vertices,
        output_faces,
        output_uvs,
        output_normals,
        vmaps,
        uv_raster_depths,
    ) = timed("atlas_vertex_expansion", build_atlas)
    occupied_pixels = timed(
        "atlas_coverage_rasterization",
        lambda: _atlas_occupancy(
            context, output_uvs, output_faces, config.atlas_size
        ),
    )

    columns, rows = atlas_grid(config.num_views)
    inner_width = config.atlas_size / columns - 2 * config.padding_pixels
    inner_height = config.atlas_size / rows - 2 * config.padding_pixels
    area_scale = (inner_width * inner_height) / (
        float(config.assignment_resolution) ** 2
    )
    projected_atlas_pixels = float((selected_projected * area_scale).sum().item())
    low_visibility = selected_ratio < config.low_visibility_ratio
    visible = ~fallback
    face_histogram = torch.bincount(selected_views, minlength=config.num_views)
    diagnostics = {
        "num_views": config.num_views,
        "atlas_grid": [columns, rows],
        "assignment_resolution": config.assignment_resolution,
        "atlas_size": config.atlas_size,
        "padding_pixels": config.padding_pixels,
        "front_facing_mode": config.front_facing_mode,
        "view_assignment_mode": config.view_assignment_mode,
        "preferred_visibility_ratio": config.preferred_visibility_ratio,
        "uv_raster_depth_mode": "projection_camera_depth",
        "projection_radius": float(radius.item()),
        "input_vertices": int(vertices.shape[0]),
        "input_faces": int(faces.shape[0]),
        "output_vertices_uv_expanded": int(output_vertices.shape[0]),
        "output_faces": int(output_faces.shape[0]),
        "visible_faces": int(visible.sum().item()),
        "visible_face_fraction": float(visible.float().mean().item()),
        "visible_surface_area_fraction": _weighted_fraction(visible, face_areas),
        "fallback_unseen_faces": int(fallback.sum().item()),
        "fallback_unseen_surface_area_fraction": _weighted_fraction(
            fallback, face_areas
        ),
        "low_visibility_faces": int(low_visibility.sum().item()),
        "low_visibility_face_fraction": float(low_visibility.float().mean().item()),
        "low_visibility_surface_area_fraction": _weighted_fraction(
            low_visibility, face_areas
        ),
        "preferred_visibility_faces": int(selected_preferred.sum().item()),
        "preferred_visibility_face_fraction": float(
            selected_preferred.float().mean().item()
        ),
        "preferred_visibility_surface_area_fraction": _weighted_fraction(
            selected_preferred, face_areas
        ),
        "selected_visibility_ratio_mean": float(selected_ratio.mean().item()),
        "selected_visibility_ratio_median": float(selected_ratio.median().item()),
        "selected_visibility_ratio_min": float(selected_ratio.min().item()),
        "selected_front_alignment_mean": float(selected_alignment.mean().item()),
        "selected_front_alignment_min": float(selected_alignment.min().item()),
        "selected_visible_pixels": int(selected_counts.sum().item()),
        "face_count_per_view": [int(value) for value in face_histogram.tolist()],
        "atlas_occupied_pixels": occupied_pixels,
        "atlas_occupancy_fraction": float(
            occupied_pixels / (config.atlas_size**2)
        ),
        "projected_atlas_pixels": projected_atlas_pixels,
        "projected_overdraw_ratio": float(
            projected_atlas_pixels / max(occupied_pixels, 1)
        ),
        "uv_min": [float(value) for value in output_uvs.amin(dim=0).tolist()],
        "uv_max": [float(value) for value in output_uvs.amax(dim=0).tolist()],
    }
    production_safe, quality_issues = _quality_assessment(diagnostics)
    diagnostics["production_safe"] = production_safe
    diagnostics["quality_issues"] = quality_issues
    return ProjectionUVResult(
        vertices=output_vertices,
        faces=output_faces,
        uvs=output_uvs,
        normals=output_normals,
        vmaps=vmaps,
        uv_raster_depths=uv_raster_depths,
        face_view_ids=selected_views,
        face_seen=visible,
        diagnostics=diagnostics,
        timings=timings,
    )


@torch.inference_mode()
def projection_uv_unwrap_batch(
    *,
    vertices: Sequence[torch.Tensor],
    faces: Sequence[torch.Tensor],
    vertex_normals: Sequence[torch.Tensor],
    device: torch.device,
    config: ProjectionAtlasConfig = ProjectionAtlasConfig(),
    raster_context: Any = None,
) -> list[ProjectionUVResult]:
    """Generate projection atlases with visibility rasterized across objects."""

    import nvdiffrast.torch as dr

    config.validate()
    if not vertices or len(vertices) != len(faces) or len(vertices) != len(vertex_normals):
        raise ValueError("vertices, faces, and vertex_normals must be equal-size batches")
    context = raster_context or dr.RasterizeCudaContext(device=device)
    directions = view_directions(config.num_views, device=device)
    vertices_gpu = [
        value.to(device=device, dtype=torch.float32).contiguous()
        for value in vertices
    ]
    faces_gpu = [
        value.to(device=device, dtype=torch.int32).contiguous() for value in faces
    ]
    normals_gpu = [
        value.to(device=device, dtype=torch.float32).contiguous()
        for value in vertex_normals
    ]
    clip_positions = [
        _project_vertices(value, directions, config.projection_margin)[0]
        for value in vertices_gpu
    ]
    _sync(device)
    raster_started = time.perf_counter()
    visible_counts = _visibility_counts_batch(
        context,
        clip_positions,
        faces_gpu,
        config.assignment_resolution,
        config.raster_instance_batch_size,
    )
    _sync(device)
    shared_raster_seconds = float(time.perf_counter() - raster_started)
    results = []
    for object_vertices, object_faces, object_normals, counts in zip(
        vertices_gpu, faces_gpu, normals_gpu, visible_counts
    ):
        result = projection_uv_unwrap(
            vertices=object_vertices,
            faces=object_faces,
            vertex_normals=object_normals,
            device=device,
            config=config,
            raster_context=context,
            _visible_counts_override=counts,
        )
        result.timings["multiview_rasterization_shared"] = shared_raster_seconds
        results.append(result)
    return results


@torch.inference_mode()
def postprocess_mesh_projection_uv(
    mesh: MeshInput,
    *,
    device: torch.device,
    topology_config: PostprocessConfig = PostprocessConfig(),
    atlas_config: ProjectionAtlasConfig = ProjectionAtlasConfig(),
    raster_context: Any = None,
) -> tuple[PostprocessedMesh, dict[str, Any]]:
    """Run production topology cleanup followed by projection-atlas UVs."""

    if device.type != "cuda":
        raise ValueError("CuMesh projection postprocessing requires CUDA")
    topology_config.validate()
    atlas_config.validate()
    _validate_input(mesh)
    _sync(device)
    wall_started = time.perf_counter()
    cumesh, _ = _load_cumesh_modules()
    vertices = mesh.vertices.detach().to(
        device=device, dtype=torch.float32
    ).contiguous()
    faces = mesh.faces.detach().to(device=device, dtype=torch.int32).contiguous()
    handler = cumesh.CuMesh()
    handler.init(vertices, faces)
    _sync(device)
    topology_started = time.perf_counter()
    topology_stage_timings: dict[str, float] = {}
    _run_topology(
        handler,
        mesh.decimation_target,
        topology_config,
        device=device,
        stage_timings=topology_stage_timings,
    )
    _sync(device)
    topology_seconds = float(time.perf_counter() - topology_started)
    _sync(device)
    degenerate_started = time.perf_counter()
    handler.remove_degenerate_faces()
    _sync(device)
    degenerate_seconds = float(time.perf_counter() - degenerate_started)
    _sync(device)
    normals_started = time.perf_counter()
    handler.compute_vertex_normals()
    clean_vertices, clean_faces = handler.read()
    clean_normals = handler.read_vertex_normals()
    _sync(device)
    normals_seconds = float(time.perf_counter() - normals_started)
    projection = projection_uv_unwrap(
        vertices=clean_vertices,
        faces=clean_faces,
        vertex_normals=clean_normals,
        device=device,
        config=atlas_config,
        raster_context=raster_context,
    )
    values = (
        projection.vertices.detach().cpu().contiguous(),
        projection.faces.detach().cpu().contiguous(),
        projection.uvs.detach().cpu().contiguous(),
        projection.normals.detach().cpu().contiguous(),
        projection.vmaps.detach().cpu().contiguous(),
    )
    uv_raster_depths = projection.uv_raster_depths.detach().cpu().contiguous()
    _validate_output(mesh.name, *values)
    output_vertices, output_faces, output_uvs, output_normals, output_vmaps = values
    projection_timings = {
        "remove_degenerate_faces": degenerate_seconds,
        "compute_vertex_normals": normals_seconds,
        **projection.timings,
    }
    projection_seconds = float(sum(projection_timings.values()))
    _sync(device)
    wall_seconds = float(time.perf_counter() - wall_started)
    record = {
        "name": mesh.name,
        "backend": "nvdiffrast_projection_atlas",
        "decimation_target": int(mesh.decimation_target),
        "input_vertices": int(mesh.vertices.shape[0]),
        "input_faces": int(mesh.faces.shape[0]),
        "output_vertices_uv_expanded": int(output_vertices.shape[0]),
        "output_faces": int(output_faces.shape[0]),
        "input_bounds": _bounds(mesh.vertices.detach().cpu()),
        "output_bounds": _bounds(output_vertices),
        "topology_seconds_shared": topology_seconds,
        "uv_seconds_shared": projection_seconds,
        "total_seconds_shared": topology_seconds + projection_seconds,
        "topology_config": asdict(topology_config),
        "topology_stage_timings": topology_stage_timings,
        "projection_atlas_config": asdict(atlas_config),
        "projection_timings": projection_timings,
        "projection_diagnostics": projection.diagnostics,
        "metadata": mesh.metadata,
    }
    processed = PostprocessedMesh(
        name=mesh.name,
        vertices=output_vertices,
        faces=output_faces,
        uvs=output_uvs,
        normals=output_normals,
        vmaps=output_vmaps,
        record=record,
        uv_raster_depths=uv_raster_depths,
    )
    return processed, {
        "face_view_ids": projection.face_view_ids.detach().cpu().contiguous(),
        "timings": projection_timings,
        "topology_stage_timings": topology_stage_timings,
        "diagnostics": projection.diagnostics,
        "wall_seconds": wall_seconds,
    }


@torch.inference_mode()
def postprocess_meshes_projection_uv(
    meshes: Sequence[MeshInput],
    *,
    device: torch.device,
    batch_size: int = 4,
    topology_config: PostprocessConfig = PostprocessConfig(),
    atlas_config: ProjectionAtlasConfig = ProjectionAtlasConfig(),
    raster_context: Any = None,
    allow_scalar_fallback: bool = True,
) -> tuple[list[PostprocessedMesh], dict[str, Any]]:
    """Batch CuMesh topology and cross-object projection visibility rasterization."""

    import nvdiffrast.torch as dr

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    topology_config.validate()
    atlas_config.validate()
    prepared = _prepare_inputs(meshes, device)
    context = raster_context or dr.RasterizeCudaContext(device=device)
    _, cumesh_batch = _load_cumesh_modules()
    results: list[PostprocessedMesh] = []
    batch_records = []
    wall_started = time.perf_counter()
    for batch_id, start in enumerate(range(0, len(prepared), batch_size)):
        chunk = prepared[start : start + batch_size]
        chunk_started = time.perf_counter()
        try:
            handler = cumesh_batch.CuMesh()
            handler.init(
                [mesh.vertices for mesh in chunk],
                [mesh.faces for mesh in chunk],
            )
            topology_stage_timings: dict[str, float] = {}
            topology_started = time.perf_counter()
            _run_topology(
                handler,
                [mesh.decimation_target for mesh in chunk],
                topology_config,
                device=device,
                stage_timings=topology_stage_timings,
            )
            _sync(device)
            topology_seconds = float(time.perf_counter() - topology_started)
            handler.remove_degenerate_faces()
            handler.compute_vertex_normals()
            clean_vertices, clean_faces = handler.read()
            clean_normals = handler.read_vertex_normals()
            projections = projection_uv_unwrap_batch(
                vertices=clean_vertices,
                faces=clean_faces,
                vertex_normals=clean_normals,
                device=device,
                config=atlas_config,
                raster_context=context,
            )
            for mesh, projection in zip(chunk, projections):
                values = (
                    projection.vertices.detach().cpu().contiguous(),
                    projection.faces.detach().cpu().contiguous(),
                    projection.uvs.detach().cpu().contiguous(),
                    projection.normals.detach().cpu().contiguous(),
                    projection.vmaps.detach().cpu().contiguous(),
                )
                _validate_output(mesh.name, *values)
                projection_seconds = float(sum(projection.timings.values()))
                record = {
                    "name": mesh.name,
                    "backend": "cumesh_batch_nvdiffrast_projection_atlas",
                    "batch_id": batch_id,
                    "batch_size": len(chunk),
                    "fallback_reason": None,
                    "decimation_target": int(mesh.decimation_target),
                    "input_vertices": int(mesh.vertices.shape[0]),
                    "input_faces": int(mesh.faces.shape[0]),
                    "output_vertices_uv_expanded": int(values[0].shape[0]),
                    "output_faces": int(values[1].shape[0]),
                    "input_bounds": _bounds(mesh.vertices.detach().cpu()),
                    "output_bounds": _bounds(values[0]),
                    "topology_seconds_shared": topology_seconds,
                    "uv_seconds_shared": projection_seconds,
                    "total_seconds_shared": topology_seconds + projection_seconds,
                    "topology_config": asdict(topology_config),
                    "topology_stage_timings": topology_stage_timings,
                    "projection_atlas_config": asdict(atlas_config),
                    "projection_timings": projection.timings,
                    "projection_diagnostics": projection.diagnostics,
                    "metadata": mesh.metadata,
                }
                results.append(
                    PostprocessedMesh(
                        name=mesh.name,
                        vertices=values[0],
                        faces=values[1],
                        uvs=values[2],
                        normals=values[3],
                        vmaps=values[4],
                        record=record,
                        uv_raster_depths=(
                            projection.uv_raster_depths.detach().cpu().contiguous()
                        ),
                    )
                )
            batch_records.append(
                {
                    "batch_id": batch_id,
                    "names": [mesh.name for mesh in chunk],
                    "requested_batch_size": len(chunk),
                    "actual_backend": "batch",
                    "wall_seconds": float(time.perf_counter() - chunk_started),
                    "fallback_reason": None,
                }
            )
        except Exception as error:
            if not allow_scalar_fallback:
                raise
            reason = f"{type(error).__name__}: {error}"
            for mesh in chunk:
                processed, detail = postprocess_mesh_projection_uv(
                    mesh,
                    device=device,
                    topology_config=topology_config,
                    atlas_config=atlas_config,
                    raster_context=context,
                )
                processed.record["backend"] = "scalar_projection_fallback"
                processed.record["fallback_reason"] = reason
                results.append(processed)
            batch_records.append(
                {
                    "batch_id": batch_id,
                    "names": [mesh.name for mesh in chunk],
                    "requested_batch_size": len(chunk),
                    "actual_backend": "scalar_fallback",
                    "wall_seconds": float(time.perf_counter() - chunk_started),
                    "fallback_reason": reason,
                }
            )
    return results, {
        "backend": "cumesh_batch_nvdiffrast_projection_atlas",
        "requested_batch_size": batch_size,
        "allow_scalar_fallback": allow_scalar_fallback,
        "num_meshes": len(results),
        "wall_seconds": float(time.perf_counter() - wall_started),
        "batches": batch_records,
    }
