from __future__ import annotations

import os
import sys
from typing import Any, List, Tuple

import torch

from .bvh import cuBVH
from ._utils import normalize_batch_arg, normalize_mesh_batch, restore_mesh_batch


def _init_hashmap(resolution: int, batch_size: int, capacity: int, device: torch.device):
    volume = batch_size * resolution * resolution * resolution
    if volume < 2**32:
        hashmap_keys = torch.full((capacity,), torch.iinfo(torch.uint32).max, dtype=torch.uint32, device=device)
    elif volume < 2**64:
        hashmap_keys = torch.full((capacity,), torch.iinfo(torch.uint64).max, dtype=torch.uint64, device=device)
    else:
        raise ValueError(f"The batched spatial volume is too large: {volume} > 2^64.")
    hashmap_vals = torch.empty((capacity,), dtype=torch.uint32, device=device)
    return hashmap_keys, hashmap_vals


def _ensure_batched_constants(device: torch.device):
    if not hasattr(remesh_narrow_band_dc, "edge_neighbor_voxel_offset"):
        remesh_narrow_band_dc.edge_neighbor_voxel_offset = torch.tensor([
            [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0]],
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 1], [0, 0, 0, 1]],
            [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0]],
        ], dtype=torch.int32).unsqueeze(0)
        remesh_narrow_band_dc.quad_split_1_n = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long)
        remesh_narrow_band_dc.quad_split_1_p = torch.tensor([0, 2, 1, 0, 3, 2], dtype=torch.long)
        remesh_narrow_band_dc.quad_split_2_n = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long)
        remesh_narrow_band_dc.quad_split_2_p = torch.tensor([0, 3, 1, 3, 2, 1], dtype=torch.long)
    remesh_narrow_band_dc.edge_neighbor_voxel_offset = remesh_narrow_band_dc.edge_neighbor_voxel_offset.to(device)
    remesh_narrow_band_dc.quad_split_1_n = remesh_narrow_band_dc.quad_split_1_n.to(device)
    remesh_narrow_band_dc.quad_split_1_p = remesh_narrow_band_dc.quad_split_1_p.to(device)
    remesh_narrow_band_dc.quad_split_2_n = remesh_narrow_band_dc.quad_split_2_n.to(device)
    remesh_narrow_band_dc.quad_split_2_p = remesh_narrow_band_dc.quad_split_2_p.to(device)
    return torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=torch.int32, device=device)


def _ensure_tensor_1d(value: Any, batch_size: int, name: str, device: torch.device, dtype) -> torch.Tensor:
    values = normalize_batch_arg(value, batch_size, name)
    tensors = []
    for item in values:
        if torch.is_tensor(item):
            tensor = item.to(device=device, dtype=dtype)
        else:
            tensor = torch.tensor(item, device=device, dtype=dtype)
        tensors.append(tensor)
    return torch.stack(tensors, dim=0)


def _ensure_centers(center: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(center):
        if center.ndim == 1 and center.shape[0] == 3:
            return center.to(device=device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1).contiguous()
        if center.ndim == 2 and center.shape == (batch_size, 3):
            return center.to(device=device, dtype=torch.float32)
    if isinstance(center, (list, tuple)) and len(center) == 3 and not torch.is_tensor(center[0]):
        return torch.tensor(center, device=device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1).contiguous()
    return _ensure_tensor_1d(center, batch_size, "center", device, torch.float32)


def _split_points_by_batch(points: torch.Tensor, batch_ids: torch.Tensor, batch_size: int) -> List[torch.Tensor]:
    return [points[batch_ids == batch_idx] for batch_idx in range(batch_size)]


def _bvh_unsigned_distance(
    bvh: cuBVH,
    positions: torch.Tensor,
    batch_ids: torch.Tensor,
    batch_size: int,
    return_uvw: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if positions.shape[0] == 0:
        empty_float = positions.new_empty((0,))
        empty_long = torch.empty((0,), dtype=torch.long, device=positions.device)
        empty_uvw = positions.new_empty((0, 3)) if return_uvw else None
        return empty_float, empty_long, empty_uvw

    index_list = [(batch_ids == batch_idx).nonzero(as_tuple=False).flatten() for batch_idx in range(batch_size)]
    positions_list = [positions[index_i] for index_i in index_list]
    packed_positions = torch.cat(
        [
            positions_i + bvh._translations[batch_idx].to(positions_i.device, positions_i.dtype)
            for batch_idx, positions_i in enumerate(positions_list)
        ],
        dim=0,
    )
    distances, face_id, uvw = bvh._impl.unsigned_distance(packed_positions, return_uvw=return_uvw)
    counts = [positions_i.shape[0] for positions_i in positions_list]
    distance_list = list(torch.split(distances, counts, dim=0))
    face_list = list(torch.split(face_id, counts, dim=0))
    uvw_list = list(torch.split(uvw, counts, dim=0)) if uvw is not None else None

    out_distances = torch.empty_like(distances)
    out_face_id = torch.empty_like(face_id)
    out_uvw = torch.empty_like(uvw) if uvw is not None else None
    for batch_idx, index_i in enumerate(index_list):
        out_distances[index_i] = distance_list[batch_idx]
        out_face_id[index_i] = face_list[batch_idx] - bvh._face_offsets[batch_idx].to(face_id.device)
        if out_uvw is not None:
            out_uvw[index_i] = uvw_list[batch_idx]
    return out_distances, out_face_id, out_uvw


def _bvh_signed_distance(
    bvh: cuBVH,
    positions: torch.Tensor,
    batch_ids: torch.Tensor,
    batch_size: int,
    mode: str = "watertight",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if positions.shape[0] == 0:
        empty_float = positions.new_empty((0,))
        empty_long = torch.empty((0,), dtype=torch.long, device=positions.device)
        return empty_float, empty_long

    index_list = [(batch_ids == batch_idx).nonzero(as_tuple=False).flatten() for batch_idx in range(batch_size)]
    positions_list = [positions[index_i] for index_i in index_list]
    packed_positions = torch.cat(
        [
            positions_i + bvh._translations[batch_idx].to(positions_i.device, positions_i.dtype)
            for batch_idx, positions_i in enumerate(positions_list)
        ],
        dim=0,
    )
    distances, face_id, _ = bvh._impl.signed_distance(packed_positions, return_uvw=False, mode=mode)
    counts = [positions_i.shape[0] for positions_i in positions_list]
    distance_list = list(torch.split(distances, counts, dim=0))
    face_list = list(torch.split(face_id, counts, dim=0))

    out_distances = torch.empty_like(distances)
    out_face_id = torch.empty_like(face_id)
    for batch_idx, index_i in enumerate(index_list):
        out_distances[index_i] = distance_list[batch_idx]
        out_face_id[index_i] = face_list[batch_idx] - bvh._face_offsets[batch_idx].to(face_id.device)
    return out_distances, out_face_id


def _import_cubvh_backend():
    try:
        from cumesh import _cubvh
        return _cubvh
    except ImportError:
        cumesh_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if cumesh_root not in sys.path:
            sys.path.append(cumesh_root)
        from cumesh import _cubvh
        return _cubvh


def _project_mesh_vertices_single(
    bvh: cuBVH,
    batch_idx: int,
    batch_size: int,
    verts: torch.Tensor,
    tris: torch.Tensor,
    orig_vertices: torch.Tensor,
    orig_faces: torch.Tensor,
    project_back: float,
    project_mode: str,
    normal_smooth_iterations: int = 5,
    snap_uvw_margin: float = 0.05,
) -> torch.Tensor:
    if project_mode not in {"nearest", "plane", "normal"}:
        raise ValueError(f"Unsupported project_mode: {project_mode}. Expected 'nearest', 'plane', or 'normal'.")
    device = verts.device

    def query(points: torch.Tensor):
        bidx = torch.full((points.shape[0],), batch_idx, dtype=torch.long, device=device)
        distances, face_id, uvw = _bvh_unsigned_distance(bvh, points, bidx, batch_size, return_uvw=True)
        nearest = (orig_vertices[orig_faces[face_id.long()]] * uvw.unsqueeze(-1)).sum(dim=1)
        return distances, face_id, uvw, nearest

    distances, face_id, _, nearest = query(verts)
    if project_mode == "nearest":
        return verts - project_back * (verts - nearest)

    if project_mode == "plane":
        tri_verts = orig_vertices[orig_faces[face_id.long()]]
        normals = torch.cross(tri_verts[:, 1] - tri_verts[:, 0], tri_verts[:, 2] - tri_verts[:, 0], dim=1)
        normals = normals / normals.norm(dim=1, keepdim=True).clamp(min=1e-12)
        offset = ((verts - nearest) * normals).sum(dim=1, keepdim=True) * normals
        return verts - project_back * offset

    # "normal": deflate along the mesh's own smoothed vertex normals; the
    # direction field is continuous so source rims/edges do not cause spikes.
    tris_long = tris.long()
    face_normals = torch.cross(
        verts[tris_long[:, 1]] - verts[tris_long[:, 0]],
        verts[tris_long[:, 2]] - verts[tris_long[:, 0]],
        dim=1,
    )
    vert_normals = torch.zeros_like(verts)
    for corner in range(3):
        vert_normals.index_add_(0, tris_long[:, corner], face_normals)
    vert_normals = vert_normals / vert_normals.norm(dim=1, keepdim=True).clamp(min=1e-12)

    flip = ((verts - nearest) * vert_normals).sum(dim=1, keepdim=True) < 0.0
    vert_normals = torch.where(flip, -vert_normals, vert_normals)

    edge_src = tris_long[:, [0, 1, 2, 1, 2, 0]].reshape(-1)
    edge_dst = tris_long[:, [1, 2, 0, 0, 1, 2]].reshape(-1)
    for _ in range(max(0, int(normal_smooth_iterations))):
        neighbor_sum = torch.zeros_like(vert_normals)
        neighbor_sum.index_add_(0, edge_dst, vert_normals[edge_src])
        vert_normals = vert_normals + neighbor_sum
        vert_normals = vert_normals / vert_normals.norm(dim=1, keepdim=True).clamp(min=1e-12)

    moved = verts - project_back * distances.unsqueeze(1) * vert_normals

    if snap_uvw_margin >= 0.0:
        _, _, uvw2, nearest2 = query(moved)
        interior = (uvw2.min(dim=1).values > snap_uvw_margin).unsqueeze(1)
        moved = torch.where(interior, moved - project_back * (moved - nearest2), moved)
    return moved


def remesh_sdf_marching_cubes(
    vertices: Any,
    faces: Any,
    center: Any,
    scale: Any,
    resolution: Any,
    sdf_mode: str = "watertight",
    sign_source: str = "bvh",
    sign_samples: int = 5,
    sign_jitter_voxels: float = 0.25,
    sign_denoise_iterations: int = 2,
    floodfill_eps_voxels: float = 2.0,
    floodfill_closing_voxels: int = 0,
    band: Any = 1,
    project_back: Any = 0,
    project_mode: str = "nearest",
    project_normal_smooth_iterations: int = 5,
    project_snap_uvw_margin: float = 0.05,
    chunk_size: int = 262144,
    smooth_weight: float = 0.0,
    smooth_iterations: int = 2,
    expand_cells: int = 1,
    ensure_consistency: bool = True,
    bvh: Any = None,
    vertex_mask: torch.Tensor | None = None,
    face_mask: torch.Tensor | None = None,
):
    cubvh_backend = _import_cubvh_backend()

    vertices_list, faces_list, batch_kind = normalize_mesh_batch(vertices, faces, vertex_mask, face_mask)
    batch_size = len(vertices_list)
    device = vertices_list[0].device

    centers = _ensure_centers(center, batch_size, device)
    scales = _ensure_tensor_1d(scale, batch_size, "scale", device, torch.float32).reshape(batch_size)
    resolutions = _ensure_tensor_1d(resolution, batch_size, "resolution", device, torch.int32).reshape(batch_size)
    bands = _ensure_tensor_1d(band, batch_size, "band", device, torch.float32).reshape(batch_size)
    project_backs = _ensure_tensor_1d(project_back, batch_size, "project_back", device, torch.float32).reshape(batch_size)

    if not torch.equal(resolutions, resolutions[:1].expand_as(resolutions)):
        raise NotImplementedError("Batched SDF marching cubes currently requires the same resolution for every mesh in the batch.")
    resolution_value = int(resolutions[0].item())

    if bvh is None:
        bvh = cuBVH(vertices, faces, vertex_mask=vertex_mask, face_mask=face_mask)
    elif not isinstance(bvh, cuBVH):
        raise TypeError("Batched SDF marching cubes expects a cumesh_batch.cuBVH instance or None.")
    if bvh.batch_size != batch_size:
        raise ValueError(f"BVH batch size mismatch: expected {batch_size}, got {bvh.batch_size}.")
    if sign_source not in {"bvh", "floodfill"}:
        raise ValueError(f"Unsupported sign_source: {sign_source}")

    offsets = _ensure_batched_constants(device)
    mc_corner_offsets = torch.tensor([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=torch.int32, device=device)
    smooth_weight = float(max(0.0, min(1.0, smooth_weight)))
    smooth_iterations = max(0, int(smooth_iterations))
    expand_cells = max(0, int(expand_cells))
    if expand_cells > 0:
        expand_range = torch.arange(-expand_cells, expand_cells + 1, device=device, dtype=torch.int32)
        expand_offsets = torch.stack(
            torch.meshgrid(expand_range, expand_range, expand_range, indexing="ij"),
            dim=-1,
        ).reshape(-1, 3)
    else:
        expand_offsets = None
    smooth_offsets = torch.tensor([
        [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
    ], dtype=torch.int64, device=device)

    base_resolution = resolution_value
    while base_resolution > 32:
        if base_resolution % 2 != 0:
            raise ValueError("resolution must be divisible by 2 until it reaches the base resolution.")
        base_resolution //= 2

    initial_coords = torch.stack(
        torch.meshgrid(
            torch.arange(base_resolution, device=device),
            torch.arange(base_resolution, device=device),
            torch.arange(base_resolution, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).int().reshape(-1, 3)

    def query_unsigned(points: torch.Tensor, batch_idx: int) -> torch.Tensor:
        chunks = []
        for start in range(0, points.shape[0], chunk_size):
            pts_chunk = points[start : start + chunk_size]
            batch_chunk = torch.full((pts_chunk.shape[0],), batch_idx, dtype=torch.long, device=device)
            distance_chunk, _, _ = _bvh_unsigned_distance(bvh, pts_chunk, batch_chunk, batch_size, return_uvw=False)
            chunks.append(distance_chunk)
        return torch.cat(chunks, dim=0)

    def query_signed(points: torch.Tensor, batch_idx: int) -> torch.Tensor:
        chunks = []
        for start in range(0, points.shape[0], chunk_size):
            pts_chunk = points[start : start + chunk_size]
            batch_chunk = torch.full((pts_chunk.shape[0],), batch_idx, dtype=torch.long, device=device)
            distance_chunk, _ = _bvh_signed_distance(bvh, pts_chunk, batch_chunk, batch_size, mode=sdf_mode)
            chunks.append(distance_chunk)
        return torch.cat(chunks, dim=0)

    def denoise_sparse_sdf_signs(coords_i: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        # Flip the sign of grid corners whose sign disagrees with a clear majority
        # of their 6-neighbors. Ray-stab / nearest-normal sign estimates can fail in
        # coherent patches (grazing rays on axis-aligned planes), which shows up as
        # floating shards or missing chunks after marching cubes.
        if sign_denoise_iterations <= 0 or coords_i.shape[0] == 0:
            return values

        coords_long = coords_i.long()
        grid_stride_y = resolution_value + 1
        grid_stride_x = grid_stride_y * grid_stride_y
        keys = coords_long[:, 0] * grid_stride_x + coords_long[:, 1] * grid_stride_y + coords_long[:, 2]
        sorted_keys, order = torch.sort(keys)

        for _ in range(int(sign_denoise_iterations)):
            sorted_values = values[order]
            same = torch.zeros_like(values)
            opposite = torch.zeros_like(values)
            for offset in smooth_offsets:
                neighbor_coords = coords_long + offset
                valid = ((neighbor_coords >= 0) & (neighbor_coords <= resolution_value)).all(dim=1)
                neighbor_keys = (
                    neighbor_coords[:, 0] * grid_stride_x
                    + neighbor_coords[:, 1] * grid_stride_y
                    + neighbor_coords[:, 2]
                )
                pos = torch.searchsorted(sorted_keys, neighbor_keys.clamp(min=0))
                pos_safe = pos.clamp(max=sorted_keys.shape[0] - 1)
                found = valid & (pos < sorted_keys.shape[0]) & (sorted_keys[pos_safe] == neighbor_keys)
                if found.any():
                    agree = torch.sign(sorted_values[pos_safe[found]]) == torch.sign(values[found])
                    same[found] += agree.to(values.dtype)
                    opposite[found] += (~agree).to(values.dtype)
            flip = opposite >= same + 2.0
            if not flip.any():
                break
            values = torch.where(flip, -values, values)
        return values

    def regularize_sparse_sdf_grid(coords_i: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if smooth_weight <= 0.0 or smooth_iterations == 0 or coords_i.shape[0] == 0:
            return values

        coords_long = coords_i.long()
        grid_stride_y = resolution_value + 1
        grid_stride_x = grid_stride_y * grid_stride_y
        keys = coords_long[:, 0] * grid_stride_x + coords_long[:, 1] * grid_stride_y + coords_long[:, 2]
        sorted_keys, order = torch.sort(keys)
        smoothed = values

        for _ in range(smooth_iterations):
            sorted_values = smoothed[order]
            neighbor_sum = torch.zeros_like(smoothed)
            neighbor_count = torch.zeros_like(smoothed)
            for offset in smooth_offsets:
                neighbor_coords = coords_long + offset
                valid = ((neighbor_coords >= 0) & (neighbor_coords <= resolution_value)).all(dim=1)
                neighbor_keys = (
                    neighbor_coords[:, 0] * grid_stride_x
                    + neighbor_coords[:, 1] * grid_stride_y
                    + neighbor_coords[:, 2]
                )
                pos = torch.searchsorted(sorted_keys, neighbor_keys.clamp(min=0))
                pos_safe = pos.clamp(max=sorted_keys.shape[0] - 1)
                found = valid & (pos < sorted_keys.shape[0]) & (sorted_keys[pos_safe] == neighbor_keys)
                if found.any():
                    neighbor_sum[found] += sorted_values[pos[found]]
                    neighbor_count[found] += 1.0

            has_neighbors = neighbor_count > 0.0
            averaged = smoothed.clone()
            averaged[has_neighbors] = neighbor_sum[has_neighbors] / neighbor_count[has_neighbors]
            candidate = (1.0 - smooth_weight) * values + smooth_weight * averaged
            same_sign = (values == 0.0) | (candidate == 0.0) | (torch.sign(candidate) == torch.sign(values))
            smoothed = torch.where(same_sign, candidate, values * 0.25)

        return smoothed

    def make_floodfill_sdf_grid(batch_idx: int) -> torch.Tensor:
        grid_coords = torch.stack(
            torch.meshgrid(
                torch.arange(resolution_value + 1, device=device),
                torch.arange(resolution_value + 1, device=device),
                torch.arange(resolution_value + 1, device=device),
                indexing="ij",
            ),
            dim=-1,
        ).int()
        grid_pts = (grid_coords.reshape(-1, 3).float() / float(resolution_value) - 0.5) * scales[batch_idx] + centers[batch_idx]
        udf = query_unsigned(grid_pts, batch_idx).reshape(resolution_value + 1, resolution_value + 1, resolution_value + 1)
        eps = float(floodfill_eps_voxels) * scales[batch_idx] / float(resolution_value)
        sdf_grid = udf - eps
        occ = udf < eps
        closing_voxels = max(0, int(floodfill_closing_voxels))
        if closing_voxels > 0:
            # Morphological closing: blocks flood leaks through holes up to
            # ~2*closing_voxels wide without inflating the extracted iso-surface
            # (which stays at udf == eps). Cells solidified by the closing get
            # a non-exterior label and are sign-flipped below like any interior.
            kernel = 2 * closing_voxels + 1
            occ_f = occ[None, None].half()
            occ_f = torch.nn.functional.max_pool3d(occ_f, kernel, stride=1, padding=closing_voxels)
            occ_f = 1.0 - torch.nn.functional.max_pool3d(1.0 - occ_f, kernel, stride=1, padding=closing_voxels)
            occ = occ_f[0, 0] > 0.5
        labels = cubvh_backend.floodfill(occ.contiguous().unsqueeze(0)).squeeze(0)
        exterior_label = labels[0, 0, 0]
        inner_mask = (labels != exterior_label) & (sdf_grid > 0.0)
        sdf_grid = sdf_grid.clone()
        sdf_grid[inner_mask] *= -1.0
        if smooth_weight > 0.0 and smooth_iterations > 0:
            sdf = sdf_grid[None, None]
            original = sdf
            for _ in range(smooth_iterations):
                padded = torch.nn.functional.pad(sdf, (1, 1, 1, 1, 1, 1), mode="replicate")
                averaged = (
                    padded[:, :, 1:-1, 1:-1, :-2]
                    + padded[:, :, 1:-1, 1:-1, 2:]
                    + padded[:, :, 1:-1, :-2, 1:-1]
                    + padded[:, :, 1:-1, 2:, 1:-1]
                    + padded[:, :, :-2, 1:-1, 1:-1]
                    + padded[:, :, 2:, 1:-1, 1:-1]
                ) / 6.0
                candidate = (1.0 - smooth_weight) * original + smooth_weight * averaged
                same_sign = (original == 0.0) | (candidate == 0.0) | (torch.sign(candidate) == torch.sign(original))
                sdf = torch.where(same_sign, candidate, original * 0.25)
            sdf_grid = sdf[0, 0]
        return sdf_grid

    out_vertices = []
    out_faces = []
    for batch_idx in range(batch_size):
        coords = initial_coords
        current_resolution = base_resolution
        while True:
            cell_size = scales[batch_idx] / float(current_resolution)
            pts = ((coords.float() + 0.5) / float(current_resolution) - 0.5) * scales[batch_idx] + centers[batch_idx]
            batch_for_pts = torch.full((pts.shape[0],), batch_idx, dtype=torch.long, device=device)
            distances, _, _ = _bvh_unsigned_distance(bvh, pts, batch_for_pts, batch_size, return_uvw=False)
            # Marching cubes extracts the signed zero level, so candidate cells must
            # stay near distance 0. The dual-contour path instead tracks an offset
            # UDF shell at distance eps; using that shell here drops most zero-crossing
            # cells and leaves only sparse face fragments.
            surface_band = (torch.clamp(bands[batch_idx], min=1.0) + 0.87) * cell_size
            coords = coords[distances < surface_band]

            if current_resolution >= resolution_value:
                break

            current_resolution *= 2
            coords = coords * 2
            coords = (coords[:, None, :] + offsets[None, :, :]).reshape(-1, 3).contiguous()

        if coords.shape[0] == 0:
            out_vertices.append(vertices_list[batch_idx].new_zeros((0, 3)))
            out_faces.append(faces_list[batch_idx].new_zeros((0, 3)))
            continue
        coords = torch.unique(coords, dim=0)
        if expand_offsets is not None:
            coords = (coords[:, None, :] + expand_offsets[None, :, :]).reshape(-1, 3)
            valid_coords = ((coords >= 0) & (coords < resolution_value)).all(dim=1)
            coords = torch.unique(coords[valid_coords], dim=0)

        corner_coords = coords[:, None, :] + mc_corner_offsets[None, :, :]
        if sign_source == "floodfill":
            sdf_grid = make_floodfill_sdf_grid(batch_idx)
            flat_corner_coords = corner_coords.reshape(-1, 3).long()
            corner_values = sdf_grid[
                flat_corner_coords[:, 0],
                flat_corner_coords[:, 1],
                flat_corner_coords[:, 2],
            ]
            corners = corner_values.reshape(-1, 8).contiguous()
        else:
            flat_corner_coords = corner_coords.reshape(-1, 3)
            unique_corner_coords, inverse_corner = torch.unique(flat_corner_coords, dim=0, return_inverse=True)
            corner_pts = (
                unique_corner_coords.float() / float(resolution_value) - 0.5
            ) * scales[batch_idx] + centers[batch_idx]
            unique_sdf = query_signed(corner_pts, batch_idx)
            if sign_samples > 1:
                # The query grid is axis-aligned with the dominant planes of the
                # geometry, so a single sign query can fail coherently (grazing
                # rays / ambiguous nearest faces). Take a sign majority over
                # jittered sample positions; keep the magnitude of the exact query.
                jitter_amp = float(sign_jitter_voxels) * scales[batch_idx] / float(resolution_value)
                sign_votes = torch.sign(unique_sdf)
                for _ in range(int(sign_samples) - 1):
                    jitter = (torch.rand_like(corner_pts) * 2.0 - 1.0) * jitter_amp
                    sign_votes = sign_votes + torch.sign(query_signed(corner_pts + jitter, batch_idx))
                majority = torch.where(sign_votes == 0, torch.sign(unique_sdf), torch.sign(sign_votes))
                unique_sdf = majority * unique_sdf.abs().clamp(min=1e-12)
            unique_sdf = denoise_sparse_sdf_signs(unique_corner_coords, unique_sdf)
            unique_sdf = regularize_sparse_sdf_grid(unique_corner_coords, unique_sdf)
            corners = unique_sdf[inverse_corner].reshape(-1, 8).contiguous()

        active = ((corners < 0.0).any(dim=1) & (corners >= 0.0).any(dim=1))
        active_coords = coords[active]
        active_corners = corners[active]
        if active_coords.shape[0] == 0:
            out_vertices.append(vertices_list[batch_idx].new_zeros((0, 3)))
            out_faces.append(faces_list[batch_idx].new_zeros((0, 3)))
            continue

        verts_grid, faces_i = cubvh_backend.sparse_marching_cubes(
            active_coords,
            active_corners,
            0.0,
            ensure_consistency=ensure_consistency,
        )
        verts_i = (verts_grid.float() / float(resolution_value) - 0.5) * scales[batch_idx] + centers[batch_idx]
        faces_i = faces_i.int()

        project_back_i = float(project_backs[batch_idx].item())
        if project_back_i > 0.0 and verts_i.numel() > 0:
            verts_i = _project_mesh_vertices_single(
                bvh,
                batch_idx,
                batch_size,
                verts_i,
                faces_i,
                vertices_list[batch_idx],
                faces_list[batch_idx],
                project_back_i,
                project_mode,
                normal_smooth_iterations=project_normal_smooth_iterations,
                snap_uvw_margin=project_snap_uvw_margin,
            )

        out_vertices.append(verts_i)
        out_faces.append(faces_i)

    return restore_mesh_batch(out_vertices, out_faces, batch_kind)


def _sparse_binary_closing_voxels(
    coords: torch.Tensor,
    resolution: int,
    radius: int = 1,
    iterations: int = 1,
    connectivity: int = 1,
    chunk_size: int = 32768,
) -> torch.Tensor:
    if coords.shape[0] == 0 or radius <= 0 or iterations <= 0:
        return coords

    device = coords.device
    resolution = int(resolution)
    radius = int(radius)
    iterations = int(iterations)
    connectivity = int(connectivity)
    axis = torch.arange(-radius, radius + 1, device=device, dtype=torch.int64)
    offsets = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
    if connectivity == 1:
        offsets = offsets[offsets.abs().sum(dim=1) <= radius]
    elif connectivity == 2:
        offsets = offsets[offsets.abs().sum(dim=1) <= 2 * radius]
    elif connectivity == 3:
        pass
    else:
        raise ValueError(f"Unsupported closing_connectivity: {connectivity}. Expected 1, 2, or 3.")

    res = torch.tensor(resolution, dtype=torch.int64, device=device)
    res2 = res * res
    res3 = res2 * res

    def linearize(batch: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        xyz64 = xyz.to(torch.int64)
        batch64 = batch.to(torch.int64)
        return batch64 * res3 + xyz64[:, 0] * res2 + xyz64[:, 1] * res + xyz64[:, 2]

    def decode(keys: torch.Tensor) -> torch.Tensor:
        batch = keys // res3
        rem = keys - batch * res3
        x = rem // res2
        rem = rem - x * res2
        y = rem // res
        z = rem - y * res
        return torch.stack([batch, x, y, z], dim=1).to(torch.int32)

    closed_keys = torch.sort(torch.unique(linearize(coords[:, 0], coords[:, 1:]))).values
    for _ in range(iterations):
        dilated_parts = []
        for start in range(0, closed_keys.shape[0], chunk_size):
            chunk = decode(closed_keys[start : start + chunk_size])
            batch = chunk[:, 0].to(torch.int64)
            xyz = chunk[:, 1:].to(torch.int64)
            neighbor_xyz = xyz[:, None, :] + offsets[None, :, :]
            in_bounds = ((neighbor_xyz >= 0) & (neighbor_xyz < resolution)).all(dim=2)
            if not in_bounds.any():
                continue
            neighbor_batch = batch[:, None].expand(-1, offsets.shape[0])[in_bounds]
            neighbor_xyz = neighbor_xyz[in_bounds]
            dilated_parts.append(linearize(neighbor_batch, neighbor_xyz))
        if len(dilated_parts) == 0:
            return coords.new_empty((0, 4))
        dilated_keys = torch.sort(torch.unique(torch.cat(dilated_parts, dim=0))).values
        if dilated_keys.shape[0] == 0:
            return coords.new_empty((0, 4))

        kept_parts = []
        for start in range(0, dilated_keys.shape[0], chunk_size):
            chunk_keys = dilated_keys[start : start + chunk_size]
            chunk = decode(chunk_keys)
            batch = chunk[:, 0].to(torch.int64)
            xyz = chunk[:, 1:].to(torch.int64)
            neighbor_xyz = xyz[:, None, :] + offsets[None, :, :]
            in_bounds = ((neighbor_xyz >= 0) & (neighbor_xyz < resolution)).all(dim=2)
            neighbor_batch = batch[:, None].expand(-1, offsets.shape[0]).reshape(-1)
            neighbor_keys = linearize(neighbor_batch, neighbor_xyz.reshape(-1, 3)).reshape(chunk_keys.shape[0], offsets.shape[0])
            lookup = torch.searchsorted(dilated_keys, neighbor_keys.reshape(-1)).reshape_as(neighbor_keys)
            lookup = torch.clamp(lookup, max=dilated_keys.shape[0] - 1)
            exists = dilated_keys[lookup] == neighbor_keys
            keep = (exists | ~in_bounds).all(dim=1)
            if keep.any():
                kept_parts.append(chunk_keys[keep])
        if len(kept_parts) == 0:
            return coords.new_empty((0, 4))
        closed_keys = torch.sort(torch.unique(torch.cat(kept_parts, dim=0))).values
        if closed_keys.shape[0] == 0:
            return coords.new_empty((0, 4))

    return decode(closed_keys).contiguous()


def remesh_narrow_band_dc(
    vertices: Any,
    faces: Any,
    center: Any,
    scale: Any,
    resolution: Any,
    band: Any = 1,
    project_back: Any = 0,
    project_mode: str = "nearest",
    project_normal_smooth_iterations: int = 5,
    project_snap_uvw_margin: float = 0.05,
    closing_radius: int = 0,
    closing_iterations: int = 1,
    closing_connectivity: int = 1,
    verbose: bool = False,
    bvh: Any = None,
    vertex_mask: torch.Tensor | None = None,
    face_mask: torch.Tensor | None = None,
):
    from cumesh import _C

    vertices_list, faces_list, batch_kind = normalize_mesh_batch(vertices, faces, vertex_mask, face_mask)
    batch_size = len(vertices_list)
    device = vertices_list[0].device

    centers = _ensure_centers(center, batch_size, device)
    scales = _ensure_tensor_1d(scale, batch_size, "scale", device, torch.float32).reshape(batch_size)
    resolutions = _ensure_tensor_1d(resolution, batch_size, "resolution", device, torch.int32).reshape(batch_size)
    bands = _ensure_tensor_1d(band, batch_size, "band", device, torch.float32).reshape(batch_size)
    project_backs = _ensure_tensor_1d(project_back, batch_size, "project_back", device, torch.float32).reshape(batch_size)

    if not torch.equal(resolutions, resolutions[:1].expand_as(resolutions)):
        raise NotImplementedError("True batched remeshing currently requires the same resolution for every mesh in the batch.")
    resolution_value = int(resolutions[0].item())

    if bvh is None:
        bvh = cuBVH(vertices, faces, vertex_mask=vertex_mask, face_mask=face_mask)
    elif not isinstance(bvh, cuBVH):
        raise TypeError("Batched remeshing expects a cumesh_batch.cuBVH instance or None.")
    if bvh.batch_size != batch_size:
        raise ValueError(f"BVH batch size mismatch: expected {batch_size}, got {bvh.batch_size}.")

    offsets = _ensure_batched_constants(device)
    eps = bands * scales / resolution_value
    closing_radius = max(0, int(closing_radius))
    closing_iterations = max(0, int(closing_iterations))
    closing_connectivity = int(closing_connectivity)

    base_resolution = resolution_value
    while base_resolution > 32:
        if base_resolution % 2 != 0:
            raise ValueError("resolution must be divisible by 2 until it reaches the base resolution.")
        base_resolution //= 2

    base_coords = torch.stack(
        torch.meshgrid(
            torch.arange(base_resolution, device=device),
            torch.arange(base_resolution, device=device),
            torch.arange(base_resolution, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).int().reshape(-1, 3)
    batch_ids = torch.arange(batch_size, device=device, dtype=torch.int32).repeat_interleave(base_coords.shape[0])
    coords = torch.cat([batch_ids.unsqueeze(1), base_coords.repeat(batch_size, 1)], dim=1).contiguous()

    while True:
        batch_idx = coords[:, 0].long()
        cell_size = scales[batch_idx] / float(base_resolution)
        pts = ((coords[:, 1:].float() + 0.5) / float(base_resolution) - 0.5) * scales[batch_idx].unsqueeze(1) + centers[batch_idx]
        distances, _, _ = _bvh_unsigned_distance(bvh, pts, batch_idx, batch_size, return_uvw=False)
        distances = (distances - eps[batch_idx]).abs()
        coords = coords[distances < 0.87 * cell_size]

        if base_resolution >= resolution_value:
            break

        base_resolution *= 2
        coords[:, 1:] *= 2
        expanded = coords[:, None, :].expand(-1, 8, -1).clone()
        expanded[:, :, 1:] += offsets[None, :, :]
        coords = expanded.reshape(-1, 4).contiguous()

    if closing_radius > 0 and closing_iterations > 0:
        coords = _sparse_binary_closing_voxels(
            coords,
            resolution_value,
            radius=closing_radius,
            iterations=closing_iterations,
            connectivity=closing_connectivity,
        )

    num_voxels = coords.shape[0]
    if num_voxels == 0:
        empty_vertices = [vertices_list[i].new_zeros((0, 3)) for i in range(batch_size)]
        empty_faces = [faces_list[i].new_zeros((0, 3)) for i in range(batch_size)]
        return restore_mesh_batch(empty_vertices, empty_faces, batch_kind)

    hashmap_vox = _init_hashmap(resolution_value, batch_size, max(2 * max(num_voxels, 1), 1), device)
    _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap_vox, coords, resolution_value, resolution_value, resolution_value)
    grid_verts = _C.get_sparse_voxel_grid_active_vertices_batched(*hashmap_vox, coords, resolution_value, resolution_value, resolution_value)

    vert_batch_idx = grid_verts[:, 0].long()
    pts_vert = (grid_verts[:, 1:].float() / float(resolution_value) - 0.5) * scales[vert_batch_idx].unsqueeze(1) + centers[vert_batch_idx]
    distances_vert, _, _ = _bvh_unsigned_distance(bvh, pts_vert, vert_batch_idx, batch_size, return_uvw=False)
    distances_vert = distances_vert - eps[vert_batch_idx]

    hashmap_vert = _init_hashmap(resolution_value + 1, batch_size, max(2 * max(grid_verts.shape[0], 1), 1), device)
    _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap_vert, grid_verts, resolution_value + 1, resolution_value + 1, resolution_value + 1)
    dual_verts, intersected = _C.simple_dual_contour_batched(
        *hashmap_vert, coords, distances_vert, resolution_value + 1, resolution_value + 1, resolution_value + 1
    )

    edge_neighbor_voxel = coords.reshape(num_voxels, 1, 1, 4) + remesh_narrow_band_dc.edge_neighbor_voxel_offset
    connected_voxel = edge_neighbor_voxel[intersected != 0]
    intersected_flat = intersected[intersected != 0]
    connected_voxel_indices = _C.hashmap_lookup_3d_cuda(
        *hashmap_vox,
        connected_voxel.reshape(-1, 4).contiguous(),
        resolution_value,
        resolution_value,
        resolution_value,
    ).reshape(-1, 4).int()
    valid = (connected_voxel_indices != 0xFFFFFFFF).all(dim=1)
    quad_indices = connected_voxel_indices[valid]
    intersected_dir = intersected_flat[valid].int()

    if quad_indices.numel() == 0:
        empty_vertices = [vertices_list[i].new_zeros((0, 3)) for i in range(batch_size)]
        empty_faces = [faces_list[i].new_zeros((0, 3)) for i in range(batch_size)]
        return restore_mesh_batch(empty_vertices, empty_faces, batch_kind)

    unique_verts = torch.unique(quad_indices.reshape(-1))
    dual_verts = dual_verts[unique_verts]
    dual_batch = coords[unique_verts, 0].long()
    vert_map = torch.zeros((num_voxels,), dtype=torch.int32, device=device)
    vert_map[unique_verts] = torch.arange(unique_verts.shape[0], dtype=torch.int32, device=device)
    quad_indices = vert_map[quad_indices]

    mesh_vertices = (dual_verts / float(resolution_value) - 0.5) * scales[dual_batch].unsqueeze(1) + centers[dual_batch]
    attempt_triangles_0 = torch.where(
        (intersected_dir == 1).unsqueeze(1),
        quad_indices[:, remesh_narrow_band_dc.quad_split_1_p],
        quad_indices[:, remesh_narrow_band_dc.quad_split_1_n],
    )
    normals0 = torch.cross(
        mesh_vertices[attempt_triangles_0[:, 1]] - mesh_vertices[attempt_triangles_0[:, 0]],
        mesh_vertices[attempt_triangles_0[:, 2]] - mesh_vertices[attempt_triangles_0[:, 0]],
        dim=1,
    )
    normals1 = torch.cross(
        mesh_vertices[attempt_triangles_0[:, 2]] - mesh_vertices[attempt_triangles_0[:, 1]],
        mesh_vertices[attempt_triangles_0[:, 3]] - mesh_vertices[attempt_triangles_0[:, 1]],
        dim=1,
    )
    align0 = (normals0 * normals1).sum(dim=1).abs()

    attempt_triangles_1 = torch.where(
        (intersected_dir == 1).unsqueeze(1),
        quad_indices[:, remesh_narrow_band_dc.quad_split_2_p],
        quad_indices[:, remesh_narrow_band_dc.quad_split_2_n],
    )
    normals0 = torch.cross(
        mesh_vertices[attempt_triangles_1[:, 1]] - mesh_vertices[attempt_triangles_1[:, 0]],
        mesh_vertices[attempt_triangles_1[:, 2]] - mesh_vertices[attempt_triangles_1[:, 0]],
        dim=1,
    )
    normals1 = torch.cross(
        mesh_vertices[attempt_triangles_1[:, 2]] - mesh_vertices[attempt_triangles_1[:, 1]],
        mesh_vertices[attempt_triangles_1[:, 3]] - mesh_vertices[attempt_triangles_1[:, 1]],
        dim=1,
    )
    align1 = (normals0 * normals1).sum(dim=1).abs()
    mesh_triangles = torch.where((align0 > align1).unsqueeze(1), attempt_triangles_0, attempt_triangles_1).reshape(-1, 3).int()

    if project_mode not in {"nearest", "plane", "normal"}:
        raise ValueError(f"Unsupported project_mode: {project_mode}. Expected 'nearest', 'plane', or 'normal'.")
    if project_mode == "normal" and project_backs.max().item() > 0 and mesh_vertices.numel() > 0:
        # Deflate the offset shell along its own (smoothed) vertex normals by the
        # measured distance to the source surface. Unlike nearest-point or
        # nearest-plane projection, the displacement direction field is continuous
        # over the shell, so rims/edges of the source mesh do not cause spikes.
        face_normals = torch.cross(
            mesh_vertices[mesh_triangles[:, 1].long()] - mesh_vertices[mesh_triangles[:, 0].long()],
            mesh_vertices[mesh_triangles[:, 2].long()] - mesh_vertices[mesh_triangles[:, 0].long()],
            dim=1,
        )
        vert_normals = torch.zeros_like(mesh_vertices)
        for corner in range(3):
            vert_normals.index_add_(0, mesh_triangles[:, corner].long(), face_normals)
        vert_normals = vert_normals / vert_normals.norm(dim=1, keepdim=True).clamp(min=1e-12)

        distances, face_id, uvw = _bvh_unsigned_distance(bvh, mesh_vertices, dual_batch, batch_size, return_uvw=True)
        nearest = torch.empty_like(mesh_vertices)
        for batch_idx in range(batch_size):
            mask = dual_batch == batch_idx
            if not mask.any():
                continue
            orig_tri_verts = vertices_list[batch_idx][faces_list[batch_idx][face_id[mask].long()]]
            nearest[mask] = (orig_tri_verts * uvw[mask].unsqueeze(-1)).sum(dim=1)
        # Orient normals to point away from the source surface so "-normal" moves
        # toward it on both the outer and inner sheet of the shell.
        away = mesh_vertices - nearest
        flip = (vert_normals * away).sum(dim=1, keepdim=True) < 0.0
        vert_normals = torch.where(flip, -vert_normals, vert_normals)

        edge_src = mesh_triangles[:, [0, 1, 2, 1, 2, 0]].reshape(-1).long()
        edge_dst = mesh_triangles[:, [1, 2, 0, 0, 1, 2]].reshape(-1).long()
        for _ in range(max(0, int(project_normal_smooth_iterations))):
            neighbor_sum = torch.zeros_like(vert_normals)
            neighbor_sum.index_add_(0, edge_dst, vert_normals[edge_src])
            vert_normals = vert_normals + neighbor_sum
            vert_normals = vert_normals / vert_normals.norm(dim=1, keepdim=True).clamp(min=1e-12)

        scale_back = project_backs[dual_batch].unsqueeze(1)
        mesh_vertices = mesh_vertices - scale_back * distances.unsqueeze(1) * vert_normals

        # Snap vertices whose nearest point now lies strictly inside a source face
        # back onto the surface. Vertices nearest to rims/edges (hole caps, sharp
        # corners) are left at the deflated position to avoid rim collapse.
        if project_snap_uvw_margin >= 0.0:
            _, face_id, uvw = _bvh_unsigned_distance(bvh, mesh_vertices, dual_batch, batch_size, return_uvw=True)
            nearest = torch.empty_like(mesh_vertices)
            for batch_idx in range(batch_size):
                mask = dual_batch == batch_idx
                if not mask.any():
                    continue
                orig_tri_verts = vertices_list[batch_idx][faces_list[batch_idx][face_id[mask].long()]]
                nearest[mask] = (orig_tri_verts * uvw[mask].unsqueeze(-1)).sum(dim=1)
            interior = (uvw.min(dim=1).values > project_snap_uvw_margin).unsqueeze(1)
            mesh_vertices = torch.where(
                interior,
                mesh_vertices - scale_back * (mesh_vertices - nearest),
                mesh_vertices,
            )
    elif project_backs.max().item() > 0 and mesh_vertices.numel() > 0:
        _, face_id, uvw = _bvh_unsigned_distance(bvh, mesh_vertices, dual_batch, batch_size, return_uvw=True)
        displacement = torch.zeros_like(mesh_vertices)
        for batch_idx in range(batch_size):
            mask = dual_batch == batch_idx
            if not mask.any():
                continue
            local_face_id = face_id[mask].long()
            orig_tri_verts = vertices_list[batch_idx][faces_list[batch_idx][local_face_id]]
            projected = (orig_tri_verts * uvw[mask].unsqueeze(-1)).sum(dim=1)
            offset = mesh_vertices[mask] - projected
            if project_mode == "plane":
                # Move along the nearest face's normal onto its supporting plane.
                # For vertices whose nearest point lies in a face interior this equals
                # nearest-point projection; for vertices nearest to a rim/edge (hole
                # caps, sharp corners) it keeps each vertex in the plane of its
                # nearest face instead of collapsing it onto the rim line.
                normals = torch.cross(
                    orig_tri_verts[:, 1] - orig_tri_verts[:, 0],
                    orig_tri_verts[:, 2] - orig_tri_verts[:, 0],
                    dim=1,
                )
                normals = normals / normals.norm(dim=1, keepdim=True).clamp(min=1e-12)
                offset = (offset * normals).sum(dim=1, keepdim=True) * normals
            displacement[mask] = offset
        mesh_vertices = mesh_vertices - project_backs[dual_batch].unsqueeze(1) * displacement

    out_vertices = []
    out_faces = []
    for batch_idx in range(batch_size):
        vertex_ids = (dual_batch == batch_idx).nonzero(as_tuple=False).flatten()
        face_mask_i = dual_batch[mesh_triangles[:, 0]] == batch_idx
        faces_i = mesh_triangles[face_mask_i]
        remap = torch.full((mesh_vertices.shape[0],), -1, dtype=torch.long, device=device)
        remap[vertex_ids] = torch.arange(vertex_ids.shape[0], device=device)
        out_vertices.append(mesh_vertices[vertex_ids])
        out_faces.append(remap[faces_i.long()].int())

    return restore_mesh_batch(out_vertices, out_faces, batch_kind)
