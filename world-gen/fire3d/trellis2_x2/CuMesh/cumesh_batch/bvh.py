from __future__ import annotations

from typing import Any

import torch

from ._utils import (
    normalize_batch_arg,
    normalize_mesh_batch,
    pack_mesh_batch,
    restore_tensor_batch,
)


class cuBVH:
    def __init__(self, vertices: Any, triangles: Any, vertex_mask: torch.Tensor | None = None, face_mask: torch.Tensor | None = None):
        from cumesh.bvh import cuBVH as _SingleBVH

        vertices_list, triangles_list, batch_kind = normalize_mesh_batch(vertices, triangles, vertex_mask, face_mask)
        packed_vertices, packed_triangles, translations, stride = pack_mesh_batch(vertices_list, triangles_list)
        self._impl = _SingleBVH(packed_vertices, packed_triangles)
        self._batch_kind = batch_kind
        self._batch_size = len(vertices_list)
        self._translations = translations
        self._stride = stride
        self._face_offsets = []
        offset = 0
        for triangles_i in triangles_list:
            self._face_offsets.append(offset)
            offset += triangles_i.shape[0]
        self._face_offsets = torch.tensor(self._face_offsets, dtype=torch.long)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _translate_points(self, positions: Any, name: str):
        positions_list = normalize_batch_arg(positions, self._batch_size, name)
        translated = []
        for batch_idx, points in enumerate(positions_list):
            translated.append(points + self._translations[batch_idx].to(points.device, points.dtype))
        return positions_list, translated

    def _restore_face_ids(self, face_ids: list[torch.Tensor]) -> list[torch.Tensor]:
        restored = []
        for batch_idx, face_id in enumerate(face_ids):
            restored.append(face_id - self._face_offsets[batch_idx].to(face_id.device))
        return restored

    def ray_trace(self, rays_o: Any, rays_d: Any):
        rays_o_list, translated_rays_o = self._translate_points(rays_o, "rays_o")
        _, translated_rays_d = self._translate_points(rays_d, "rays_d")
        packed_rays_o = torch.cat(translated_rays_o, dim=0)
        packed_rays_d = torch.cat(translated_rays_d, dim=0)
        positions, face_id, depth = self._impl.ray_trace(packed_rays_o, packed_rays_d)

        counts = [rays.shape[0] for rays in rays_o_list]
        positions_list = list(torch.split(positions, counts, dim=0))
        face_list = list(torch.split(face_id, counts, dim=0))
        depth_list = list(torch.split(depth, counts, dim=0))
        for batch_idx in range(self._batch_size):
            positions_list[batch_idx] = positions_list[batch_idx] - self._translations[batch_idx].to(positions.device, positions.dtype)
        face_list = self._restore_face_ids(face_list)
        return (
            restore_tensor_batch(positions_list, self._batch_kind),
            restore_tensor_batch(face_list, self._batch_kind),
            restore_tensor_batch(depth_list, self._batch_kind),
        )

    def unsigned_distance(self, positions: Any, return_uvw: bool = False):
        positions_list, translated = self._translate_points(positions, "positions")
        packed_positions = torch.cat(translated, dim=0)
        distances, face_id, uvw = self._impl.unsigned_distance(packed_positions, return_uvw=return_uvw)
        counts = [points.shape[0] for points in positions_list]
        distances_list = list(torch.split(distances, counts, dim=0))
        face_list = self._restore_face_ids(list(torch.split(face_id, counts, dim=0)))
        uvw_list = list(torch.split(uvw, counts, dim=0)) if uvw is not None else None
        return (
            restore_tensor_batch(distances_list, self._batch_kind),
            restore_tensor_batch(face_list, self._batch_kind),
            restore_tensor_batch(uvw_list, self._batch_kind) if uvw_list is not None else None,
        )

    def signed_distance(self, positions: Any, return_uvw: bool = False, mode: str = "watertight"):
        positions_list, translated = self._translate_points(positions, "positions")
        packed_positions = torch.cat(translated, dim=0)
        distances, face_id, uvw = self._impl.signed_distance(packed_positions, return_uvw=return_uvw, mode=mode)
        counts = [points.shape[0] for points in positions_list]
        distances_list = list(torch.split(distances, counts, dim=0))
        face_list = self._restore_face_ids(list(torch.split(face_id, counts, dim=0)))
        uvw_list = list(torch.split(uvw, counts, dim=0)) if uvw is not None else None
        return (
            restore_tensor_batch(distances_list, self._batch_kind),
            restore_tensor_batch(face_list, self._batch_kind),
            restore_tensor_batch(uvw_list, self._batch_kind) if uvw_list is not None else None,
        )
