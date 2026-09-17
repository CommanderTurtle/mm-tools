from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import torch


BatchKind = str


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def normalize_mesh_batch(
    vertices: Any,
    faces: Any,
    vertex_mask: torch.Tensor | None = None,
    face_mask: torch.Tensor | None = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], BatchKind]:
    if torch.is_tensor(vertices) and torch.is_tensor(faces):
        if vertices.ndim == 2 and faces.ndim == 2:
            return [vertices], [faces], "single"
        if vertices.ndim == 3 and faces.ndim == 3 and vertices.shape[0] == faces.shape[0]:
            batch_size = vertices.shape[0]
            if vertex_mask is None:
                vertex_mask = torch.ones(vertices.shape[:2], dtype=torch.bool, device=vertices.device)
            if face_mask is None:
                face_mask = torch.ones(faces.shape[:2], dtype=torch.bool, device=faces.device)
            vertices_list = [vertices[i][vertex_mask[i]] for i in range(batch_size)]
            faces_list = [faces[i][face_mask[i]] for i in range(batch_size)]
            return vertices_list, faces_list, "padded"
        raise ValueError("Expected [V, 3]/[F, 3] or padded [B, V, 3]/[B, F, 3] tensors.")

    if _is_sequence(vertices) and _is_sequence(faces):
        vertices_list = _as_list(vertices)
        faces_list = _as_list(faces)
        if len(vertices_list) != len(faces_list):
            raise ValueError("vertices and faces batches must have the same length.")
        return vertices_list, faces_list, "sequence"

    raise TypeError("vertices and faces must both be tensors or both be sequences of tensors.")


def normalize_batch_arg(value: Any, batch_size: int, name: str) -> List[Any]:
    if batch_size == 1 and not _is_sequence(value):
        return [value]
    if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == batch_size:
        return list(value.unbind(0))
    if _is_sequence(value):
        value_list = _as_list(value)
        if len(value_list) != batch_size:
            raise ValueError(f"{name} batch size mismatch: expected {batch_size}, got {len(value_list)}.")
        return value_list
    return [value for _ in range(batch_size)]


def _safe_center(vertices: torch.Tensor) -> torch.Tensor:
    if vertices.numel() == 0:
        return torch.zeros(3, dtype=torch.float32, device=vertices.device)
    vertices_f = vertices.float()
    return 0.5 * (vertices_f.min(dim=0).values + vertices_f.max(dim=0).values)


def _global_radius(vertices_list: Sequence[torch.Tensor]) -> float:
    radius = 0.0
    for vertices in vertices_list:
        if vertices.numel() == 0:
            continue
        centered = vertices.float() - _safe_center(vertices)
        radius = max(radius, centered.abs().amax().item())
    return radius


def make_transforms(vertices_list: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, float]:
    if not vertices_list:
        raise ValueError("Empty mesh batch.")
    device = vertices_list[0].device
    radius = _global_radius(vertices_list)
    stride = max(4.0 * radius + 1.0, 1.0)
    translations = []
    for batch_idx, vertices in enumerate(vertices_list):
        center = _safe_center(vertices)
        target = torch.tensor([batch_idx * stride, 0.0, 0.0], dtype=torch.float32, device=device)
        translations.append(target - center)
    return torch.stack(translations, dim=0), stride


def pack_mesh_batch(
    vertices_list: Sequence[torch.Tensor],
    faces_list: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    translations, stride = make_transforms(vertices_list)
    packed_vertices = []
    packed_faces = []
    vertex_offset = 0
    for batch_idx, (vertices, faces) in enumerate(zip(vertices_list, faces_list)):
        translation = translations[batch_idx].to(vertices.device, vertices.dtype)
        packed_vertices.append(vertices + translation)
        packed_faces.append(faces + vertex_offset)
        vertex_offset += vertices.shape[0]
    return torch.cat(packed_vertices, dim=0), torch.cat(packed_faces, dim=0), translations, stride


def infer_batch_ids_from_positions(
    positions: torch.Tensor,
    stride: float,
    batch_size: int,
) -> torch.Tensor:
    if positions.numel() == 0:
        return torch.empty((positions.shape[0],), dtype=torch.long, device=positions.device)
    batch_ids = torch.round(positions[:, 0] / stride).long()
    return batch_ids.clamp_(0, batch_size - 1)


def split_mesh_batch(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    translations: torch.Tensor,
    stride: float,
    batch_size: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, List[torch.Tensor], torch.Tensor]:
    vertex_batch = infer_batch_ids_from_positions(vertices, stride, batch_size)
    batch_vertex_ids = [(vertex_batch == batch_idx).nonzero(as_tuple=False).flatten() for batch_idx in range(batch_size)]

    remap = torch.full((vertices.shape[0],), -1, dtype=torch.long, device=vertices.device)
    vertices_list = []
    for batch_idx, global_ids in enumerate(batch_vertex_ids):
        remap[global_ids] = torch.arange(global_ids.shape[0], device=vertices.device)
        translation = translations[batch_idx].to(vertices.device, vertices.dtype)
        vertices_list.append(vertices[global_ids] - translation)

    face_batch = vertex_batch[faces[:, 0]] if faces.numel() > 0 else torch.empty((0,), dtype=torch.long, device=faces.device)
    faces_list = []
    for batch_idx in range(batch_size):
        batch_faces = faces[face_batch == batch_idx]
        faces_list.append(remap[batch_faces.long()].to(faces.dtype))

    return vertices_list, faces_list, vertex_batch, batch_vertex_ids, face_batch


def split_vertex_tensor_batch(
    values: torch.Tensor,
    batch_vertex_ids: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    return [values[global_ids] for global_ids in batch_vertex_ids]


def split_face_tensor_batch(
    values: torch.Tensor,
    face_batch: torch.Tensor,
    batch_size: int,
) -> List[torch.Tensor]:
    return [values[face_batch == batch_idx] for batch_idx in range(batch_size)]


def split_index_tensor_batch(
    values: torch.Tensor,
    owner_batch: torch.Tensor,
    batch_size: int,
) -> List[torch.Tensor]:
    return [values[owner_batch == batch_idx] for batch_idx in range(batch_size)]


def restore_mesh_batch(vertices_list: Sequence[torch.Tensor], faces_list: Sequence[torch.Tensor], kind: BatchKind):
    if kind == "single":
        return vertices_list[0], faces_list[0]
    if kind == "padded":
        return pad_mesh_batch(vertices_list, faces_list)
    return list(vertices_list), list(faces_list)


def restore_tensor_batch(values_list: Sequence[torch.Tensor], kind: BatchKind):
    if kind == "single":
        return values_list[0]
    if kind == "padded":
        return pad_tensor_batch(values_list)
    return list(values_list)


def pad_tensor_batch(values_list: Sequence[torch.Tensor]):
    if not values_list:
        raise ValueError("Empty batch.")
    batch_size = len(values_list)
    max_len = max(value.shape[0] for value in values_list)
    tail_shape = values_list[0].shape[1:]
    out = values_list[0].new_zeros((batch_size, max_len, *tail_shape))
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=values_list[0].device)
    for batch_idx, value in enumerate(values_list):
        out[batch_idx, : value.shape[0]] = value
        mask[batch_idx, : value.shape[0]] = True
    return out, mask


def pad_mesh_batch(vertices_list: Sequence[torch.Tensor], faces_list: Sequence[torch.Tensor]):
    vertices, vertex_mask = pad_tensor_batch(vertices_list)
    faces, face_mask = pad_tensor_batch(faces_list)
    return vertices, faces, vertex_mask, face_mask
