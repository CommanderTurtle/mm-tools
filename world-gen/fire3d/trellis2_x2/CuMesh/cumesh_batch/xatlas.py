from __future__ import annotations

from typing import Any

from ._utils import normalize_batch_arg, normalize_mesh_batch, restore_tensor_batch


class Atlas:
    def __init__(self):
        self._atlas = None
        self._batch_kind = "single"
        self._batch_size = 0

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def add_mesh(
        self,
        vertices: Any,
        faces: Any,
        normals: Any = None,
        uvs: Any = None,
        vertex_mask=None,
        face_mask=None,
    ):
        from cumesh.xatlas import Atlas as _SingleAtlas

        vertices_list, faces_list, batch_kind = normalize_mesh_batch(vertices, faces, vertex_mask, face_mask)
        normals_list = normalize_batch_arg(normals, len(vertices_list), "normals") if normals is not None else [None] * len(vertices_list)
        uvs_list = normalize_batch_arg(uvs, len(vertices_list), "uvs") if uvs is not None else [None] * len(vertices_list)
        self._batch_kind = batch_kind
        self._batch_size = len(vertices_list)
        if self._atlas is None:
            self._atlas = _SingleAtlas()
        for vertices_i, faces_i, normals_i, uvs_i in zip(vertices_list, faces_list, normals_list, uvs_list):
            self._atlas.add_mesh(vertices_i, faces_i, normals_i, uvs_i)

    def compute_charts(self, **kwargs: Any):
        if self._atlas is None:
            raise RuntimeError("No meshes have been added to the batch atlas.")
        self._atlas.compute_charts(**kwargs)

    def pack_charts(self, **kwargs: Any):
        if self._atlas is None:
            raise RuntimeError("No meshes have been added to the batch atlas.")
        self._atlas.pack_charts(**kwargs)

    def get_mesh(self, index: int | None = None):
        if self._atlas is None:
            raise RuntimeError("No meshes have been added to the batch atlas.")
        if index is not None:
            return self._atlas.get_mesh(index)
        xrefs = []
        faces = []
        uvs = []
        for mesh_idx in range(self._batch_size):
            xref_i, faces_i, uvs_i = self._atlas.get_mesh(mesh_idx)
            xrefs.append(xref_i)
            faces.append(faces_i)
            uvs.append(uvs_i)
        return (
            restore_tensor_batch(xrefs, self._batch_kind),
            restore_tensor_batch(faces, self._batch_kind),
            restore_tensor_batch(uvs, self._batch_kind),
        )
