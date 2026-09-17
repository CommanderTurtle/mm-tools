from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from ._utils import (
    normalize_batch_arg,
    normalize_mesh_batch,
    pack_mesh_batch,
    restore_mesh_batch,
    restore_tensor_batch,
    split_face_tensor_batch,
    split_index_tensor_batch,
    split_mesh_batch,
    split_vertex_tensor_batch,
)


class CuMesh:
    def __init__(self):
        self.mesh = None
        self._batch_kind = "single"
        self._batch_size = 0
        self._translations = None
        self._stride = None
        self._partition_cache = None

    def _invalidate_cache(self):
        self._partition_cache = None

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _ensure_initialized(self):
        if self.mesh is None:
            raise RuntimeError("CuMesh batch has not been initialized.")

    def init(
        self,
        vertices: Any,
        faces: Any,
        vertex_mask: torch.Tensor | None = None,
        face_mask: torch.Tensor | None = None,
    ):
        from cumesh.cumesh import CuMesh as _SingleCuMesh

        vertices_list, faces_list, batch_kind = normalize_mesh_batch(vertices, faces, vertex_mask, face_mask)
        packed_vertices, packed_faces, translations, stride = pack_mesh_batch(vertices_list, faces_list)
        self.mesh = _SingleCuMesh()
        self.mesh.init(packed_vertices.contiguous(), packed_faces.contiguous())
        self._batch_kind = batch_kind
        self._batch_size = len(vertices_list)
        self._translations = translations
        self._stride = stride
        self._invalidate_cache()
        return self

    def _read_partition(self):
        self._ensure_initialized()
        if self._partition_cache is not None:
            return self._partition_cache
        vertices, faces = self.mesh.read()
        partition = split_mesh_batch(vertices, faces, self._translations, self._stride, self._batch_size)
        self._partition_cache = partition
        return partition

    def _packed_face_mask(self, face_mask: Any) -> torch.Tensor:
        _, _, _, _, face_batch = self._read_partition()
        local_masks = normalize_batch_arg(face_mask, self._batch_size, "face_mask")
        packed_mask = torch.zeros((face_batch.shape[0],), dtype=torch.bool, device=face_batch.device)
        for batch_idx, local_mask in enumerate(local_masks):
            batch_face_ids = (face_batch == batch_idx).nonzero(as_tuple=False).flatten()
            packed_mask[batch_face_ids] = local_mask.to(device=packed_mask.device, dtype=torch.bool)
        return packed_mask

    def _split_vertex_values(self, values: torch.Tensor):
        _, _, _, batch_vertex_ids, _ = self._read_partition()
        return restore_tensor_batch(split_vertex_tensor_batch(values, batch_vertex_ids), self._batch_kind)

    def _split_face_values(self, values: torch.Tensor):
        _, _, _, _, face_batch = self._read_partition()
        return restore_tensor_batch(split_face_tensor_batch(values, face_batch, self._batch_size), self._batch_kind)

    def _split_edge_values(self, values: torch.Tensor):
        vertices, _, _, _, _ = self._read_partition()
        packed_vertices, _ = self.mesh.read()
        vertex_batch = torch.round(packed_vertices[:, 0] / self._stride).long().clamp(0, self._batch_size - 1)
        edge_batch = vertex_batch[values[:, 0]]
        return restore_tensor_batch(split_index_tensor_batch(values, edge_batch, self._batch_size), self._batch_kind)

    @property
    def num_vertices(self):
        vertices, _, _, _, _ = self._read_partition()
        values = [vertex.shape[0] for vertex in vertices]
        return values[0] if self._batch_kind == "single" else values

    @property
    def num_faces(self):
        _, faces, _, _, _ = self._read_partition()
        values = [face.shape[0] for face in faces]
        return values[0] if self._batch_kind == "single" else values

    @property
    def num_edges(self):
        edges = self.read_edges()
        if self._batch_kind == "single":
            return edges.shape[0]
        return [edge.shape[0] for edge in edges]

    @property
    def num_boundaries(self):
        boundaries = self.read_boundaries()
        if self._batch_kind == "single":
            return boundaries.shape[0]
        return [boundary.shape[0] for boundary in boundaries]

    @property
    def num_conneted_components(self):
        num_components, _ = self.read_connected_components()
        return num_components

    @property
    def num_boundary_conneted_components(self):
        num_components, _ = self.read_boundary_connected_components()
        return num_components

    @property
    def num_boundary_loops(self):
        num_loops, _, _ = self.read_boundary_loops()
        return num_loops

    def clear_cache(self):
        self._ensure_initialized()
        self.mesh.clear_cache()

    def read(self):
        vertices, faces, _, _, _ = self._read_partition()
        return restore_mesh_batch(vertices, faces, self._batch_kind)

    def read_face_normals(self):
        self._ensure_initialized()
        return self._split_face_values(self.mesh.read_face_normals())

    def read_vertex_normals(self):
        self._ensure_initialized()
        return self._split_vertex_values(self.mesh.read_vertex_normals())

    def read_edges(self):
        self._ensure_initialized()
        return self._split_edge_values(self.mesh.read_edges())

    def read_boundaries(self):
        self._ensure_initialized()
        edges = self.mesh.read_edges()
        boundaries = self.mesh.read_boundaries()
        packed_vertices, _ = self.mesh.read()
        vertex_batch = torch.round(packed_vertices[:, 0] / self._stride).long().clamp(0, self._batch_size - 1)
        edge_batch = vertex_batch[edges[:, 0]]
        boundary_batch = edge_batch[boundaries]
        return restore_tensor_batch(split_index_tensor_batch(boundaries, boundary_batch, self._batch_size), self._batch_kind)

    def read_manifold_face_adjacency(self):
        self._ensure_initialized()
        adjacency = self.mesh.read_manifold_face_adjacency()
        _, _, _, _, face_batch = self._read_partition()
        adj_batch = face_batch[adjacency[:, 0]]
        return restore_tensor_batch(split_index_tensor_batch(adjacency, adj_batch, self._batch_size), self._batch_kind)

    def read_manifold_boundary_adjacency(self):
        self._ensure_initialized()
        adjacency = self.mesh.read_manifold_boundary_adjacency()
        return restore_tensor_batch([adjacency], self._batch_kind) if self._batch_kind == "single" else [adjacency]

    def read_connected_components(self):
        self._ensure_initialized()
        _, conn_ids = self.mesh.read_connected_components()
        _, _, _, _, face_batch = self._read_partition()
        split_ids = split_face_tensor_batch(conn_ids, face_batch, self._batch_size)
        num_components = [int(ids.max().item() + 1) if ids.numel() > 0 else 0 for ids in split_ids]
        return (num_components[0], split_ids[0]) if self._batch_kind == "single" else (num_components, restore_tensor_batch(split_ids, self._batch_kind))

    def read_boundary_connected_components(self):
        self._ensure_initialized()
        _, conn_ids = self.mesh.read_boundary_connected_components()
        boundaries = self.mesh.read_boundaries()
        edges = self.mesh.read_edges()
        packed_vertices, _ = self.mesh.read()
        vertex_batch = torch.round(packed_vertices[:, 0] / self._stride).long().clamp(0, self._batch_size - 1)
        edge_batch = vertex_batch[edges[:, 0]]
        boundary_batch = edge_batch[boundaries]
        split_ids = split_index_tensor_batch(conn_ids, boundary_batch, self._batch_size)
        num_components = [int(ids.max().item() + 1) if ids.numel() > 0 else 0 for ids in split_ids]
        return (num_components[0], split_ids[0]) if self._batch_kind == "single" else (num_components, restore_tensor_batch(split_ids, self._batch_kind))

    def read_boundary_loops(self):
        self._ensure_initialized()
        num_loops, loop_edge_ids, loop_offsets = self.mesh.read_boundary_loops()
        if self._batch_kind == "single":
            return num_loops, loop_edge_ids, loop_offsets
        boundaries = self.mesh.read_boundaries()
        edges = self.mesh.read_edges()
        packed_vertices, _ = self.mesh.read()
        vertex_batch = torch.round(packed_vertices[:, 0] / self._stride).long().clamp(0, self._batch_size - 1)
        edge_batch = vertex_batch[edges[:, 0]]
        boundary_batch = edge_batch[boundaries]
        out_num_loops = []
        out_loop_ids = []
        out_offsets = []
        for batch_idx in range(self._batch_size):
            batch_loop_mask = boundary_batch[loop_edge_ids] == batch_idx
            batch_ids = loop_edge_ids[batch_loop_mask]
            out_num_loops.append(0 if batch_ids.numel() == 0 else 1)
            out_loop_ids.append(batch_ids)
            out_offsets.append(torch.tensor([0, batch_ids.shape[0]], dtype=loop_offsets.dtype, device=loop_offsets.device))
        return out_num_loops, out_loop_ids, out_offsets

    def read_all_cache(self):
        return self.mesh.read_all_cache()

    def compute_face_normals(self):
        self.mesh.compute_face_normals()

    def compute_vertex_normals(self):
        self.mesh.compute_vertex_normals()

    def get_vertex_face_adjacency(self):
        self.mesh.get_vertex_face_adjacency()

    def get_edges(self):
        self.mesh.get_edges()

    def get_edge_face_adjacency(self):
        self.mesh.get_edge_face_adjacency()

    def get_vertex_edge_adjacency(self):
        self.mesh.get_vertex_edge_adjacency()

    def get_boundary_info(self):
        self.mesh.get_boundary_info()

    def get_vertex_boundary_adjacency(self):
        self.mesh.get_vertex_boundary_adjacency()

    def get_manifold_face_adjacency(self):
        self.mesh.get_manifold_face_adjacency()

    def get_manifold_boundary_adjacency(self):
        self.mesh.get_manifold_boundary_adjacency()

    def get_connected_components(self):
        self.mesh.get_connected_components()

    def get_boundary_connected_components(self):
        self.mesh.get_boundary_connected_components()

    def get_boundary_loops(self):
        self.mesh.get_boundary_loops()

    def remove_faces(self, face_mask: Any):
        self.mesh.remove_faces(self._packed_face_mask(face_mask).contiguous())
        self._invalidate_cache()

    def remove_unreferenced_vertices(self):
        self.mesh.remove_unreferenced_vertices()
        self._invalidate_cache()

    def remove_duplicate_faces(self):
        self.mesh.remove_duplicate_faces()
        self._invalidate_cache()

    def remove_degenerate_faces(self, abs_thresh: float = 1e-24, rel_thresh: float = 1e-12):
        self.mesh.remove_degenerate_faces(abs_thresh, rel_thresh)
        self._invalidate_cache()

    def fill_holes(self, max_hole_perimeter: float = 3e-2):
        self.mesh.fill_holes(max_hole_perimeter)
        self._invalidate_cache()

    def repair_non_manifold_edges(self):
        self.mesh.repair_non_manifold_edges()
        self._invalidate_cache()

    def remove_non_manifold_faces(self):
        self.mesh.remove_non_manifold_faces()
        self._invalidate_cache()

    def remove_small_connected_components(self, min_area: float):
        self.mesh.remove_small_connected_components(min_area)
        self._invalidate_cache()

    def unify_face_orientations(self):
        self.mesh.unify_face_orientations()
        self._invalidate_cache()

    def simplify(self, target_num_faces: Any, verbose: bool = False, options: dict | None = None):
        targets = normalize_batch_arg(target_num_faces, self._batch_size, "target_num_faces")
        if self._batch_kind == "single":
            self.mesh.simplify(targets[0], verbose=verbose, options={} if options is None else options)
            self._invalidate_cache()
            return
        options = {} if options is None else options
        target_tensor = torch.tensor(
            [int(target) for target in targets],
            dtype=torch.int32,
            device=self._translations.device,
        )
        if torch.any(target_tensor <= 0):
            raise ValueError("target_num_faces values must be positive integers.")

        _, faces, vertex_batch, _, face_batch = self._read_partition()
        face_counts = torch.tensor(
            [face.shape[0] for face in faces],
            dtype=torch.int32,
            device=target_tensor.device,
        )
        if bool(torch.all(face_counts <= target_tensor).item()):
            return

        vertex_batch = vertex_batch.to(dtype=torch.int32).contiguous()
        face_batch = face_batch.to(dtype=torch.int32).contiguous()
        thresh = options.get("thresh", 1e-8)
        lambda_edge_length = options.get("lambda_edge_length", 1e-2)
        lambda_skinny = options.get("lambda_skinny", 1e-3)
        num_face = int(face_counts.sum().item())
        while True:
            vertex_batch, face_batch, face_counts = self.mesh.simplify_step_batched(
                vertex_batch,
                face_batch,
                target_tensor,
                lambda_edge_length,
                lambda_skinny,
                thresh,
                False,
            )
            if bool(torch.all(face_counts <= target_tensor).item()):
                break

            new_num_face = int(face_counts.sum().item())
            del_num_face = num_face - new_num_face
            if num_face > 0 and del_num_face / num_face < 1e-2:
                thresh *= 10
            num_face = new_num_face
        self._invalidate_cache()

    def compute_charts(
        self,
        threshold_cone_half_angle_rad: float = math.radians(90),
        refine_iterations: int = 100,
        global_iterations: int = 3,
        smooth_strength: float = 1,
        area_penalty_weight: float = 0.1,
        perimeter_area_ratio_weight: float = 0.0001,
    ):
        self.mesh.compute_charts(
            threshold_cone_half_angle_rad,
            refine_iterations,
            global_iterations,
            smooth_strength,
            area_penalty_weight,
            perimeter_area_ratio_weight,
        )

    def read_atlas_charts(self):
        return self.mesh.read_atlas_charts()

    def uv_unwrap(
        self,
        compute_charts_kwargs: dict | None = None,
        xatlas_compute_charts_kwargs: dict | None = None,
        xatlas_pack_charts_kwargs: dict | None = None,
        return_vmaps: bool = False,
        verbose: bool = False,
        parallel_workers: int | None = None,
    ):
        compute_charts_kwargs = {} if compute_charts_kwargs is None else dict(compute_charts_kwargs)
        xatlas_compute_charts_kwargs = {} if xatlas_compute_charts_kwargs is None else dict(xatlas_compute_charts_kwargs)
        xatlas_pack_charts_kwargs = {} if xatlas_pack_charts_kwargs is None else dict(xatlas_pack_charts_kwargs)

        if self._batch_kind == "single":
            outputs = self.mesh.uv_unwrap(
                compute_charts_kwargs=compute_charts_kwargs,
                xatlas_compute_charts_kwargs=xatlas_compute_charts_kwargs,
                xatlas_pack_charts_kwargs=xatlas_pack_charts_kwargs,
                return_vmaps=return_vmaps,
                verbose=verbose,
            )
            self._invalidate_cache()
            return outputs

        return self._uv_unwrap_batched_parallel(
            compute_charts_kwargs=compute_charts_kwargs,
            xatlas_compute_charts_kwargs=xatlas_compute_charts_kwargs,
            xatlas_pack_charts_kwargs=xatlas_pack_charts_kwargs,
            return_vmaps=return_vmaps,
            verbose=verbose,
            parallel_workers=parallel_workers,
        )

    def _uv_unwrap_batched_parallel(
        self,
        compute_charts_kwargs: dict,
        xatlas_compute_charts_kwargs: dict,
        xatlas_pack_charts_kwargs: dict,
        return_vmaps: bool,
        verbose: bool,
        parallel_workers: int | None,
    ):
        import time as _time

        from cumesh.xatlas import Atlas

        xatlas_compute_charts_kwargs["verbose"] = verbose
        xatlas_pack_charts_kwargs["verbose"] = verbose

        # 1. Run shared GPU stages once on the packed mesh.
        _t0 = _time.perf_counter()
        self.mesh.remove_degenerate_faces()
        self._invalidate_cache()
        self.mesh.compute_charts(**compute_charts_kwargs)
        torch.cuda.synchronize()
        _t_charts = _time.perf_counter()

        new_vertices_packed, _ = self.mesh.read()
        num_charts, _, chart_vmap, chart_faces, chart_vertex_offset, chart_face_offset = self.mesh.read_atlas_charts()

        # 2. Move per-chart bookkeeping to CPU once.
        chart_vmap_cpu = chart_vmap.cpu()
        chart_faces_cpu = chart_faces.cpu()
        chart_vertex_offset_cpu = chart_vertex_offset.cpu()
        chart_face_offset_cpu = chart_face_offset.cpu()
        chart_vertices_packed_cpu = new_vertices_packed[chart_vmap].cpu()  # already chart-vmap-ordered

        # 3. Determine which batch each chart belongs to (charts cannot span batches because
        #    spatial X-stride packing keeps meshes disconnected).
        chart_batch_ids = [0] * int(num_charts)
        if num_charts > 0:
            first_vertex_idx = chart_vertex_offset_cpu[: int(num_charts)].long()
            first_x = chart_vertices_packed_cpu.index_select(0, first_vertex_idx)[:, 0]
            inferred = torch.round(first_x / self._stride).long().clamp_(0, self._batch_size - 1)
            chart_batch_ids = inferred.tolist()

        batch_chart_indices: list[list[int]] = [[] for _ in range(self._batch_size)]
        for chart_idx, batch_id in enumerate(chart_batch_ids):
            batch_chart_indices[batch_id].append(chart_idx)

        # 4. Per-batch xatlas pipeline. xatlas releases the GIL, so threading parallelizes truly.
        def process_batch(batch_id: int):
            chart_ids = batch_chart_indices[batch_id]
            if not chart_ids:
                empty_long = torch.empty((0,), dtype=torch.long)
                empty_faces = torch.empty((0, 3), dtype=torch.int32)
                empty_uvs = torch.empty((0, 2), dtype=torch.float32)
                return empty_long, empty_faces, empty_uvs

            atlas = Atlas()
            local_chart_vmaps = []
            for chart_idx in chart_ids:
                v_start = int(chart_vertex_offset_cpu[chart_idx].item())
                v_end = int(chart_vertex_offset_cpu[chart_idx + 1].item())
                f_start = int(chart_face_offset_cpu[chart_idx].item())
                f_end = int(chart_face_offset_cpu[chart_idx + 1].item())
                chart_v = chart_vertices_packed_cpu[v_start:v_end].contiguous().to(torch.float32)
                chart_f = (chart_faces_cpu[f_start:f_end] - v_start).contiguous().to(torch.int32)
                chart_vmap_i = chart_vmap_cpu[v_start:v_end]
                local_chart_vmaps.append(chart_vmap_i)
                atlas.add_mesh(chart_v, chart_f)

            _tc0 = _time.perf_counter()
            atlas.compute_charts(**xatlas_compute_charts_kwargs)
            _tc1 = _time.perf_counter()
            atlas.pack_charts(**xatlas_pack_charts_kwargs)
            _tc2 = _time.perf_counter()
            with _uv_lock:
                _uv_substage["xatlas_parameterize"] += _tc1 - _tc0
                _uv_substage["xatlas_pack"] += _tc2 - _tc1

            vmaps_local = []
            faces_local = []
            uvs_local = []
            cnt = 0
            for k in range(len(chart_ids)):
                vmap, x_faces, x_uvs = atlas.get_mesh(k)
                vmaps_local.append(local_chart_vmaps[k][vmap.long()])
                faces_local.append(x_faces + cnt)
                uvs_local.append(x_uvs)
                cnt += int(vmap.shape[0])

            return (
                torch.cat(vmaps_local, dim=0),
                torch.cat(faces_local, dim=0),
                torch.cat(uvs_local, dim=0),
            )

        import threading as _threading

        _uv_lock = _threading.Lock()
        _uv_substage = {"xatlas_parameterize": 0.0, "xatlas_pack": 0.0}
        _t_prep = _time.perf_counter()
        if parallel_workers is not None and parallel_workers <= 0:
            raise ValueError("parallel_workers must be positive")
        requested_workers = parallel_workers or (os.cpu_count() or 1)
        max_workers = min(self._batch_size, max(1, int(requested_workers)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_batch, range(self._batch_size)))
        _t_xatlas = _time.perf_counter()
        # Stage split for profiling: GPU chart computation vs per-mesh xatlas
        # chartify+pack (threaded CPU) vs bookkeeping. Read by callers after
        # uv_unwrap; overwritten per call.
        self.last_uv_timings = {
            "charts_gpu_seconds": _t_charts - _t0,
            "chart_readout_seconds": _t_prep - _t_charts,
            "xatlas_pack_seconds": _t_xatlas - _t_prep,
            # thread-summed CPU time inside the xatlas stage (can exceed the
            # wall-clock xatlas_pack_seconds because workers run in parallel)
            "xatlas_parameterize_cpu_seconds": _uv_substage["xatlas_parameterize"],
            "xatlas_pack_cpu_seconds": _uv_substage["xatlas_pack"],
            "parallel_workers": max_workers,
        }

        # 5. Recover per-batch vertex positions (in original local coordinates).
        vertices_packed_cpu = new_vertices_packed.cpu()
        translations_cpu = self._translations.detach().cpu().to(vertices_packed_cpu.dtype)

        vertices_list = []
        faces_list = []
        uvs_list = []
        vmaps_list = []
        for batch_id, (vmaps_b, faces_b, uvs_b) in enumerate(results):
            if vmaps_b.numel() == 0:
                vertices_list.append(torch.empty((0, 3), dtype=vertices_packed_cpu.dtype))
            else:
                vertices_list.append(vertices_packed_cpu.index_select(0, vmaps_b.long()) - translations_cpu[batch_id])
            faces_list.append(faces_b)
            uvs_list.append(uvs_b)
            vmaps_list.append(vmaps_b)

        out = [
            restore_tensor_batch(vertices_list, self._batch_kind),
            restore_tensor_batch(faces_list, self._batch_kind),
            restore_tensor_batch(uvs_list, self._batch_kind),
        ]
        if return_vmaps:
            out.append(restore_tensor_batch(vmaps_list, self._batch_kind))
        return tuple(out)
