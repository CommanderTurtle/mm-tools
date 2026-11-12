"""Exact type-0 range unions and bounded-memory PyTorch BlockMask construction.

The portable representation contains integers, not device tensors or pickled
callables. Sparse block occupancy is separate from the exact token predicate.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1


def _integer_array(value, name, columns=None):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size == 0:
        array = np.empty((0, columns) if columns else (0,), dtype=np.int64)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers")
    if columns and (array.ndim != 2 or array.shape[1] != columns):
        raise ValueError(f"{name} must have shape [R,{columns}]")
    if not columns and array.ndim != 1:
        raise ValueError(f"{name} must have shape [R]")
    return np.array(array, dtype=np.int64, copy=True)


@dataclass(frozen=True, eq=False)
class RangeSpec:
    q_len: int
    kv_len: int
    q_ranges: np.ndarray
    k_ranges: np.ndarray
    attn_type_map: np.ndarray | None = None

    def __post_init__(self):
        for name in ("q_len", "kv_len"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))
        qr = _integer_array(self.q_ranges, "q_ranges", 2)
        kr = _integer_array(self.k_ranges, "k_ranges", 2)
        at = (
            np.zeros(len(qr), np.int64)
            if self.attn_type_map is None
            else _integer_array(self.attn_type_map, "attn_type_map")
        )
        if len(qr) != len(kr) or len(qr) != len(at):
            raise ValueError("Range/type counts must match")
        if np.any(at != 0):
            raise NotImplementedError("Only type-0 rectangle unions are supported")
        for name, array, length in [("q", qr, self.q_len), ("k", kr, self.kv_len)]:
            if (
                np.any(array < 0)
                or np.any(array > length)
                or np.any(array[:, 0] > array[:, 1])
            ):
                raise ValueError(f"Invalid {name} range bounds for length {length}")
        for name, array in [("q_ranges", qr), ("k_ranges", kr), ("attn_type_map", at)]:
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def fingerprint(self):
        obj = {
            "q_len": self.q_len,
            "kv_len": self.kv_len,
            "q_ranges": self.q_ranges.tolist(),
            "k_ranges": self.k_ranges.tolist(),
            "attn_type_map": self.attn_type_map.tolist(),
        }
        return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

    def save(self, path):
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            schema_version=FORMAT_VERSION,
            q_len=self.q_len,
            kv_len=self.kv_len,
            q_ranges=self.q_ranges,
            k_ranges=self.k_ranges,
            attn_type_map=self.attn_type_map,
        )
        Path(path).write_bytes(buffer.getvalue())

    @classmethod
    def load(cls, path, metadata=None):
        with np.load(path, allow_pickle=False) as data:
            if "q_len" in data:
                q_len, kv_len = int(data["q_len"]), int(data["kv_len"])
            elif metadata:
                q_len, kv_len = metadata["q_len"], metadata["kv_len"]
            else:
                raise ValueError("Fixture without shape requires metadata")
            return cls(
                q_len, kv_len, data["q_ranges"], data["k_ranges"], data["attn_type_map"]
            )

    def overlaps(self):
        """Report true QK rectangle overlap, not merely overlapping query spans."""
        result = []
        for i, (q, k) in enumerate(zip(self.q_ranges, self.k_ranges)):
            for j in range(i):
                if max(q[0], self.q_ranges[j, 0]) < min(
                    q[1], self.q_ranges[j, 1]
                ) and max(k[0], self.k_ranges[j, 0]) < min(k[1], self.k_ranges[j, 1]):
                    result.append((j, i))
        return result

    def oracle(self, queries, keys):
        """Independent rectangle paint oracle on arbitrary coordinate grids."""
        q = np.asarray(queries, np.int64).reshape(-1, 1)
        k = np.asarray(keys, np.int64).reshape(1, -1)
        result = np.zeros((q.size, k.size), dtype=bool)
        for (qs, qe), (ks, ke) in zip(self.q_ranges, self.k_ranges):
            result |= (q >= qs) & (q < qe) & (k >= ks) & (k < ke)
        return result

    def interval_tables(self):
        """Piecewise-constant query rows with merged disjoint key intervals."""
        boundaries = np.unique(np.r_[0, self.q_ranges.ravel(), self.q_len])
        rows = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            intervals = sorted(
                (int(k[0]), int(k[1]))
                for q, k in zip(self.q_ranges, self.k_ranges)
                if q[0] <= start < q[1] and k[0] < k[1]
            )
            merged = []
            for a, b in intervals:
                if merged and a <= merged[-1][1]:
                    merged[-1][1] = max(b, merged[-1][1])
                else:
                    merged.append([a, b])
            rows.append(merged)
        width = max(1, max(map(len, rows), default=0))
        starts = np.zeros((len(rows) + 1, width), np.int64)
        ends = np.zeros_like(starts)
        row_ids = np.full(self.q_len + 1, len(rows), np.int64)
        for index, intervals in enumerate(rows):
            row_ids[boundaries[index] : boundaries[index + 1]] = index
            for col, (a, b) in enumerate(intervals):
                starts[index, col], ends[index, col] = a, b
        return row_ids, starts, ends, boundaries

    def mask_mod(self, device="cpu", query_mask=None, key_mask=None):
        import torch

        row_ids, starts, ends, _ = self.interval_tables()
        lookup = torch.from_numpy(row_ids).to(device)
        width = starts.shape[1]
        lo = torch.from_numpy(starts.ravel()).to(device)
        hi = torch.from_numpy(ends.ravel()).to(device)
        q_len, kv_len = self.q_len, self.kv_len
        masked = query_mask is not None
        if masked:
            query_visible = torch.from_numpy(np.r_[query_mask, False]).to(device)
            key_visible = torch.from_numpy(np.r_[key_mask, False]).to(device)

        def mask_mod(batch, head, q_idx, kv_idx):
            row = lookup[q_idx.clamp(0, q_len)]
            allowed = (kv_idx >= lo[row * width]) & (kv_idx < hi[row * width])
            for col in range(1, width):
                allowed = allowed | (
                    (kv_idx >= lo[row * width + col]) & (kv_idx < hi[row * width + col])
                )
            if masked:
                allowed = allowed & query_visible[q_idx.clamp(0, q_len)] & key_visible[kv_idx.clamp(0, kv_len)]
            return (
                allowed
                & (q_idx >= 0)
                & (q_idx < q_len)
                & (kv_idx >= 0)
                & (kv_idx < kv_len)
            )

        return mask_mod

    def block_arrays(self, block_size=(128, 128), query_mask=None, key_mask=None):
        qb, kb = block_size
        if type(qb) is not int or type(kb) is not int or qb <= 0 or kb <= 0:
            raise ValueError("Block sizes must be positive integers")
        nq = (self.q_len + qb - 1) // qb
        nk = (self.kv_len + kb - 1) // kb
        if (query_mask is None) != (key_mask is None):
            raise ValueError("Query and key visibility must be supplied together")
        if query_mask is not None:
            for mask, length in ((query_mask, self.q_len), (key_mask, self.kv_len)):
                if mask.dtype != np.bool_ or mask.shape != (length,):
                    raise ValueError("Invalid token visibility mask")
        occupancy = np.zeros((nq, nk), dtype=np.uint8)
        full = np.zeros_like(occupancy, dtype=bool)
        for (a, b), (c, d) in zip(self.q_ranges, self.k_ranges):
            if a == b or c == d:
                continue
            occupancy[a // qb : (b + qb - 1) // qb, c // kb : (d + kb - 1) // kb] = 1
            # Only complete physical tiles covered by a single rectangle.
            full[(a + qb - 1) // qb : b // qb, (c + kb - 1) // kb : d // kb] = True
        if query_mask is not None:
            q_tiles = np.pad(query_mask, (0, nq * qb - self.q_len)).reshape(nq, qb)
            k_tiles = np.pad(key_mask, (0, nk * kb - self.kv_len)).reshape(nk, kb)
            occupancy &= q_tiles.any(1)[:, None] & k_tiles.any(1)[None, :]
            # Full tiles bypass mask_mod, so every token must be visible.
            full &= q_tiles.all(1)[:, None] & k_tiles.all(1)[None, :]
        occupancy[full] = 2
        arrays = {}
        for name, selected in [("kv", occupancy == 1), ("full_kv", occupancy == 2)]:
            counts = selected.sum(axis=-1, dtype=np.int32)
            # Full width is intentional: torch's sparse transpose assumes it.
            indices = np.zeros((nq, max(nk, 1)), dtype=np.int32)
            for row in range(nq):
                cols = np.flatnonzero(selected[row]).astype(np.int32)
                indices[row, : len(cols)] = cols
            arrays[name + "_num_blocks"] = counts[None, None, :]
            arrays[name + "_indices"] = indices[None, None, :, :]
        arrays.update(
            block_size=np.array(block_size, np.int64),
            seq_lengths=np.array([self.q_len, self.kv_len], np.int64),
            occupancy=occupancy,
            range_hash=np.array(self.fingerprint),
        )
        return arrays

    def block_mask(self, device="cpu", block_size=(128, 128), arrays=None,
                   query_mask=None, key_mask=None):
        import torch
        from torch.nn.attention.flex_attention import BlockMask

        if self.q_len == 0 or self.kv_len == 0:
            raise ValueError(
                "Empty attention uses adapter zero-output path, not BlockMask"
            )
        if arrays is not None and (query_mask is not None or key_mask is not None):
            raise ValueError("Precomputed block arrays cannot override token visibility")
        a = self.block_arrays(block_size, query_mask, key_mask) if arrays is None else arrays
        if str(a["range_hash"]) != self.fingerprint or tuple(a["seq_lengths"]) != (
            self.q_len,
            self.kv_len,
        ):
            raise ValueError("Block indices belong to a different mask/shape")
        if tuple(a["block_size"]) != tuple(block_size):
            raise ValueError("Block size mismatch")
        fields = {
            name: torch.from_numpy(np.array(a[name], copy=True)).to(device)
            for name in [
                "kv_num_blocks",
                "kv_indices",
                "full_kv_num_blocks",
                "full_kv_indices",
            ]
        }
        return BlockMask.from_kv_blocks(
            **fields,
            BLOCK_SIZE=tuple(block_size),
            mask_mod=self.mask_mod(device, query_mask, key_mask),
            seq_lengths=(self.q_len, self.kv_len),
        )


def exact_equivalence(spec, block_size=(128, 128), device="cpu", compiled=False):
    """Finite-partition exhaustive proof, with all query lookup entries checked.

    Rectangle and generated-predicate values are constant between every range
    endpoint/block boundary. Weighted cells cover the entire token domain.
    """
    import torch

    qb, kb = block_size
    q_edges = np.unique(
        np.r_[0, spec.q_ranges.ravel(), np.arange(0, spec.q_len, qb), spec.q_len]
    )
    k_edges = np.unique(
        np.r_[0, spec.k_ranges.ravel(), np.arange(0, spec.kv_len, kb), spec.kv_len]
    )
    rows, _lo, _hi, slab = spec.interval_tables()
    for i, (a, b) in enumerate(zip(slab[:-1], slab[1:])):
        if not np.all(rows[a:b] == i):
            raise AssertionError("Query lookup varies within proof slab")
    assert rows[-1] == len(slab) - 1
    arrays = spec.block_arrays(block_size)
    predicate = spec.mask_mod(device)

    def evaluate(q, k):
        zero = torch.zeros((), device=q.device, dtype=torch.int64)
        return predicate(zero, zero, q[:, None], k[None, :])

    if compiled:
        evaluate = torch.compile(evaluate, fullgraph=True)
    mismatch = block_mismatch = allowed = 0
    boundary_mismatch = 0
    k_coords = k_edges[:-1]
    for start in range(0, len(q_edges) - 1, 64):
        q_coords = q_edges[start : min(start + 64, len(q_edges) - 1)]
        oracle = spec.oracle(q_coords, k_coords)
        actual = (
            evaluate(
                torch.tensor(q_coords, device=device),
                torch.tensor(k_coords, device=device),
            )
            .cpu()
            .numpy()
        )
        weights = (
            np.diff(q_edges)[start : start + len(q_coords), None]
            * np.diff(k_edges)[None, :]
        )
        mismatch += int(weights[oracle != actual].sum())
        allowed += int(weights[oracle].sum())
        if q_coords.size and k_coords.size:
            kind = arrays["occupancy"][q_coords[:, None] // qb, k_coords[None, :] // kb]
            effective = (kind == 2) | ((kind == 1) & actual)
            block_mismatch += int(weights[effective != oracle].sum())
    # Physical padded boundary/sentinel checks on both sides of real domain.
    qc = np.unique(
        np.r_[
            -1,
            0,
            spec.q_ranges.ravel() - 1,
            spec.q_ranges.ravel(),
            spec.q_len - 1,
            spec.q_len,
            spec.q_len + qb,
        ]
    )
    kc = np.unique(
        np.r_[
            -1,
            0,
            spec.k_ranges.ravel() - 1,
            spec.k_ranges.ravel(),
            spec.kv_len - 1,
            spec.kv_len,
            spec.kv_len + kb,
        ]
    )
    for start in range(0, len(qc), 64):
        actual = (
            evaluate(
                torch.tensor(qc[start : start + 64], device=device),
                torch.tensor(kc, device=device),
            )
            .cpu()
            .numpy()
        )
        boundary_mismatch += int(
            np.count_nonzero(actual != spec.oracle(qc[start : start + 64], kc))
        )
    report = {
        "method": "exact_endpoint_and_block_partition_with_full_query_lookup_check",
        "q_len": spec.q_len,
        "kv_len": spec.kv_len,
        "domain_elements": spec.q_len * spec.kv_len,
        "partition_cells": max(0, len(q_edges) - 1) * max(0, len(k_edges) - 1),
        "allowed_elements": allowed,
        "mask_mod_mismatch_count": mismatch,
        "block_mask_mismatch_count": block_mismatch,
        "boundary_mismatch_count": boundary_mismatch,
        "overlapping_rectangle_pairs": len(spec.overlaps()),
        "compiled": compiled,
        "device": str(device),
        "range_hash": spec.fingerprint,
        "block_metadata_bytes": sum(a.nbytes for a in arrays.values()),
        "token_dense_allocated": False,
    }
    report["passed"] = mismatch == block_mismatch == boundary_mismatch == 0
    return report
