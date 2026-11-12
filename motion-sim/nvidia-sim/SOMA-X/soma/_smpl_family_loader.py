# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal SMPL-family asset loading helpers."""

import inspect
import pickle
import sys
import types
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import issparse


def ensure_chumpy_compat() -> None:
    """Install compatibility shims needed by legacy Chumpy-backed model pickles."""
    if not hasattr(inspect, "getargspec"):
        ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

        def getargspec(func):
            spec = inspect.getfullargspec(func)
            return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

        inspect.getargspec = getargspec

    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, value)

    try:
        import chumpy  # noqa: F401
    except ModuleNotFoundError:
        _install_chumpy_pickle_stub()


def _install_chumpy_pickle_stub() -> None:
    """Install the small Chumpy subset needed to unpickle legacy model shapedirs."""

    class Ch:
        def __setstate__(self, state: dict[str, Any]) -> None:
            self.__dict__.update(state)

        @property
        def r(self) -> np.ndarray:
            return np.asarray(self)

        @property
        def shape(self) -> tuple[int, ...]:
            return self.r.shape

        def __array__(self, dtype: np.dtype | type | None = None) -> np.ndarray:
            if not hasattr(self, "x"):
                raise TypeError("Unsupported Chumpy pickle object without 'x' data.")
            array = np.asarray(self.x)
            if dtype is not None:
                array = array.astype(dtype, copy=False)
            return array

    class Select(Ch):
        def __array__(self, dtype: np.dtype | type | None = None) -> np.ndarray:
            source = np.asarray(self.a).reshape(-1)
            result = source[np.asarray(self.idxs, dtype=np.int64)]
            if getattr(self, "preferred_shape", None) is not None:
                result = result.reshape(self.preferred_shape)
            if dtype is not None:
                result = result.astype(dtype, copy=False)
            return result

    Ch.__module__ = "chumpy.ch"
    Select.__module__ = "chumpy.reordering"

    chumpy_mod = types.ModuleType("chumpy")
    ch_mod = types.ModuleType("chumpy.ch")
    reordering_mod = types.ModuleType("chumpy.reordering")
    ch_mod.Ch = Ch
    reordering_mod.Select = Select
    chumpy_mod.Ch = Ch
    chumpy_mod.ch = ch_mod
    chumpy_mod.reordering = reordering_mod

    sys.modules.setdefault("chumpy", chumpy_mod)
    sys.modules.setdefault("chumpy.ch", ch_mod)
    sys.modules.setdefault("chumpy.reordering", reordering_mod)


def _to_numpy(value: Any, dtype: np.dtype | type | None = None) -> np.ndarray:
    if issparse(value):
        value = value.toarray()
    array = np.asarray(value)
    if array.shape == () and array.dtype == object:
        array = np.asarray(array.item())
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _read_model_file(model_path: Path) -> dict[str, Any]:
    suffix = model_path.suffix.lower()
    if suffix == ".npz":
        with np.load(model_path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    if suffix == ".pkl":
        ensure_chumpy_compat()
        with open(model_path, "rb") as f:
            return pickle.load(f, encoding="latin1")
    raise ValueError(f"Unsupported SMPL-family model file extension: {model_path.suffix!r}.")


def _get_required(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"SMPL-family model is missing required key; tried {keys}.")


def parent_ids_from_kintree(kintree_table: np.ndarray) -> np.ndarray:
    """Convert an SMPL-family kintree table into parent column indices."""
    kintree = _to_numpy(kintree_table, np.int64)
    if kintree.shape[0] != 2:
        raise ValueError(f"Expected kintree_table shape (2, J), got {kintree.shape}.")

    child_ids = kintree[1]
    id_to_col = {int(joint_id): idx for idx, joint_id in enumerate(child_ids)}
    parents = np.zeros(child_ids.shape[0], dtype=np.int64)
    for idx in range(1, child_ids.shape[0]):
        parent_id = int(kintree[0, idx])
        if parent_id not in id_to_col:
            raise ValueError(f"Parent joint id {parent_id} was not found in kintree children.")
        parents[idx] = id_to_col[parent_id]
    return parents


def _normalize_shapedirs(
    shapedirs: np.ndarray,
    *,
    vertex_count: int,
    num_betas: int | None,
) -> np.ndarray:
    shapedirs = _to_numpy(shapedirs, np.float32)
    if shapedirs.ndim != 3:
        raise ValueError(f"Expected shapedirs rank 3, got shape {shapedirs.shape}.")
    if shapedirs.shape[0] != vertex_count and shapedirs.shape[1:] == (vertex_count, 3):
        shapedirs = np.transpose(shapedirs, (1, 2, 0))
    if shapedirs.shape[0] != vertex_count or shapedirs.shape[1] != 3:
        raise ValueError(
            f"Expected shapedirs shape (V, 3, B), got {shapedirs.shape} for V={vertex_count}."
        )
    if num_betas is not None:
        shapedirs = shapedirs[:, :, :num_betas]
    return shapedirs


def _normalize_posedirs(posedirs: np.ndarray, *, vertex_count: int) -> np.ndarray:
    posedirs = _to_numpy(posedirs, np.float32)
    if posedirs.ndim == 3:
        if posedirs.shape[0] == vertex_count and posedirs.shape[1] == 3:
            return posedirs.reshape(vertex_count * 3, posedirs.shape[2]).T
        if posedirs.shape[1] == vertex_count and posedirs.shape[2] == 3:
            return posedirs.reshape(posedirs.shape[0], vertex_count * 3)
    if posedirs.ndim == 2:
        if posedirs.shape[1] == vertex_count * 3:
            return posedirs
        if posedirs.shape[0] == vertex_count * 3:
            return posedirs.T
    raise ValueError(f"Could not normalize posedirs with shape {posedirs.shape}.")


def load_smpl_family_model(
    model_path: str | Path,
    *,
    model_type: str,
    num_betas: int | None = 10,
) -> dict[str, np.ndarray]:
    """Load an SMPL, SMPL-H, or SMPL-X model into plain NumPy arrays.

    The returned tensors match the subset previously taken from the
    third-party ``smplx`` package: template vertices, shape directions,
    joint regressor, skinning weights, parents, faces, and pose directions.
    """
    model_type = model_type.lower()
    if model_type not in {"smpl", "smplh", "smplx"}:
        raise ValueError(f"Unsupported SMPL-family model type: {model_type!r}.")
    model_path = Path(model_path)
    data = _read_model_file(model_path)

    v_template = _to_numpy(_get_required(data, "v_template"), np.float32)
    vertex_count = v_template.shape[0]
    shapedirs = _normalize_shapedirs(
        _get_required(data, "shapedirs"),
        vertex_count=vertex_count,
        num_betas=num_betas,
    )
    j_regressor = _to_numpy(_get_required(data, "J_regressor"), np.float32)
    weights = _to_numpy(_get_required(data, "weights", "lbs_weights"), np.float32)
    faces = _to_numpy(_get_required(data, "f", "faces"), np.int64)
    parents = parent_ids_from_kintree(_get_required(data, "kintree_table"))
    posedirs = _normalize_posedirs(_get_required(data, "posedirs"), vertex_count=vertex_count)

    return {
        "v_template": v_template,
        "shapedirs": shapedirs,
        "J_regressor": j_regressor,
        "lbs_weights": weights,
        "parents": parents,
        "faces": faces,
        "posedirs": posedirs,
    }
