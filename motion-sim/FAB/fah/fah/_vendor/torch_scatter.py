"""Minimal API-compatible torch_scatter shim for DPVO inference (scatter_sum / scatter_mean / scatter_softmax with a 1-D
index along one dim). fah.world, scripts/benchmark_world_e2e.py and the docs/world.md commands always put this directory
first on the DPVO subprocess path: it reproduces the reported world numbers exactly (PyG's torch-scatter also works,
with slightly different numerics)."""
import torch


def scatter_sum(src, index, dim=-1, dim_size=None):
    assert index.dim() == 1
    dim = dim % src.dim()
    n = int(index.max()) + 1 if dim_size is None else dim_size
    shape = list(src.shape)
    shape[dim] = n
    return src.new_zeros(shape).index_add_(dim, index, src)


def scatter_mean(src, index, dim=-1, dim_size=None):
    dim = dim % src.dim()
    out = scatter_sum(src, index, dim, dim_size)
    cnt = src.new_zeros(out.shape[dim]).index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    view = [1] * src.dim()
    view[dim] = -1
    return out / cnt.clamp(min=1).view(view)


def scatter_softmax(src, index, dim=-1):  # DPVO SoftAgg
    assert index.dim() == 1
    dim = dim % src.dim()
    shape = list(src.shape)
    shape[dim] = int(index.max()) + 1
    idx = index.view([-1 if d == dim else 1 for d in range(src.dim())]).expand_as(src)
    mx = src.new_full(shape, float("-inf")).scatter_reduce(dim, idx, src, reduce="amax", include_self=True)
    e = (src - mx.gather(dim, idx)).exp()
    return e / src.new_zeros(shape).scatter_add(dim, idx, e).gather(dim, idx)
