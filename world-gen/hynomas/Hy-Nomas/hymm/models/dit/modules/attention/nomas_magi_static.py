"""Static range metadata only; selected KV positions are rebuilt every call."""
from collections import OrderedDict
import inspect
from functools import lru_cache
import torch

_CACHE=OrderedDict()
_BUILD=None


def key_ranges(selected, extras):
    starts=selected.to(torch.int32)*64
    dynamic=torch.stack((starts,starts+64),dim=-1).reshape(-1,2)
    return torch.cat((dynamic,extras),dim=0)


@lru_cache(maxsize=1)
def backend_supported():
    try:
        import importlib
        f=importlib.import_module('magi_attention.functional.flex_flash_attn')
        names=set(inspect.signature(f._flex_flash_attn_forward).parameters)
        return {'auto_range_merge','merge_q_ranges','fwd_qk_map','fwd_unique_count','sparse_load_loop_count','equal_k_range_size'}<=names
    except (ImportError,AttributeError):
        return False


def supported(q,k,v,selected,nq,deterministic):
    return (not deterministic and not torch.is_grad_enabled() and q.is_cuda
            and q.dtype==k.dtype==v.dtype==torch.bfloat16
            and q.shape[-1]==k.shape[-1]==v.shape[-1]==128
            and selected.ndim==2 and nq==selected.shape[0]*64
            and backend_supported())


def attention(q,k,v,selected,extras,geometry_key,scale):
    from .nomas_sparse_geometry import geometry, video_ranges
    from .magi_static_geometry import prepare,attention as run
    global _BUILD
    key=(geometry_key,tuple(selected.shape),str(q.device))
    if key not in _CACHE:
        nq,nv,nh,nt,offsets,windows=geometry_key
        geo=geometry(nq,nv,nh,nt,offsets,windows,q.device)
        qr,kr,ty=video_ranges(selected,geo)
        meta=prepare(qr,kr,ty)
        _CACHE[key]=meta
        if len(_CACHE)>4:_CACHE.popitem(last=False)
    else:
        _CACHE.move_to_end(key)
        meta=_CACHE[key]
    if _BUILD is None:_BUILD=torch.compile(key_ranges,fullgraph=True,dynamic=True)
    kr=_BUILD(selected,extras[1])
    return run(q,k,v,kr,meta,scale)
