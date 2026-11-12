"""Opt-in compiled pointwise/reduction helpers; no GEMM or visibility changes."""
from functools import lru_cache
import torch


def finite(q,k):
    return torch.isfinite(q).all() & torch.isfinite(k).all()


@lru_cache(maxsize=1)
def compiled(name):
    return torch.compile(globals()[name],fullgraph=True,dynamic=True)


def checked(name,*args):
    from .nomas_inference_optim import mode
    policy=mode('GUARDS')
    reference=globals()[name]
    if policy=='off':return reference(*args)
    result=compiled(name)(*args)
    if policy=='shadow':
        if not torch.equal(reference(*args),result):
            raise RuntimeError('Compiled finite guard changed: '+name)
        print('NOMAS_OPT_GUARD_EQUAL '+name,flush=True)
    return result
