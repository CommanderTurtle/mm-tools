"""Opt-in rectangular multi-head CuTe BF16 router for Nomas inference.

SM90, FP32 pooled prototypes, 16 prototypes/block, head dim 128. BF16
rounding may change near-tied selections versus the FP32 legacy route.
"""
from functools import lru_cache
import torch
import math


@lru_cache(maxsize=1)
def dependencies():
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from cuda.bindings import driver as cuda
    from .score_sm90 import ScoreSm90
    from .ops import cast, SelectGroups
    return cute, from_dlpack, cuda, ScoreSm90, cast, SelectGroups


def groups_and_budgets(nk, bpf, config):
    selection = config.get('selection_scope', 'per_time_group')
    if selection == 'global_fraction':
        fraction = float(config.get('global_keep_fraction', .25))
        if not 0 < fraction <= 1 or bpf <= 0 or nk <= 0 or nk % bpf:
            raise ValueError('Global routing requires fraction in (0,1] and complete KV frames')
        return ((0, nk),), (math.ceil(nk*fraction),)
    if selection not in ('per_time_group', 'local_plus_global'):
        raise ValueError('Unknown selection_scope: '+str(selection))
    if selection == 'local_plus_global':
        fraction = float(config.get('global_keep_fraction', .125))
        if not 0 < fraction <= 1 or int(config.get('kv_time_group_size', 1)) != 1:
            raise ValueError('Local-plus-global requires fraction in (0,1] and single-frame local groups')
    group = int(config.get('kv_time_group_size', 1))
    topk = min(int(config.get('top_k_per_kv_frame', 5)), bpf)
    if min(group, topk, bpf, nk) <= 0 or nk % bpf:
        raise ValueError('Positive group/Top-k and complete KV frames required')
    groups = tuple((s, min(s+group*bpf, nk)) for s in range(0, nk, group*bpf))
    return groups, tuple((e-s)//bpf*topk for s, e in groups)


def eligibility(q, k, qi, ki, bpf, config):
    if torch.is_grad_enabled() or q.requires_grad or k.requires_grad:
        return 'grad_enabled', 0
    if (q.ndim != 4 or k.ndim != 4 or q.shape[0] != k.shape[0]
            or q.shape[2:] != (16, 128) or k.shape[2:] != (16, 128)
            or min(*q.shape, *k.shape) <= 0):
        return 'unsupported_geometry', 0
    if (not q.is_cuda or q.device != k.device or q.dtype != torch.float32
            or k.dtype != torch.float32 or not q.is_contiguous() or not k.is_contiguous()):
        return 'unsupported_layout_dtype_device', 0
    if q.data_ptr() % 16 or k.data_ptr() % 16:
        return 'unsupported_alignment', 0
    if (qi.shape != (q.shape[1],) or ki.shape != (k.shape[1],)
            or qi.device != q.device or ki.device != q.device):
        return 'unsupported_ids', 0
    if torch.cuda.get_device_capability(q.device)[0] != 9:
        return 'requires_sm90', 0
    groups, budgets = groups_and_budgets(k.shape[1], bpf, config)
    if config.get('selection_scope','per_time_group') != 'global_fraction' and max(e-s for s,e in groups) > 1024:
        return 'group_width_limit', 0
    h,nq,_,_=q.shape; nk=k.shape[1]
    workspace=(q.numel()+k.numel())*2+h*nq*nk*4+nq*nk+h*nq*sum(budgets)*8+64*2**20
    if config.get('selection_scope','per_time_group') in ('global_fraction', 'local_plus_global'):
        workspace += h*nq*nk*16  # stable sort indices and sorting workspace reserve
    if config.get('selection_scope') == 'local_plus_global':
        workspace += h*nq*nk*12  # masked scores plus concatenated output
    if int(config.get('query_time_group_size',1)) > 1:
        workspace += h*nq*nk*4  # temporal score reduction/expansion safety margin
    free,_=torch.cuda.mem_get_info(q.device)
    available=free+torch.cuda.memory_reserved(q.device)-torch.cuda.memory_allocated(q.device)
    if workspace>min(available,float(config.get('router_workspace_gib',16))*2**30):
        return 'workspace_limit', workspace
    try:
        dependencies()
    except (ImportError, ModuleNotFoundError) as exc:
        return 'missing_cute_dependency:'+str(exc.name), workspace
    from hymm.models.dit.modules.attention.nomas_compiled_aux import checked
    if not bool(checked('finite', q, k)):
        return 'nonfinite_input', workspace
    return None, workspace


# Per-process compiled entry cache. Device is part of every key, stream is
# supplied at invocation; no output/workspace tensor is reused between calls.
_CACHE={}


def _run(key, fn, args):
    cute,*_=dependencies()
    if key not in _CACHE:
        _CACHE[key]=cute.compile(fn,*args)
    _CACHE[key](*args)


def score(q,k):
    cute,from_dlpack,cuda,ScoreSm90,cast,_=dependencies()
    stream=cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    tensor=lambda x: from_dlpack(x,assumed_align=16)
    converted=[]
    for x in (q,k):
        y=torch.empty_like(x,dtype=torch.bfloat16)
        args=(tensor(x.view(-1)),tensor(y.view(-1)),x.numel(),stream)
        _run(('cast',str(x.device),tuple(x.shape)),cast,args)
        converted.append(y)
    qb,kb=converted
    h,nq,r,d=q.shape; nk=k.shape[1]
    out=torch.empty((h,nq,nk),device=q.device,dtype=torch.float32)
    args=(tensor(qb.reshape(h,nq*r,d).permute(1,2,0)),
          tensor(kb.reshape(h,nk*r,d).permute(1,2,0)),tensor(out),stream)
    from hymm.models.dit.modules.attention.nomas_inference_optim import mode, compare, note
    policy=mode('ROUTER')
    if policy=='off':
        _run(('score',str(q.device),h,nq,nk),ScoreSm90(64),args)
    else:
        from .score_top4_redux import ScoreTop4Redux
        _run(('redux',str(q.device),h,nq,nk),ScoreTop4Redux(128),args)
        note('router.redux')
        if policy=='shadow':
            ref=torch.empty_like(out)
            _run(('score',str(q.device),h,nq,nk),ScoreSm90(64),(*args[:-2],tensor(ref),stream))
            compare('router',ref,out,exact=True)
            return ref
    return out


def full_route(q,k,qi,ki,bpf,config,*,return_scores=False,_eligibility_checked=False):
    reason=None
    if not _eligibility_checked:
        reason,_=eligibility(q,k,qi,ki,bpf,config)
    if reason:
        raise ValueError('CuTe route unsupported: '+reason)
    query_group=int(config.get('query_time_group_size',1))
    if query_group < 1:
        raise ValueError('query_time_group_size must be positive')
    if query_group > 1:
        if int(config.get('kv_time_group_size',1)) != 1:
            raise ValueError('Shared Query routes require per-KV-frame budgets')
        if q.shape[1] % bpf or int(qi[0]) % bpf or not torch.equal(qi,qi[0]+torch.arange(q.shape[1],device=qi.device)):
            raise ValueError('Shared Query routes require complete contiguous raster frame IDs')
    _,from_dlpack,cuda,_,_,SelectGroups=dependencies()
    groups,budgets=groups_and_budgets(k.shape[1],bpf,config)
    with torch.cuda.device(q.device):
        scores=score(q,k)
        if config.get('force_self_block',True):
            scores.masked_fill_(qi[:,None]==ki[None,:],torch.inf)
        if config.get('selection_scope','per_time_group') == 'global_fraction':
            if query_group != 1:
                raise ValueError('Global routing requires query_time_group_size=1')
            out=scores.argsort(dim=-1,descending=True,stable=True)[...,:budgets[0]].contiguous()
            return (out,scores) if return_scores else out
        row_map=None
        if query_group > 1:
            # Mean router scores, not mean Q prototypes: scoring has a nonlinear
            # top-four prototype reduction. Self infinities above preserve the
            # union of all members' self blocks (one per KV frame/spatial site).
            h,nq,nk=scores.shape; frames=nq//bpf
            by_time=scores.view(h,frames,bpf,nk)
            complete=frames//query_group*query_group
            parts=[]
            if complete:
                parts.append(by_time[:,:complete].reshape(h,-1,query_group,bpf,nk).mean(2))
            if complete < frames:
                parts.append(by_time[:,complete:].mean(1,keepdim=True))
            scores=torch.cat(parts,dim=1).reshape(h,-1,nk).contiguous()
            row_map=((torch.arange(frames,device=q.device)//query_group)[:,None]*bpf+
                     torch.arange(bpf,device=q.device)[None,:]).flatten()
        out=torch.empty((q.shape[0],scores.shape[1],sum(budgets)),device=q.device,dtype=torch.int64)
        args=(from_dlpack(scores,assumed_align=16),from_dlpack(out,assumed_align=16),
              cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream))
        _run(('select',str(q.device),tuple(scores.shape),groups,budgets),SelectGroups(groups,budgets),args)
        if config.get('selection_scope') == 'local_plus_global':
            if query_group != 1:
                raise ValueError('Local-plus-global requires query_time_group_size=1')
            # Local quota is guaranteed. Global quota is measured against ALL
            # visible video blocks, then filled from unselected blocks only.
            # This keeps rows rectangular and avoids duplicate softmax mass.
            remaining = scores.clone()
            remaining.scatter_(-1, out, -torch.inf)
            extra = min(math.ceil(k.shape[1]*float(config.get('global_keep_fraction', .125))),
                        k.shape[1]-out.shape[-1])
            global_out = remaining.argsort(dim=-1, descending=True, stable=True)[..., :extra]
            out = torch.cat((out, global_out), dim=-1)
        if row_map is not None:
            out=out.index_select(1,row_map)
            if return_scores:
                scores=scores.index_select(1,row_map)
    return (out,scores) if return_scores else out
