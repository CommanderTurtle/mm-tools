"""Explicit experimental Triton router. Stable selection matches torch_reference.

This mode opts into the fused kernel for all supported inputs; it never silently
changes router or attention backend. torch_reference is the portable rollback.
There is no shape whitelist. The score launch grid scales with H, Nq and Nk.
"""
import math
import torch


def full_route(q,k,qi,ki,bpf,config,*,return_scores=False):
    if torch.is_grad_enabled() or q.requires_grad or k.requires_grad:
        raise RuntimeError('Triton routing is inference-only')
    if (q.ndim!=4 or k.ndim!=4 or q.shape[0]!=k.shape[0]
            or q.shape[2:]!=(16,128) or k.shape[2:]!=(16,128)
            or not q.is_cuda or q.device!=k.device
            or q.dtype!=torch.float32 or k.dtype!=torch.float32
            or not q.is_contiguous() or not k.is_contiguous()):
        raise ValueError('Triton router requires contiguous CUDA FP32 [H,N,16,128] prototypes')
    h,nq=q.shape[:2];nk=k.shape[1]
    fraction=float(config.get('global_keep_fraction',.25))
    if (config.get('selection_scope')!='global_fraction' or not 0<fraction<=1
            or min(h,nq,nk,bpf)<=0 or nk%bpf or qi.shape!=(nq,) or ki.shape!=(nk,)
            or qi.device!=q.device or ki.device!=q.device):
        raise ValueError('Invalid global_fraction route geometry or IDs')
    budget=math.ceil(nk*fraction)
    from kernel.sparse_router_score import score
    efficient=bool(config.get('memory_efficient',False))
    free,total=torch.cuda.mem_get_info(q.device)
    fraction_limit=torch.cuda.get_per_process_memory_fraction(q.device)
    allocated=torch.cuda.memory_allocated(q.device)
    available=min(free+torch.cuda.memory_reserved(q.device)-allocated,
                  int(total*fraction_limit)-allocated)
    limit=min(available,int(float(config.get('router_workspace_gib',16))*2**30))
    output_bytes=h*nq*budget*8
    if not efficient:
        required=h*nq*nk*24+output_bytes
        if required>limit:raise ValueError('Triton router workspace limit exceeded')
        scores=score(q,k)
        if config.get('force_self_block',True):
            scores.masked_fill_(qi[:,None]==ki[None,:],torch.inf)
        result=scores.argsort(dim=-1,descending=True,stable=True)[...,:budget].contiguous()
        return (result,scores) if return_scores else result
    # Independent rows retain exactly the same stable ordering, including ties.
    # Bound sorting temporaries; account separately for retained outputs.
    score_bytes=h*nq*nk*4 if return_scores else 0
    workspace=min(256*2**20,limit-output_bytes-score_bytes)
    row_bytes=nk*32+16*128*4
    rows=int(workspace//row_bytes)
    if rows<4:raise ValueError('Triton router workspace cannot fit four query rows')
    heads=max(1,min(h,rows//nq))
    query_chunk=nq if rows>=nq else max(4,(rows//4)*4)
    result=torch.empty((h,nq,budget),device=q.device,dtype=torch.int64)
    all_scores=torch.empty((h,nq,nk),device=q.device,dtype=torch.float32) if return_scores else None
    for first_head in range(0,h,heads):
        for first_query in range(0,nq,query_chunk):
            hs=slice(first_head,min(h,first_head+heads))
            qs=slice(first_query,min(nq,first_query+query_chunk))
            scores=score(q[hs,qs].contiguous(),k[hs])
            if config.get('force_self_block',True):
                scores.masked_fill_(qi[qs,None]==ki[None,:],torch.inf)
            indices=scores.argsort(dim=-1,descending=True,stable=True)
            result[hs,qs].copy_(indices[...,:budget])
            if all_scores is not None:all_scores[hs,qs].copy_(scores)
            del indices,scores
    return (result,all_scores) if return_scores else result
