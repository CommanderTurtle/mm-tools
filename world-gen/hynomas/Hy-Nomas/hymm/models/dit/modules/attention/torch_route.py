"""Portable exact top-four prototype routing, without importing CuTe.

CUDA BF16 bmm uses FP32 output (no intermediate BF16 score rounding).
Only one query/key prototype tile is materialized at a time. Final selection
is a stable descending sort, preserving KV order for ties.
"""
import math
import torch


def full_route(q, k, qi, ki, bpf, config, *, return_scores=False, precision='bf16'):
    if torch.is_grad_enabled() or q.requires_grad or k.requires_grad:
        raise RuntimeError('Routing is inference-only')
    if (q.ndim != 4 or k.ndim != 4 or q.shape[0] != k.shape[0]
            or q.shape[2:] != (16, 128) or k.shape[2:] != (16, 128)
            or q.device != k.device or q.dtype != torch.float32 or k.dtype != torch.float32):
        raise ValueError('Expected matching FP32 [H,N,16,128] prototypes')
    h, nq = q.shape[:2]; nk = k.shape[1]
    fraction = float(config.get('global_keep_fraction', .25))
    if (config.get('selection_scope') != 'global_fraction' or not 0 < fraction <= 1
            or min(h,nq,nk,bpf) <= 0 or nk % bpf or qi.shape != (nq,) or ki.shape != (nk,)
            or qi.device != q.device or ki.device != q.device):
        raise ValueError('Expected global_fraction routing with complete KV frames and matching IDs')
    if precision not in ('bf16', 'fp32'):
        raise ValueError('Unknown scoring precision')
    budget = math.ceil(nk * fraction)
    workspace = int(float(config.get('router_workspace_gib', 16)) * 2**30)
    # Include result, converted prototypes, sort workspace and row scores.
    fixed = h*nq*budget*8 + (q.numel()+k.numel())*4
    if return_scores:
        fixed += h*nq*nk*4
    available = workspace-fixed
    if available < h*nk*24 + h*256*8:
        raise ValueError('Router workspace too small for one query row')
    qt = min(nq, 16, max(1, available//(h*(nk*24+256*8))))
    kt = min(nk, 128, max(1, (available-h*qt*nk*24)//(h*qt*256*8)))
    qb, kb = ((x.to(torch.bfloat16) for x in (q,k)) if precision == 'bf16' else (q,k))
    if q.device.type != 'cuda':
        qb, kb = qb.float(), kb.float()
    result = torch.empty((h,nq,budget),device=q.device,dtype=torch.int64)
    all_scores = torch.empty((h,nq,nk),device=q.device) if return_scores else None
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    if precision == 'fp32':
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for qs in range(0,nq,qt):
            qe = min(qs+qt,nq)
            scores = torch.empty((h,qe-qs,nk),device=q.device,dtype=torch.float32)
            a = qb[:,qs:qe].reshape(h,-1,128)
            for ks in range(0,nk,kt):
                ke = min(ks+kt,nk)
                b = kb[:,ks:ke].reshape(h,-1,128).transpose(1,2)
                dots = torch.bmm(a,b,out_dtype=torch.float32) if a.is_cuda and a.dtype == torch.bfloat16 else torch.bmm(a,b)
                pairs = dots.reshape(h,qe-qs,16,ke-ks,16).permute(0,1,3,2,4).reshape(h,qe-qs,ke-ks,256)
                scores[:,:,ks:ke] = pairs.topk(4,dim=-1).values.mean(-1)*(128**-.5)
            if config.get('force_self_block', True):
                scores.masked_fill_(qi[qs:qe,None] == ki[None,:], torch.inf)
            result[:,qs:qe] = scores.argsort(dim=-1,descending=True,stable=True)[...,:budget]
            if return_scores:
                all_scores[:,qs:qe] = scores
    finally:
        if precision == 'fp32':
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
    return (result,all_scores) if return_scores else result
