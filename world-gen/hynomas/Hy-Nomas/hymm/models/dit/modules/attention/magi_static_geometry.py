"""Standalone inference metadata reuse experiment; no production dispatch.

Only query geometry, attention types and key-range LENGTHS are retained.
Dynamic selected key positions are reordered afresh on every invocation.
BF16 QK/PV and the original Magi FP32 accumulation/softmax kernel are unchanged.
Caller supplies aligned 64-token video query ranges and full-attention pairs.
"""
import importlib
import torch


def prepare(qr, kr, ty):
    f=importlib.import_module('magi_attention.functional.flex_flash_attn')
    assert qr.dtype==kr.dtype==torch.int32 and qr.is_cuda
    assert bool((qr[:,1]-qr[:,0]==64).all())
    assert bool((ty==0).all())
    # Stable ordering for equal query ranges, independent of selected K values.
    order=qr[:,0].argsort(stable=True)
    sq=qr.index_select(0,order)
    st=ty.index_select(0,order)
    lengths=(kr[:,1]-kr[:,0]).index_select(0,order)
    geometry_k=torch.stack((torch.zeros_like(lengths),lengths),-1)
    merged,sq,_,st,qk,unique=f.merge_ranges(sq,geometry_k,st)
    loop,invalid,equal=f.magi_attn_ext.compute_sparse_load_metadata(geometry_k,qk,unique,st,128)
    ready=torch.cuda.Event();ready.record()
    creation=torch.cuda.current_stream(qr.device)
    return dict(order=order,qr=sq,ty=st,merged=merged,qk=qk,unique=unique,loop=loop,invalid=invalid,equal=equal,ready=ready,streams={creation.cuda_stream:creation})


def attention(q,k,v,kr,meta,softmax_scale=None):
    f=importlib.import_module('magi_attention.functional.flex_flash_attn')
    stream=torch.cuda.current_stream(q.device)
    if stream.cuda_stream not in meta['streams']:
        stream.wait_event(meta['ready'])
        for tensor in meta.values():
            if isinstance(tensor,torch.Tensor):tensor.record_stream(stream)
        meta['streams'][stream.cuda_stream]=stream
    sorted_k=kr.index_select(0,meta['order'])
    y,aux=f._flex_flash_attn_forward(q=q,k=k,v=v,sink=None,sink_layout='sh',out=None,lse=None,
        q_ranges=meta['qr'],k_ranges=sorted_k,attn_type_map=meta['ty'],
        softmax_scale=q.shape[-1]**-.5 if softmax_scale is None else softmax_scale,
        softcap=0.,out_type=torch.float32,disable_fwd_atomic_reduction=False,deterministic=False,sm_margin=0,
        ref_block_size=(64,128),max_seqlen_q=64,auto_range_merge=True,merge_q_ranges=meta['merged'],
        fwd_qk_map=meta['qk'],fwd_unique_count=meta['unique'],sparse_load=True,
        sparse_load_loop_count=meta['loop'],sparse_load_invalid_count=meta['invalid'],equal_k_range_size=meta['equal'])
    return y.to(q.dtype),aux.lse
