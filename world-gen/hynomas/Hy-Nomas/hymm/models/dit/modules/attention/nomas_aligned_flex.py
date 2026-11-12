"""Optional aligned 64-token Flex layout; padding is excluded from softmax.

Each selected logical video block maps to exactly one physical KV block.
Full valid video/video tiles bypass mask_mod; edge and text tiles retain the
exact per-token predicate. Dynamic selections are rebuilt on every call.
"""
from functools import lru_cache
import inspect
import torch

@lru_cache(maxsize=16)
@torch.inference_mode(False)
def _layout(lengths, ntxt, device):
    lengths=tuple(lengths)
    if not lengths or min(lengths)<1 or max(lengths)>64:
        raise ValueError('Aligned Flex requires video blocks of 1..64 tokens')
    lens=torch.tensor(lengths,device=device,dtype=torch.long)
    starts=lens.cumsum(0)-lens
    offsets=torch.arange(64,device=device)
    gather=(starts[:,None]+offsets).clamp(max=sum(lengths)-1).flatten()
    valid=(offsets[None,:]<lens[:,None]).flatten()
    restore=valid.nonzero().flatten()
    if ntxt:
        tail=torch.arange(ntxt,device=device)+sum(lengths)
        gather=torch.cat((gather,tail))
        restore=torch.cat((restore,torch.arange(ntxt,device=device)+len(lengths)*64))
        valid=torch.cat((valid,torch.ones(ntxt,device=device,dtype=torch.bool)))
    ready=torch.cuda.Event();ready.record()
    return gather,restore,valid,ready

def layout(lengths,ntxt,device):
    gather,restore,valid,ready=_layout(tuple(lengths),ntxt,str(device))
    stream=torch.cuda.current_stream(device);stream.wait_event(ready)
    for tensor in (gather,restore,valid):tensor.record_stream(stream)
    return gather,restore,valid

@lru_cache(maxsize=1)
def compiled_attention():
    from torch.nn.attention.flex_attention import flex_attention
    return torch.compile(flex_attention,dynamic=True,fullgraph=True)

@lru_cache(maxsize=1)
def _can_skip_q_blocks():
    from torch.nn.attention.flex_attention import BlockMask
    return 'compute_q_blocks' in inspect.signature(BlockMask.from_kv_blocks).parameters

def build_mask(selected,nq,nv,nhisttxt,ntxt,offsets,windows,q_lengths,k_lengths,qvalid,kvalid):
    from torch.nn.attention.flex_attention import BlockMask
    h,lq,count=selected.shape;lk=len(k_lengths);device=selected.device
    if len(q_lengths)!=lq or sum(q_lengths)!=nq or sum(k_lengths)!=nv:
        raise ValueError('Invalid aligned video lengths')
    offs=[ntxt] if offsets is None else [max(0,min(int(o),ntxt)) for o in offsets]
    if not offs or offs!=sorted(offs) or offs[-1]!=ntxt:raise ValueError('Invalid text offsets')
    ng=len(offs)
    # Label 0=common text, 1..=action. Video queries carry their action id+1.
    qa=torch.full((lq,),-1,device=device,dtype=torch.long)
    bounds=[0]
    for length in q_lengths:bounds.append(bounds[-1]+length)
    if offsets is not None:
        last=0
        for left,right,action in windows:
            if left!=last or right<left or left not in bounds or right not in bounds or action < -1 or action>=ng-1:
                raise ValueError('Action windows must partition logical blocks')
            qa[bounds.index(left):bounds.index(right)]=action+1 if action>=0 else -1
            last=right
        if last!=nq:raise ValueError('Action windows do not cover queries')
    tl=[offs[0]]+[b-a for a,b in zip(offs,offs[1:])]
    text_group=torch.repeat_interleave(torch.arange(ng,device=device),torch.tensor(tl,device=device))
    # Zero-length lookup guards allow ntxt==0.
    text_group=torch.cat((text_group,torch.zeros(1,device=device,dtype=torch.long)))
    allowed=torch.zeros((h,lq,lk),device=device,dtype=torch.bool)
    allowed.scatter_(2,selected,True)
    qsize=lq*64+ntxt;ksize=lk*64+nhisttxt+ntxt
    def mask_mod(b,head,qidx,kidx):
        qi=qidx.clamp(max=qsize-1);ki=kidx.clamp(max=ksize-1)
        qvideo=qi<lq*64;kvideo=ki<lk*64
        qb=(qi//64).clamp(max=lq-1);kb=(ki//64).clamp(max=lk-1)
        qaction=torch.where(qvideo,qa[qb],text_group[(qi-lq*64).clamp(min=0,max=ntxt)])
        khistory=(ki>=lk*64)&(ki<lk*64+nhisttxt)
        kaction=text_group[(ki-lk*64-nhisttxt).clamp(min=0,max=ntxt)]
        text_ok=khistory|(kaction==0)|((qaction>0)&(kaction==qaction))
        visible=torch.where(kvideo,torch.where(qvideo,allowed[head,qb,kb],True),text_ok)
        return (qidx<qsize)&(kidx<ksize)&qvalid[qi]&kvalid[ki]&visible
    pq=(qsize+63)//64;pk=(ksize+63)//64
    coarse=torch.zeros((h,pq,pk),device=device,dtype=torch.bool)
    coarse[:,:lq,:lk]=allowed
    if ntxt:coarse[:,lq:,:]=True
    if nhisttxt+ntxt:coarse[:,:lq,lk:]=True
    full=torch.zeros_like(coarse)
    qfull=torch.tensor(q_lengths,device=device)==64
    kfull=torch.tensor(k_lengths,device=device)==64
    full[:,:lq,:lk]=allowed&qfull[None,:,None]&kfull[None,None,:]
    partial=coarse&~full
    def arrays(x):
        counts=x.sum(-1,dtype=torch.int32)[None]
        # Row-independent stable sort: limit temporary int64 indices/workspace.
        flat=x.reshape(-1,x.shape[-1])
        indices=torch.empty(flat.shape,device=x.device,dtype=torch.int32)
        rows=max(1,(64*2**20)//(flat.shape[-1]*32))
        for first in range(0,flat.shape[0],rows):
            chunk=flat[first:first+rows].to(torch.int8)
            ordered=chunk.argsort(dim=-1,descending=True,stable=True)
            indices[first:first+rows].copy_(ordered)
            del chunk,ordered
        return counts,indices.reshape(x.shape)[None]
    pn,pi=arrays(partial);fn,fi=arrays(full)
    inference_options={}
    if not torch.is_grad_enabled() and _can_skip_q_blocks():
        inference_options['compute_q_blocks']=False
    mask=BlockMask.from_kv_blocks(pn,pi,full_kv_num_blocks=fn,full_kv_indices=fi,
                                 BLOCK_SIZE=(64,64),mask_mod=mask_mod,seq_lengths=(qsize,ksize),**inference_options)
    return mask,mask_mod,coarse,full

def choose_head_group(heads,maximum,q_lengths,k_lengths,ntxt,nhisttxt,device,head_dim=128):
    """Conservative workspace estimate; bound by live allocator and device headroom."""
    maximum=min(heads,maximum)
    qsize=len(q_lengths)*64+ntxt
    ksize=len(k_lengths)*64+nhisttxt+ntxt
    pq=(qsize+63)//64;pk=(ksize+63)//64
    # Padded Q/K/V, attention/output restoration and mask/index intermediates.
    per_head=(4*qsize+2*ksize)*head_dim*2+pq*pk*32+qsize*4
    free,total=torch.cuda.mem_get_info(device)
    allocated=torch.cuda.memory_allocated(device)
    cached=max(0,torch.cuda.memory_reserved(device)-allocated)
    fraction=torch.cuda.get_per_process_memory_fraction(device)
    # Respect live device headroom and the configured allocator limit on any GPU.
    available=min(free+cached,int(total*fraction)-allocated)
    available=max(0,available-192*2**20)
    selected=1
    while selected*2<=maximum and per_head*(selected*2)<=available:
        selected*=2
    return selected

def composed_gather_plan(q_lengths,k_lengths,ntxt,nhisttxt,qorder,korder,device):
    """Compose raster->logical and logical->padded maps once per layer call."""
    qg,restore,qvalid=layout(q_lengths,ntxt,device)
    kg,_,kvalid=layout(k_lengths,nhisttxt+ntxt,device)
    return (qorder.index_select(0,qg), korder.index_select(0,kg),
            restore,qvalid,kvalid)

def _attention(q,k,v,selected,nq,nv,nhisttxt,ntxt,offsets,windows,ranges_fn,stats,*,q_lengths,k_lengths,gather_plan=None):
    if torch.is_grad_enabled():raise RuntimeError('Aligned Flex is inference-only')
    if not q.is_cuda or q.dtype!=torch.bfloat16 or k.dtype!=q.dtype or v.dtype!=q.dtype:
        raise ValueError('Aligned Flex expects CUDA BF16')
    if gather_plan is None:
        if q.shape!=(nq+ntxt,k.shape[1],128) or k.shape!=v.shape or k.shape[0]!=nv+nhisttxt+ntxt:
            raise ValueError('Invalid aligned Flex QKV shape')
        qg,restore,qvalid=layout(q_lengths,ntxt,q.device)
        kg,_,kvalid=layout(k_lengths,nhisttxt+ntxt,k.device)
    else:
        if q.ndim!=3 or k.ndim!=3 or q.shape[1:]!=k.shape[1:] or k.shape!=v.shape or q.shape[-1]!=128:
            raise ValueError('Invalid raster Flex QKV shape')
        qg,kg,restore,qvalid,kvalid=gather_plan
    mask,_,coarse,full=build_mask(selected,nq,nv,nhisttxt,ntxt,offsets,windows,q_lengths,k_lengths,qvalid,kvalid)
    physical_q_blocks,physical_kv_blocks=coarse.shape[-2:]
    del coarse,full
    qq=q.index_select(0,qg);kk=k.index_select(0,kg);vv=v.index_select(0,kg)
    qq,kk,vv=(x.transpose(0,1)[None] for x in (qq,kk,vv))
    output=compiled_attention()(qq,kk,vv,block_mask=mask,scale=q.shape[-1]**-.5,
        kernel_options={'ROWS_GUARANTEED_SAFE':False,'BLOCKS_ARE_CONTIGUOUS':False,'BLOCK_M':64,'BLOCK_N':64,'FORCE_USE_FLEX_ATTENTION':True})
    result=output.squeeze(0).transpose(0,1).index_select(0,restore)
    stats['sparse_flex_calls']=stats.get('sparse_flex_calls',0)+1
    stats['text_flex_calls']=stats.get('text_flex_calls',0)+int(ntxt>0)
    stats['aligned_flex_calls']=stats.get('aligned_flex_calls',0)+1
    stats['physical_q_blocks']=physical_q_blocks;stats['physical_kv_blocks']=physical_kv_blocks
    return result


def attention(q,k,v,selected,nq,nv,nhisttxt,ntxt,offsets,windows,ranges_fn,stats,
              *,q_lengths,k_lengths,head_chunk=0,gather_plan=None):
    """Bound mask/padding storage by processing independent attention heads."""
    if type(head_chunk) is not int or head_chunk < 0:
        raise ValueError('head_chunk must be a nonnegative integer')
    heads=q.shape[1]
    if not head_chunk or head_chunk >= heads:
        return _attention(q,k,v,selected,nq,nv,nhisttxt,ntxt,offsets,windows,
                          ranges_fn,stats,q_lengths=q_lengths,k_lengths=k_lengths,gather_plan=gather_plan)
    result=torch.empty_like(q)
    for first in range(0,heads,head_chunk):
        last=min(first+head_chunk,heads)
        part=_attention(q[:,first:last],k[:,first:last],v[:,first:last],
                        selected[first:last],nq,nv,nhisttxt,ntxt,offsets,windows,
                        ranges_fn,stats,q_lengths=q_lengths,k_lengths=k_lengths,gather_plan=gather_plan)
        result[:,first:last].copy_(part)
        del part
    stats['head_chunk_calls']=stats.get('head_chunk_calls',0)+1
    return result
