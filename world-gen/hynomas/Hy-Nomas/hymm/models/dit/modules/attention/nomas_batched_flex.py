"""Head-aware sparse Flex attention with exact short-tile/text visibility.

No token QxK mask or CPU routing copy. Coarse physical adjacency is conservative;
mask_mod applies the exact logical adjacency to every token. All visible video
and text keys share one softmax. Dynamic masks are never cached across calls.
"""
from functools import lru_cache
import torch


@lru_cache(maxsize=1)
def compiled_attention():
    from torch.nn.attention.flex_attention import flex_attention
    return torch.compile(flex_attention, dynamic=True, fullgraph=True)


def build_mask(selected, nq, nv, nhisttxt, ntxt, offsets, windows,
               q_lengths, k_lengths):
    from torch.nn.attention.flex_attention import BlockMask
    h,lq,count=selected.shape; lk=len(k_lengths); device=selected.device
    if len(q_lengths)!=lq or sum(q_lengths)!=nq or sum(k_lengths)!=nv:
        raise ValueError('Logical tiles must cover Q/K video exactly')
    if min((*q_lengths,*k_lengths)) <= 0 or max((*q_lengths,*k_lengths)) > 64:
        raise ValueError('Expected real 8x8 tiles of 1..64 tokens')
    offs=[ntxt] if offsets is None else [max(0,min(int(o),ntxt)) for o in offsets]
    if not offs or offs!=sorted(offs) or offs[-1]!=ntxt:
        raise ValueError('Action offsets must cover text')
    ng=len(offs)
    # Logical video tiles + historical text + current text groups.
    allowed=torch.zeros((h,lq+ng,lk+1+ng),device=device,dtype=torch.bool)
    allowed[:,:lq,:lk].scatter_(2,selected,True)
    allowed[:,:lq,lk]=True
    allowed[:,lq:,:lk+1]=True
    allowed[:,:,lk+1]=True  # common text (or all text with no actions)
    qmap=torch.repeat_interleave(torch.arange(lq,device=device),torch.tensor(q_lengths,device=device))
    kmap=torch.repeat_interleave(torch.arange(lk,device=device),torch.tensor(k_lengths,device=device))
    # Windows are frame-aligned and cannot cut an 8x8 spatial tile.
    bounds=[0]
    for size in q_lengths: bounds.append(bounds[-1]+size)
    if offsets is not None:
        last=0
        for left,right,action in windows:
            if left!=last or right<left or left not in bounds or right not in bounds or action < -1 or action>=ng-1:
                raise ValueError('Action windows must partition video along logical tile boundaries')
            if action>=0:
                allowed[:,bounds.index(left):bounds.index(right),lk+2+action]=True
            last=right
        if last!=nq: raise ValueError('Action windows must cover video')
        for action in range(ng-1):
            allowed[:,lq+1+action,lk+2+action]=True
    text_lengths=[offs[0]]+[b-a for a,b in zip(offs,offs[1:])]
    qt=torch.repeat_interleave(torch.arange(ng,device=device)+lq,torch.tensor(text_lengths,device=device))
    kt=torch.repeat_interleave(torch.arange(ng,device=device)+lk+1,torch.tensor(text_lengths,device=device))
    qmap=torch.cat((qmap,qt)); kmap=torch.cat((kmap,torch.full((nhisttxt,),lk,device=device),kt))
    qsize=nq+ntxt; ksize=nv+nhisttxt+ntxt
    def mask_mod(b,head,qidx,kidx):
        # Flex evaluates padded tiles too: clamp indices before the final guard.
        return (qidx<qsize)&(kidx<ksize)&allowed[head,qmap[qidx.clamp(max=qsize-1)],kmap[kidx.clamp(max=ksize-1)]]
    # Logical blocks <=64 overlap at most two physical Q64/K128 tiles.
    qb,kb=64,128
    pq=(qsize+qb-1)//qb; pk=(ksize+kb-1)//kb
    coarse=torch.zeros((h,pq,pk),device=device,dtype=torch.bool)
    qe=torch.tensor(q_lengths,device=device).cumsum(0); qs=qe-torch.tensor(q_lengths,device=device)
    ke=torch.tensor(k_lengths,device=device).cumsum(0); ks=ke-torch.tensor(k_lengths,device=device)
    hi=torch.arange(h,device=device)[:,None,None]
    for qphys in (qs//qb,(qe-1)//qb):
        for kphys in (ks//kb,(ke-1)//kb):
            coarse[hi,qphys[None,:,None],kphys[selected]]=True
    if ntxt:
        coarse[:,nq//qb:,:]=True
    if nhisttxt+ntxt:
        coarse[:,:,nv//kb:]=True
    counts=coarse.sum(-1,dtype=torch.int32).unsqueeze(0)
    # Stable sort puts occupied blocks first, in ascending KV order.
    indices=coarse.to(torch.int8).argsort(dim=-1,descending=True,stable=True).to(torch.int32).unsqueeze(0)
    mask=BlockMask.from_kv_blocks(counts,indices,BLOCK_SIZE=(qb,kb),mask_mod=mask_mod,seq_lengths=(qsize,ksize))
    return mask,mask_mod,coarse


def attention(q,k,v,selected,nq,nv,nhisttxt,ntxt,offsets,windows,ranges_fn,stats,
              *,q_lengths,k_lengths):
    if torch.is_grad_enabled(): raise RuntimeError('Sparse Flex is inference-only')
    if q.dtype!=torch.bfloat16 or k.dtype!=q.dtype or v.dtype!=q.dtype or not q.is_cuda:
        raise ValueError('Sparse Flex expects CUDA BF16 Q/K/V')
    if q.shape!=(nq+ntxt,k.shape[1],128) or k.shape!=v.shape or k.shape[0]!=nv+nhisttxt+ntxt:
        raise ValueError('Sparse Flex Q/K/V geometry mismatch')
    mask,_,coarse=build_mask(selected,nq,nv,nhisttxt,ntxt,offsets,windows,q_lengths,k_lengths)
    qq,kk,vv=(x.transpose(0,1).unsqueeze(0) for x in (q,k,v))
    out=compiled_attention()(qq,kk,vv,block_mask=mask,scale=q.shape[-1]**-.5,
        kernel_options={'ROWS_GUARANTEED_SAFE':False,'BLOCKS_ARE_CONTIGUOUS':False,'BLOCK_M':64,'BLOCK_N':64,'FORCE_USE_FLEX_ATTENTION':True})
    stats['sparse_flex_calls']=stats.get('sparse_flex_calls',0)+1
    stats['text_flex_calls']=stats.get('text_flex_calls',0)+int(ntxt>0)
    stats['physical_blocks']=coarse.numel()
    # Avoid a GPU synchronization per layer just to format diagnostics.
    stats['physical_q_blocks']=coarse.shape[-2]; stats['physical_kv_blocks']=coarse.shape[-1]
    return out.squeeze(0).transpose(0,1).contiguous()
