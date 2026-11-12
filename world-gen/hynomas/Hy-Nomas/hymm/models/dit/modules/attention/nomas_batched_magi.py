"""Multi-head protected Magi, optional grouped video-head packing.

H3 28c7435 batches protected FA3 queries. Nomas must retain action-specific
visibility, so protected queries use ONE multi-head Magi call with shared ranges.
Video heads have different routes: encode head identity by disjoint sequence
intervals. Every query still has one softmax over its complete allowed KV set.
"""
import torch


def _reference_attention(q,k,v,selected,nq,nv,nhisttxt,ntxt,offsets,windows,ranges_fn,stats):
    from magi_attention.functional import flex_flash_attn_func
    if torch.is_grad_enabled():
        raise RuntimeError('Batched Nomas Magi is inference-only')
    heads=q.shape[1]; nk=k.shape[0]
    deterministic=torch.are_deterministic_algorithms_enabled()
    sparse_load=not deterministic
    out=torch.empty_like(q)
    text_ranges=None
    for hs in range(heads):
        he=hs+1; qr_all=[];kr_all=[];ty_all=[]
        for head in range(hs,he):
            qr,kr,ty=ranges_fn(selected[head],nq,nv,nhisttxt,ntxt,offsets,windows,q.device)
            video=qr[:,0]<nq
            if text_ranges is None and ntxt:
                text_ranges=(qr[~video]-nq,kr[~video],ty[~video])
            qr_all.append(qr[video]+(head-hs)*nq)
            kr_all.append(kr[video]+(head-hs)*nk)
            ty_all.append(ty[video])
        # flatten head-major sequences, each original head becomes a separate
        # interval in the one-head Magi invocation; ranges forbid cross-head KV.
        pack=lambda x:x[:,hs:he].transpose(0,1).contiguous().reshape(-1,1,x.shape[-1])
        result,_=flex_flash_attn_func(q=pack(q[:nq]),k=pack(k),v=pack(v),
            q_ranges=torch.cat(qr_all),k_ranges=torch.cat(kr_all),attn_type_map=torch.cat(ty_all),
            softmax_scale=q.shape[-1]**-.5,deterministic=deterministic,
            auto_range_merge=not deterministic,sparse_load=sparse_load,max_seqlen_q=64,
            ref_block_size=(64,128),
            disable_fwd_atomic_reduction=False)
        out[:nq,hs:he]=result.reshape(he-hs,nq,q.shape[-1]).transpose(0,1)
        stats['video_calls']=stats.get('video_calls',0)+1
    if ntxt:
        qr,kr,ty=text_ranges
        result,_=flex_flash_attn_func(q=q[nq:].contiguous(),k=k.contiguous(),v=v.contiguous(),
            q_ranges=qr,k_ranges=kr,attn_type_map=ty,softmax_scale=q.shape[-1]**-.5,
            deterministic=deterministic)
        out[nq:]=result
        stats['text_calls']=stats.get('text_calls',0)+1
    stats['batched_calls']=stats.get('batched_calls',0)+1
    return out


def attention(q,k,v,selected,nq,nv,nhisttxt,ntxt,offsets,windows,ranges_fn,stats, *, q_lengths=None,k_lengths=None):
    from hymm.models.dit.modules.attention.nomas_inference_optim import mode
    if not any(mode(name)!='off' for name in ('RANGES','TEXT','MAGI_STATIC','HEAD_VIEW')):
        return _reference_attention(q,k,v,selected,nq,nv,nhisttxt,ntxt,offsets,windows,ranges_fn,stats)
    from hymm.models.dit.modules.attention.nomas_optimized_magi import attention as optimized
    return optimized(q,k,v,selected,nq,nv,nhisttxt,ntxt,offsets,windows,ranges_fn,stats,q_lengths=q_lengths,k_lengths=k_lengths)
