"""Multi-head protected Magi, optional grouped video-head packing.

H3 28c7435 batches protected FA3 queries. Nomas must retain action-specific
visibility, so protected queries use ONE multi-head Magi call with shared ranges.
Video heads have different routes: encode head identity by disjoint sequence
intervals. Every query still has one softmax over its complete allowed KV set.
"""
import torch
from hymm.models.dit.modules.attention.nomas_inference_optim import mode, compare, note, option
from hymm.models.dit.modules.attention.nomas_sparse_geometry import geometry, video_ranges, lengths


def attention(q,k,v,selected,nq,nv,nhisttxt,ntxt,offsets,windows,ranges_fn,stats, *, q_lengths=None,k_lengths=None):
    from magi_attention.functional import flex_flash_attn_func
    if torch.is_grad_enabled():
        raise RuntimeError('Batched Nomas Magi is inference-only')
    heads=q.shape[1]; nk=k.shape[0]
    deterministic=torch.are_deterministic_algorithms_enabled()
    sparse_load=not deterministic
    out=torch.empty_like(q)
    text_ranges=None
    range_mode=mode('RANGES');text_mode=mode('TEXT')
    qchunk=64 if text_mode!='off' else 0
    kchunk=option('text_kchunk',0) if text_mode!='off' else 0
    if kchunk<0:raise ValueError('NOMAS_OPT_TEXT_KCHUNK must be nonnegative')
    geo=geometry(nq,nv,nhisttxt,ntxt,offsets,windows,q.device,q_lengths,k_lengths,qchunk,kchunk)
    cached_video,cached_text=geo[2:4]
    ql,kl=lengths(nq,nv,q_lengths,k_lengths)
    from hymm.models.dit.modules.attention import nomas_magi_static
    static_mode=mode('MAGI_STATIC')
    stride_mode=mode('HEAD_VIEW')
    if static_mode!='off' and (any(n!=64 for n in ql+kl) or not nomas_magi_static.supported(q,k,v,selected[0],nq,deterministic)):
        note("magi_static.fallback_geometry_or_backend")
        stats['magi_static_fallbacks']=stats.get('magi_static_fallbacks',0)+1
        static_mode='off'
    geometry_key=(nq,nv,nhisttxt,ntxt,None if offsets is None else tuple(offsets),tuple(tuple(w) for w in (windows or [])))
    for hs in range(heads):
        he=hs+1; qr_all=[];kr_all=[];ty_all=[]
        if static_mode!='on':
            if range_mode!='on':
                qr,kr,ty=ranges_fn(selected[hs],nq,nv,nhisttxt,ntxt,offsets,windows,q.device)
                video=qr[:,0]<nq
                if text_ranges is None and ntxt:
                    text_ranges=(qr[~video]-nq,kr[~video],ty[~video])
                old_ranges=(qr[video],kr[video],ty[video])
            if range_mode!='off':
                note('ranges.hit')
                new_ranges=video_ranges(selected[hs],geo)
                if range_mode=='shadow':
                    for old,new in zip(old_ranges,new_ranges):
                        if not torch.equal(old,new):raise RuntimeError('Video ranges changed')
                    print('NOMAS_OPT_RANGES_EQUAL',flush=True)
                else:old_ranges=new_ranges
            qr_all.append(old_ranges[0]);kr_all.append(old_ranges[1]);ty_all.append(old_ranges[2])
        # flatten head-major sequences, each original head becomes a separate
        # interval in the one-head Magi invocation; ranges forbid cross-head KV.
        pack=lambda x:x[:,hs:he].transpose(0,1).contiguous().reshape(-1,1,x.shape[-1])
        # FFA accepts row strides; keep the head slice instead of copying Q/K/V.
        view_supported=all(x.stride(-1)==1 and x.stride(0)>0 for x in (q,k,v))
        if stride_mode!='off':note('head_view.hit' if view_supported else 'head_view.fallback')
        views=dict(q=q[:nq,hs:he],k=k[:,hs:he],v=v[:,hs:he])
        operands=views if stride_mode=='on' and view_supported else dict(q=pack(q[:nq]),k=pack(k),v=pack(v))
        if static_mode!='on':
            result,_=flex_flash_attn_func(**operands,
                q_ranges=qr_all[0],k_ranges=kr_all[0],attn_type_map=ty_all[0],
                softmax_scale=q.shape[-1]**-.5,deterministic=deterministic,
                auto_range_merge=not deterministic,sparse_load=sparse_load,max_seqlen_q=64,
                ref_block_size=(64,128),
                disable_fwd_atomic_reduction=False)
        if static_mode!='off':
            optimized,_=nomas_magi_static.attention(**operands,selected=selected[hs],extras=cached_video,geometry_key=geometry_key,scale=q.shape[-1]**-.5)
            if static_mode=='shadow':compare('magi_static',result,optimized)
            else:result=optimized
            note('magi_static.hit')
            stats['magi_static_hits']=stats.get('magi_static_hits',0)+1
        if stride_mode=='shadow' and view_supported:
            if static_mode=='on':
                view_result,_=nomas_magi_static.attention(**views,selected=selected[hs],extras=cached_video,geometry_key=geometry_key,scale=q.shape[-1]**-.5)
            else:
                view_result,_=flex_flash_attn_func(**views,
                    q_ranges=qr_all[0],k_ranges=kr_all[0],attn_type_map=ty_all[0],
                    softmax_scale=q.shape[-1]**-.5,deterministic=deterministic,
                    auto_range_merge=not deterministic,sparse_load=sparse_load,max_seqlen_q=64,
                    ref_block_size=(64,128),disable_fwd_atomic_reduction=False)
            compare('head_view',result,view_result)
        out[:nq,hs:he]=result.reshape(he-hs,nq,q.shape[-1]).transpose(0,1)
        stats['video_calls']=stats.get('video_calls',0)+1
    if ntxt:
        if text_ranges is None:
            text_ranges=geometry(nq,nv,nhisttxt,ntxt,offsets,windows,q.device,q_lengths,k_lengths)[3]
        qr,kr,ty=text_ranges
        operands=dict(q=q[nq:].contiguous(),k=k.contiguous(),v=v.contiguous(),softmax_scale=q.shape[-1]**-.5,deterministic=deterministic)
        if text_mode!='on':
            result,_=flex_flash_attn_func(**operands,q_ranges=qr,k_ranges=kr,attn_type_map=ty)
        if text_mode!='off':
            note('text.hit')
            sq,sk,st=cached_text
            candidate,_=flex_flash_attn_func(**operands,q_ranges=sq,k_ranges=sk,attn_type_map=st,
                max_seqlen_q=qchunk,ref_block_size=(qchunk,128),auto_range_merge=False,
                disable_fwd_atomic_reduction=False)
            if text_mode=='shadow':compare('text',result,candidate)
            else:result=candidate
        out[nq:]=result
        stats['text_calls']=stats.get('text_calls',0)+1
    stats['batched_calls']=stats.get('batched_calls',0)+1
    return out
