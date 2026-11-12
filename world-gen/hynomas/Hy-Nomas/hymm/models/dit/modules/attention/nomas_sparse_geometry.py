"""Bounded static geometry cache, including compact edge blocks (no padded KV)."""
from functools import lru_cache
import torch


def lengths(nq, nv, q_lengths, k_lengths):
    ql = tuple(q_lengths) if q_lengths is not None else (64,)*(nq//64)
    kl = tuple(k_lengths) if k_lengths is not None else (64,)*(nv//64)
    if sum(ql)!=nq or sum(kl)!=nv or any(n<1 or n>64 for n in ql+kl):
        raise ValueError('Invalid compact Sparse block lengths')
    return ql,kl


@lru_cache(maxsize=16)
@torch.inference_mode(False)
def _build(nq,nv,nh,nt,offs,wins,device,ql,kl,qchunk,kchunk):
    qb=[0];kb=[0]
    for n in ql:qb.append(qb[-1]+n)
    for n in kl:kb.append(kb[-1]+n)
    video=[];text=[]
    def emit(a,b,c,d):
        if b<=a or d<=c:return
        if a<nq:
            end=min(b,nq)
            video.extend((max(a,l),min(end,r),c,d) for l,r in zip(qb,qb[1:]) if r>a and l<end)
            a=end
        if b>a:text.append((a-nq,b-nq,c,d))
    emit(nq,nq+nt,0,nv+nh);emit(0,nq,nv,nv+nh)
    start=nv+nh
    if offs is None:emit(0,nq+nt,start,start+nt)
    else:
        offs=tuple(max(0,min(int(o),nt)) for o in offs)
        if not offs or list(offs)!=sorted(offs) or offs[-1]!=nt:raise ValueError('Invalid text offsets')
        emit(0,nq+nt,start,start+offs[0])
        for a,b,i in wins:
            if i>=0:emit(a,b,start+offs[i],start+offs[i+1])
        for i in range(len(offs)-1):emit(nq+offs[i],nq+offs[i+1],start+offs[i],start+offs[i+1])
    split=[]
    for a,b,c,d in text:
        for qa in range(a,b,qchunk or b-a):
            for ka in range(c,d,kchunk or d-c):
                split.append((qa,min(qa+(qchunk or b-a),b),ka,min(ka+(kchunk or d-c),d)))
    def tensors(rows):
        t=torch.tensor(rows,device=device,dtype=torch.int32).reshape(-1,4)
        return t[:,:2].contiguous(),t[:,2:].contiguous(),torch.zeros(len(rows),device=device,dtype=torch.int32)
    q=torch.tensor(qb,device=device,dtype=torch.int32)
    k=torch.tensor(kb,device=device,dtype=torch.int32)
    v,t=tensors(video),tensors(split)
    ready=None
    if torch.device(device).type=='cuda':
        ready=torch.cuda.Event();ready.record(torch.cuda.current_stream(device))
    return q,k,v,t,ready


def geometry(nq,nv,nh,nt,offsets,windows,device,q_lengths=None,k_lengths=None,qchunk=0,kchunk=0):
    ql,kl=lengths(nq,nv,q_lengths,k_lengths)
    geo=_build(nq,nv,nh,nt,None if offsets is None else tuple(offsets),
               tuple(tuple(w) for w in (windows or [])),str(device),ql,kl,qchunk,kchunk)
    if geo[4] is not None:
        stream=torch.cuda.current_stream(device);stream.wait_event(geo[4])
        for tensor in (geo[0],geo[1],*geo[2],*geo[3]):tensor.record_stream(stream)
    return geo


def video_ranges(selected, geo):
    qb,kb,extras,_,_=geo
    n,count=selected.shape
    if n!=len(qb)-1:raise ValueError('Selected queries do not match geometry')
    qr=torch.stack((qb[:-1],qb[1:]),-1)[:,None,:].expand(n,count,2).reshape(-1,2)
    kr=torch.stack((kb[selected],kb[selected+1]),-1).reshape(-1,2)
    qr=torch.cat((qr,extras[0]));kr=torch.cat((kr,extras[1]))
    return qr,kr,torch.zeros(qr.shape[0],device=selected.device,dtype=torch.int32)
