"""Inference-only BF16 dot + FP32 top-four prototype reduction.

FP32 contiguous inputs [H,N,16,128] are rounded to BF16 on load, matching
CuTe's input precision. Outputs [H,Nq,Nk] FP32. No padded prototype score
matrix is stored in HBM. Each program owns BQ*BK block pairs; workspace is
only the output. Standard tl.dot permits native MMA on SM89 and SM90;
there is no explicit TMA/WGMMA or SM90 dependency. GPU tests required per arch.
No backward. Raw score_kernel is independently benchmarkable with preallocated
output; score() is an allocation-owning convenience wrapper.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def score_kernel(Q,K,S,NQ:tl.constexpr,NK:tl.constexpr,BQ:tl.constexpr,BK:tl.constexpr):
    iq=tl.program_id(0)*BQ;ik=tl.program_id(1)*BK;head=tl.program_id(2)
    qm=iq*16+tl.arange(0,BQ*16)
    km=ik*16+tl.arange(0,BK*16)
    d=tl.arange(0,128)
    q=tl.load(Q+(head*NQ*16+qm[:,None])*128+d[None,:],qm[:,None]<NQ*16,other=0).to(tl.bfloat16)
    k=tl.load(K+(head*NK*16+km[None,:])*128+d[:,None],km[None,:]<NK*16,other=0).to(tl.bfloat16)
    dots=tl.dot(q,k,out_dtype=tl.float32)
    pairs=tl.reshape(tl.permute(tl.reshape(dots,(BQ,16,BK,16)),(0,2,1,3)),(BQ*BK,256))
    total=tl.full((BQ*BK,),0,tl.float32)
    pos=tl.arange(0,256)
    for _ in tl.static_range(4):
        val,idx=tl.max(pairs,1,return_indices=True)
        total+=val
        pairs=tl.where(pos[None,:]==idx[:,None],-float('inf'),pairs)
    scores=tl.reshape(total*(0.25*0.08838834764831845),(BQ,BK))
    qr=iq+tl.arange(0,BQ);kr=ik+tl.arange(0,BK)
    tl.store(S+(head*NQ+qr[:,None])*NK+kr[None,:],scores,(qr[:,None]<NQ)&(kr[None,:]<NK))


def score(q,k,*,bq=4,bk=4,warps=4):
    h,nq,_,_=q.shape;nk=k.shape[1]
    output=torch.empty((h,nq,nk),device=q.device,dtype=torch.float32)
    score_kernel[(triton.cdiv(nq,bq),triton.cdiv(nk,bk),h)](q,k,output,nq,nk,bq,bk,num_warps=warps)
    return output
