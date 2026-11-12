"""CuTe inference kernels for PixelUMM QK norm/RoPE and residual RMSNorm.

QK: BF16 contiguous flattened [B,N,H,128]; FP32 cosine/sine [N,128];
BF16 norm weights [64]. FP32 two independent RMS reductions over 64 dimensions,
then learned scale, section-local rotate-half (64,32,32), final BF16 cast.
K path also repeats K/V heads into token-major storage. No intermediate tensors.
Hidden RMS: BF16 [rows,D] + optional BF16 residual; rounded BF16 sum, FP32 RMS,
BF16 normalization rounding followed by BF16 learned-scale multiplication.
Inference only; no backward, no atomics/workspace except small CTA RMS sums.
QK requires head_dim128; row/token/head counts are not whitelisted.
"""
import cutlass
import cutlass.cute as cute
from cutlass import Float32, BFloat16
from cuda.bindings import driver as cuda
import cutlass.utils as utils


class QKNormRoPE:
    def __init__(self, heads, repeat=1, warps=4, with_v=False, eps=1e-6):
        self.heads=heads;self.repeat=repeat;self.warps=warps;self.with_v=with_v;self.eps=eps

    @cute.jit
    def __call__(self,x:cute.Tensor,v:cute.Tensor,wt:cute.Tensor,wh:cute.Tensor,
                 co:cute.Tensor,si:cute.Tensor,y:cute.Tensor,vo:cute.Tensor,
                 tokens:cutlass.Int32,stream:cuda.CUstream):
        self.kernel(x,v,wt,wh,co,si,y,vo,tokens).launch(
            grid=(cute.ceil_div(x.shape[0]//128,self.warps),1,1),block=(32*self.warps,1,1),stream=stream)

    @cute.kernel
    def kernel(self,x:cute.Tensor,v:cute.Tensor,wt:cute.Tensor,wh:cute.Tensor,
               co:cute.Tensor,si:cute.Tensor,y:cute.Tensor,vo:cute.Tensor,tokens:cutlass.Int32):
        tid,_,_=cute.arch.thread_idx();bid,_,_=cute.arch.block_idx()
        row=bid*self.warps+tid//32;lane=tid%32
        if row<x.shape[0]//128:
            a0=x[row*128+lane].to(Float32);a1=x[row*128+lane+32].to(Float32)
            a2=x[row*128+lane+64].to(Float32);a3=x[row*128+lane+96].to(Float32)
            st=a0*a0+a1*a1;sh=a2*a2+a3*a3
            for i in cutlass.range_constexpr(5):
                st+=cute.arch.shuffle_sync_bfly(st,1<<i)
                sh+=cute.arch.shuffle_sync_bfly(sh,1<<i)
            rt=cute.math.rsqrt(st*(1.0/64)+self.eps,fastmath=True)
            rh=cute.math.rsqrt(sh*(1.0/64)+self.eps,fastmath=True)
            z0=(a0*rt)*wt[lane].to(Float32);z1=(a1*rt)*wt[lane+32].to(Float32)
            z2=(a2*rh)*wh[lane].to(Float32);z3=(a3*rh)*wh[lane+32].to(Float32)
            r2=cute.arch.shuffle_sync_bfly(z2,16);r3=cute.arch.shuffle_sync_bfly(z3,16)
            if lane<16:r2=-r2;r3=-r3
            token=(row//self.heads)%tokens
            o0=z0*co[token*128+lane]-z1*si[token*128+lane]
            o1=z1*co[token*128+lane+32]+z0*si[token*128+lane+32]
            o2=z2*co[token*128+lane+64]+r2*si[token*128+lane+64]
            o3=z3*co[token*128+lane+96]+r3*si[token*128+lane+96]
            for rep in cutlass.range_constexpr(self.repeat):
                dest=((row//self.heads)*(self.heads*self.repeat)+(row%self.heads)*self.repeat+rep)*128+lane
                y[dest]=o0.to(BFloat16);y[dest+32]=o1.to(BFloat16)
                y[dest+64]=o2.to(BFloat16);y[dest+96]=o3.to(BFloat16)
                if cutlass.const_expr(self.with_v):
                    for j in cutlass.range_constexpr(4):vo[dest+j*32]=v[row*128+lane+j*32]


class ResidualRMSNorm:
    def __init__(self,dim,threads=128,residual=True,eps=1e-6):
        self.dim=dim;self.threads=threads;self.residual=residual;self.eps=eps
    @cute.jit
    def __call__(self,x:cute.Tensor,r:cute.Tensor,w:cute.Tensor,y:cute.Tensor,s:cute.Tensor,stream:cuda.CUstream):
        self.kernel(x,r,w,y,s).launch(grid=(x.shape[0]//self.dim,1,1),block=(self.threads,1,1),stream=stream)
    @cute.kernel
    def kernel(self,x:cute.Tensor,r:cute.Tensor,w:cute.Tensor,y:cute.Tensor,s:cute.Tensor):
        tid,_,_=cute.arch.thread_idx();row,_,_=cute.arch.block_idx()
        vals=cute.make_rmem_tensor((self.dim//self.threads,),Float32)
        sums=utils.SmemAllocator().allocate_tensor(Float32,cute.make_layout((self.threads//32,)))
        total=Float32(0)
        for j in cutlass.range_constexpr(self.dim//self.threads):
            idx=row*self.dim+tid+j*self.threads
            val=x[idx].to(Float32)
            if cutlass.const_expr(self.residual):
                val=(val+r[idx].to(Float32)).to(BFloat16).to(Float32)
                s[idx]=val.to(BFloat16)
            vals[j]=val;total+=val*val
        for i in cutlass.range_constexpr(5):total+=cute.arch.shuffle_sync_bfly(total,1<<i)
        if tid%32==0:sums[tid//32]=total
        cute.arch.sync_threads()
        total=Float32(0)
        for j in cutlass.range_constexpr(self.threads//32):total+=sums[j]
        inv=cute.math.rsqrt(total/self.dim+self.eps,fastmath=True)
        for j in cutlass.range_constexpr(self.dim//self.threads):
            col=tid+j*self.threads
            z=(vals[j]*inv).to(BFloat16).to(Float32)*w[col].to(Float32)
            y[row*self.dim+col]=z.to(BFloat16)
