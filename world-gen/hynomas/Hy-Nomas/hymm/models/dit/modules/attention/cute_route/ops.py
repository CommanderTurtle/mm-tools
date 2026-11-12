"""CuTe BF16 conversion and exact stable group selection (no Triton)."""
import cutlass
import cutlass.cute as cute
from cutlass import Float32,Int32,Int64,BFloat16
import cutlass.utils as utils
from cuda.bindings import driver as cuda

@cute.kernel
def cast_kernel(x:cute.Tensor,y:cute.Tensor,n:Int32):
    t,_,_=cute.arch.thread_idx(); b,_,_=cute.arch.block_idx()
    i=b*256+t
    if i<n: y[i]=BFloat16(x[i])

@cute.jit
def cast(x:cute.Tensor,y:cute.Tensor,n:Int32,stream:cuda.CUstream):
    cast_kernel(x,y,n).launch(grid=(cute.ceil_div(n,256),1,1),block=(256,1,1),stream=stream)

class SelectGroups:
    def __init__(self,groups,budgets):
        self.groups=tuple(groups);self.budgets=tuple(budgets)
        self.width=1<<max(1,max(e-s for s,e in groups)-1).bit_length()
    @cute.jit
    def __call__(self,s:cute.Tensor,o:cute.Tensor,stream:cuda.CUstream):
        self.kernel(s,o).launch(grid=(s.shape[1],len(self.groups),s.shape[0]),block=(self.width,1,1),stream=stream)
    @cute.kernel
    def kernel(self,s:cute.Tensor,o:cute.Tensor):
        t,_,_=cute.arch.thread_idx();q,g,h=cute.arch.block_idx()
        start=Int32(0); length=Int32(0);budget=Int32(0); offset=Int32(0)
        off=0
        for i in cutlass.range_constexpr(len(self.groups)):
            if g==i:
                start=Int32(self.groups[i][0]);length=Int32(self.groups[i][1]-self.groups[i][0])
                budget=Int32(self.budgets[i]);offset=Int32(off)
            off+=self.budgets[i]
        sm=utils.SmemAllocator()
        values=sm.allocate_tensor(Float32,cute.make_layout(self.width),byte_alignment=16)
        indices=sm.allocate_tensor(Int32,cute.make_layout(self.width),byte_alignment=16)
        v=Float32(float('-inf'))
        if t<length: v=s[h,q,start+t]
        values[t]=v;indices[t]=t
        cute.arch.sync_threads()
        for stage in cutlass.range_constexpr(self.width.bit_length()-1):
            for sub in cutlass.range_constexpr(stage+1):
                delta=1<<(stage-sub)
                v=values[t]; idx=indices[t]
                ov=values[t^delta]; oi=indices[t^delta]
                better=(ov>v)|((ov==v)&(oi<idx))
                want_better=((t&(1<<(stage+1)))==0)==((t&delta)==0)
                cute.arch.sync_threads()
                if better==want_better:
                    v=ov;idx=oi
                values[t]=v;indices[t]=idx
                cute.arch.sync_threads()
        if t<budget: o[h,q,offset+t]=Int64(start+indices[t])
