"""BF16 SM90 TMA/WGMMA prototype GEMM with on-chip 16x16 top-four reduction.

TMA/pipeline API patterns follow NVIDIA CUTLASS dense_gemm.py (BSD-3-Clause,
see UPSTREAM.md). No prototype dot matrix reaches HBM.
"""
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, BFloat16
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass import pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90
from cuda.bindings import driver as cuda


class ScoreSm90:
    def __init__(self, tile_n=64):
        self.bn = tile_n

    @cute.jit
    def __call__(self, q: cute.Tensor, k: cute.Tensor, scores: cute.Tensor,
                 stream: cuda.CUstream):
        # Input views (N*16,D,H), K-major, output (H,Nq,Nk).
        d = cute.size(q, mode=[1])
        shape = (64, self.bn, d)
        la = sm90.make_smem_layout_a(utils.LayoutEnum.ROW_MAJOR, shape, BFloat16, 1)
        lb = sm90.make_smem_layout_b(utils.LayoutEnum.ROW_MAJOR, shape, BFloat16, 1)
        mma = sm90.make_trivial_tiled_mma(BFloat16, BFloat16,
                warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
                Float32, atom_layout_mnk=(1,1,1), tiler_mn=(64,self.bn))
        ta, qa = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), q,
                    cute.slice_(la,(None,None,0)), (64,d))
        tb, kb = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), k,
                    cute.slice_(lb,(None,None,0)), (self.bn,d))
        @cute.struct
        class Storage:
            barriers: cute.struct.MemRange[Int64, 2]
            a: cute.struct.Align[cute.struct.MemRange[BFloat16,cute.cosize(la)],1024]
            b: cute.struct.Align[cute.struct.MemRange[BFloat16,cute.cosize(lb)],1024]
        self.storage = Storage
        self.kernel(ta,qa,tb,kb,scores,la,lb,mma).launch(
            grid=(cute.ceil_div(scores.shape[1],4),cute.ceil_div(scores.shape[2],self.bn//16),scores.shape[0]),
            block=(128,1,1),stream=stream)

    @cute.kernel
    def kernel(self, ta: cute.CopyAtom, q: cute.Tensor, tb: cute.CopyAtom,
               k: cute.Tensor, out: cute.Tensor, la: cute.ComposedLayout,
               lb: cute.ComposedLayout, mma: cute.TiledMma):
        tid,_,_ = cute.arch.thread_idx()
        bx,by,h = cute.arch.block_idx()
        warp = tid//32
        lane = tid%32
        storage = utils.SmemAllocator().allocate(self.storage)
        a = storage.a.get_tensor(la.outer,swizzle=la.inner)
        b = storage.b.get_tensor(lb.outer,swizzle=lb.inner)
        pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.barriers.data_ptr(),num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,4),
            tx_count=(64+self.bn)*q.shape[1]*2)
        gq = cute.local_tile(q,(64,q.shape[1]),(bx,0,h))
        gk = cute.local_tile(k,(self.bn,k.shape[1]),(by,0,h))
        sa,ga = cpasync.tma_partition(ta,0,cute.make_layout(1),cute.group_modes(a,0,2),cute.group_modes(gq,0,2))
        sb,gb = cpasync.tma_partition(tb,0,cute.make_layout(1),cute.group_modes(b,0,2),cute.group_modes(gk,0,2))
        ps=pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer,1)
        cs=pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer,1)
        if warp==0:
            pipe.producer_acquire(ps)
            cute.copy(ta,ga,sa[(None,0)],tma_bar_ptr=pipe.producer_get_barrier(ps))
            cute.copy(tb,gb,sb[(None,0)],tma_bar_ptr=pipe.producer_get_barrier(ps))
        pipe.consumer_wait(cs)
        thr=mma.get_slice(tid)
        fa=mma.make_fragment_A(thr.partition_A(a))
        fb=mma.make_fragment_B(thr.partition_B(b))
        coords=thr.partition_C(cute.make_identity_tensor((64,self.bn)))
        acc=cute.make_rmem_tensor(coords.shape,Float32)
        acc.fill(0.0)
        warpgroup.fence()
        mma.set(warpgroup.Field.ACCUMULATE,False)
        for z in cutlass.range_constexpr(cute.size(fa,mode=[2])):
            cute.gemm(mma,acc,fa[(None,None,z,0)],fb[(None,None,z,0)],acc)
            mma.set(warpgroup.Field.ACCUMULATE,True)
        warpgroup.commit_group()
        warpgroup.wait_group(0)
        # SM90 accumulator: each warp owns 16 rows; eight registers/lane per 16 columns.
        for p in cutlass.range_constexpr(self.bn//16):
            qi=warp
            ki=p
            vals=cute.make_rmem_tensor((8,),Float32)
            for j in cutlass.range_constexpr(8):
                vals[j]=acc[p*8+j]
            total=Float32(0)
            for repeat in cutlass.range_constexpr(4):
                best=Float32(float('-inf'))
                ix=Int32(256)
                for j in cutlass.range_constexpr(8):
                    c=coords[p*8+j]
                    idx=(c[0]%16)*16+c[1]%16
                    if vals[j]>best:
                        best=vals[j]; ix=idx
                for delta in cutlass.range_constexpr(5):
                    other=cute.arch.shuffle_sync_bfly(best,1<<delta)
                    oi=cute.arch.shuffle_sync_bfly(ix,1<<delta)
                    if (other>best) | ((other==best) & (oi<ix)):
                        best=other; ix=oi
                total+=best
                for j in cutlass.range_constexpr(8):
                    c=coords[p*8+j]
                    if (c[0]%16)*16+c[1]%16==ix:
                        vals[j]=Float32(float('-inf'))
            row=bx*4+qi
            col=by*(self.bn//16)+ki
            if (lane==0) & (row<out.shape[1]) & (col<out.shape[2]):
                score=total*0.25*(q.shape[1]**-0.5)
                out[h,row,col]=score
