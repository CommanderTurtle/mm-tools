"""Standalone exact Top-4 reduction with SM80+ integer warp REDUX.
BF16 WGMMA -> FP32 dot products -> reversible signed-float ordering keys ->
four warp max/winner rounds -> original descending FP32 sum and scale.
Inputs: (Nq*16,D,H)/(Nk*16,D,H) K-major BF16, output (H,Nq,Nk) FP32.
Same TMA tile/tail semantics as ScoreSm90. Finite dot products, inference only;
no workspace beyond per-CTA shared memory/registers. No quantized GEMM.
Grid ceil(Nq/4)*ceil(Nk/(tile_n/16))*H; real case ~520k CTAs / 78 SM.
Purpose: reduce cross-lane selection instructions; independently benchmarked
before any model integration. NaN dot products are outside this contract.
"""
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, BFloat16
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass import pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90
from cuda.bindings import driver as cuda


class ScoreTop4Redux:
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
        # Order register consumption before integer reinterpretation of WGMMA outputs.
        cute.arch.sync_threads()
        # SM90 accumulator: each warp owns 16 rows; eight registers/lane per 16 columns.
        for p in cutlass.range_constexpr(self.bn//16):
            qi=warp
            ki=p
            # Exact, order-preserving float-bit keys; BF16 WGMMA is unchanged.
            vals=cute.make_rmem_tensor((8,),Float32)
            for j in cutlass.range_constexpr(8):
                vals[j]=acc[p*8+j]
            for width in cutlass.range_constexpr(1,4):
                for dist_exp in cutlass.range_constexpr(width-1,-1,-1):
                    for i in cutlass.range_constexpr(8):
                        j=i ^ (1 << dist_exp)
                        if cutlass.const_expr(j>i):
                            vi=vals[i];vj=vals[j]
                            if cutlass.const_expr((i & (1 << width))==0):
                                vals[i]=(vi if vi>vj else vj)
                                vals[j]=(vi if vi<vj else vj)
                            else:
                                vals[i]=(vi if vi<vj else vj)
                                vals[j]=(vi if vi>vj else vj)
            top=cute.make_rmem_tensor((4,),Float32)
            for i in cutlass.range_constexpr(4):top[i]=vals[i]
            total=Float32(0)
            for rank in cutlass.range_constexpr(4):
                bits=top[0].bitcast(Int32)
                key=bits ^ ((bits >> 31) & 0x7fffffff)
                maximum=cute.arch.warp_redux_sync(key,kind="max")
                winner=cute.arch.warp_redux_sync(Int32(lane if key==maximum else 32),kind="min")
                bits=maximum ^ ((maximum >> 31) & 0x7fffffff)
                total+=bits.bitcast(Float32)
                for i in cutlass.range_constexpr(3):
                    top[i]=(top[i+1] if lane==winner else top[i])
                top[3]=(Float32(float("-inf")) if lane==winner else top[3])
            row=bx*4+qi
            col=by*(self.bn//16)+ki
            if (lane==0) & (row<out.shape[1]) & (col<out.shape[2]):
                score=total*0.25*(q.shape[1]**-0.5)
                out[h,row,col]=score
