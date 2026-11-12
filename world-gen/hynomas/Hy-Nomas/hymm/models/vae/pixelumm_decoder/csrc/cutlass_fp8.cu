#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "cutlass/gemm/device/gemm_blockwise.h"
#include "cutlass/epilogue/thread/linear_combination.h"

using E = cutlass::float_e4m3_t;
using O = cutlass::bfloat16_t;
using R = cutlass::layout::RowMajor;
using C = cutlass::layout::ColumnMajor;
using Epilogue = cutlass::epilogue::thread::LinearCombination<O, 8, float, float>;
using Gemm = cutlass::gemm::device::GemmBlockwise<
    E, R, E, C, O, R, float, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm89, cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<32, 32, 128>, cutlass::gemm::GemmShape<16, 8, 32>,
    float, R, 128, Epilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3,
    16, 16, false, cutlass::arch::OpMultiplyAdd>;

at::Tensor gemm(at::Tensor a, at::Tensor b, at::Tensor sa, at::Tensor sb) {
    TORCH_CHECK(a.is_cuda() && a.dim()==2 && b.dim()==2, "Expected CUDA matrices");
    c10::cuda::CUDAGuard guard(a.device());
    for (auto t : {a, b, sa, sb}) {
        TORCH_CHECK(t.device()==a.device() && t.is_contiguous(), "Expected contiguous tensors on one device");
    }
    TORCH_CHECK(a.scalar_type()==at::kFloat8_e4m3fn && b.scalar_type()==at::kFloat8_e4m3fn,
                "Expected E4M3 FP8 inputs");
    TORCH_CHECK(sa.scalar_type()==at::kFloat && sb.scalar_type()==at::kFloat, "Expected FP32 scales");
    int m=a.size(0), n=b.size(0), k=a.size(1);
    TORCH_CHECK(m>0 && n>0 && k>0 && b.size(1)==k && n%128==0 && k%128==0, "Invalid GEMM shape");
    TORCH_CHECK(sa.dim()==2 && sa.size(0)==(m+127)/128 && sa.size(1)==k/128, "Invalid activation scales");
    TORCH_CHECK(sb.dim()==2 && sb.size(0)==n/128 && sb.size(1)==k/128, "Invalid weight scales");
    auto prop=at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(prop->major==8 && prop->minor==9, "CUTLASS FP8 requires SM89");
    auto out=at::empty({m,n}, a.options().dtype(at::kBFloat16));
    // Bound each launch's activation working set relative to L2. Chunk boundaries
    // stay on scale-block boundaries; rows are independent GEMM outputs.
    int chunk = std::max(128, int(prop->l2CacheSize / (8LL * k) / 128) * 128);
    auto stream = at::cuda::getCurrentCUDAStream();
    for (int row=0; row<m; row+=chunk) {
        int rows=std::min(chunk,m-row);
        typename Gemm::Arguments args(
            {rows,n,k}, {reinterpret_cast<E const*>(a.data_ptr())+int64_t(row)*k,k},
            {reinterpret_cast<E const*>(b.data_ptr()),k},
            {reinterpret_cast<O const*>(out.data_ptr())+int64_t(row)*n,n},
            {reinterpret_cast<O*>(out.data_ptr())+int64_t(row)*n,n},
            {sa.data_ptr<float>()+int64_t(row/128)*(k/128),k/128},
            {sb.data_ptr<float>(),k/128}, {1.f,0.f});
        Gemm op;
        auto status=op.can_implement(args);
        TORCH_CHECK(status==cutlass::Status::kSuccess, "Unsupported CUTLASS GEMM: ",cutlassGetStatusString(status));
        TORCH_CHECK(op.get_workspace_size(args)==0, "Unexpected CUTLASS workspace");
        status=op(args,nullptr,stream);
        TORCH_CHECK(status==cutlass::Status::kSuccess, "CUTLASS GEMM failed: ",cutlassGetStatusString(status));
    }
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemm", &gemm); }
