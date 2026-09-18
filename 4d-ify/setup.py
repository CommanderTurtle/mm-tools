from pathlib import Path
import os

from setuptools import find_namespace_packages, setup
import torch.utils.cpp_extension as cpp_extension


if os.environ.get("MMTOOLS_FORWARD_CUDA") == "1":
    cpp_extension._check_cuda_version = lambda *_args, **_kwargs: None

BuildExtension = cpp_extension.BuildExtension
CUDAExtension = cpp_extension.CUDAExtension


ROOT = Path(__file__).resolve().parent
EIGEN = ROOT / "eigen-3.4.0"


def extension(name, sources, *, include_dirs=()):
    return CUDAExtension(
        name,
        sources=sources,
        include_dirs=[str(path) for path in include_dirs],
        extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
    )


setup(
    name="dpvo",
    version="0.0.0",
    packages=find_namespace_packages(include=("dpvo", "dpvo.*")),
    ext_modules=[
        extension(
            "cuda_corr",
            ["dpvo/altcorr/correlation.cpp", "dpvo/altcorr/correlation_kernel.cu"],
        ),
        extension(
            "cuda_ba",
            ["dpvo/fastba/ba.cpp", "dpvo/fastba/ba_cuda.cu", "dpvo/fastba/block_e.cu"],
            include_dirs=(EIGEN,),
        ),
        extension(
            "lietorch_backends",
            [
                "dpvo/lietorch/src/lietorch.cpp",
                "dpvo/lietorch/src/lietorch_gpu.cu",
                "dpvo/lietorch/src/lietorch_cpu.cpp",
            ],
            include_dirs=(ROOT / "dpvo" / "lietorch" / "include", EIGEN),
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
