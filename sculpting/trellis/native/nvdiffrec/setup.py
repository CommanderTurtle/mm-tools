import os
import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


c_flags = ['-DNVDR_TORCH']
nvcc_flags = ['-DNVDR_TORCH']

ld_flags = []

if os.name == 'posix':
    ld_flags = ['-lcuda', '-lnvrtc']
elif os.name == 'nt':
    ld_flags = ['cuda.lib', 'advapi32.lib', 'nvrtc.lib']

if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
    os.environ['TORCH_CUDA_ARCH_LIST'] = ''

setup(
    name='nvdiffrec_render',
    packages=[
        'nvdiffrec_render',
        'nvdiffrec_render.renderutils',
    ],
    package_data={
        'nvdiffrec_render': ['bsdf_256_256.bin']
    },
    ext_modules=[
        CUDAExtension(
            name='nvdiffrec_render.renderutils._C',
            sources=[
                'nvdiffrec_render/renderutils/c_src/mesh.cu',
                'nvdiffrec_render/renderutils/c_src/loss.cu',
                'nvdiffrec_render/renderutils/c_src/bsdf.cu',
                'nvdiffrec_render/renderutils/c_src/normal.cu',
                'nvdiffrec_render/renderutils/c_src/cubemap.cu',
                'nvdiffrec_render/renderutils/c_src/common.cpp',
                'nvdiffrec_render/renderutils/c_src/torch_bindings.cpp'
            ],
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            },
            extra_link_args=ld_flags,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
