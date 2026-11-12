
#if defined(EIGEN_CORE_GPU_HIP_CUDA_DEFINES_H)

#ifndef EIGEN_PERMANENTLY_ENABLE_GPU_HIP_CUDA_DEFINES

#undef gpuStream_t
#undef gpuDeviceProp_t
#undef gpuError_t
#undef gpuSuccess
#undef gpuErrorNotReady
#undef gpuGetDeviceCount
#undef gpuGetErrorString
#undef gpuGetDeviceProperties
#undef gpuStreamDefault
#undef gpuGetDevice
#undef gpuSetDevice
#undef gpuMalloc
#undef gpuFree
#undef gpuMemsetAsync
#undef gpuMemset2DAsync
#undef gpuMemcpyAsync
#undef gpuMemcpyDeviceToDevice
#undef gpuMemcpyDeviceToHost
#undef gpuMemcpyHostToDevice
#undef gpuStreamQuery
#undef gpuSharedMemConfig
#undef gpuDeviceSetSharedMemConfig
#undef gpuStreamSynchronize
#undef gpuDeviceSynchronize
#undef gpuMemcpy

#endif  // EIGEN_PERMANENTLY_ENABLE_GPU_HIP_CUDA_DEFINES

#undef EIGEN_CORE_GPU_HIP_CUDA_DEFINES_H

#endif  // EIGEN_CORE_GPU_HIP_CUDA_DEFINES_H
