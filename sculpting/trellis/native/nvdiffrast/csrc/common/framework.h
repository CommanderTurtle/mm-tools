
#pragma once

// Framework-specific macros to enable code sharing.

//------------------------------------------------------------------------
// PyTorch.

#ifdef NVDR_TORCH
#ifndef __CUDACC__
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/numpy.h>
#endif
#define NVDR_CHECK(COND, ERR) do { TORCH_CHECK(COND, ERR) } while(0)
#define NVDR_CHECK_CUDA_ERROR(CUDA_CALL) do { cudaError_t err = CUDA_CALL; TORCH_CHECK(!err, "Cuda error: ", cudaGetLastError(), "[", #CUDA_CALL, ";]"); } while(0)
#endif

//------------------------------------------------------------------------
