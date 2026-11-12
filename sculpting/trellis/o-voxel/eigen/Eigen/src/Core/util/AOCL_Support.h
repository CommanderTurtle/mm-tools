
#ifndef EIGEN_AOCL_SUPPORT_H
#define EIGEN_AOCL_SUPPORT_H

#if defined(EIGEN_USE_AOCL_ALL) || defined(EIGEN_USE_AOCL_MT)

#include <complex>

// Define AOCL component flags based on main flags
#ifdef EIGEN_USE_AOCL_ALL
#define EIGEN_USE_AOCL_VML  // Enable AOCL Vector Math Library
#define EIGEN_USE_AOCL_BLAS // Enable AOCL BLAS (BLIS)

// Enable Eigen BLAS backend only if BLIS provides compatible interface
#if defined(EIGEN_AOCL_BLIS_COMPATIBLE)
#define EIGEN_USE_BLAS // Enable Eigen BLAS backend
#endif

#define EIGEN_USE_LAPACKE // Enable LAPACK backend (FLAME)
#endif

#ifdef EIGEN_USE_AOCL_MT
#define EIGEN_USE_AOCL_VML  // Enable AOCL Vector Math Library
#define EIGEN_USE_AOCL_BLAS // Enable AOCL BLAS (BLIS)

// For multithreaded: disable EIGEN_USE_BLAS to avoid signature conflicts
// Use direct BLIS calls instead through EIGEN_USE_AOCL_BLAS
// #define EIGEN_USE_BLAS       // Commented out - causes conflicts with BLIS
// interface

// Note: LAPACKE disabled in MT mode to avoid header conflicts
#define EIGEN_USE_LAPACKE // Commented out - causes conflicts with BLIS LAPACKE
#define EIGEN_AOCL_USE_BLIS_MT 1 // Enable multithreaded BLIS
#endif

// Handle standalone EIGEN_USE_AOCL_VML flag
#ifndef EIGEN_USE_AOCL_VML
#ifdef EIGEN_USE_AOCL_ALL
#define EIGEN_USE_AOCL_VML
#endif
#ifdef EIGEN_USE_AOCL_MT
#define EIGEN_USE_AOCL_VML
#endif
#endif

// Configuration constants - define these for any AOCL usage
#ifndef EIGEN_AOCL_VML_THRESHOLD
#define EIGEN_AOCL_VML_THRESHOLD 128 // Threshold for VML dispatch
#endif

#ifndef AOCL_SIMD_WIDTH
#define AOCL_SIMD_WIDTH 8 // AVX-512: 512 bits / 64 bits per double
#endif

// Include AOCL Math Library headers for VML
#if defined(EIGEN_USE_AOCL_VML) || defined(EIGEN_USE_AOCL_ALL) ||              \
    defined(EIGEN_USE_AOCL_MT)
#if defined(__has_include)
#if __has_include("amdlibm.h")
#include "amdlibm.h"
#ifndef AMD_LIBM_VEC_EXPERIMENTAL
#define AMD_LIBM_VEC_EXPERIMENTAL
#endif
#if __has_include("amdlibm_vec.h")
#include "amdlibm_vec.h"
#endif
#endif
#else
// Fallback for compilers without __has_include
#include "amdlibm.h"
#ifndef AMD_LIBM_VEC_EXPERIMENTAL
#define AMD_LIBM_VEC_EXPERIMENTAL
#endif
#include "amdlibm_vec.h"
#endif
#endif

// Include CBLAS headers when BLAS is enabled
#ifdef EIGEN_USE_AOCL_BLAS
#if defined(__has_include)
#if __has_include("cblas.h")
#include "cblas.h"
#elif __has_include("blis.h")
#include "blis.h"
#endif
#else
// Fallback
#include "cblas.h"
#endif
#endif

namespace Eigen {
// AOCL-specific type definitions
typedef std::complex<double> dcomplex;
typedef std::complex<float> scomplex;
typedef int BlasIndex; // Standard BLAS index type
} // namespace Eigen

#endif // EIGEN_USE_AOCL_ALL || EIGEN_USE_AOCL_MT

#endif // EIGEN_AOCL_SUPPORT_H
