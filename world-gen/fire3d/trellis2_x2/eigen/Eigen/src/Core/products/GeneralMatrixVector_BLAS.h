#ifndef EIGEN_GENERAL_MATRIX_VECTOR_BLAS_H
#define EIGEN_GENERAL_MATRIX_VECTOR_BLAS_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

/**********************************************************************
 * This file implements general matrix-vector multiplication using BLAS
 * gemv function via partial specialization of
 * general_matrix_vector_product::run(..) method for float, double,
 * std::complex<float> and std::complex<double> types
 **********************************************************************/

// gemv specialization

template <typename Index, typename LhsScalar, int StorageOrder, bool ConjugateLhs, typename RhsScalar,
          bool ConjugateRhs>
struct general_matrix_vector_product_gemv;

#define EIGEN_BLAS_GEMV_SPECIALIZE(Scalar)                                                                       \
  template <typename Index, bool ConjugateLhs, bool ConjugateRhs>                                                \
  struct general_matrix_vector_product<Index, Scalar, const_blas_data_mapper<Scalar, Index, ColMajor>, ColMajor, \
                                       ConjugateLhs, Scalar, const_blas_data_mapper<Scalar, Index, RowMajor>,    \
                                       ConjugateRhs, Specialized> {                                              \
    static void run(Index rows, Index cols, const const_blas_data_mapper<Scalar, Index, ColMajor>& lhs,          \
                    const const_blas_data_mapper<Scalar, Index, RowMajor>& rhs, Scalar* res, Index resIncr,      \
                    Scalar alpha) {                                                                              \
      if (ConjugateLhs) {                                                                                        \
        general_matrix_vector_product<Index, Scalar, const_blas_data_mapper<Scalar, Index, ColMajor>, ColMajor,  \
                                      ConjugateLhs, Scalar, const_blas_data_mapper<Scalar, Index, RowMajor>,     \
                                      ConjugateRhs, BuiltIn>::run(rows, cols, lhs, rhs, res, resIncr, alpha);    \
      } else {                                                                                                   \
        general_matrix_vector_product_gemv<Index, Scalar, ColMajor, ConjugateLhs, Scalar, ConjugateRhs>::run(    \
            rows, cols, lhs.data(), lhs.stride(), rhs.data(), rhs.stride(), res, resIncr, alpha);                \
      }                                                                                                          \
    }                                                                                                            \
  };                                                                                                             \
  template <typename Index, bool ConjugateLhs, bool ConjugateRhs>                                                \
  struct general_matrix_vector_product<Index, Scalar, const_blas_data_mapper<Scalar, Index, RowMajor>, RowMajor, \
                                       ConjugateLhs, Scalar, const_blas_data_mapper<Scalar, Index, ColMajor>,    \
                                       ConjugateRhs, Specialized> {                                              \
    static void run(Index rows, Index cols, const const_blas_data_mapper<Scalar, Index, RowMajor>& lhs,          \
                    const const_blas_data_mapper<Scalar, Index, ColMajor>& rhs, Scalar* res, Index resIncr,      \
                    Scalar alpha) {                                                                              \
      general_matrix_vector_product_gemv<Index, Scalar, RowMajor, ConjugateLhs, Scalar, ConjugateRhs>::run(      \
          rows, cols, lhs.data(), lhs.stride(), rhs.data(), rhs.stride(), res, resIncr, alpha);                  \
    }                                                                                                            \
  };

EIGEN_BLAS_GEMV_SPECIALIZE(double)
EIGEN_BLAS_GEMV_SPECIALIZE(float)
EIGEN_BLAS_GEMV_SPECIALIZE(dcomplex)
EIGEN_BLAS_GEMV_SPECIALIZE(scomplex)

#define EIGEN_BLAS_GEMV_SPECIALIZATION(EIGTYPE, BLASTYPE, BLASFUNC)                                                 \
  template <typename Index, int LhsStorageOrder, bool ConjugateLhs, bool ConjugateRhs>                              \
  struct general_matrix_vector_product_gemv<Index, EIGTYPE, LhsStorageOrder, ConjugateLhs, EIGTYPE, ConjugateRhs> { \
    typedef Matrix<EIGTYPE, Dynamic, 1, ColMajor> GEMVVector;                                                       \
                                                                                                                    \
    static void run(Index rows, Index cols, const EIGTYPE* lhs, Index lhsStride, const EIGTYPE* rhs, Index rhsIncr, \
                    EIGTYPE* res, Index resIncr, EIGTYPE alpha) {                                                   \
      if (rows == 0 || cols == 0) return;                                                                           \
      BlasIndex m = convert_index<BlasIndex>(rows), n = convert_index<BlasIndex>(cols),                             \
                lda = convert_index<BlasIndex>(lhsStride), incx = convert_index<BlasIndex>(rhsIncr),                \
                incy = convert_index<BlasIndex>(resIncr);                                                           \
      const EIGTYPE beta(1);                                                                                        \
      const EIGTYPE* x_ptr;                                                                                         \
      char trans = (LhsStorageOrder == ColMajor) ? 'N' : (ConjugateLhs) ? 'C' : 'T';                                \
      if (LhsStorageOrder == RowMajor) {                                                                            \
        m = convert_index<BlasIndex>(cols);                                                                         \
        n = convert_index<BlasIndex>(rows);                                                                         \
      }                                                                                                             \
      GEMVVector x_tmp;                                                                                             \
      if (ConjugateRhs) {                                                                                           \
        Map<const GEMVVector, 0, InnerStride<> > map_x(rhs, cols, 1, InnerStride<>(incx));                          \
        x_tmp = map_x.conjugate();                                                                                  \
        x_ptr = x_tmp.data();                                                                                       \
        incx = 1;                                                                                                   \
      } else {                                                                                                      \
        x_ptr = rhs;                                                                                                \
      }                                                                                                             \
      BLASFUNC(&trans, &m, &n, (const BLASTYPE*)&numext::real_ref(alpha), (const BLASTYPE*)lhs, &lda,               \
               (const BLASTYPE*)x_ptr, &incx, (const BLASTYPE*)&numext::real_ref(beta), (BLASTYPE*)res, &incy);     \
    }                                                                                                               \
  };

#ifdef EIGEN_USE_MKL
EIGEN_BLAS_GEMV_SPECIALIZATION(double, double, dgemv)
EIGEN_BLAS_GEMV_SPECIALIZATION(float, float, sgemv)
EIGEN_BLAS_GEMV_SPECIALIZATION(dcomplex, MKL_Complex16, zgemv)
EIGEN_BLAS_GEMV_SPECIALIZATION(scomplex, MKL_Complex8, cgemv)
#else
EIGEN_BLAS_GEMV_SPECIALIZATION(double, double, dgemv_)
EIGEN_BLAS_GEMV_SPECIALIZATION(float, float, sgemv_)
EIGEN_BLAS_GEMV_SPECIALIZATION(dcomplex, double, zgemv_)
EIGEN_BLAS_GEMV_SPECIALIZATION(scomplex, float, cgemv_)
#endif

}  // namespace internal

}  // end namespace Eigen

#endif  // EIGEN_GENERAL_MATRIX_VECTOR_BLAS_H
