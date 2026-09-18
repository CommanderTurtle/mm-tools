
#ifndef EIGEN_COMPLEX_SCHUR_LAPACKE_H
#define EIGEN_COMPLEX_SCHUR_LAPACKE_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \internal Specialization for the data types supported by LAPACKe */

#define EIGEN_LAPACKE_SCHUR_COMPLEX(EIGTYPE, LAPACKE_TYPE, LAPACKE_PREFIX, LAPACKE_PREFIX_U, EIGCOLROW,            \
                                    LAPACKE_COLROW)                                                                \
  template <>                                                                                                      \
  template <typename InputType>                                                                                    \
  inline ComplexSchur<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW> >&                                              \
  ComplexSchur<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW> >::compute(const EigenBase<InputType>& matrix,         \
                                                                       bool computeU) {                            \
    typedef Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW> MatrixType;                                               \
    typedef MatrixType::RealScalar RealScalar;                                                                     \
    typedef std::complex<RealScalar> ComplexScalar;                                                                \
                                                                                                                   \
    eigen_assert(matrix.cols() == matrix.rows());                                                                  \
                                                                                                                   \
    m_matUisUptodate = false;                                                                                      \
    if (matrix.cols() == 1) {                                                                                      \
      m_matT = matrix.derived().template cast<ComplexScalar>();                                                    \
      if (computeU) m_matU = ComplexMatrixType::Identity(1, 1);                                                    \
      m_info = Success;                                                                                            \
      m_isInitialized = true;                                                                                      \
      m_matUisUptodate = computeU;                                                                                 \
      return *this;                                                                                                \
    }                                                                                                              \
    lapack_int n = internal::convert_index<lapack_int>(matrix.cols()), sdim, info;                                 \
    lapack_int matrix_order = LAPACKE_COLROW;                                                                      \
    char jobvs, sort = 'N';                                                                                        \
    LAPACK_##LAPACKE_PREFIX_U##_SELECT1 select = 0;                                                                \
    jobvs = (computeU) ? 'V' : 'N';                                                                                \
    m_matU.resize(n, n);                                                                                           \
    lapack_int ldvs = internal::convert_index<lapack_int>(m_matU.outerStride());                                   \
    m_matT = matrix;                                                                                               \
    lapack_int lda = internal::convert_index<lapack_int>(m_matT.outerStride());                                    \
    Matrix<EIGTYPE, Dynamic, Dynamic> w;                                                                           \
    w.resize(n, 1);                                                                                                \
    info = LAPACKE_##LAPACKE_PREFIX##gees(matrix_order, jobvs, sort, select, n, (LAPACKE_TYPE*)m_matT.data(), lda, \
                                          &sdim, (LAPACKE_TYPE*)w.data(), (LAPACKE_TYPE*)m_matU.data(), ldvs);     \
    if (info == 0)                                                                                                 \
      m_info = Success;                                                                                            \
    else                                                                                                           \
      m_info = NoConvergence;                                                                                      \
                                                                                                                   \
    m_isInitialized = true;                                                                                        \
    m_matUisUptodate = computeU;                                                                                   \
    return *this;                                                                                                  \
  }

EIGEN_LAPACKE_SCHUR_COMPLEX(dcomplex, lapack_complex_double, z, Z, ColMajor, LAPACK_COL_MAJOR)
EIGEN_LAPACKE_SCHUR_COMPLEX(scomplex, lapack_complex_float, c, C, ColMajor, LAPACK_COL_MAJOR)
EIGEN_LAPACKE_SCHUR_COMPLEX(dcomplex, lapack_complex_double, z, Z, RowMajor, LAPACK_ROW_MAJOR)
EIGEN_LAPACKE_SCHUR_COMPLEX(scomplex, lapack_complex_float, c, C, RowMajor, LAPACK_ROW_MAJOR)

}  // end namespace Eigen

#endif  // EIGEN_COMPLEX_SCHUR_LAPACKE_H
