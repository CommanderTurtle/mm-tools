
#ifndef EIGEN_REAL_SCHUR_LAPACKE_H
#define EIGEN_REAL_SCHUR_LAPACKE_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \internal Specialization for the data types supported by LAPACKe */

#define EIGEN_LAPACKE_SCHUR_REAL(EIGTYPE, LAPACKE_TYPE, LAPACKE_PREFIX, LAPACKE_PREFIX_U, EIGCOLROW, LAPACKE_COLROW) \
  template <>                                                                                                        \
  template <typename InputType>                                                                                      \
  inline RealSchur<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW> >&                                                   \
  RealSchur<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW> >::compute(const EigenBase<InputType>& matrix,              \
                                                                    bool computeU) {                                 \
    eigen_assert(matrix.cols() == matrix.rows());                                                                    \
                                                                                                                     \
    lapack_int n = internal::convert_index<lapack_int>(matrix.cols()), sdim, info;                                   \
    lapack_int matrix_order = LAPACKE_COLROW;                                                                        \
    char jobvs, sort = 'N';                                                                                          \
    LAPACK_##LAPACKE_PREFIX_U##_SELECT2 select = 0;                                                                  \
    jobvs = (computeU) ? 'V' : 'N';                                                                                  \
    m_matU.resize(n, n);                                                                                             \
    lapack_int ldvs = internal::convert_index<lapack_int>(m_matU.outerStride());                                     \
    m_matT = matrix;                                                                                                 \
    lapack_int lda = internal::convert_index<lapack_int>(m_matT.outerStride());                                      \
    Matrix<EIGTYPE, Dynamic, Dynamic> wr, wi;                                                                        \
    wr.resize(n, 1);                                                                                                 \
    wi.resize(n, 1);                                                                                                 \
    info = LAPACKE_##LAPACKE_PREFIX##gees(matrix_order, jobvs, sort, select, n, (LAPACKE_TYPE*)m_matT.data(), lda,   \
                                          &sdim, (LAPACKE_TYPE*)wr.data(), (LAPACKE_TYPE*)wi.data(),                 \
                                          (LAPACKE_TYPE*)m_matU.data(), ldvs);                                       \
    if (info == 0)                                                                                                   \
      m_info = Success;                                                                                              \
    else                                                                                                             \
      m_info = NoConvergence;                                                                                        \
                                                                                                                     \
    m_isInitialized = true;                                                                                          \
    m_matUisUptodate = computeU;                                                                                     \
    return *this;                                                                                                    \
  }

EIGEN_LAPACKE_SCHUR_REAL(double, double, d, D, ColMajor, LAPACK_COL_MAJOR)
EIGEN_LAPACKE_SCHUR_REAL(float, float, s, S, ColMajor, LAPACK_COL_MAJOR)
EIGEN_LAPACKE_SCHUR_REAL(double, double, d, D, RowMajor, LAPACK_ROW_MAJOR)
EIGEN_LAPACKE_SCHUR_REAL(float, float, s, S, RowMajor, LAPACK_ROW_MAJOR)

}  // end namespace Eigen

#endif  // EIGEN_REAL_SCHUR_LAPACKE_H
