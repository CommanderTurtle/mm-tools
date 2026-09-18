#ifndef EIGEN_SAEIGENSOLVER_LAPACKE_H
#define EIGEN_SAEIGENSOLVER_LAPACKE_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \internal Specialization for the data types supported by LAPACKe */

#define EIGEN_LAPACKE_EIG_SELFADJ_2(EIGTYPE, LAPACKE_TYPE, LAPACKE_RTYPE, LAPACKE_NAME, EIGCOLROW)                   \
  template <>                                                                                                        \
  template <typename InputType>                                                                                      \
  inline SelfAdjointEigenSolver<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW> >&                                      \
  SelfAdjointEigenSolver<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW> >::compute(const EigenBase<InputType>& matrix, \
                                                                                 int options) {                      \
    eigen_assert(matrix.cols() == matrix.rows());                                                                    \
    eigen_assert((options & ~(EigVecMask | GenEigMask)) == 0 && (options & EigVecMask) != EigVecMask &&              \
                 "invalid option parameter");                                                                        \
    bool computeEigenvectors = (options & ComputeEigenvectors) == ComputeEigenvectors;                               \
    lapack_int n = internal::convert_index<lapack_int>(matrix.cols()), lda, info;                                    \
    m_eivalues.resize(n, 1);                                                                                         \
    m_subdiag.resize(n - 1);                                                                                         \
    m_eivec = matrix;                                                                                                \
                                                                                                                     \
    if (n == 1) {                                                                                                    \
      m_eivalues.coeffRef(0, 0) = numext::real(m_eivec.coeff(0, 0));                                                 \
      if (computeEigenvectors) m_eivec.setOnes(n, n);                                                                \
      m_info = Success;                                                                                              \
      m_isInitialized = true;                                                                                        \
      m_eigenvectorsOk = computeEigenvectors;                                                                        \
      return *this;                                                                                                  \
    }                                                                                                                \
                                                                                                                     \
    lda = internal::convert_index<lapack_int>(m_eivec.outerStride());                                                \
    char jobz, uplo = 'L' /*, range='A'*/;                                                                           \
    jobz = computeEigenvectors ? 'V' : 'N';                                                                          \
                                                                                                                     \
    info = LAPACKE_##LAPACKE_NAME(LAPACK_COL_MAJOR, jobz, uplo, n, (LAPACKE_TYPE*)m_eivec.data(), lda,               \
                                  (LAPACKE_RTYPE*)m_eivalues.data());                                                \
    m_info = (info == 0) ? Success : NoConvergence;                                                                  \
    m_isInitialized = true;                                                                                          \
    m_eigenvectorsOk = computeEigenvectors;                                                                          \
    return *this;                                                                                                    \
  }

#define EIGEN_LAPACKE_EIG_SELFADJ(EIGTYPE, LAPACKE_TYPE, LAPACKE_RTYPE, LAPACKE_NAME)       \
  EIGEN_LAPACKE_EIG_SELFADJ_2(EIGTYPE, LAPACKE_TYPE, LAPACKE_RTYPE, LAPACKE_NAME, ColMajor) \
  EIGEN_LAPACKE_EIG_SELFADJ_2(EIGTYPE, LAPACKE_TYPE, LAPACKE_RTYPE, LAPACKE_NAME, RowMajor)

EIGEN_LAPACKE_EIG_SELFADJ(double, double, double, dsyev)
EIGEN_LAPACKE_EIG_SELFADJ(float, float, float, ssyev)
EIGEN_LAPACKE_EIG_SELFADJ(dcomplex, lapack_complex_double, double, zheev)
EIGEN_LAPACKE_EIG_SELFADJ(scomplex, lapack_complex_float, float, cheev)

}  // end namespace Eigen

#endif  // EIGEN_SAEIGENSOLVER_H
