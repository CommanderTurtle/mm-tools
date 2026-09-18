#ifndef EIGEN_JACOBISVD_LAPACKE_H
#define EIGEN_JACOBISVD_LAPACKE_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \internal Specialization for the data types supported by LAPACKe */

#define EIGEN_LAPACKE_SVD(EIGTYPE, LAPACKE_TYPE, LAPACKE_RTYPE, LAPACKE_PREFIX, EIGCOLROW, LAPACKE_COLROW, OPTIONS) \
  template <>                                                                                                       \
  template <typename Derived>                                                                                       \
  inline JacobiSVD<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic>, OPTIONS>&                        \
  JacobiSVD<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic>, OPTIONS>::compute_impl(                 \
      const MatrixBase<Derived>& matrix, unsigned int computationOptions) {                                         \
    /*typedef MatrixType::Scalar Scalar;*/                                                                          \
    /*typedef MatrixType::RealScalar RealScalar;*/                                                                  \
    allocate(matrix.rows(), matrix.cols(), computationOptions);                                                     \
                                                                                                                    \
    /*const RealScalar precision = RealScalar(2) * NumTraits<Scalar>::epsilon();*/                                  \
    m_nonzeroSingularValues = diagSize();                                                                           \
                                                                                                                    \
    lapack_int lda = internal::convert_index<lapack_int>(matrix.outerStride()), ldu, ldvt;                          \
    lapack_int matrix_order = LAPACKE_COLROW;                                                                       \
    char jobu, jobvt;                                                                                               \
    LAPACKE_TYPE *u, *vt, dummy;                                                                                    \
    jobu = (m_computeFullU) ? 'A' : (m_computeThinU) ? 'S' : 'N';                                                   \
    jobvt = (m_computeFullV) ? 'A' : (m_computeThinV) ? 'S' : 'N';                                                  \
    if (computeU()) {                                                                                               \
      ldu = internal::convert_index<lapack_int>(m_matrixU.outerStride());                                           \
      u = (LAPACKE_TYPE*)m_matrixU.data();                                                                          \
    } else {                                                                                                        \
      ldu = 1;                                                                                                      \
      u = &dummy;                                                                                                   \
    }                                                                                                               \
    MatrixType localV;                                                                                              \
    lapack_int vt_rows = (m_computeFullV)   ? internal::convert_index<lapack_int>(cols())                           \
                         : (m_computeThinV) ? internal::convert_index<lapack_int>(diagSize())                       \
                                            : 1;                                                                    \
    if (computeV()) {                                                                                               \
      localV.resize(vt_rows, cols());                                                                               \
      ldvt = internal::convert_index<lapack_int>(localV.outerStride());                                             \
      vt = (LAPACKE_TYPE*)localV.data();                                                                            \
    } else {                                                                                                        \
      ldvt = 1;                                                                                                     \
      vt = &dummy;                                                                                                  \
    }                                                                                                               \
    Matrix<LAPACKE_RTYPE, Dynamic, Dynamic> superb;                                                                 \
    superb.resize(diagSize(), 1);                                                                                   \
    MatrixType m_temp;                                                                                              \
    m_temp = matrix;                                                                                                \
    lapack_int info = LAPACKE_##LAPACKE_PREFIX##gesvd(                                                              \
        matrix_order, jobu, jobvt, internal::convert_index<lapack_int>(rows()),                                     \
        internal::convert_index<lapack_int>(cols()), (LAPACKE_TYPE*)m_temp.data(), lda,                             \
        (LAPACKE_RTYPE*)m_singularValues.data(), u, ldu, vt, ldvt, superb.data());                                  \
    /* Check the result of the LAPACK call */                                                                       \
    if (info < 0 || !m_singularValues.allFinite()) {                                                                \
      m_info = InvalidInput;                                                                                        \
    } else if (info > 0) {                                                                                          \
      m_info = NoConvergence;                                                                                       \
    } else {                                                                                                        \
      m_info = Success;                                                                                             \
      if (computeV()) m_matrixV = localV.adjoint();                                                                 \
    }                                                                                                               \
    /* for(int i=0;i<diagSize();i++) if (m_singularValues.coeffRef(i) < precision) { m_nonzeroSingularValues--;     \
     * m_singularValues.coeffRef(i)=RealScalar(0);}*/                                                               \
    m_isInitialized = true;                                                                                         \
    return *this;                                                                                                   \
  }

#define EIGEN_LAPACK_SVD_OPTIONS(OPTIONS)                                                            \
  EIGEN_LAPACKE_SVD(double, double, double, d, ColMajor, LAPACK_COL_MAJOR, OPTIONS)                  \
  EIGEN_LAPACKE_SVD(float, float, float, s, ColMajor, LAPACK_COL_MAJOR, OPTIONS)                     \
  EIGEN_LAPACKE_SVD(dcomplex, lapack_complex_double, double, z, ColMajor, LAPACK_COL_MAJOR, OPTIONS) \
  EIGEN_LAPACKE_SVD(scomplex, lapack_complex_float, float, c, ColMajor, LAPACK_COL_MAJOR, OPTIONS)   \
                                                                                                     \
  EIGEN_LAPACKE_SVD(double, double, double, d, RowMajor, LAPACK_ROW_MAJOR, OPTIONS)                  \
  EIGEN_LAPACKE_SVD(float, float, float, s, RowMajor, LAPACK_ROW_MAJOR, OPTIONS)                     \
  EIGEN_LAPACKE_SVD(dcomplex, lapack_complex_double, double, z, RowMajor, LAPACK_ROW_MAJOR, OPTIONS) \
  EIGEN_LAPACKE_SVD(scomplex, lapack_complex_float, float, c, RowMajor, LAPACK_ROW_MAJOR, OPTIONS)

EIGEN_LAPACK_SVD_OPTIONS(0)
EIGEN_LAPACK_SVD_OPTIONS(ComputeThinU)
EIGEN_LAPACK_SVD_OPTIONS(ComputeThinV)
EIGEN_LAPACK_SVD_OPTIONS(ComputeFullU)
EIGEN_LAPACK_SVD_OPTIONS(ComputeFullV)
EIGEN_LAPACK_SVD_OPTIONS(ComputeThinU | ComputeThinV)
EIGEN_LAPACK_SVD_OPTIONS(ComputeFullU | ComputeFullV)
EIGEN_LAPACK_SVD_OPTIONS(ComputeThinU | ComputeFullV)
EIGEN_LAPACK_SVD_OPTIONS(ComputeFullU | ComputeThinV)

}  // end namespace Eigen

#endif  // EIGEN_JACOBISVD_LAPACKE_H
