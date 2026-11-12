
#ifndef EIGEN_QR_LAPACKE_H
#define EIGEN_QR_LAPACKE_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

namespace lapacke_helpers {

template <typename MatrixQR, typename HCoeffs>
struct lapacke_hqr {
  static void run(MatrixQR& mat, HCoeffs& hCoeffs, Index = 32, typename MatrixQR::Scalar* = 0) {
    lapack_int m = to_lapack(mat.rows());
    lapack_int n = to_lapack(mat.cols());
    lapack_int lda = to_lapack(mat.outerStride());
    lapack_int matrix_order = lapack_storage_of(mat);
    geqrf(matrix_order, m, n, to_lapack(mat.data()), lda, to_lapack(hCoeffs.data()));
    hCoeffs.adjointInPlace();
  }
};

}  // namespace lapacke_helpers

/** \internal Specialization for the data types supported by LAPACKe */
#define EIGEN_LAPACKE_HH_QR(EIGTYPE)                                      \
  template <typename MatrixQR, typename HCoeffs>                          \
  struct householder_qr_inplace_blocked<MatrixQR, HCoeffs, EIGTYPE, true> \
      : public lapacke_helpers::lapacke_hqr<MatrixQR, HCoeffs> {};

EIGEN_LAPACKE_HH_QR(double)
EIGEN_LAPACKE_HH_QR(float)
EIGEN_LAPACKE_HH_QR(std::complex<double>)
EIGEN_LAPACKE_HH_QR(std::complex<float>)

#undef EIGEN_LAPACKE_HH_QR

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_QR_LAPACKE_H
