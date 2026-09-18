
#ifndef EIGEN_TRIANGULAR_SOLVER_MATRIX_BLAS_H
#define EIGEN_TRIANGULAR_SOLVER_MATRIX_BLAS_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// implements LeftSide op(triangular)^-1 * general
#define EIGEN_BLAS_TRSM_L(EIGTYPE, BLASTYPE, BLASFUNC)                                                              \
  template <typename Index, int Mode, bool Conjugate, int TriStorageOrder>                                          \
  struct triangular_solve_matrix<EIGTYPE, Index, OnTheLeft, Mode, Conjugate, TriStorageOrder, ColMajor, 1> {        \
    enum {                                                                                                          \
      IsLower = (Mode & Lower) == Lower,                                                                            \
      IsUnitDiag = (Mode & UnitDiag) ? 1 : 0,                                                                       \
      IsZeroDiag = (Mode & ZeroDiag) ? 1 : 0,                                                                       \
      conjA = ((TriStorageOrder == ColMajor) && Conjugate) ? 1 : 0                                                  \
    };                                                                                                              \
    static void run(Index size, Index otherSize, const EIGTYPE* _tri, Index triStride, EIGTYPE* _other,             \
                    Index otherIncr, Index otherStride, level3_blocking<EIGTYPE, EIGTYPE>& /*blocking*/) {          \
      if (size == 0 || otherSize == 0) return;                                                                      \
      EIGEN_ONLY_USED_FOR_DEBUG(otherIncr);                                                                         \
      eigen_assert(otherIncr == 1);                                                                                 \
      BlasIndex m = convert_index<BlasIndex>(size), n = convert_index<BlasIndex>(otherSize), lda, ldb;              \
      char side = 'L', uplo, diag = 'N', transa;                                                                    \
      /* Set alpha_ */                                                                                              \
      EIGTYPE alpha(1);                                                                                             \
      ldb = convert_index<BlasIndex>(otherStride);                                                                  \
                                                                                                                    \
      const EIGTYPE* a;                                                                                             \
      /* Set trans */                                                                                               \
      transa = (TriStorageOrder == RowMajor) ? ((Conjugate) ? 'C' : 'T') : 'N';                                     \
      /* Set uplo */                                                                                                \
      uplo = IsLower ? 'L' : 'U';                                                                                   \
      if (TriStorageOrder == RowMajor) uplo = (uplo == 'L') ? 'U' : 'L';                                            \
      /* Set a, lda */                                                                                              \
      typedef Matrix<EIGTYPE, Dynamic, Dynamic, TriStorageOrder> MatrixTri;                                         \
      Map<const MatrixTri, 0, OuterStride<> > tri(_tri, size, size, OuterStride<>(triStride));                      \
      MatrixTri a_tmp;                                                                                              \
                                                                                                                    \
      if (conjA) {                                                                                                  \
        a_tmp = tri.conjugate();                                                                                    \
        a = a_tmp.data();                                                                                           \
        lda = convert_index<BlasIndex>(a_tmp.outerStride());                                                        \
      } else {                                                                                                      \
        a = _tri;                                                                                                   \
        lda = convert_index<BlasIndex>(triStride);                                                                  \
      }                                                                                                             \
      if (IsUnitDiag) diag = 'U';                                                                                   \
      /* call ?trsm*/                                                                                               \
      BLASFUNC(&side, &uplo, &transa, &diag, &m, &n, (const BLASTYPE*)&numext::real_ref(alpha), (const BLASTYPE*)a, \
               &lda, (BLASTYPE*)_other, &ldb);                                                                      \
    }                                                                                                               \
  };

#ifdef EIGEN_USE_MKL
EIGEN_BLAS_TRSM_L(double, double, dtrsm)
EIGEN_BLAS_TRSM_L(dcomplex, MKL_Complex16, ztrsm)
EIGEN_BLAS_TRSM_L(float, float, strsm)
EIGEN_BLAS_TRSM_L(scomplex, MKL_Complex8, ctrsm)
#else
EIGEN_BLAS_TRSM_L(double, double, dtrsm_)
EIGEN_BLAS_TRSM_L(dcomplex, double, ztrsm_)
EIGEN_BLAS_TRSM_L(float, float, strsm_)
EIGEN_BLAS_TRSM_L(scomplex, float, ctrsm_)
#endif

// implements RightSide general * op(triangular)^-1
#define EIGEN_BLAS_TRSM_R(EIGTYPE, BLASTYPE, BLASFUNC)                                                              \
  template <typename Index, int Mode, bool Conjugate, int TriStorageOrder>                                          \
  struct triangular_solve_matrix<EIGTYPE, Index, OnTheRight, Mode, Conjugate, TriStorageOrder, ColMajor, 1> {       \
    enum {                                                                                                          \
      IsLower = (Mode & Lower) == Lower,                                                                            \
      IsUnitDiag = (Mode & UnitDiag) ? 1 : 0,                                                                       \
      IsZeroDiag = (Mode & ZeroDiag) ? 1 : 0,                                                                       \
      conjA = ((TriStorageOrder == ColMajor) && Conjugate) ? 1 : 0                                                  \
    };                                                                                                              \
    static void run(Index size, Index otherSize, const EIGTYPE* _tri, Index triStride, EIGTYPE* _other,             \
                    Index otherIncr, Index otherStride, level3_blocking<EIGTYPE, EIGTYPE>& /*blocking*/) {          \
      if (size == 0 || otherSize == 0) return;                                                                      \
      EIGEN_ONLY_USED_FOR_DEBUG(otherIncr);                                                                         \
      eigen_assert(otherIncr == 1);                                                                                 \
      BlasIndex m = convert_index<BlasIndex>(otherSize), n = convert_index<BlasIndex>(size), lda, ldb;              \
      char side = 'R', uplo, diag = 'N', transa;                                                                    \
      /* Set alpha_ */                                                                                              \
      EIGTYPE alpha(1);                                                                                             \
      ldb = convert_index<BlasIndex>(otherStride);                                                                  \
                                                                                                                    \
      const EIGTYPE* a;                                                                                             \
      /* Set trans */                                                                                               \
      transa = (TriStorageOrder == RowMajor) ? ((Conjugate) ? 'C' : 'T') : 'N';                                     \
      /* Set uplo */                                                                                                \
      uplo = IsLower ? 'L' : 'U';                                                                                   \
      if (TriStorageOrder == RowMajor) uplo = (uplo == 'L') ? 'U' : 'L';                                            \
      /* Set a, lda */                                                                                              \
      typedef Matrix<EIGTYPE, Dynamic, Dynamic, TriStorageOrder> MatrixTri;                                         \
      Map<const MatrixTri, 0, OuterStride<> > tri(_tri, size, size, OuterStride<>(triStride));                      \
      MatrixTri a_tmp;                                                                                              \
                                                                                                                    \
      if (conjA) {                                                                                                  \
        a_tmp = tri.conjugate();                                                                                    \
        a = a_tmp.data();                                                                                           \
        lda = convert_index<BlasIndex>(a_tmp.outerStride());                                                        \
      } else {                                                                                                      \
        a = _tri;                                                                                                   \
        lda = convert_index<BlasIndex>(triStride);                                                                  \
      }                                                                                                             \
      if (IsUnitDiag) diag = 'U';                                                                                   \
      /* call ?trsm*/                                                                                               \
      BLASFUNC(&side, &uplo, &transa, &diag, &m, &n, (const BLASTYPE*)&numext::real_ref(alpha), (const BLASTYPE*)a, \
               &lda, (BLASTYPE*)_other, &ldb);                                                                      \
      /*std::cout << "TRMS_L specialization!\n";*/                                                                  \
    }                                                                                                               \
  };

#ifdef EIGEN_USE_MKL
EIGEN_BLAS_TRSM_R(double, double, dtrsm)
EIGEN_BLAS_TRSM_R(dcomplex, MKL_Complex16, ztrsm)
EIGEN_BLAS_TRSM_R(float, float, strsm)
EIGEN_BLAS_TRSM_R(scomplex, MKL_Complex8, ctrsm)
#else
EIGEN_BLAS_TRSM_R(double, double, dtrsm_)
EIGEN_BLAS_TRSM_R(dcomplex, double, ztrsm_)
EIGEN_BLAS_TRSM_R(float, float, strsm_)
EIGEN_BLAS_TRSM_R(scomplex, float, ctrsm_)
#endif

}  // end namespace internal

}  // end namespace Eigen

#endif  // EIGEN_TRIANGULAR_SOLVER_MATRIX_BLAS_H
