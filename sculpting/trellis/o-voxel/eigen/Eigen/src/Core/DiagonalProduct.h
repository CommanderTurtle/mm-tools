
#ifndef EIGEN_DIAGONALPRODUCT_H
#define EIGEN_DIAGONALPRODUCT_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

/** \returns the diagonal matrix product of \c *this by the diagonal matrix \a diagonal.
 */
template <typename Derived>
template <typename DiagonalDerived>
EIGEN_DEVICE_FUNC inline const Product<Derived, DiagonalDerived, LazyProduct> MatrixBase<Derived>::operator*(
    const DiagonalBase<DiagonalDerived> &a_diagonal) const {
  return Product<Derived, DiagonalDerived, LazyProduct>(derived(), a_diagonal.derived());
}

}  // end namespace Eigen

#endif  // EIGEN_DIAGONALPRODUCT_H
