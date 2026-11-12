
#ifndef EIGEN_SPARSEREDUX_H
#define EIGEN_SPARSEREDUX_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename Derived>
typename internal::traits<Derived>::Scalar SparseMatrixBase<Derived>::sum() const {
  eigen_assert(rows() > 0 && cols() > 0 && "you are using a non initialized matrix");
  Scalar res(0);
  internal::evaluator<Derived> thisEval(derived());
  for (Index j = 0; j < outerSize(); ++j)
    for (typename internal::evaluator<Derived>::InnerIterator iter(thisEval, j); iter; ++iter) res += iter.value();
  return res;
}

template <typename Scalar_, int Options_, typename Index_>
typename internal::traits<SparseMatrix<Scalar_, Options_, Index_> >::Scalar
SparseMatrix<Scalar_, Options_, Index_>::sum() const {
  eigen_assert(rows() > 0 && cols() > 0 && "you are using a non initialized matrix");
  if (this->isCompressed())
    return Matrix<Scalar, 1, Dynamic>::Map(m_data.valuePtr(), m_data.size()).sum();
  else
    return Base::sum();
}

template <typename Scalar_, int Options_, typename Index_>
typename internal::traits<SparseVector<Scalar_, Options_, Index_> >::Scalar
SparseVector<Scalar_, Options_, Index_>::sum() const {
  eigen_assert(rows() > 0 && cols() > 0 && "you are using a non initialized matrix");
  return Matrix<Scalar, 1, Dynamic>::Map(m_data.valuePtr(), m_data.size()).sum();
}

}  // end namespace Eigen

#endif  // EIGEN_SPARSEREDUX_H
