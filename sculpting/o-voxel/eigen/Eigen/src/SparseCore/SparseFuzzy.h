
#ifndef EIGEN_SPARSE_FUZZY_H
#define EIGEN_SPARSE_FUZZY_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename Derived>
template <typename OtherDerived>
bool SparseMatrixBase<Derived>::isApprox(const SparseMatrixBase<OtherDerived>& other, const RealScalar& prec) const {
  const typename internal::nested_eval<Derived, 2, PlainObject>::type actualA(derived());
  std::conditional_t<bool(IsRowMajor) == bool(OtherDerived::IsRowMajor),
                     const typename internal::nested_eval<OtherDerived, 2, PlainObject>::type, const PlainObject>
      actualB(other.derived());

  return (actualA - actualB).squaredNorm() <= prec * prec * numext::mini(actualA.squaredNorm(), actualB.squaredNorm());
}

}  // end namespace Eigen

#endif  // EIGEN_SPARSE_FUZZY_H
