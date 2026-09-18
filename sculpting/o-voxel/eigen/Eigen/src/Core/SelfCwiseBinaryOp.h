
#ifndef EIGEN_SELFCWISEBINARYOP_H
#define EIGEN_SELFCWISEBINARYOP_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator*=(const Scalar& other) {
  using ConstantExpr = typename internal::plain_constant_type<Derived, Scalar>::type;
  using Op = internal::mul_assign_op<Scalar>;
  internal::call_assignment(derived(), ConstantExpr(rows(), cols(), other), Op());
  return derived();
}

template <typename Derived>
template <bool Enable, typename>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator*=(const RealScalar& other) {
  realView() *= other;
  return derived();
}

template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator/=(const Scalar& other) {
  using ConstantExpr = typename internal::plain_constant_type<Derived, Scalar>::type;
  using Op = internal::div_assign_op<Scalar>;
  internal::call_assignment(derived(), ConstantExpr(rows(), cols(), other), Op());
  return derived();
}

template <typename Derived>
template <bool Enable, typename>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator/=(const RealScalar& other) {
  realView() /= other;
  return derived();
}

}  // end namespace Eigen

#endif  // EIGEN_SELFCWISEBINARYOP_H
