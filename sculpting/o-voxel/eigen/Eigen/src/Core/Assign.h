
#ifndef EIGEN_ASSIGN_H
#define EIGEN_ASSIGN_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

template <typename Derived>
template <typename OtherDerived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::lazyAssign(const DenseBase<OtherDerived>& other) {
  enum { SameType = internal::is_same<typename Derived::Scalar, typename OtherDerived::Scalar>::value };

  EIGEN_STATIC_ASSERT_LVALUE(Derived)
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Derived, OtherDerived)
  EIGEN_STATIC_ASSERT(
      SameType,
      YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)

  eigen_assert(rows() == other.rows() && cols() == other.cols());
  internal::call_assignment_no_alias(derived(), other.derived());

  return derived();
}

template <typename Derived>
template <typename OtherDerived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator=(const DenseBase<OtherDerived>& other) {
  internal::call_assignment(derived(), other.derived());
  return derived();
}

template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator=(const DenseBase& other) {
  internal::call_assignment(derived(), other.derived());
  return derived();
}

template <typename Derived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& MatrixBase<Derived>::operator=(const MatrixBase& other) {
  internal::call_assignment(derived(), other.derived());
  return derived();
}

template <typename Derived>
template <typename OtherDerived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& MatrixBase<Derived>::operator=(const DenseBase<OtherDerived>& other) {
  internal::call_assignment(derived(), other.derived());
  return derived();
}

template <typename Derived>
template <typename OtherDerived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& MatrixBase<Derived>::operator=(const EigenBase<OtherDerived>& other) {
  internal::call_assignment(derived(), other.derived());
  return derived();
}

template <typename Derived>
template <typename OtherDerived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& MatrixBase<Derived>::operator=(
    const ReturnByValue<OtherDerived>& other) {
  other.derived().evalTo(derived());
  return derived();
}

}  // end namespace Eigen

#endif  // EIGEN_ASSIGN_H
