
#ifndef EIGEN_MATH_FUNCTIONS_SVE_H
#define EIGEN_MATH_FUNCTIONS_SVE_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

template <>
EIGEN_STRONG_INLINE PacketXf pexp<PacketXf>(const PacketXf& x) {
  return pexp_float(x);
}

template <>
EIGEN_STRONG_INLINE PacketXf plog<PacketXf>(const PacketXf& x) {
  return plog_float(x);
}

template <>
EIGEN_STRONG_INLINE PacketXf psin<PacketXf>(const PacketXf& x) {
  return psin_float(x);
}

template <>
EIGEN_STRONG_INLINE PacketXf pcos<PacketXf>(const PacketXf& x) {
  return pcos_float(x);
}

template <>
EIGEN_STRONG_INLINE PacketXf ptan<PacketXf>(const PacketXf& x) {
  return ptan_float(x);
}

// Hyperbolic Tangent function.
template <>
EIGEN_STRONG_INLINE PacketXf ptanh<PacketXf>(const PacketXf& x) {
  return ptanh_float(x);
}

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_MATH_FUNCTIONS_SVE_H
