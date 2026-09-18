
#ifndef EIGEN_TYPE_CASTING_CLANG_H
#define EIGEN_TYPE_CASTING_CLANG_H

namespace Eigen {
namespace internal {

//==============================================================================
// preinterpret
//==============================================================================
template <>
EIGEN_STRONG_INLINE Packet16f preinterpret<Packet16f, Packet16i>(const Packet16i& a) {
  return reinterpret_cast<Packet16f>(a);
}
template <>
EIGEN_STRONG_INLINE Packet16i preinterpret<Packet16i, Packet16f>(const Packet16f& a) {
  return reinterpret_cast<Packet16i>(a);
}

template <>
EIGEN_STRONG_INLINE Packet8d preinterpret<Packet8d, Packet8l>(const Packet8l& a) {
  return reinterpret_cast<Packet8d>(a);
}
template <>
EIGEN_STRONG_INLINE Packet8l preinterpret<Packet8l, Packet8d>(const Packet8d& a) {
  return reinterpret_cast<Packet8l>(a);
}

//==============================================================================
// pcast
//==============================================================================
#if EIGEN_HAS_BUILTIN(__builtin_convertvector)
template <>
EIGEN_STRONG_INLINE Packet16i pcast<Packet16f, Packet16i>(const Packet16f& a) {
  return __builtin_convertvector(a, Packet16i);
}
template <>
EIGEN_STRONG_INLINE Packet16f pcast<Packet16i, Packet16f>(const Packet16i& a) {
  return __builtin_convertvector(a, Packet16f);
}

template <>
EIGEN_STRONG_INLINE Packet8l pcast<Packet8d, Packet8l>(const Packet8d& a) {
  return __builtin_convertvector(a, Packet8l);
}
template <>
EIGEN_STRONG_INLINE Packet8d pcast<Packet8l, Packet8d>(const Packet8l& a) {
  return __builtin_convertvector(a, Packet8d);
}
#endif

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_TYPE_CASTING_CLANG_H
