#ifndef EIGEN_RESHAPED_HELPER_H
#define EIGEN_RESHAPED_HELPER_H

// IWYU pragma: private
#include "../InternalHeaderCheck.h"

namespace Eigen {

enum AutoSize_t { AutoSize };
const int AutoOrder = 2;

namespace internal {

template <typename SizeType, typename OtherSize, int TotalSize>
struct get_compiletime_reshape_size {
  enum { value = get_fixed_value<SizeType>::value };
};

template <typename SizeType>
Index get_runtime_reshape_size(SizeType size, Index /*other*/, Index /*total*/) {
  return internal::get_runtime_value(size);
}

template <typename OtherSize, int TotalSize>
struct get_compiletime_reshape_size<AutoSize_t, OtherSize, TotalSize> {
  enum {
    other_size = get_fixed_value<OtherSize>::value,
    value = (TotalSize == Dynamic || other_size == Dynamic) ? Dynamic : TotalSize / other_size
  };
};

inline Index get_runtime_reshape_size(AutoSize_t /*size*/, Index other, Index total) { return total / other; }

constexpr int get_compiletime_reshape_order(int flags, int order) {
  return order == AutoOrder ? flags & RowMajorBit : order;
}

}  // namespace internal

}  // end namespace Eigen

#endif  // EIGEN_RESHAPED_HELPER_H
