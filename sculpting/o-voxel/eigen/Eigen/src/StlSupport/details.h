
#ifndef EIGEN_STL_DETAILS_H
#define EIGEN_STL_DETAILS_H

#ifndef EIGEN_ALIGNED_ALLOCATOR
#define EIGEN_ALIGNED_ALLOCATOR Eigen::aligned_allocator
#endif

namespace Eigen {

// This one is needed to prevent reimplementing the whole std::vector.
template <class T>
class aligned_allocator_indirection : public EIGEN_ALIGNED_ALLOCATOR<T> {
 public:
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef T value_type;

  template <class U>
  struct rebind {
    typedef aligned_allocator_indirection<U> other;
  };

  aligned_allocator_indirection() {}
  aligned_allocator_indirection(const aligned_allocator_indirection&) : EIGEN_ALIGNED_ALLOCATOR<T>() {}
  aligned_allocator_indirection(const EIGEN_ALIGNED_ALLOCATOR<T>&) {}
  template <class U>
  aligned_allocator_indirection(const aligned_allocator_indirection<U>&) {}
  template <class U>
  aligned_allocator_indirection(const EIGEN_ALIGNED_ALLOCATOR<U>&) {}
  ~aligned_allocator_indirection() {}
};

#if EIGEN_COMP_MSVC

// sometimes, MSVC detects, at compile time, that the argument x
// in std::vector::resize(size_t s,T x) won't be aligned and generate an error
// even if this function is never called. Whence this little wrapper.
#define EIGEN_WORKAROUND_MSVC_STL_SUPPORT(T) \
  std::conditional_t<Eigen::internal::is_arithmetic<T>::value, T, Eigen::internal::workaround_msvc_stl_support<T> >

namespace internal {
template <typename T>
struct workaround_msvc_stl_support : public T {
  inline workaround_msvc_stl_support() : T() {}
  inline workaround_msvc_stl_support(const T& other) : T(other) {}
  inline operator T&() { return *static_cast<T*>(this); }
  inline operator const T&() const { return *static_cast<const T*>(this); }
  template <typename OtherT>
  inline T& operator=(const OtherT& other) {
    T::operator=(other);
    return *this;
  }
  inline workaround_msvc_stl_support& operator=(const workaround_msvc_stl_support& other) {
    T::operator=(other);
    return *this;
  }
};
}  // namespace internal

#else

#define EIGEN_WORKAROUND_MSVC_STL_SUPPORT(T) T

#endif

}  // namespace Eigen

#endif  // EIGEN_STL_DETAILS_H
