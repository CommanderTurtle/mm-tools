#ifndef EIGEN_STDVECTOR_H
#define EIGEN_STDVECTOR_H

#ifndef EIGEN_STDVECTOR_MODULE_H
#error "Please include Eigen/StdVector instead of including this file directly."
#endif

#include "details.h"

/**
 * This section contains a convenience MACRO which allows an easy specialization of
 * std::vector such that for data types with alignment issues the correct allocator
 * is used automatically.
 */
#define EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(...)                                                 \
  namespace std {                                                                                   \
  template <>                                                                                       \
  class vector<__VA_ARGS__, std::allocator<__VA_ARGS__> >                                           \
      : public vector<__VA_ARGS__, EIGEN_ALIGNED_ALLOCATOR<__VA_ARGS__> > {                         \
    typedef vector<__VA_ARGS__, EIGEN_ALIGNED_ALLOCATOR<__VA_ARGS__> > vector_base;                 \
                                                                                                    \
   public:                                                                                          \
    typedef __VA_ARGS__ value_type;                                                                 \
    typedef vector_base::allocator_type allocator_type;                                             \
    typedef vector_base::size_type size_type;                                                       \
    typedef vector_base::iterator iterator;                                                         \
    explicit vector(const allocator_type& a = allocator_type()) : vector_base(a) {}                 \
    template <typename InputIterator>                                                               \
    vector(InputIterator first, InputIterator last, const allocator_type& a = allocator_type())     \
        : vector_base(first, last, a) {}                                                            \
    vector(const vector& c) : vector_base(c) {}                                                     \
    explicit vector(size_type num, const value_type& val = value_type()) : vector_base(num, val) {} \
    vector(iterator start_, iterator end_) : vector_base(start_, end_) {}                           \
    vector& operator=(const vector& x) {                                                            \
      vector_base::operator=(x);                                                                    \
      return *this;                                                                                 \
    }                                                                                               \
  };                                                                                                \
  }

#endif  // EIGEN_STDVECTOR_H
