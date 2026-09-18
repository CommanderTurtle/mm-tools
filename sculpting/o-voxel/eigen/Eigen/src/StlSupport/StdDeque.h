
#ifndef EIGEN_STDDEQUE_H
#define EIGEN_STDDEQUE_H

#ifndef EIGEN_STDDEQUE_MODULE_H
#error "Please include Eigen/StdDeque instead of including this file directly."
#endif

#include "details.h"

/**
 * This section contains a convenience MACRO which allows an easy specialization of
 * std::deque such that for data types with alignment issues the correct allocator
 * is used automatically.
 */
#define EIGEN_DEFINE_STL_DEQUE_SPECIALIZATION(...)                                                \
  namespace std {                                                                                 \
  template <>                                                                                     \
  class deque<__VA_ARGS__, std::allocator<__VA_ARGS__> >                                          \
      : public deque<__VA_ARGS__, EIGEN_ALIGNED_ALLOCATOR<__VA_ARGS__> > {                        \
    typedef deque<__VA_ARGS__, EIGEN_ALIGNED_ALLOCATOR<__VA_ARGS__> > deque_base;                 \
                                                                                                  \
   public:                                                                                        \
    typedef __VA_ARGS__ value_type;                                                               \
    typedef deque_base::allocator_type allocator_type;                                            \
    typedef deque_base::size_type size_type;                                                      \
    typedef deque_base::iterator iterator;                                                        \
    explicit deque(const allocator_type& a = allocator_type()) : deque_base(a) {}                 \
    template <typename InputIterator>                                                             \
    deque(InputIterator first, InputIterator last, const allocator_type& a = allocator_type())    \
        : deque_base(first, last, a) {}                                                           \
    deque(const deque& c) : deque_base(c) {}                                                      \
    explicit deque(size_type num, const value_type& val = value_type()) : deque_base(num, val) {} \
    deque(iterator start_, iterator end_) : deque_base(start_, end_) {}                           \
    deque& operator=(const deque& x) {                                                            \
      deque_base::operator=(x);                                                                   \
      return *this;                                                                               \
    }                                                                                             \
  };                                                                                              \
  }

#endif  // EIGEN_STDDEQUE_H
