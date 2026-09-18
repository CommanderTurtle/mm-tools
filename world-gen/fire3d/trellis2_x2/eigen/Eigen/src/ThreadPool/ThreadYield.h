#ifndef EIGEN_CXX11_THREADPOOL_THREAD_YIELD_H
#define EIGEN_CXX11_THREADPOOL_THREAD_YIELD_H

// Try to come up with a portable way to yield
#define EIGEN_THREAD_YIELD() std::this_thread::yield()

#endif  // EIGEN_CXX11_THREADPOOL_THREAD_YIELD_H
