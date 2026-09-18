
#ifndef EIGEN_CXX11_THREADPOOL_THREAD_CANCEL_H
#define EIGEN_CXX11_THREADPOOL_THREAD_CANCEL_H

// Try to come up with a portable way to cancel a thread
#if EIGEN_OS_GNULINUX
#define EIGEN_THREAD_CANCEL(t) pthread_cancel(t.native_handle());
#define EIGEN_SUPPORTS_THREAD_CANCELLATION 1
#else
#define EIGEN_THREAD_CANCEL(t)
#endif

#endif  // EIGEN_CXX11_THREADPOOL_THREAD_CANCEL_H
