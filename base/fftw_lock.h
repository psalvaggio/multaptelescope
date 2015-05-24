// File Description
// Author: Philip Salvaggio

#ifndef FFTW_LOCK_H
#define FFTW_LOCK_H

#include <mutex>

std::mutex& fftw_lock();

class FftwLock {
 public:
  static std::mutex fftw_lock_;
};

#endif  // FFTW_LOCK_H
