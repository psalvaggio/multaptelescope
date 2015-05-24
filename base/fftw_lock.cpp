// File Description
// Author: Philip Salvaggio

#include "fftw_lock.h"

std::mutex FftwLock::fftw_lock_;

std::mutex& fftw_lock() {
  return FftwLock::fftw_lock_;
}
