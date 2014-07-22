// File Description
// Author: Philip Salvaggio

#ifndef HDF5_READER_H
#define HDF5_READER_H

#include "base/macros.h"

#include <opencv/cv.h>

namespace mats_io {

class HDF5Reader {
 public:
  static bool Read(const std::string& filename,
                   const std::string& dataset,
                   cv::Mat* data);

 private:
  NO_CONSTRUCTION(HDF5Reader)
};

}

#endif  // HDF5_READER_H
