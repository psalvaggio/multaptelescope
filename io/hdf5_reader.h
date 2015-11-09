// File Description
// Author: Philip Salvaggio

#ifndef HDF5_READER_H
#define HDF5_READER_H

#include <opencv2/core/core.hpp>

namespace mats_io {

class HDF5Reader {
 public:
  HDF5Reader() = delete;

  static bool Read(const std::string& filename,
                   const std::string& dataset,
                   cv::Mat* data);
};

}

#endif  // HDF5_READER_H
