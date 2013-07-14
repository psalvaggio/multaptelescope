// File reader for the detectors parameter file.
// Author: Philip Salvaggio

#ifndef DETECTOR_READER_H
#define DETECTOR_READER_H

#include "base/detector_parameters.pb.h"

#include <string>

namespace mats_io {

class DetectorReader {
 public:
  DetectorReader();
  ~DetectorReader();

  bool Read(const std::string& filename, 
            mats::DetectorParameters* detector);
};

}

#endif  // DETECTOR_WRITER_H
