// File writer for the detectors parameter file.
// Author: Philip Salvaggio

#ifndef DETECTOR_WRITER_H
#define DETECTOR_WRITER_H

#include "base/detector_parameters.pb.h"

#include <string>

namespace mats_io {

class DetectorWriter {
 public:
  DetectorWriter();
  ~DetectorWriter();

  bool Write(const std::string& filename, 
             const mats::DetectorParameters& detector);
};

}

#endif  // DETECTOR_WRITER_H
