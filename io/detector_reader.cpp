// File reader for the detectors parameter file.
// Author: Philip Salvaggio

#include "detector_reader.h"

#include "io/logging.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <fstream>

using namespace google::protobuf;
using namespace google::protobuf::io;

namespace mats_io {

DetectorReader::DetectorReader() {}
DetectorReader::~DetectorReader() {}

bool DetectorReader::Read(const std::string& filename, 
                          mats::DetectorParameters* detector) {
  std::ifstream ifs(filename.c_str());
  if (!ifs.is_open()) {
    mainLog() << "Could not open file: " << filename << std::endl;
    return false;
  }

  IstreamInputStream is(&ifs);
  return TextFormat::Parse(&is, detector);
}

}
