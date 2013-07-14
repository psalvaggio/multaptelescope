// File writer for the detectors parameter file.
// Author: Philip Salvaggio

#include "detector_writer.h"

#include "io/logging.h"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <fstream>

using namespace google::protobuf;
using namespace google::protobuf::io;

namespace mats_io {

DetectorWriter::DetectorWriter() {}
DetectorWriter::~DetectorWriter() {}

bool DetectorWriter::Write(const std::string& filename, 
                           const mats::DetectorParameters& detector) {
  std::ofstream ofs(filename.c_str());
  if (!ofs.is_open()) {
    mainLog() << "Could not open file: " << filename << std::endl;
    return false;
  }

  OstreamOutputStream os(&ofs);
  return TextFormat::Print(detector, &os);
}

}
