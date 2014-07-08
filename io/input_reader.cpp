// File Description
// Author: Philip Salvaggio

#include "input_reader.h"

#include "io/logging.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

using namespace google::protobuf;
using namespace google::protobuf::io;
 
using mats::SimulationConfig;
using mats::Simulation;

namespace mats_io {

InputReader::InputReader() {}
InputReader::~InputReader() {}

bool InputReader::Read(const std::string& filename,
                       SimulationConfig* simulation) {
  std::ifstream ifs(filename.c_str());
  if (!ifs.is_open()) {
    mainLog() << "Could not open file: " << filename << std::endl;
    return false;
  }

  IstreamInputStream is(&ifs);
  return TextFormat::Parse(&is, simulation);
}

}
