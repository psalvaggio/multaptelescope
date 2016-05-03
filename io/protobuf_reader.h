// A generic class for reading in protobufs.
// Author: Philip Salvaggio

#ifndef PROTOBUF_READER_H
#define PROTOBUF_READER_H

#include "io/logging.h"

#include <string>
#include <fstream>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

namespace mats_io {

class ProtobufReader {
 public:
  ProtobufReader() = delete;

  template <typename T>
  static bool Read(const std::string& filename, T* output) {
    std::ifstream ifs(filename.c_str());
    if (!ifs.is_open()) {
      mainLog() << "Could not open file: " << filename << std::endl;
      return false;
    }

    google::protobuf::io::IstreamInputStream is(&ifs);
    return google::protobuf::TextFormat::Parse(&is, output);
  }
};

}

#endif  // PROTOBUF_READER_H
