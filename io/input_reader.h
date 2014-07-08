// File Description
// Author: Philip Salvaggio

#ifndef INPUT_READER_H
#define INPUT_READER_H

#include "base/simulation_config.pb.h"

#include <string>
#include <vector>
#include <iostream>

namespace mats_io {

class InputReader {
 public:
  InputReader();
  ~InputReader();

  bool Read(const std::string& filename,
            mats::SimulationConfig* simulation);
};

}

#endif  // INPUT_READER_H
