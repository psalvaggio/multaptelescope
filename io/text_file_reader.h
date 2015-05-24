// File Description
// Author: Philip Salvaggio

#ifndef TEXT_FILE_READER_H
#define TEXT_FILE_READER_H

#include <vector>

namespace mats_io {

class TextFileReader {
 public:
  static bool Parse(const std::string& filename,
                    std::vector<std::vector<double>>* data);

  static bool Resample(const std::string& filename,
                       const std::vector<double>& independent_var,
                       std::vector<std::vector<double>>* data);
};

}  // namespace mats_io

#endif  // TEXT_FILE_READER_H
