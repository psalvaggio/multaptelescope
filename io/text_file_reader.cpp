// File Description
// Author: Philip Salvaggio

#include "text_file_reader.h"

#include "base/linear_interpolator.h"

#include <fstream>
#include <regex>

using namespace std;

namespace mats_io {

bool TextFileReader::Parse(const string& filename,
                           vector<vector<double>>* data) {
  ifstream ifs(filename);
  if (!ifs.is_open()) return false;

  int line_idx = 0;
  string line;
  regex delim("[\\s\\t,]+");

  while (getline(ifs, line)) {
    smatch sm;
    regex_match(line, sm, delim);

    size_t num_fields = sm.size() + 1;
    if (num_fields > data->size()) {
      for (size_t i = data->size(); i < num_fields; i++) {
        data->emplace_back(max(0, line_idx - 1), 0);
      }
    }

    int start = 0;
    for (size_t i = 0; i < num_fields; i++) {
      int end = sm.position(i);
      data->at(i).push_back(atof(line.substr(start, end - start + 1).c_str()));
    }
  }

  return true;
}

bool TextFileReader::Resample(const string& filename,
                              const vector<double>& independent_var,
                              vector<vector<double>>* data) {
  vector<vector<double>> raw_data;
  if (!TextFileReader::Parse(filename, data)) return false;

  for (size_t i = 1; i < raw_data.size(); i++) {
    data->emplace_back();
    mats::LinearInterpolator::Interpolate(
        raw_data[0], raw_data[i], independent_var, &(data->back()));
  }

  return true;
}

}  // namespace mats_io
