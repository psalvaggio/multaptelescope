// Implementation file for the RANSAC framework.
// Author: Philip Salvaggio

#include "ransac.h"

namespace ransac {

bool RansacHasValidResults(Error_t error) {
  return error == RansacSuccess || error == RansacMaxTrials;
}


std::string RansacErrorString(Error_t error) {
  switch (error) {
    case RansacSuccess: return "Success";
    case RansacMaxTrials: return "Max trials exceeded.";
    case RansacDegenerateModel: return "No non-degenerate model could be fit.";
    case RansacInvalidInput: return "Invalid input arguments were given.";
    default: return "Invalid error code.";
  }
}


void RandomSample(size_t num_to_select,
                  size_t max_index,
                  std::vector<int>* indices) {
  std::set<int> uniq_indices;
  while (uniq_indices.size() < num_to_select) {
    int idx = rand() % max_index;
    uniq_indices.insert(idx);
  }

  std::set<int>::iterator it;
  int i = 0;
  for (it = uniq_indices.begin(); it != uniq_indices.end(); it++, i++) {
    (*indices)[i] = *it;
  }
}

}
