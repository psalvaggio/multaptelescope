// File Description
// Author: Philip Salvaggio

#include "hamming_window.h"

#include <cmath>
#include <algorithm>

void HammingWindow(int size, int center, double* hamming) {
  int left_size = center;
  int right_size = size - center - 1;
  int width = std::max(left_size, right_size);

  for (int i = 0; i < size; i++) {
    int x = i - center;
    hamming[i] = 0.54 + 0.46 * cos((M_PI * x) / width);
  }
}
