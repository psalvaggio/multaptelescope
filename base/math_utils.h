// File Description
// Author: Philip Salvaggio

#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <cmath>
#include <iostream>

namespace mats {

double Gaussian1D(double param, double mean, double std_dev) {
  return exp(-pow(param - mean, 2) / (2 * std_dev * std_dev));
}

double Normal1D(double param, double mean, double std_dev) {
  return Gaussian1D(param, mean, std_dev) / (std_dev * sqrt(2 * M_PI));
}

}

#endif  // MATH_UTILS_H
