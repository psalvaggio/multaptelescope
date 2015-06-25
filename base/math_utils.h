// File Description
// Author: Philip Salvaggio

#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <cmath>
#include <functional>
#include <vector>

namespace mats {

double Gaussian1D(double param, double mean, double std_dev) {
  return exp(-pow(param - mean, 2) / (2 * std_dev * std_dev));
}

double Normal1D(double param, double mean, double std_dev) {
  return Gaussian1D(param, mean, std_dev) / (std_dev * sqrt(2 * M_PI));
}

void range(double start, double incr, double end_inc,
           std::vector<double>* range) {
  range->clear();
  for (double i = start; i <= end_inc; i+= incr) range->push_back(i);
}

void range(double start, double incr, double end_inc,
           std::vector<double>* range, std::function<double(double)> func) {
  range->clear();
  for (double i = start; i <= end_inc; i+= incr) range->push_back(func(i));
}

}

#endif  // MATH_UTILS_H
