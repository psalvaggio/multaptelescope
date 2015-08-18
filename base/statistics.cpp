// File Description
// Author: Philip Salvaggio

#include "statistics.h"

#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <iostream>

using namespace std;

namespace mats {

double TConfidenceInterval(double stddev, int size, double p) {
  boost::math::students_t t_dist(size - 1);
  double t_val = boost::math::quantile(
                     boost::math::complement(t_dist, (1 - p) / 2));
  return t_val * stddev / sqrt(size);
}

double ZConfidenceInterval(double stddev, int size, double p) {
  boost::math::normal_distribution<double> z_dist;
  double z_val = boost::math::quantile(
                     boost::math::complement(z_dist, (1 - p) / 2));
  return z_val * stddev / sqrt(size);
}

}
