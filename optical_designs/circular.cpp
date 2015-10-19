// File Description
// Author: Philip Salvaggio

#include "circular.h"

#include "base/zernike_aberrations.h"
#include "base/aperture_parameters.pb.h"
#include "base/simulation_config.pb.h"
#include "base/pupil_function.h"
#include "io/logging.h"

#include <cmath>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <vector>

using mats::Simulation;
using namespace cv;

Circular::Circular(const Simulation& params) : Aperture(params) {}

Circular::~Circular() {}

void Circular::GetApertureTemplate(Mat_<double>* output) const {
  Mat_<double>& mask = *output;

  const size_t kSize = mask.rows;
  const double kHalfSize = kSize / 2.0;
  const double kHalfSize2 = kHalfSize * kHalfSize;
  const double kPrimaryR2 = 1;

  for (size_t i = 0; i < kSize; i++) {
    double y = i - kHalfSize;
    for (size_t j = 0; j < kSize; j++) {
      double x = j - kHalfSize;

      double r2 = (x*x + y*y) / kHalfSize2;
      mask(i, j) = (r2 < kPrimaryR2) ? 1 : 0;
    }
  }
}
