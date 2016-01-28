// File Description
// Author: Philip Salvaggio

#include "cassegrain.h"
#include "base/zernike_aberrations.h"
#include "base/simulation_config.pb.h"
#include "base/opencv_utils.h"
#include "base/pupil_function.h"
#include "io/logging.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace cv;

namespace mats {

Cassegrain::Cassegrain(const Simulation& params)
    : Aperture(params) {}

Cassegrain::~Cassegrain() {}

void Cassegrain::GetApertureTemplate(Mat_<double>* output) const {
  Mat_<double>& mask = *output;

  const size_t kSize = mask.rows;
  const double kHalfSize = 0.5 * kSize;
  const double kHalfSize2 = kHalfSize * kHalfSize;

  double primary_r2 = 1;
  double secondary_r2 = 1 - aperture_params().fill_factor();

  for (size_t i = 0; i < kSize; i++) {
    double y = i - kHalfSize;
    for (size_t j = 0; j < kSize; j++) {
      double x = j - kHalfSize;

      double r2 = (x*x + y*y) / kHalfSize2;
      mask(i, j) = (r2 < primary_r2 && r2 >= secondary_r2) ? 1 : 0;
    }
  }
}

}
