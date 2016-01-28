// File Description
// Author: Philip Salvaggio

#include "axicon.h"
#include "base/simulation_config.pb.h"
#include "base/pupil_function.h"
#include "io/logging.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace cv;

namespace mats {

Axicon::Axicon(const Simulation& params)
    : Circular(params) {
  axicon_params_ = aperture_params().GetExtension(axicon_params);
}

Axicon::~Axicon() {}

void Axicon::GetOpticalPathLengthDiff(double image_height,
                                      double angle,
                                      Mat_<double>* output) const {
  ZernikeWavefrontError(image_height, angle, output);

  Mat_<double>& opd = *output;

  const size_t kSize = opd.rows;
  const double kHalfSize = kSize / 2.0;
  const double kHalfSize2 = kHalfSize * kHalfSize;
  const double kPrimaryR2 = 1;
  const double kNumCycles = axicon_params_.num_cycles();

  for (size_t i = 0; i < kSize; i++) {
    double y = i - kHalfSize;
    for (size_t j = 0; j < kSize; j++) {
      double x = j - kHalfSize;
      double r2 = (x*x + y*y) / kHalfSize2;

      opd(i, j) += (r2 < kPrimaryR2) ? sqrt(r2) * kNumCycles : 0;
    }
  }
}

}
