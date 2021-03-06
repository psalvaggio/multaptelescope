// File Description
// Author: Philip Salvaggio

#include "optical_vortex.h"
#include "base/simulation_config.pb.h"
#include "base/pupil_function.h"
#include "io/logging.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace cv;

namespace mats {

OpticalVortex::OpticalVortex(const Simulation& params) : Circular(params) {
  vortex_params_ = aperture_params().GetExtension(optical_vortex_params);
}

OpticalVortex::~OpticalVortex() {}

void OpticalVortex::GetOpticalPathLengthDiff(double image_height,
                                             double angle,
                                             Mat_<double>* output) const {
  ZernikeWavefrontError(image_height, angle, output);
  Mat_<double>& opd = *output;

  const size_t kSize = opd.rows;
  const double kHalfSize = kSize / 2.0;
  const double kHalfSize2 = kHalfSize * kHalfSize;
  const double kPrimaryR2 = 1;
  const double kNumCycles = vortex_params_.num_cycles();
  const double k2Pi = 2 * M_PI;

  for (size_t i = 0; i < kSize; i++) {
    double y = i - kHalfSize;
    for (size_t j = 0; j < kSize; j++) {
      double x = j - kHalfSize;
      double r2 = (x*x + y*y) / kHalfSize2;

      if (r2 < kPrimaryR2) {
        double phase = atan2(y, x) / k2Pi;
        if (phase < 0) phase += 1;
        opd(i, j) += phase * kNumCycles;
      }
    }
  }
}

}
