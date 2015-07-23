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

using mats::ApertureParameters;
using mats::SimulationConfig;
using mats::Simulation;
using mats::PupilFunction;
using namespace cv;

OpticalVortex::OpticalVortex(const SimulationConfig& params, int sim_index)
    : Circular(params, sim_index) {
  vortex_params_ = this->aperture_params().GetExtension(optical_vortex_params);
}

OpticalVortex::~OpticalVortex() {}

void OpticalVortex::GetOpticalPathLengthDiff(Mat_<double>* output) const {
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
        opd(i, j) = phase * kNumCycles;
      } else {
        opd(i, j) = 0;
      }
    }
  }
}

void OpticalVortex::GetOpticalPathLengthDiffEstimate(
    Mat_<double>* output) const {
  GetOpticalPathLengthDiff(output);
}
