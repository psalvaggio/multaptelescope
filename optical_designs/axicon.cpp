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

using mats::ApertureParameters;
using mats::SimulationConfig;
using mats::Simulation;
using mats::PupilFunction;
using namespace cv;

Axicon::Axicon(const SimulationConfig& params, int sim_index)
    : Circular(params, sim_index) {
  axicon_params_ = this->aperture_params().GetExtension(axicon_params);
}

Axicon::~Axicon() {}

Mat Axicon::GetOpticalPathLengthDiff() const {
  const size_t kSize = params().array_size();
  const double kHalfSize = kSize / 2.0;
  const double kHalfSize2 = kHalfSize * kHalfSize;
  const double kPrimaryR2 = 1;
  const double kNumCycles = axicon_params_.num_cycles();

  Mat opd(kSize, kSize, CV_64FC1);
  double* opd_data = reinterpret_cast<double*>(opd.data);

  for (size_t i = 0; i < kSize; i++) {
    double y = i - kHalfSize;
    for (size_t j = 0; j < kSize; j++) {
      double x = j - kHalfSize;
      double r2 = (x*x + y*y) / kHalfSize2;

      if (r2 < kPrimaryR2) {
        opd_data[i*kSize + j] = sqrt(r2) * kNumCycles;
      } else {
        opd_data[i*kSize + j] = 0;
      }
    }
  }

  return opd;
}

Mat Axicon::GetOpticalPathLengthDiffEstimate() const {
  return GetOpticalPathLengthDiff();
}
