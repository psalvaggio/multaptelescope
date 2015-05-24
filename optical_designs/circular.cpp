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

using mats::ApertureParameters;
using mats::SimulationConfig;
using mats::Simulation;
using mats::PupilFunction;
using namespace cv;

Circular::Circular(const SimulationConfig& params, int sim_index)
    : Aperture(params, sim_index) {}

Circular::~Circular() {}

Mat Circular::GetOpticalPathLengthDiff() const {
  Mat opd;
  ZernikeAberrations& ab_factory(ZernikeAberrations::getInstance());
  ab_factory.aberrations(aberrations(), params().array_size(), &opd);
  return opd;
}

Mat Circular::GetOpticalPathLengthDiffEstimate() const {
  if (simulation_params().wfe_knowledge() == Simulation::NONE) {
    return Mat(params().array_size(), params().array_size(), CV_64FC1);
  }

  double knowledge_level = 0;
  switch (simulation_params().wfe_knowledge()) {
    case Simulation::HIGH: knowledge_level = 0.05; break;
    case Simulation::MEDIUM: knowledge_level = 0.1; break;
    default: knowledge_level = 0.2; break;
  }

  const std::vector<double>& real_weights = aberrations();
  std::vector<double> wrong_weights;
  for (size_t i = 0; i < real_weights.size(); i++) {
    wrong_weights.push_back(real_weights[i] +
        (2 * (rand() % 2) - 1) * knowledge_level);
  }

  Mat opd_est;
  ZernikeAberrations& ab_factory(ZernikeAberrations::getInstance());
  ab_factory.aberrations(wrong_weights, params().array_size(), &opd_est);
  return opd_est;
}

Mat Circular::GetApertureTemplate() const {
  const size_t kSize = params().array_size();
  const double kHalfSize = kSize / 2.0;
  const double kHalfSize2 = kHalfSize * kHalfSize;
  const double kPrimaryR2 = 1;

  Mat output(kSize, kSize, CV_64FC1);
  double* output_data = (double*) output.data;

  for (size_t i = 0; i < kSize; i++) {
    double y = i - kHalfSize;
    for (size_t j = 0; j < kSize; j++) {
      double x = j - kHalfSize;

      double r2 = (x*x + y*y) / kHalfSize2;
      output_data[i*kSize + j] = (r2 < kPrimaryR2) ? 1 : 0;
    }
  }

  return output;
}
