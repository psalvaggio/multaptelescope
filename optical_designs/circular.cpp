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

void Circular::GetOpticalPathLengthDiff(Mat_<double>* output) const {
  ZernikeAberrations& ab_factory(ZernikeAberrations::getInstance());
  ab_factory.aberrations(aberrations(), output->rows, output);
}

void Circular::GetOpticalPathLengthDiffEstimate(Mat_<double>* output) const {
  if (simulation_params().wfe_knowledge() == Simulation::NONE) {
    output->setTo(Scalar(0));
    return;
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

  ZernikeAberrations& ab_factory(ZernikeAberrations::getInstance());
  ab_factory.aberrations(wrong_weights, output->rows, output);
}

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
