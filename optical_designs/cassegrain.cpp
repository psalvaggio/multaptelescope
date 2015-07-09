// File Description
// Author: Philip Salvaggio

#include "cassegrain.h"
#include "base/zernike_aberrations.h"
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

Cassegrain::Cassegrain(const SimulationConfig& params, int sim_index)
    : Aperture(params, sim_index) {}

Cassegrain::~Cassegrain() {}

Mat Cassegrain::GetOpticalPathLengthDiff() const {
  Mat opd;
  ZernikeAberrations& ab_factory(ZernikeAberrations::getInstance());
  ab_factory.aberrations(aberrations(), params().array_size(), &opd);
  return opd;
}

Mat Cassegrain::GetOpticalPathLengthDiffEstimate() const {
  if (simulation_params().wfe_knowledge() == Simulation::NONE) {
    return Mat::zeros(params().array_size(), params().array_size(), CV_64FC1);
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

Mat Cassegrain::GetApertureTemplate() const {
  const size_t size = params().array_size();
  const double half_size = size / 2.0;
  const double half_size2 = half_size * half_size;

  Mat output(size, size, CV_64FC1);
  double* output_data = (double*) output.data;

  double primary_r2 = 1;
  double secondary_r2 = 1 - aperture_params().fill_factor();

  for (size_t i = 0; i < size; i++) {
    double y = i - half_size;
    for (size_t j = 0; j < size; j++) {
      double x = j - half_size;

      double r2 = (x*x + y*y) / half_size2;
      output_data[i*size + j] = (r2 < primary_r2 && r2 >= secondary_r2)
                                ? 1 : 0;
    }
  }

  return output;
}
