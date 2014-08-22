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

using mats::ApertureParameters;
using mats::SimulationConfig;
using mats::Simulation;
using mats::PupilFunction;
using namespace cv;

Cassegrain::Cassegrain(const SimulationConfig& params, int sim_index)
    : Aperture(params, sim_index), mask_(), opd_(), opd_est_() {}

Cassegrain::~Cassegrain() {}

Mat Cassegrain::GetOpticalPathLengthDiff() {
  if (opd_.rows > 0) return opd_;

  ZernikeAberrations& ab_factory(ZernikeAberrations::getInstance());
  ab_factory.aberrations(aberrations(), params().array_size(), &opd_);

  return opd_;
}

Mat Cassegrain::GetOpticalPathLengthDiffEstimate() {
  if (simulation_params().wfe_knowledge() == Simulation::NONE) {
    return Mat(params().array_size(), params().array_size(), CV_64FC1);
  }

  if (opd_.rows == 0) GetOpticalPathLengthDiff();
  if (opd_est_.rows > 0) return opd_est_;

  double knowledge_level = 0;
  switch (simulation_params().wfe_knowledge()) {
    case Simulation::HIGH: knowledge_level = 0.05; break;
    case Simulation::MEDIUM: knowledge_level = 0.1; break;
    default: knowledge_level = 0.2; break;
  }

  mainLog() << "Error in the estimates of piston/tip/tilt: "
            << knowledge_level << " [waves]" << std::endl;

  vector<double>& real_weights = aberrations();
  vector<double> wrong_weights;
  for (size_t i = 0; i < real_weights.size(); i++) {
    wrong_weights.push_back(real_weights[i] +
        (2 * (rand() % 2) - 1) * knowledge_level);
  }

  ZernikeAberrations& ab_factory(ZernikeAberrations::getInstance());
  ab_factory.aberrations(wrong_weights, params().array_size(), &opd_est_);

  return opd_est_;
}

Mat Cassegrain::GetApertureTemplate() {
  if (mask_.rows < 0) return mask_;

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

  mask_ = output;

  return mask_;
}
