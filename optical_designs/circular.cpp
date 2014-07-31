// File Description
// Author: Philip Salvaggio

#include "circular.h"

#include "base/aperture_parameters.pb.h"
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

Circular::Circular(const SimulationConfig& params, int sim_index)
    : Aperture(params, sim_index), mask_(), opd_(), opd_est_() {}

Circular::~Circular() {}

Mat Circular::GetOpticalPathLengthDiff() {
  if (opd_.rows > 0) return opd_;

  AberrationFactory::ZernikeAberrations(aberrations(),
      params().array_size(), &opd_);

  return opd_;
}

Mat Circular::GetOpticalPathLengthDiffEstimate() {
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

  AberrationFactory::ZernikeAberrations(aberrations(),
      params().array_size(), &opd_est_);

  return opd_est_;
}

Mat Circular::GetApertureTemplate() {
  const size_t size = params().array_size();
  const double half_size = size / 2.0;
  const double half_size2 = half_size * half_size;

  Mat output(size, size, CV_64FC1);
  double* output_data = (double*) output.data;

  double primary_r2 = 1;

  for (size_t i = 0; i < size; i++) {
    double y = i - half_size;
    for (size_t j = 0; j < size; j++) {
      double x = j - half_size;

      double r2 = (x*x + y*y) / half_size2;
      output_data[i*size + j] = (r2 < primary_r2) ? 1 : 0;
    }
  }

  return output;
}
