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

  mainLog() << "Error in the estimates of piston/tip/tilt: "
            << knowledge_level << " [waves]" << std::endl;

  const vector<double>& real_weights = aberrations();
  vector<double> wrong_weights;
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
  const size_t size = params().array_size();
  const double half_size = size / 2.0;
  const double half_size2 = half_size * half_size;
  double primary_r2 = 1;

  Mat output(size, size, CV_64FC1);
  double* output_data = (double*) output.data;

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
