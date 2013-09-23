// File Description
// Author: Philip Salvaggio

#include "circular.h"

#include "optical_designs/aperture_parameters.pb.h"
#include "base/simulation_config.pb.h"
#include "base/pupil_function.h"
#include "io/logging.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

using mats::SimulationConfig;
using mats::Simulation;
using mats::PupilFunction;
using namespace cv;

Circular::Circular(const SimulationConfig& params,
                   int sim_index,
                   const ApertureParameters& aperture_params)
    : Aperture(params, sim_index, aperture_params),
      diameter_(simulation_params().encircled_diameter()), 
      ptt_vals_() {
  Mat ptt_mat(3, 1, CV_64FC1);
  randn(ptt_mat, 0, 1);

  double* ptt_vals = (double*) ptt_mat.data;
  memcpy(ptt_vals_, ptt_vals, 3*sizeof(double));
}

Circular::~Circular() {}

Mat Circular::GetOpticalPathLengthDiff() {
  Mat mask = GetApertureTemplate();
  Mat ptt = GetPistonTipTilt(ptt_vals_[0], ptt_vals_[1], ptt_vals_[2]);

  ptt = ptt.mul(mask);
  double* ptt_data = (double*) ptt.data;

  int total_elements = 0;
  double total_wfe_sq = 0;
  for (int i = 0; i < ptt.rows * ptt.cols; i++) {
    if (fabs(ptt_data[i]) > 1e-13) {
      total_elements++;
      total_wfe_sq += ptt_data[i] * ptt_data[i];
    }
  }
  double ptt_rms_scale = simulation_params().ptt_opd_rms() /
                         sqrt(total_wfe_sq / total_elements);
  ptt *= ptt_rms_scale;
  for (int i = 0; i < 3; i++) ptt_vals_[i] *= ptt_rms_scale;

  mainLog() << "Piston: " << ptt_vals_[0] << " [waves]" << std::endl;
  mainLog() << "Tip: " << ptt_vals_[1] << " [waves]" << std::endl;
  mainLog() << "Tilt: " << ptt_vals_[2] << " [waves]" << std::endl;

  return ptt;
}

Mat Circular::GetOpticalPathLengthDiffEstimate() {
  if (simulation_params().wfe_knowledge() == Simulation::NONE) {
    return Mat(params().array_size(), params().array_size(), CV_64FC1);
  }

  int piston_adj = 2 * (rand() % 2) - 1;
  int tip_adj = 2 * (rand() % 2) - 1;
  int tilt_adj = 2 * (rand() % 2) - 1;

  double knowledge_level = 0;
  switch (simulation_params().wfe_knowledge()) {
    case Simulation::HIGH: knowledge_level = 0.05; break;
    case Simulation::MEDIUM: knowledge_level = 0.1; break;
    default: knowledge_level = 0.2; break;
  }

  mainLog() << "Error in the estimates of piston/tip/tilt: "
            << knowledge_level << " [waves]" << std::endl;

  double piston_est = ptt_vals_[0] + piston_adj * knowledge_level;
  double tip_est = ptt_vals_[1] + tip_adj * knowledge_level;
  double tilt_est = ptt_vals_[2] + tilt_adj * knowledge_level;

  Mat mask = GetApertureTemplate();
  Mat ptt = GetPistonTipTilt(piston_est, tip_est, tilt_est);

  ptt = ptt.mul(mask);
  double* ptt_data = (double*) ptt.data;

  int total_elements = 0;
  double total_wfe_sq = 0;
  for (int i = 0; i < ptt.rows * ptt.cols; i++) {
    if (fabs(ptt_data[i]) > 1e-13) {
      total_elements++;
      total_wfe_sq += ptt_data[i] * ptt_data[i];
    }
  }
  mainLog() << "RMS OPD of estimated wavefront error: "
            << sqrt(total_wfe_sq / total_elements) << std::endl;

  return ptt;
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
