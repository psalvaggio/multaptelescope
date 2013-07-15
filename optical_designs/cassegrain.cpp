// File Description
// Author: Philip Salvaggio

#include "cassegrain.h"
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

Cassegrain::Cassegrain(const SimulationConfig& params, int sim_index)
    : Aperture(params, sim_index),
      diameter_(simulation_params().encircled_diameter()), 
      secondary_diameter_(diameter_ *
                          sqrt(1 - simulation_params().fill_factor())) {}

Cassegrain::~Cassegrain() {}

Mat Cassegrain::GetOpticalPathLengthDiff() const {
  double ptt_vals[3];

  Mat mask = GetApertureTemplate();
  Mat ptt = GetRandomPistonTipTilt(ptt_vals);

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
  for (int i = 0; i < 3; i++) ptt_vals[i] *= ptt_rms_scale;

  mainLog() << "Piston: " << ptt_vals[0] << " [waves]" << std::endl;
  mainLog() << "Tip: " << ptt_vals[1] << " [waves]" << std::endl;
  mainLog() << "Tilt: " << ptt_vals[2] << " [waves]" << std::endl;

  return ptt;
}

Mat Cassegrain::GetApertureTemplate() const {
  const size_t size = params().array_size();
  const double half_size = size / 2.0;
  const double half_size2 = half_size * half_size;

  Mat output(size, size, CV_64FC1);
  double* output_data = (double*) output.data;

  double primary_r2 = 1;
  double secondary_r2 = 1 - simulation_params().fill_factor();

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
