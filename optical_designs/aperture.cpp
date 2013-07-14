// File Description
// Author: Philip Salvaggio

#include "aperture.h"

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

Aperture::Aperture(const SimulationConfig& params, int sim_index)
    : params_(params),
      sim_params_(params.simulation(sim_index)) {}

Aperture::~Aperture() {}

void Aperture::GetPupilFunction(PupilFunction* pupil) const {
  const size_t size = params_.array_size();
  const double half_size = size / 2;
  const double kTargetDiameter = size / 2.0;

  Mat& pupil_real = pupil->real_part();
  Mat& pupil_imag = pupil->imaginary_part();
  pupil_real.create(size, size, CV_64FC1);
  pupil_imag.create(size, size, CV_64FC1);

  pupil->set_meters_per_pixel(sim_params_.encircled_diameter() /
                              kTargetDiameter);

  Mat scaled_aperture(size, size, CV_64FC1);
  Mat unscaled_aperture = GetApertureTemplate();
  Mat opd = GetOpticalPathLengthDiff();

  Mat unscaled_aberrated_real(size, size, CV_64FC1);
  Mat unscaled_aberrated_imag(size, size, CV_64FC1);

  double* aberrated_real = (double*) unscaled_aberrated_real.data;
  double* aberrated_imag = (double*) unscaled_aberrated_imag.data;
  double* unaberrated = (double*) unscaled_aperture.data;
  double* opd_data = (double*) opd.data;
  for (size_t i = 0; i < size*size; i++) {
    aberrated_real[i] = unaberrated[i] * cos(2 * M_PI * opd_data[i]);
    aberrated_imag[i] = unaberrated[i] * sin(2 * M_PI * opd_data[i]);
  }

  Range scaling_range;
  scaling_range.start = half_size / 2;
  scaling_range.end = 3 * half_size / 2 + 1;

  Mat real_center = pupil_real(scaling_range, scaling_range);
  Mat imag_center = pupil_imag(scaling_range, scaling_range);

  resize(unscaled_aberrated_real, real_center, real_center.size(), 0, 0,
         INTER_NEAREST);
  resize(unscaled_aberrated_imag, imag_center, imag_center.size(), 0, 0,
         INTER_NEAREST);
}

Mat Aperture::GetRandomPistonTipTilt(double* ptt) const {
  Mat ptt_mat(3, 1, CV_64FC1);
  randn(ptt_mat, 0, 1);

  double* ptt_vals = (double*) ptt_mat.data;

  if (ptt != NULL) {
    memcpy(ptt, ptt_vals, 3*sizeof(double));
  }

  return GetPistonTipTilt(ptt_vals);
}

Mat Aperture::GetPistonTipTiltEstimate(double* ptt_truth,
                                       double* ptt_estimates) const {
  int piston_adj = 2 * (rand() % 2) - 1;
  int tip_adj = 2 * (rand() % 2) - 1;
  int tilt_adj = 2 * (rand() % 2) - 1;

  double knowledge_level = 0;
  switch (sim_params_.wfe_knowledge()) {
    case Simulation::HIGH: knowledge_level = 0.05; break;
    case Simulation::MEDIUM: knowledge_level = 0.1; break;
    default: knowledge_level = 0.2; break;
  }
  mainLog() << "Error in the estimates of piston/tip/tilt: "
            << knowledge_level << " [waves]" << std::endl;

  ptt_estimates[0] = ptt_truth[0] + piston_adj * knowledge_level;
  ptt_estimates[1] = ptt_truth[1] + tip_adj * knowledge_level;
  ptt_estimates[2] = ptt_truth[2] + tilt_adj * knowledge_level;

  return GetPistonTipTilt(ptt_estimates);
}

Mat Aperture::GetPistonTipTilt(double* ptt) const {
  const size_t size = params_.array_size();
  const double half_size = size / 2.0;

  Mat output(size, size, CV_64FC1);

  double* output_data = (double*) output.data;

  for (size_t i = 0; i < size; i++) {
    double y = (i - half_size) / half_size;
    for (size_t j = 0; j < size; j++) {
      double x = (j - half_size) / half_size;

      output_data[i*size + j] =
          ptt[0] + ptt[1] * x + ptt[2] * y;
    }
  }

  return output;
}
