// File Description
// Author: Philip Salvaggio

#include "aperture.h"

#include "base/simulation_config.pb.h"
#include "base/pupil_function.h"
#include "base/opencv_utils.h"
#include "io/logging.h"

#include "optical_designs/cassegrain.h"
#include "optical_designs/triarm9.h"
#include "optical_designs/circular.h"
#include "optical_designs/two_cassegrain.h"
#include "optical_designs/triarm3.h"
#include "optical_designs/hdf5_wfe.h"

#include <opencv/highgui.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

using mats::SimulationConfig;
using mats::Simulation;
using mats::PupilFunction;
using mats::ApertureParameters;
using std::cout;
using std::endl;
using namespace cv;

Aperture::Aperture(const SimulationConfig& params, int sim_index)
    : params_(params),
      sim_params_(params.simulation(sim_index)),
      aperture_params_(params.simulation(sim_index).aperture_params()),
      aberrations_() {
  int size = 0;
  for (int i = 0; i < aperture_params_.aberration_size(); i++) {
    size = std::max(size,
        static_cast<int>(aperture_params_.aberration(i).type()) + 1);
  }

  aberrations_.resize(size, 0);
  for (int i = 0; i < aperture_params_.aberration_size(); i++) {
    aberrations_[aperture_params_.aberration(i).type()] =
      aperture_params_.aberration(i).value();
  }
}

Aperture::~Aperture() {}

void Aperture::GetPupilFunction(const Mat& wfe,
                                double wavelength,
                                PupilFunction* pupil) {
  const size_t kSize = params_.array_size();
  
  // We want to pupil function to take up the center half of the array. This
  // is a decent tradeoff so the user has enough resolution to upsample and a
  // decent range (2x) over which to upsample.
  int target_diameter = kSize / 2;

  // Set the scale of the pupil function, so the user can rescale the
  // function with physical units if they need to.
  pupil->set_meters_per_pixel(aperture_params_.encircled_diameter() /
                              target_diameter);

  // Resize the wavefront error and aperture template to the desired
  // resolution.
  Mat scaled_aperture, scaled_wfe;
  resize(GetApertureTemplate(), scaled_aperture,
         Size(target_diameter, target_diameter));
  resize(wfe, scaled_wfe, Size(target_diameter, target_diameter));

  // The wavefront error is in units of the reference wavelength. So, we will
  // need to scale it to be in terms of the requested wavelength, with this
  // scale factor.
  double wavelength_scale = params_.reference_wavelength() / wavelength;

  // Create the aberrated pupil function. This consists of putting the
  // wavefront error into the phase of the complex pupil function.
  Mat scaled_aberrated_real(target_diameter, target_diameter, CV_64FC1);
  Mat scaled_aberrated_imag(target_diameter, target_diameter, CV_64FC1);

  double* aberrated_real = (double*) scaled_aberrated_real.data;
  double* aberrated_imag = (double*) scaled_aberrated_imag.data;
  double* unaberrated = (double*) scaled_aperture.data;
  double* opd_data = (double*) scaled_wfe.data;
  for (int i = 0; i < target_diameter*target_diameter; i++) {
    aberrated_real[i] = unaberrated[i] * cos(2 * M_PI * opd_data[i] *
                        wavelength_scale);
    aberrated_imag[i] = unaberrated[i] * sin(2 * M_PI * opd_data[i] *
                        wavelength_scale);
  }

  // Copy the aberrated pupil function into the output variables.
  Range crop_range;
  crop_range.start = kSize / 2 - (target_diameter + 1) / 2;
  crop_range.end = crop_range.start + target_diameter;

  Mat& pupil_real(pupil->real_part());
  Mat& pupil_imag(pupil->imaginary_part());

  pupil_real = Mat::zeros(kSize, kSize, CV_64FC1);
  pupil_imag = Mat::zeros(kSize, kSize, CV_64FC1);

  scaled_aberrated_real.copyTo(pupil_real(crop_range, crop_range));
  scaled_aberrated_imag.copyTo(pupil_imag(crop_range, crop_range));
}

Mat Aperture::GetWavefrontError() {
  return GetOpticalPathLengthDiff();
}

Mat Aperture::GetWavefrontErrorEstimate() {
  return GetOpticalPathLengthDiffEstimate();
}

Mat Aperture::GetApertureMask() {
  return GetApertureTemplate();
}

Mat Aperture::GetPistonTipTilt(double piston, double tip, double tilt,
                               size_t rows, size_t cols) const {
  const double kHalfRow = rows / 2.0;
  const double kHalfCol = cols / 2.0;

  Mat output(rows, cols, CV_64FC1);

  double* output_data = (double*) output.data;

  for (size_t i = 0; i < rows; i++) {
    double y = (i - kHalfRow) / kHalfRow;
    for (size_t j = 0; j < cols; j++) {
      double x = (j - kHalfCol) / kHalfCol;

      output_data[i*cols + j] = piston + tip * x + tilt * y;
    }
  }

  return output;
}

Mat Aperture::GetPistonTipTilt(double piston, double tip, double tilt) const {
  const size_t size = params_.array_size();
  return GetPistonTipTilt(piston, tip, tilt, size, size);
}


Aperture* ApertureFactory::Create(const mats::SimulationConfig& params, 
                                  int sim_index) {
  if (sim_index > params.simulation_size()) {
    mainLog() << "ApertureFactory error: Given simulation index out of bounds."
              << std::endl;
    return NULL;
  }

  const Simulation& sim(params.simulation(sim_index));
  const ApertureParameters& ap_params(sim.aperture_params());
  ApertureParameters::ApertureType ap_type = ap_params.type();
  if (ap_type == ApertureParameters::TRIARM9) {
    return new Triarm9(params, sim_index);
  } else if (ap_type == ApertureParameters::CASSEGRAIN) {
    return new Cassegrain(params, sim_index);
  } else if (ap_type == ApertureParameters::CIRCULAR) {
    return new Circular(params, sim_index);
  } else if (ap_type == ApertureParameters::TWO_CASSEGRAIN) {
    return new TwoCassegrain(params, sim_index);
  } else if (ap_type == ApertureParameters::TRIARM3) {
    return new Triarm3(params, sim_index);
  } else if (ap_type == ApertureParameters::HDF5_WFE) {
    return new Hdf5Wfe(params, sim_index);
  }

  mainLog() << "ApertureFactory error: Unsupported aperture type." << std::endl;
  return NULL;
}
