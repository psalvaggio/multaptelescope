// File Description
// Author: Philip Salvaggio

#include "aperture.h"

#include "base/pupil_function.h"
#include "base/opencv_utils.h"
#include "io/logging.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>

using mats::SimulationConfig;
using mats::Simulation;
using mats::PupilFunction;
using mats::ApertureParameters;
using namespace std;
using namespace cv;

Aperture::Aperture(const SimulationConfig& params, int sim_index)
    : params_(),
      aperture_params_(params.simulation(sim_index).aperture_params()),
      aberrations_(),
      mask_(),
      opd_(),
      opd_est_(),
      encircled_diameter_(-1),
      fill_factor_(-1) {

  params_.CopyFrom(params);
  params_.clear_simulation();
  params_.add_simulation()->CopyFrom(params.simulation(sim_index));

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


double Aperture::fill_factor() const {
  if (fill_factor_ >= 0) return fill_factor_;

  if (aperture_params_.has_fill_factor()) {
    fill_factor_ = aperture_params_.fill_factor();
  } else {
    Mat mask = GetApertureMask();
    Scalar sum = cv::sum(mask);
    fill_factor_ = sum[0] / (M_PI * pow(params().array_size() / 2, 2));
  }
  return fill_factor_;
}


double Aperture::encircled_diameter() const {
  if (encircled_diameter_ < 0) {
    encircled_diameter_ = GetEncircledDiameter();
  }
  return encircled_diameter_;
}

double Aperture::GetEncircledDiameter() const {
  return aperture_params_.encircled_diameter();
}


void Aperture::GetPupilFunction(double wavelength,
                                PupilFunction* pupil) const {
  GetPupilFunctionHelper(wavelength, pupil, [this] (Mat_<double>* output) {
    GetWavefrontError(output);
  });
}

void Aperture::GetPupilFunctionEstimate(double wavelength,
                                        PupilFunction* pupil) const {
  GetPupilFunctionHelper(wavelength, pupil, [this] (Mat_<double>* output) {
    GetWavefrontErrorEstimate(output);
  });
}

void Aperture::GetPupilFunctionHelper(
    double wavelength,
    PupilFunction* pupil,
    function<void(Mat_<double>*)> wfe_generator) const {
  // We want to pupil function to take up the center half of the array. This
  // is a decent tradeoff so the user has enough resolution to upsample and a
  // decent range (2x) over which to upsample.
  const size_t kSize = params_.array_size();
  int target_diameter = kSize / 2;

  Range mid_range(kSize / 4, kSize / 4 + target_diameter);

  // Set the scale of the pupil function, so the user can rescale the
  // function with physical units if they need to.
  pupil->set_meters_per_pixel(encircled_diameter() / target_diameter);

  Mat_<double> mask(params_.array_size(), params_.array_size()),
               wfe(params_.array_size(), params_.array_size());
  mask = 0; wfe = 0;
  Mat_<double> mask_center = mask(mid_range, mid_range),
               wfe_center = wfe(mid_range, mid_range);

  GetApertureMask(&mask_center);
  wfe_generator(&wfe_center);

  // The wavefront error is in units of the reference wavelength. So, we need
  // to scale it to be in terms of the requested wavelength.
  wfe *= params_.reference_wavelength() / wavelength;

  // Create the aberrated pupil function. This consists of putting the
  // wavefront error into the phase of the complex pupil function.
  Mat& pupil_real = pupil->real_part();
  Mat& pupil_imag = pupil->imaginary_part();
  pupil_real = Mat::zeros(kSize, kSize, CV_64FC1);
  pupil_imag = Mat::zeros(kSize, kSize, CV_64FC1);

  for (int i = mid_range.start; i < mid_range.end; i++) {
    for (int j = mid_range.start; j < mid_range.end; j++) {
      pupil_real.at<double>(i, j) = mask(i, j) * cos(2 * M_PI * wfe(i, j));
      pupil_imag.at<double>(i, j) = mask(i, j) * sin(2 * M_PI * wfe(i, j));
    }
  }
}


// Accessors for wavefront error.
Mat Aperture::GetWavefrontError(int size) const {
  size = (size == -1) ? params_.array_size() : size;
  if (opd_.rows != size) {
    opd_.create(size, size);
    GetOpticalPathLengthDiff(&opd_);
  }
  return opd_;
}

void Aperture::GetWavefrontError(cv::Mat_<double>* output) const {
  if (output->rows > 0 && output->rows != opd_.rows) {
    opd_.create(output->rows, output->rows);
    GetOpticalPathLengthDiff(&opd_);
  }
  opd_.copyTo(*output);
}


// Accessors for wavefron error estimates.
Mat Aperture::GetWavefrontErrorEstimate(int size) const {
  size = (size == -1) ? params_.array_size() : size;
  if (opd_est_.rows != size) {
    opd_est_.create(size, size);
    GetOpticalPathLengthDiffEstimate(&opd_est_);
  }
  return opd_est_;
}

void Aperture::GetWavefrontErrorEstimate(cv::Mat_<double>* output) const {
  if (output->rows > 0 && output->rows != opd_est_.rows) {
    opd_est_.create(output->rows, output->rows);
    GetOpticalPathLengthDiffEstimate(&opd_est_);
  }
  opd_est_.copyTo(*output);
}


// Accessors for aperture masks.
Mat Aperture::GetApertureMask(int size) const {
  size = (size == -1) ? params_.array_size() : size;
  if (mask_.rows != params_.array_size()) {
    mask_.create(size, size);
    GetApertureTemplate(&mask_);
  }
  return mask_;
}

void Aperture::GetApertureMask(cv::Mat_<double>* output) const {
  if (output->rows > 0 && output->rows != mask_.rows) {
    mask_.create(output->rows, output->rows);
    GetApertureTemplate(&mask_);
  }
  mask_.copyTo(*output);
}


// DEPRECATED TO BE REMOVED
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


// ApertureFactory Implementation.
Aperture* ApertureFactory::Create(const mats::SimulationConfig& params, 
                                  int sim_index) {
  if (sim_index > params.simulation_size()) {
    mainLog() << "ApertureFactory error: Given simulation index out of bounds."
              << endl;
    return NULL;
  }

  ApertureParameters::ApertureType ap_type =
      params.simulation(sim_index).aperture_params().type();
  Aperture* ap = ApertureFactoryImpl::Create(
      ApertureParameters::ApertureType_Name(ap_type), params, sim_index);
  if (ap) return ap;

  mainLog() << "ApertureFactory error: Unsupported aperture type." << endl;
  return NULL;
}
