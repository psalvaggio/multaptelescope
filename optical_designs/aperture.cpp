// File Description
// Author: Philip Salvaggio

#include "aperture.h"

#include "base/pupil_function.h"
#include "base/opencv_utils.h"
#include "base/zernike_aberrations.h"
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
using mats::ZernikeCoefficient;
using namespace std;
using namespace cv;

Aperture::Aperture(const Simulation& params)
    : sim_params_(params),
      aperture_params_(params.aperture_params()),
      on_axis_aberrations_(),
      off_axis_aberrations_(),
      mask_(),
      on_axis_opd_(),
      encircled_diameter_(-1),
      fill_factor_(-1) {
  int size = 0;
  for (int i = 0; i < aperture_params_.aberration_size(); i++) {
    size = std::max(size,
        static_cast<int>(aperture_params_.aberration(i).type()) + 1);
  }

  on_axis_aberrations_.resize(size, 0);
  off_axis_aberrations_.resize(size, 0);
  for (int i = 0; i < aperture_params_.aberration_size(); i++) {
    auto type = aperture_params_.aberration(i).type();
    if (IsOffAxis(type)) {
      off_axis_aberrations_[type] = aperture_params_.aberration(i).value();
    } else {
      on_axis_aberrations_[type] = aperture_params_.aberration(i).value();
    }
  }
}

Aperture::~Aperture() {}


double Aperture::fill_factor() const {
  if (fill_factor_ >= 0) return fill_factor_;

  if (aperture_params_.has_fill_factor()) {
    fill_factor_ = aperture_params_.fill_factor();
  } else {
    Mat mask = GetApertureMask(512);
    Scalar sum = cv::sum(mask);
    fill_factor_ = sum[0] / (M_PI * pow(mask.rows / 2, 2));
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


bool Aperture::IsOffAxis(ZernikeCoefficient::AberrationType type) const {
  return type == ZernikeCoefficient::ASTIG_X ||
         type == ZernikeCoefficient::ASTIG_Y ||
         type == ZernikeCoefficient::COMA_X ||
         type == ZernikeCoefficient::COMA_Y;
}


bool Aperture::HasAberration(ZernikeCoefficient::AberrationType type) const {
  if (IsOffAxis(type)) {
    return off_axis_aberrations_.size() > type &&
           abs(off_axis_aberrations_[type]) > 1e-10;
  } else {
    return on_axis_aberrations_.size() > type &&
           abs(off_axis_aberrations_[type]) > 1e-10;
  }
}


double Aperture::GetAberration(ZernikeCoefficient::AberrationType type) const {
  if (HasAberration(type)) {
    return IsOffAxis(type) ? off_axis_aberrations_[type]
                           : on_axis_aberrations_[type];
  }
  return 0;
}


bool Aperture::HasOffAxisAberration() const {
  return HasAberration(ZernikeCoefficient::COMA) ||
         HasAberration(ZernikeCoefficient::ASTIGMATISM);
}


void Aperture::GetPupilFunction(double wavelength,
                                double image_height,
                                double angle,
                                PupilFunction* pupil) const {
  vector<double> wavelengths{wavelength};
  vector<PupilFunction> pupils;
  GetPupilFunction(wavelengths, image_height, angle,
                   pupil->size(), pupil->reference_wavelength(), &pupils);
  *pupil = move(pupils[0]);
}

void Aperture::GetPupilFunction(const vector<double>& wavelength,
                                double image_height,
                                double angle,
                                int size,
                                double reference_wavelength,
                                vector<PupilFunction>* output) const {

  // We want to pupil function to take up the center half of the array. This
  // is a decent tradeoff so the user has enough resolution to upsample and a
  // decent range (2x) over which to upsample.
  int target_diameter = size / 2;
  Range mid_range(size / 4, size / 4 + target_diameter);

  Mat_<double> mask(target_diameter, target_diameter),
               wfe(target_diameter, target_diameter);
  GetApertureMask(&mask);
  GetWavefrontError(image_height, angle, &wfe);

  for (const auto& lambda : wavelength) {
    // Set the scale of the pupil function, so the user can rescale the
    // function with physical units if they need to.
    output->emplace_back(size, reference_wavelength);
    auto& pupil = output->back();
    pupil.set_meters_per_pixel(encircled_diameter() / target_diameter);

    // Extract the center part of the arrays
    auto& pupil_real = pupil.real_part();
    auto& pupil_imag = pupil.imaginary_part();
    mask.copyTo(pupil_real(mid_range, mid_range));
    wfe.copyTo(pupil_imag(mid_range, mid_range));

    // The wavefront error is in units of the reference wavelength. So, we need
    // to scale it to be in terms of the requested wavelength.
    pupil_imag(mid_range, mid_range) *= reference_wavelength / lambda;

    for (int i = mid_range.start; i < mid_range.end; i++) {
      for (int j = mid_range.start; j < mid_range.end; j++) {
        double mask_val = pupil_real(i, j);
        pupil_real(i, j) = mask_val * cos(2 * M_PI * pupil_imag(i, j));
        pupil_imag(i, j) = mask_val * sin(2 * M_PI * pupil_imag(i, j));
      }
    }
  }
}


// Accessors for wavefront error.
Mat Aperture::GetWavefrontError(int size,
                                double image_height,
                                double angle) const {
  Mat_<double> opd(size, size);
  GetOpticalPathLengthDiff(image_height, angle, &opd);
  return opd;
}

void Aperture::GetWavefrontError(double image_height,
                                 double angle,
                                 cv::Mat_<double>* output) const {
  GetOpticalPathLengthDiff(image_height, angle, output);
}

// Accessors for aperture masks.
Mat Aperture::GetApertureMask(int size) const {
  if (size > 0 && size != mask_.rows) {
    mask_.create(size, size);
    GetApertureTemplate(&mask_);
  }
  Mat mask;
  mask_.copyTo(mask);
  return mask;
}


void Aperture::GetApertureMask(cv::Mat_<double>* output) const {
  if (output->rows > 0 && output->rows != mask_.rows) {
    mask_.create(output->rows, output->rows);
    GetApertureTemplate(&mask_);
  }
  mask_.copyTo(*output);
}


void Aperture::ZernikeWavefrontError(double image_height,
                                     double angle,
                                     cv::Mat_<double>* output) const {
  if (on_axis_opd_.rows != output->rows) {
    ZernikeAberrations& ab_factory(ZernikeAberrations::getInstance());
    ab_factory.aberrations(on_axis_aberrations_, output->rows, &on_axis_opd_);
  }
  on_axis_opd_.copyTo(*output);

  // If we are on axis, we're done.
  if (abs(image_height) < 1e-10) return;

  if (HasOffAxisAberration()) {
    double coma_mag = GetAberration(ZernikeCoefficient::COMA) * image_height;
    double astig_mag = GetAberration(ZernikeCoefficient::ASTIGMATISM) *
                       pow(image_height, 2);

    double cos_angle = cos(angle), sin_angle = sin(angle);
    vector<double> tmp_ab(8, 0);
    tmp_ab[ZernikeCoefficient::COMA_X] = cos_angle * coma_mag;
    tmp_ab[ZernikeCoefficient::COMA_Y] = sin_angle * coma_mag;
    tmp_ab[ZernikeCoefficient::ASTIG_X] = cos_angle * astig_mag;
    tmp_ab[ZernikeCoefficient::ASTIG_Y] = sin_angle * astig_mag;

    Mat_<double> off_axis;
    ZernikeAberrations& ab_factory(ZernikeAberrations::getInstance());
    ab_factory.aberrations(tmp_ab, output->rows, &off_axis);
    *output += off_axis;
  }
}

void Aperture::GetOpticalPathLengthDiff(double image_height,
                                        double angle,
                                        cv::Mat_<double>* output) const {
  ZernikeWavefrontError(image_height, angle, output);
}

// ApertureFactory Implementation.
Aperture* ApertureFactory::Create(const mats::Simulation& params) {
  auto ap_type = params.aperture_params().type();
  Aperture* ap = ApertureFactoryImpl::Create(
      ApertureParameters::ApertureType_Name(ap_type), params);
  if (ap) return ap;

  mainLog() << "ApertureFactory error: Unsupported aperture type." << endl;
  return NULL;
}
