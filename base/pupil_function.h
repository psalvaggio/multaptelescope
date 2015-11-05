// File Description
// Author: Philip Salvaggio

#ifndef PUPIL_FUNCTION_H
#define PUPIL_FUNCTION_H

#include "macros.h"

#include <complex>
#include <opencv2/core/core.hpp>

namespace mats {

class PupilFunction {
 public:
  PupilFunction(int size, double reference_wavelength);
  PupilFunction(PupilFunction&& other);
  ~PupilFunction();
  PupilFunction& operator=(PupilFunction&& other);

  cv::Mat_<double>& real_part() { return pupil_real_; }
  cv::Mat_<double>& imaginary_part() { return pupil_imag_; }
  const cv::Mat_<double>& real_part() const { return pupil_real_; }
  const cv::Mat_<double>& imaginary_part() const { return pupil_imag_; }

  cv::Mat magnitude() const;
  cv::Mat phase() const;

  int size() const { return pupil_real_.rows; }
  double reference_wavelength() const { return reference_wavelength_; }

  double meters_per_pixel() const { return meters_per_pixel_; }
  void set_meters_per_pixel(double meters_per_pixel) {
    meters_per_pixel_ = meters_per_pixel;
  }

  cv::Mat_<double> PointSpreadFunction();
  cv::Mat_<std::complex<double>> OpticalTransferFunction();
  cv::Mat_<double> ModulationTransferFunction();

 private:
  double reference_wavelength_;
  cv::Mat_<double> pupil_real_;
  cv::Mat_<double> pupil_imag_;
  double meters_per_pixel_;

  NO_COPY_OR_ASSIGN(PupilFunction);
};

}

#endif  // PUPIL_FUNCTION_H
