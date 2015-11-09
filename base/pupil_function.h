// A representation of a complex pupil function for an imaging system. The
// magnitude of this function is the aperture's transmission as a function
// of spatial location and the phase is the system's wavefront error as a
// function of position on the aperture, in units of waves. As such, a pupil
// function is also a function of wavelength and position in the image plane.
// Author: Philip Salvaggio

#ifndef PUPIL_FUNCTION_H
#define PUPIL_FUNCTION_H

#include <complex>
#include <opencv2/core/core.hpp>

namespace mats {

class PupilFunction {
 public:
  PupilFunction(int size, double reference_wavelength);
  PupilFunction(const PupilFunction& other) = delete;
  PupilFunction(PupilFunction&& other) = default;

  PupilFunction& operator=(const PupilFunction& other) = delete;
  PupilFunction& operator=(PupilFunction&& other) = default;

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
};

}

#endif  // PUPIL_FUNCTION_H
