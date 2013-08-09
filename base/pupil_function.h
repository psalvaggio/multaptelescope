// File Description
// Author: Philip Salvaggio

#ifndef PUPIL_FUNCTION_H
#define PUPIL_FUNCTION_H

#include "macros.h"

#include <opencv/cv.h>

namespace mats {

class PupilFunction {
 public:
  PupilFunction();
  ~PupilFunction();

  cv::Mat& real_part() { return pupil_real_; }
  cv::Mat& imaginary_part() { return pupil_imag_; }
  const cv::Mat& real_part() const { return pupil_real_; }
  const cv::Mat& imaginary_part() const { return pupil_imag_; }

  cv::Mat magnitude() const;
  cv::Mat phase() const;

  double meters_per_pixel() const { return meters_per_pixel_; }
  void set_meters_per_pixel(double meters_per_pixel) {
    meters_per_pixel_ = meters_per_pixel;
  }

  cv::Mat PointSpreadFunction();
  cv::Mat OpticalTransferFunction();
  cv::Mat ModulationTransferFunction();

 private:
  cv::Mat pupil_real_;
  cv::Mat pupil_imag_;
  double meters_per_pixel_;

  NO_COPY_OR_ASSIGN(PupilFunction);
};

}

#endif  // PUPIL_FUNCTION_H
