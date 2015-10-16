// A representation of a circular aperture.
// Author: Philip Salvaggio

#ifndef CIRCULAR_H
#define CIRCULAR_H

#include <opencv/cv.h>

#include "aperture.h"

class Circular : public Aperture {
 public:
  explicit Circular(const mats::Simulation& params);

  virtual ~Circular();

 // Virtual functions from Aperture
 private:
  void GetApertureTemplate(cv::Mat_<double>* output) const override;

  void GetOpticalPathLengthDiff(cv::Mat_<double>* output) const override;

  void GetOpticalPathLengthDiffEstimate(
      cv::Mat_<double>* output) const override;
};
REGISTER_APERTURE(Circular, CIRCULAR)

#endif  // CIRCULAR_H
