// A representation of a circular aperture.
// Author: Philip Salvaggio

#ifndef CIRCULAR_H
#define CIRCULAR_H

#include <opencv/cv.h>

#include "aperture.h"

class Circular : public Aperture {
 public:
  Circular(const mats::SimulationConfig& params, int sim_index);

  virtual ~Circular();

 // Virtual functions from Aperture
 private:
  cv::Mat GetApertureTemplate() const override;

  cv::Mat GetOpticalPathLengthDiff() const override;

  cv::Mat GetOpticalPathLengthDiffEstimate() const override;
};
REGISTER_APERTURE(Circular, CIRCULAR)

#endif  // CIRCULAR_H
