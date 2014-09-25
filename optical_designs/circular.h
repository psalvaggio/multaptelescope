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
  virtual cv::Mat GetApertureTemplate() const;

  virtual cv::Mat GetOpticalPathLengthDiff() const;

  virtual cv::Mat GetOpticalPathLengthDiffEstimate() const;
};

#endif  // CIRCULAR_H
