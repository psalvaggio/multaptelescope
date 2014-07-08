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

  double diameter() const { return diameter_; }

 // Virtual functions from Aperture
 private:
  virtual cv::Mat GetApertureTemplate();

  virtual cv::Mat GetOpticalPathLengthDiff();

  virtual cv::Mat GetOpticalPathLengthDiffEstimate();

 private:
  // The diameter of the primary mirror [m]
  double diameter_;

  // The values for piston/tip/tilt [waves]
  double ptt_vals_[3];
};

#endif  // CIRCULAR_H
