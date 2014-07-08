// A representation of the aperture of a Cassegrainian telescope.
// Author: Philip Salvaggio

#ifndef CASSEGRAIN_H
#define CASSEGRAIN_H

#include <opencv/cv.h>

#include "aperture.h"

class Cassegrain : public Aperture {
 public:
  Cassegrain(const mats::SimulationConfig& params, int sim_index);

  virtual ~Cassegrain();

  double diameter() const { return diameter_; }
  double secondary_diameter() const { return secondary_diameter_; }

 // Virtual functions from Aperture
 private:
  virtual cv::Mat GetApertureTemplate();

  virtual cv::Mat GetOpticalPathLengthDiff();

  virtual cv::Mat GetOpticalPathLengthDiffEstimate();

 private:
  // The diameter of the primary mirror [m]
  double diameter_;

  // The diameter of the secondary mirror [m]
  double secondary_diameter_;

  cv::Mat opd_;
  cv::Mat opd_est_;
};

#endif  // CASSEGRAIN_H
