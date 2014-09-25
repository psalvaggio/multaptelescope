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

 // Virtual functions from Aperture
 private:
  virtual cv::Mat GetApertureTemplate() const;

  virtual cv::Mat GetOpticalPathLengthDiff() const;

  virtual cv::Mat GetOpticalPathLengthDiffEstimate() const;
};

#endif  // CASSEGRAIN_H
