// File Description
// Author: Philip Salvaggio

#ifndef OPTICAL_VORTEX_H
#define OPTICAL_VORTEX_H

#include <opencv/cv.h>

#include "circular.h"
#include "optical_designs/optical_vortex_parameters.pb.h"

class OpticalVortex : public Circular {
 public:
  OpticalVortex(const mats::SimulationConfig& params, int sim_index);

  virtual ~OpticalVortex();

 // Virtual functions from Aperture
 private:
  void GetOpticalPathLengthDiff(cv::Mat_<double>* output) const override;

  void GetOpticalPathLengthDiffEstimate(
      cv::Mat_<double>* output) const override;

 private:
  OpticalVortexParameters vortex_params_;
};
REGISTER_APERTURE(OpticalVortex, OPTICAL_VORTEX)

#endif  // OPTICAL_VORTEX_H
