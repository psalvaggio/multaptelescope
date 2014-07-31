// File Description
// Author: Philip Salvaggio

#ifndef COMPOUND_APERTURE_H
#define COMPOUND_APERTURE_H

#include "aperture.h"
#include "base/simulation_config.pb.h"
#include "optical_designs/compound_aperture_parameters.pb.h"

#include <opencv/cv.h>

class CompoundAperture : public Aperture {
 public:
  CompoundAperture(const mats::SimulationConfig& params, int sim_index);

  virtual ~CompoundAperture();

  virtual double encircled_diameter();

 private:
  virtual cv::Mat GetApertureTemplate();

  virtual cv::Mat GetOpticalPathLengthDiff();

  virtual cv::Mat GetOpticalPathLengthDiffEstimate();

 private:
  CompoundApertureParameters compound_params_;

  std::vector<Aperture*> apertures_;
  std::vector<mats::SimulationConfig> sim_configs_;

  cv::Mat mask_;
  cv::Mat opd_;
  cv::Mat opd_est_;
};

#endif  // COMPOUND_APERTURE_H
