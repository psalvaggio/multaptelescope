// File Description
// Author: Philip Salvaggio

#ifndef COMPOUND_APERTURE_H
#define COMPOUND_APERTURE_H

#include "aperture.h"
#include "base/simulation_config.pb.h"
#include "optical_designs/compound_aperture_parameters.pb.h"

#include <memory>
#include <opencv/cv.h>

class CompoundAperture : public Aperture {
 public:
  CompoundAperture(const mats::SimulationConfig& params, int sim_index);

  virtual ~CompoundAperture();
  
 private:
  virtual double GetEncircledDiameter() const;

  virtual cv::Mat GetApertureTemplate() const;

  virtual cv::Mat GetOpticalPathLengthDiff() const;

  virtual cv::Mat GetOpticalPathLengthDiffEstimate() const;

 private:
  CompoundApertureParameters compound_params_;

  mutable std::vector<std::unique_ptr<Aperture>> apertures_;
  std::vector<mats::SimulationConfig> sim_configs_;

 private:  // Cache variables
  mutable cv::Mat opd_;
  mutable cv::Mat opd_est_;
};

#endif  // COMPOUND_APERTURE_H
