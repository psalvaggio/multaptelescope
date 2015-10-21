// File Description
// Author: Philip Salvaggio

#ifndef COMPOUND_APERTURE_H
#define COMPOUND_APERTURE_H

#include "aperture.h"
#include "base/simulation_config.pb.h"
#include "optical_designs/compound_aperture_parameters.pb.h"

#include <memory>
#include <opencv2/core/core.hpp>

class CompoundAperture : public Aperture {
 public:
  CompoundAperture(const mats::Simulation& params);

  virtual ~CompoundAperture();
  
 private:
  double GetEncircledDiameter() const override;

  void GetApertureTemplate(cv::Mat_<double>* output) const override;

  void GetOpticalPathLengthDiff(double image_height,
                                double angle,
                                cv::Mat_<double>* output) const override;

  void GenerateSubapertureHelper(
      int array_size,
      std::vector<cv::Mat_<double>>* subaps,
      std::function<void(const Aperture*, cv::Mat_<double>*)> subap_generator)
      const;

  void RotateArray(cv::Mat_<double>* array) const;

 private:
  CompoundApertureParameters compound_params_;

  mutable std::vector<std::unique_ptr<Aperture>> apertures_;
  std::vector<mats::Simulation> sim_configs_;
};
REGISTER_APERTURE(CompoundAperture, COMPOUND)

#endif  // COMPOUND_APERTURE_H
