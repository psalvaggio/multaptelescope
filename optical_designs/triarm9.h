// File Description
// Author: Philip Salvaggio

#ifndef TRIARM9_H
#define TRIARM9_H

#include "aperture.h"
#include "optical_designs/triarm9_parameters.pb.h"

#include <opencv2/core/core.hpp>

class Triarm9 : public Aperture {
 public:
  explicit Triarm9(const mats::Simulation& params);

  virtual ~Triarm9();

 // Virtual functions from Aperture
 private:
  void GetApertureTemplate(cv::Mat_<double>* output) const override;

  void GetOpticalPathLengthDiff(double image_height,
                                double angle,
                                cv::Mat_<double>* output) const override;

 // Class constants
 private:
  const static int kNumArms = 3;
  const static int kNumApertures = 9;

 private:
  Triarm9Parameters triarm9_params_;

  std::unique_ptr<Aperture> compound_aperture_;
};
REGISTER_APERTURE(Triarm9, TRIARM9)

#endif  // TRIARM9_H
