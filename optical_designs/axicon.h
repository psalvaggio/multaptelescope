// File Description
// Author: Philip Salvaggio

#ifndef AXICON_H
#define AXICON_H

#include <opencv/cv.h>

#include "circular.h"
#include "optical_designs/axicon_parameters.pb.h"

class Axicon : public Circular {
 public:
  explicit Axicon(const mats::Simulation& params);

  virtual ~Axicon();

 // Virtual functions from Aperture
 private:
  void GetOpticalPathLengthDiff(cv::Mat_<double>* output) const override;

  void GetOpticalPathLengthDiffEstimate(
      cv::Mat_<double>* output) const override;

 private:
  AxiconParameters axicon_params_;
};
REGISTER_APERTURE(Axicon, AXICON)

#endif  // AXICON_H
