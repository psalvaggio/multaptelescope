// File Description
// Author: Philip Salvaggio

#ifndef AXICON_H
#define AXICON_H

#include <opencv2/core/core.hpp>

#include "circular.h"
#include "optical_designs/axicon_parameters.pb.h"

namespace mats {

class Axicon : public Circular {
 public:
  explicit Axicon(const Simulation& params);

  virtual ~Axicon();

 // Virtual functions from Aperture
 private:
  void GetOpticalPathLengthDiff(double image_height,
                                double angle,
                                cv::Mat_<double>* output) const override;

 private:
  AxiconParameters axicon_params_;
};
REGISTER_APERTURE(Axicon, AXICON)

}

#endif  // AXICON_H
