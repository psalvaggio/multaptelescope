// A representation of a circular aperture.
// Author: Philip Salvaggio

#ifndef CIRCULAR_H
#define CIRCULAR_H

#include <opencv2/core/core.hpp>

#include "aperture.h"

namespace mats {

class Circular : public Aperture {
 public:
  explicit Circular(const Simulation& params);

  virtual ~Circular();

 // Virtual functions from Aperture
 private:
  void GetApertureTemplate(cv::Mat_<double>* output) const override;
};
REGISTER_APERTURE(Circular, CIRCULAR)

}

#endif  // CIRCULAR_H
