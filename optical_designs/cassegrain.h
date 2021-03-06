// A representation of the aperture of a Cassegrainian telescope.
// Author: Philip Salvaggio

#ifndef CASSEGRAIN_H
#define CASSEGRAIN_H

#include <opencv2/core/core.hpp>

#include "aperture.h"

namespace mats {

class Cassegrain : public Aperture {
 public:
  explicit Cassegrain(const Simulation& params);

  virtual ~Cassegrain();

 // Virtual functions from Aperture
 private:
  void GetApertureTemplate(cv::Mat_<double>* output) const override;
};
REGISTER_APERTURE(Cassegrain, CASSEGRAIN)

}

#endif  // CASSEGRAIN_H
