// A representation of the aperture of a Cassegrainian telescope.
// Author: Philip Salvaggio

#ifndef CASSEGRAIN_H
#define CASSEGRAIN_H

#include <opencv/cv.h>

#include "aperture.h"

class Cassegrain : public Aperture {
 public:
  explicit Cassegrain(const mats::Simulation& params);

  virtual ~Cassegrain();

 // Virtual functions from Aperture
 private:
  void GetApertureTemplate(cv::Mat_<double>* output) const override;
};
REGISTER_APERTURE(Cassegrain, CASSEGRAIN)

#endif  // CASSEGRAIN_H
