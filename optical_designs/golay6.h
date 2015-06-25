// Golay-6 sparse aperture design. The coordinates I used are not exact, since I
// measured them off a figure in one of Dr. Robert Fiete's papers. The real
// coordinates were not published in the original paper (Golay pls).
//
// Author: Philip Salvaggio

#ifndef GOLAY6_H
#define GOLAY6_H

#include "aperture.h"
#include "optical_designs/golay6_parameters.pb.h"

#include <opencv/cv.h>
#include <memory>
#include <vector>

class Golay6 : public Aperture {
 public:
  Golay6(const mats::SimulationConfig& params, int sim_index);

  virtual ~Golay6();

 // Virtual functions from Aperture
 private:
  cv::Mat GetApertureTemplate() const override;

  cv::Mat GetOpticalPathLengthDiff() const override;

  cv::Mat GetOpticalPathLengthDiffEstimate() const override;

 private:
  Golay6Parameters golay6_params_;

  std::unique_ptr<Aperture> compound_aperture_;
};
REGISTER_APERTURE(Golay6, GOLAY6)

#endif  // CASSEGRAIN_RING_H
