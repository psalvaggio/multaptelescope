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
  explicit Golay6(const mats::Simulation& params);

  virtual ~Golay6();

 // Virtual functions from Aperture
 private:
  void GetApertureTemplate(cv::Mat_<double>* output) const override;

  void GetOpticalPathLengthDiff(cv::Mat_<double>* output) const override;

  void GetOpticalPathLengthDiffEstimate(
      cv::Mat_<double>* output) const override;

 private:
  Golay6Parameters golay6_params_;

  std::unique_ptr<Aperture> compound_aperture_;
};
REGISTER_APERTURE(Golay6, GOLAY6)

#endif  // CASSEGRAIN_RING_H
