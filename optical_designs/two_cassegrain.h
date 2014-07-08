// File Description
// Author: Philip Salvaggio

#ifndef TWO_CASSEGRAIN_H
#define TWO_CASSEGRAIN_H

#include "aperture.h"
#include "optical_designs/two_cassegrain_parameters.pb.h"

#include <opencv/cv.h>
#include <vector>


class Cassegrain;

class TwoCassegrain : public Aperture {
 public:
  TwoCassegrain(const mats::SimulationConfig& params, int sim_index);

  virtual ~TwoCassegrain();

 // Virtual functions from Aperture
 private:
  virtual cv::Mat GetApertureTemplate();

  virtual cv::Mat GetOpticalPathLengthDiff();

  virtual cv::Mat GetOpticalPathLengthDiffEstimate();

  cv::Mat OpticalPathLengthDiffPtt(const std::vector<double>& ptt_vals);

 // Class constants
 private:
  const static int kNumArms = 2;
  const static int kNumApertures = 2;

 private:
  TwoCassegrainParameters two_cassegrain_params_;

  double diameter_;
  double subap_diameter_;
  double subap_secondary_diameter_;

  std::vector<int> subaperture_offsets_;
  std::vector<double> ptt_vals_;

  cv::Mat mask_;
  cv::Mat opd_;
  cv::Mat opd_est_;
};

#endif  // TWO_CASSEGRAIN_H
