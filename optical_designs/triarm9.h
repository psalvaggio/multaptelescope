// File Description
// Author: Philip Salvaggio

#ifndef TRIARM9_H
#define TRIARM9_H

#include "aperture.h"
#include "optical_designs/triarm9_parameters.pb.h"

#include <opencv/cv.h>
#include <vector>


class Cassegrain;

class Triarm9 : public Aperture {
 public:
  Triarm9(const mats::SimulationConfig& params, int sim_index);

  virtual ~Triarm9();

  // Exports to a Zemax UDA (User-Defined Aperture) file.
  // Units are in millimeters.
  void ExportToZemax(const std::string& aperture_filename,
                     const std::string& obstruction_filename) const;

 // Virtual functions from Aperture
 private:
  virtual cv::Mat GetApertureTemplate() const;

  virtual cv::Mat GetOpticalPathLengthDiff() const;

  virtual cv::Mat GetOpticalPathLengthDiffEstimate() const;

  cv::Mat OpticalPathLengthDiffPtt(const std::vector<double>& ptt_vals) const;

 // Class constants
 private:
  const static int kNumArms = 3;
  const static int kNumApertures = 9;

 private:
  Triarm9Parameters triarm9_params_;

  double diameter_;
  double subap_diameter_;
  double subap_secondary_diameter_;

  std::vector<int> subaperture_offsets_;
  mutable std::vector<double> ptt_vals_;
};

#endif  // TRIARM9_H
