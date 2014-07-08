// The Tri-arm3 design is a truncated version of the Tri-arm9 design.
// To match with my experimental conditions (not a good reason, I know),
// this design will be treated as a masked primary mirror. Piston/tip/tilt
// errors can be independent for each supaperture, but high-order aberrations
// will be treated globally.
//
// Author: Philip Salvaggio

#ifndef TRIARM3_H
#define TRIARM3_H

#include "aperture.h"
#include "optical_designs/triarm3_parameters.pb.h"

#include <opencv/cv.h>
#include <vector>


class Cassegrain;

class Triarm3 : public Aperture {
 public:
  Triarm3(const mats::SimulationConfig& params, int sim_index);

  virtual ~Triarm3();

  // Exports to a Zemax UDA (User-Defined Aperture) file.
  // Units are in millimeters.
  void ExportToZemax(const std::string& aperture_filename,
                     const std::string& obstruction_filename) const;

 // Virtual functions from Aperture
 private:
  virtual cv::Mat GetApertureTemplate();

  virtual cv::Mat GetOpticalPathLengthDiff();

  virtual cv::Mat GetOpticalPathLengthDiffEstimate();

  cv::Mat OpticalPathLengthDiffPtt(const std::vector<double>& ptt_vals);

 // Class constants
 private:
  const static int kNumArms = 3;
  const static int kNumApertures = 3;

 private:
  Triarm3Parameters triarm3_params_;

  double diameter_;
  double subap_diameter_;
  double subap_secondary_diameter_;

  std::vector<int> subaperture_offsets_;

  cv::Mat mask_;
  cv::Mat opd_;
  cv::Mat opd_est_;
};

#endif  // TRIARM3_H
