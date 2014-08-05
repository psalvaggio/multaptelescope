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
#include "base/scoped_ptr.h"
#include "optical_designs/triarm3_parameters.pb.h"

#include <opencv/cv.h>
#include <vector>

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

 private:
  Triarm3Parameters triarm3_params_;

  scoped_ptr<Aperture> compound_aperture_;
};

#endif  // TRIARM3_H
