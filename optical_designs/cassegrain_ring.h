// This design is a ring of apertures. The number of apertures and their fill
// factors are adjustable. To match with my experimental conditions (not a good
// reason, I know), this design will be treated as a masked primary mirror.
//
// Author: Philip Salvaggio

#ifndef CASSEGRAIN_RING_H
#define CASSEGRAIN_RING_H

#include "aperture.h"
#include "optical_designs/cassegrain_ring_parameters.pb.h"

#include <opencv/cv.h>
#include <memory>
#include <vector>

class CassegrainRing : public Aperture {
 public:
  explicit CassegrainRing(const mats::Simulation& params);

  virtual ~CassegrainRing();

  // Exports to a Zemax UDA (User-Defined Aperture) file.
  // Units are in millimeters.
  void ExportToZemax(const std::string& aperture_filename,
                     const std::string& obstruction_filename) const;

 // Virtual functions from Aperture
 private:
  void GetApertureTemplate(cv::Mat_<double>* output) const override;

  void GetOpticalPathLengthDiff(double image_height,
                                double angle,
                                cv::Mat_<double>* output) const override;
 private:
  CassegrainRingParameters ring_params_;

  std::unique_ptr<Aperture> compound_aperture_;
};
REGISTER_APERTURE(CassegrainRing, CASSEGRAIN_RING)

#endif  // CASSEGRAIN_RING_H
