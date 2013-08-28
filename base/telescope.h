// File Description
// Author: Philip Salvaggio

#ifndef TELESCOPE_H
#define TELESCOPE_H

#include "base/scoped_ptr.h"
#include "base/detector.h"
#include "base/simulation_config.pb.h"
#include "optical_designs/aperture_parameters.pb.h"

#include <vector>

class Aperture;

namespace mats {

class Telescope {
 public:
  Telescope(const SimulationConfig& sim_config,
            int sim_index,
            const ApertureParameters& ap_params,
            const DetectorParameters& det_params);


  double FocalLength() const;

  void Image(const std::vector<cv::Mat>& radiance,
             const std::vector<double>& wavelength,
             std::vector<cv::Mat>* image,
             std::vector<cv::Mat>* otf = NULL);

  void ComputeOtf(const std::vector<double>& wavelengths,
                  std::vector<cv::Mat>* otf);

 private:
  void ComputeApertureOtf(const std::vector<double>& wavelengths,
                          std::vector<cv::Mat>* otf);
 private:
  scoped_ptr<Aperture> aperture_;
  scoped_ptr<Detector> detector_;
};

}

#endif  // TELESCOPE_H
