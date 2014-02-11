// File Description
// Author: Philip Salvaggio

#ifndef TELESCOPE_H
#define TELESCOPE_H

#include "base/scoped_ptr.h"
#include <opencv/cv.h>

#include <vector>

class Aperture;
class ApertureParameters;

namespace mats {

class Detector;
class DetectorParameters;
class SimulationConfig;

class Telescope {
 public:
  Telescope(const SimulationConfig& sim_config,
            int sim_index,
            const ApertureParameters& ap_params,
            const DetectorParameters& det_params);

  virtual ~Telescope();

  // Get the focal length of the telescope. [m]
  double FocalLength() const;

  // Get the F# of the system. [unitless]
  double FNumber() const;

  // Get the G# of the telescope [sr^-1]. This describes the relationship
  // between the radiance reaching the optics and the irradiance onto the
  // detector.
  //
  // Arguments:
  //   lambda  The wavelength of interest [m]
  double GNumber(double lambda) const;

  const Detector* detector() const { return detector_.get(); }
  Detector* detector() { return detector_.get(); }
  Aperture* aperture() { return aperture_.get(); }

  void Image(const std::vector<cv::Mat>& radiance,
             const std::vector<double>& wavelength,
             std::vector<cv::Mat>* image,
             std::vector<cv::Mat>* otf = NULL);

  void ComputeOtf(const std::vector<double>& wavelengths,
                  std::vector<cv::Mat>* otf);

  void GetTransmissionSpectrum(const std::vector<double>& wavelengths,
                               std::vector<double>* transmission) const;

 private:
  void ComputeApertureOtf(const std::vector<double>& wavelengths,
                          std::vector<cv::Mat>* otf);
 private:
  scoped_ptr<Aperture> aperture_;
  scoped_ptr<Detector> detector_;
};

}

#endif  // TELESCOPE_H
