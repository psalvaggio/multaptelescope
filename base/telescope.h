// File Description
// Author: Philip Salvaggio

#ifndef TELESCOPE_H
#define TELESCOPE_H

#include <opencv/cv.h>

#include <memory>
#include <vector>

class Aperture;

namespace mats {

class Detector;
class DetectorParameters;
class SimulationConfig;
class ApertureParameters;

class Telescope {
 public:
  Telescope(const SimulationConfig& sim_config,
            int sim_index,
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
  const Aperture* aperture() const { return aperture_.get(); }
  Aperture* aperture() { return aperture_.get(); }

  const SimulationConfig& sim_params() const;

  // Gets the effective Q (lambda * F# / p) of the system over a bandpass
  //
  // Arguments:
  //  wavelengths         The bandpass wavelengths
  //  spectral_weighting  The weighting over the bandpass (must sum to 1)
  //
  // Returns:
  //  The effective Q of the system
  double GetEffectiveQ(const std::vector<double>& wavelengths,
                       const std::vector<double>& spectral_weighting) const;

  // Simulate an image through the telescope.
  //
  // Arguments:
  //  radiance    The input spectral radiance.
  //  wavelength  The wavelengths associated with each spectral band in
  //              radiance.
  //  image       The output of each band of the detector.
  //  otf         The effective OTF for each of the spectral bands of the
  //              detector.
  void Image(const std::vector<cv::Mat>& radiance,
             const std::vector<double>& wavelength,
             std::vector<cv::Mat>* image,
             std::vector<cv::Mat>* otf = NULL);

  void ComputeOtf(const std::vector<double>& wavelengths,
                  std::vector<cv::Mat>* otf) const;

  void ComputeEffectiveOtf(const std::vector<double>& wavelengths,
                           const std::vector<double>& weights,
                           cv::Mat* otf) const;

  void GetTransmissionSpectrum(const std::vector<double>& wavelengths,
                               std::vector<double>* transmission) const;

 private:
  void ComputeApertureOtf(const std::vector<double>& wavelengths,
                          std::vector<cv::Mat>* otf) const;

 private:
  std::unique_ptr<Aperture> aperture_;
  std::unique_ptr<Detector> detector_;
};

}

#endif  // TELESCOPE_H
