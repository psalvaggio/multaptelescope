// File Description
// Author: Philip Salvaggio

#ifndef TELESCOPE_H
#define TELESCOPE_H

#include "base/simulation_config.pb.h"

#include <opencv/cv.h>

#include <memory>
#include <vector>

class Aperture;

namespace mats {

class Detector;
class DetectorParameters;

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

  const SimulationConfig& sim_config() const { return sim_config_; }
  const Simulation& simulation() const;

  bool parallelism() const { return parallelism_; }
  void set_parallelism(bool use) { parallelism_ = use; }

  bool include_detector_footprint() const {
    return include_detector_footprint_;
  }
  void set_include_detector_footprint(bool include) {
    include_detector_footprint_ = include;
  }

  // Gets the effective Q (lambda * F# / p) of the system over a bandpass
  //
  // Arguments:
  //  wavelengths         The bandpass wavelengths
  //  spectral_weighting  The weighting over the bandpass (must sum to 1)
  //
  // Returns:
  //  The effective Q of the system
  double EffectiveQ(const std::vector<double>& wavelengths,
                    const std::vector<double>& spectral_weighting) const;

  // Simulate an image through the telescope.
  //
  // Arguments:
  //  radiance    The input spectral radiance.
  //  wavelength  The wavelengths associated with each spectral band in
  //              radiance.
  //  image       The output of each band of the detector.
  void Image(const std::vector<cv::Mat>& radiance,
             const std::vector<double>& wavelength,
             std::vector<cv::Mat>* image);

  // Computes the monochromatic OTF of the telescope at the provided
  // wavelengths.
  //
  // Arguments:
  //  wavelengths  Desired wavelengths for MTF computation [m]
  //  otf          Output: Will be populated with the OTFs. The size will be
  //                       the array_size() field in SimulationConfig. The OTF
  //                       will be centered at (0,0) and go to detector Nyquist.
  //                       Frequency units are [cyc/pixel].
  void ComputeOtf(const std::vector<double>& wavelengths,
                  double image_height,
                  double angle,
                  std::vector<cv::Mat>* otf) const;

  // Compute the effective OTF over a bandpass.
  //
  // Arguments:
  //  wavelengths  Wavelength domain of the bandpass [m]
  //  weights      The spectral weighting function of the bandpass.
  //  otf          Output: Output OTF (see ComputeOtf())
  void EffectiveOtf(const std::vector<double>& wavelengths,
                    const std::vector<double>& weights,
                    double image_height,
                    double angle,
                    cv::Mat* otf) const;

  // Get the transmission spectrum of the telescope optics.
  //
  // Arguments:
  //  wavelengths   The desired wavelengths [m]
  //  transmission  Output: The transmission at each wavelength.
  void GetTransmissionSpectrum(const std::vector<double>& wavelengths,
                               std::vector<double>* transmission) const;

  void IsoplanaticRegion(int radial_idx,
                         int angular_idx,
                         cv::Mat_<double>* isoplanatic_region) const;
 private:
  void ComputeApertureOtf(
      const std::vector<double>& wavelengths,
      double image_height,
      double angle,
      std::vector<cv::Mat_<std::complex<double>>>* otf) const;

  void DegradeImage(const cv::Mat& radiance,
                    const cv::Mat& spectral_otf,
                    cv::Mat* degraded) const;

  void OtfDegrade(const cv::Mat& radiance_fft,
                  const cv::Mat& spectral_otf,
                  cv::Mat* degraded) const;

  void GetImagingRegion(const cv::Mat& radiance,
                        cv::Mat* roi) const;

 private:
  mats::SimulationConfig sim_config_;
  std::unique_ptr<Aperture> aperture_;
  std::unique_ptr<Detector> detector_;
  bool include_detector_footprint_;
  bool parallelism_;
};

}

#endif  // TELESCOPE_H
