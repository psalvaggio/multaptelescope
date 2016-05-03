// File Description
// Author: Philip Salvaggio

#ifndef TELESCOPE_H
#define TELESCOPE_H

#include "base/simulation_config.pb.h"

#include <opencv2/core/core.hpp>

#include <memory>
#include <vector>

namespace mats {

class Aperture;
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

  bool include_detector_footprint() const {
    return include_detector_footprint_;
  }
  void set_include_detector_footprint(bool include) {
    include_detector_footprint_ = include;
  }

  bool parallelism() const { return parallelism_; }
  void set_parallelism(bool use) { parallelism_ = use; }

  void set_nonmodeled_mtf(const std::vector<double>& mtf) {
    nonmodeled_mtf_ = mtf;
  }
  void clear_nonmodeled_mtf() { nonmodeled_mtf_.clear(); }

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
             std::vector<cv::Mat>* image) const;

  void Restore(const cv::Mat_<double>& raw_image,
               const std::vector<double>& wavelength,
               const std::vector<double>& illumination,
               int band,
               double smoothness,
               cv::Mat_<double>* restored) const;

  void Restore(const cv::Mat_<double>& raw_image,
               const std::vector<double>& wavelength,
               const std::vector<cv::Mat>& illumination,
               int band,
               double smoothness,
               cv::Mat_<double>* restored) const;

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
                    cv::Mat_<std::complex<double>>* otf) const;

  // Get the transmission spectrum of the telescope optics.
  //
  // Arguments:
  //  wavelengths   The desired wavelengths [m]
  //  transmission  Output: The transmission at each wavelength.
  void GetTransmissionSpectrum(const std::vector<double>& wavelengths,
                               std::vector<double>* transmission) const;

 private:
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

  void ComputeApertureOtf(
      const std::vector<double>& wavelengths,
      double image_height,
      double angle,
      std::vector<cv::Mat_<std::complex<double>>>* otf) const;

  void EffectiveOtf(const std::vector<double>& weights,
                    const std::vector<cv::Mat>& spectral_otf,
                    cv::Mat_<std::complex<double>>* otf) const;

  void OtfDegrade(const cv::Mat& radiance_dft,
                  const cv::Mat& spectral_otf,
                  cv::Mat* degraded) const;

  void GetImagingRegion(const cv::Mat& radiance,
                        cv::Rect* roi) const;

  // Computes the interpolation weights for an isoplanatic region.
  //
  // Arguments:
  //  radila_idx          The radial index of the region
  //  angular_idx         The angular index of the region
  //  isoplanatic_region  Allocated output array of the desired size. Will be
  //                      filled with interpolation weights (0-1).
  void IsoplanaticRegion(int radial_idx,
                         int angular_idx,
                         cv::Mat_<double>* isoplanatic_region) const;

  void GridRegion(cv::Mat_<double>& weights,
                  int x_region, int y_region,
                  int x_regions, int y_regions) const;

  void ImageInIsoplanaticRegion(
      const std::vector<double>& wavelength,
      const std::vector<cv::Mat_<std::complex<double>>>& input_dft,
      int radial_zone,
      int angular_zone,
      std::vector<cv::Mat>* output) const;

 private:
  mats::SimulationConfig sim_config_;
  std::unique_ptr<Aperture> aperture_;
  std::unique_ptr<Detector> detector_;
  std::vector<double> nonmodeled_mtf_;
  bool include_detector_footprint_;
  bool parallelism_;
};

}

#endif  // TELESCOPE_H
