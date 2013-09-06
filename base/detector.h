// File Description
// Author: Philip Salvaggio

#ifndef DETECTOR_H
#define DETECTOR_H

#include "base/detector_parameters.pb.h"
#include "base/simulation_config.pb.h"

#include <vector>
#include <opencv/cv.h>

namespace mats {

class Detector {
 public:
  Detector(const DetectorParameters& det_params,
           const SimulationConfig& sim_params,
           int sim_index);

  // Accessors/Mutators
  const DetectorParameters& det_params() const { return det_params_; }
  const SimulationConfig& sim_params() const { return sim_params_; }
  const Simulation& simulation() const {
    return sim_params_.simulation(sim_index_);
  }

  int sim_index() const { return sim_index_; }
  void set_sim_index(int idx) { sim_index_ = idx; }


  int rows() const { return det_params_.array_rows(); }
  int cols() const { return det_params_.array_cols(); }
  double pixel_pitch() const { return det_params_.pixel_pitch(); }

  // Get the quantum efficiency of the detector at the requested wavelengths.
  //
  // Arguments:
  //  wavelengths The wavelengths at which the QE is requested. [m]
  //  qe          Output: The QE corresponding to the wavelengths.
  void GetQESpectrum(const std::vector<double>& wavelengths,
                     std::vector<double>* qe) const;

  // Convert the radiance field reaching the system into electrons on the 
  // detector.
  //
  // Arguments:
  //  radiance     The spectral radiance reaching the front of the system. Each
  //               band should be the same size as the detector. The units
  //               should be [W/m^2/sr um^-1]
  //  wavelengths  The wavelengths corresponding to the spectral radiances. [m]
  //  electrons    Output: Electron responses for each of the detector's
  //               spectral bands.
  void ResponseElectrons(const std::vector<cv::Mat>& radiance,
                         const std::vector<double>& wavelengths,
                         std::vector<cv::Mat>* electrons);

  // Aggregate a spectral signal into the bands of the detector.
  //
  // Arguments:
  //  signal      The spectral signal which is to be aggregated.
  //  wavelengths The wavelengths corresponding to the elements in signal.
  //  output      Output: The aggregated signal into bands.
  void AggregateSignal(const std::vector<cv::Mat>& signal,
                       const std::vector<double>& wavelengths,
                       std::vector<cv::Mat>* output);

  // Get the MTF that results from the sampling of the detector. This assumes
  // that the detector elements have a fill factor of 1 and are perfectly
  // uniform.
  //
  // Returns:
  //  A complex array the size of the detector that holds the OTF.
  //  Zero-frequency is at (0, 0) and the frequencies range from 0 
  //  cycles/detector to +/- N/2 cycles per detector, where N is the
  //  number of pixels along a dimension.
  cv::Mat GetSamplingOtf();

  // Get the OTF due to image smearing effects.
  //
  // Arguments:
  //   x_velocity  The velocity of the sensor in the x direction [m/s]
  //   y_velocity  The velocity of the sensor in the y direction [m/s]
  //
  // Returns:
  //  A complex array the size of the detector that holds the OTF.
  //  Zero-frequency is at (0, 0) and the frequencies range from 0 
  //  cycles/detector to +/- N/2 cycles per detector, where N is the
  //  number of pixels along a dimension.
  cv::Mat GetSmearOtf(double x_velocity, double y_velocity);

  // Get the OTF due to jittering of the optical system.
  //
  // Arguments:
  //  jitter_std_dev  Jitter is assumed to be Gaussian. This is the standard
  //                  deviation of the vibration distribution [pixels]
  //
  // Returns:
  //  A complex array the size of the detector that holds the OTF.
  //  Zero-frequency is at (0, 0) and the frequencies range from 0 
  //  cycles/detector to +/- N/2 cycles per detector, where N is the
  //  number of pixels along a dimension.
  cv::Mat GetJitterOtf(double jitter_std_dev);

 private:
  cv::Mat GetNoisePattern() const;

 private:
  DetectorParameters det_params_;
  SimulationConfig sim_params_;
  int sim_index_;
};

}

#endif  // DETECTOR_H
