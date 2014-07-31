// Abstraction of the aperture function of an optical system.
// Author: Philip Salvaggio

#ifndef APERTURE_H
#define APERTURE_H

#include <opencv/cv.h>

#include "base/macros.h"
#include "base/simulation_config.pb.h"
#include "base/aperture_parameters.pb.h"

namespace mats {
class SimulationConfig;
class Simulation;
class PupilFunction;
}

// Abstraction of an optical system's aperture function. The magnitude of the
// aperture function is a binary mask of the aperture, while the phase is the
// optical path length difference in radians.
class Aperture {
 public:
  // Constructor
  //
  // Arguments:
  //  params     The SimulationConfig object containing the parameters for this
  //             run of the model.
  //  sim_index  The index in params.simulation() of the simulation for which 
  //             this aperture is being created.
  Aperture(const mats::SimulationConfig& params, int sim_index);

  // Destructor
  virtual ~Aperture();

  // Accessors
  const mats::SimulationConfig& params() const { return params_; }
  const mats::Simulation& simulation_params() const {
    return params_.simulation(0);
  }
  mats::ApertureParameters& aperture_params() { return aperture_params_; }
  const mats::ApertureParameters& aperture_params() const {
    return aperture_params_;
  }
  std::vector<double>& aberrations() { return aberrations_; }
  const std::vector<double>& aberrations() const { return aberrations_; }

  virtual double fill_factor();
  virtual double encircled_diameter();

  // Get the complex-valued pupil function of this aperture, which can be used
  // to calculate the MTF/PSF due to the diffraction of the aperture and the
  // aberrations modeled by the specific implementation of this class.
  //
  // Arguments:
  //  wfe         The wavefront error map of the aperture [waves]
  //  wavelength  The wavelength at which the pupil function is desired [m]
  //  pupil       Output parameter to hold the resulting pupil function
  void GetPupilFunction(const cv::Mat& wfe,
                        double wavelength,
                        mats::PupilFunction* pupil);

  // Get the true wavefront error across the aperture.
  //
  // Returns:
  //  Square array with size of params().array_size() and data type CV_64FC1
  //  that represents the optical path length difference in waves.
  cv::Mat GetWavefrontError();

  // Get the estimate of the wavefront error across the aperture. The quality
  // of the estimate is based wfe_knowledge() attribute in the Simulation
  // parameters that were given.
  //
  // Returns:
  //  Square array with size of params().array_size() and data type CV_64FC1
  cv::Mat GetWavefrontErrorEstimate();

  // Get the mask of the aperture. This array will be represent the aperture
  // transmission at each point. Traditionally, it is a binary array, but it is
  // represented as a double array for generality. Unlike the pupil function,
  // the aperture fills the whole array, since the user should not be taking
  // the Fourier transform of this array.
  //
  // Retures:
  //   Square array with size of params().array_size() and data type CV_64FC1
  cv::Mat GetApertureMask();

 private:
  // Abstract method to get the mask of the aperture.
  //
  // Returns:
  //  Square array with size of params().array_size() that is the magnitude of
  //  the pupil function.
  virtual cv::Mat GetApertureTemplate() = 0;

  // Abstract method to the get the optical path length difference across the
  // aperture. This is a virtual function, as different apertures may want to
  // model this differently, such as piston/tip/tilt only, Zernike polynomials,
  // or with actual data.
  //
  // Returns:
  //  Square array with size of params().array_size() and data type CV_64FC1
  //  that represents the optical path length difference in waves.
  virtual cv::Mat GetOpticalPathLengthDiff() = 0;

  // Abstract method to the get the estimate of the optical path length 
  // difference across the aperture. The quality of the estimate is determined
  // by the wfe_knowledge() method in the simulation parameters. This quality
  // setting may mean different things to different apertures, so this function
  // is left virtual.
  //
  // Returns:
  //  Square array with size of params().array_size() and data type CV_64FC1
  //  that represents the optical path length difference estimate in waves.
  virtual cv::Mat GetOpticalPathLengthDiffEstimate() = 0;

 // Utility functions for subclasses
 protected:
  // Given piston/tip/tilt coefficients, compute the optical path length
  // differences (OPD).
  //
  // Arguments:
  //  piston  Offset of the OPD for the aperture [waves]
  //  tip     OPD between the center and edge of the aperture along the x-axis.
  //  tilt    OPD between the center and edge of the aperture along the y-axis.
  //  rows    The number of rows in the output array.
  //  cols    The number of columns in the output array.
  //
  // Returns:
  //  rows x cols array of data type CV_64FC1 that represents the PTT optical
  //  path length difference in waves. This array will have to be scaled to
  //  fit the RMS error desired for the simulation.
  cv::Mat GetPistonTipTilt(double piston, double tip, double tilt,
                           size_t rows, size_t cols) const;

  // Utility method to get the piston/tip/tilt for a square array of size
  // params_.array_size().
  cv::Mat GetPistonTipTilt(double piston, double tip, double tilt) const;

 private:
  mats::SimulationConfig params_;
  mats::ApertureParameters aperture_params_;
  std::vector<double> aberrations_;
};


// Factory class that can be used to construct Aperture subclasses, based on
// the aperture_type() in the Simulation protobuf.
class ApertureFactory {
 public:
   static Aperture* Create(const mats::SimulationConfig& params, int sim_index);

 NO_CONSTRUCTION(ApertureFactory)
};


#endif  // APERTURE_H
