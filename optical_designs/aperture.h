// Abstraction of the aperture function of an optical system.
// Author: Philip Salvaggio

#ifndef APERTURE_H
#define APERTURE_H

#include <opencv/cv.h>

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
  const mats::Simulation& simulation_params() const { return sim_params_; }

  // Get the complex-valued pupil function of this aperture, which can be used
  // to calculate the MTF/PSF due to the diffraction of the aperture and the
  // aberrations modeled by the specific implementation of this class.
  //
  // Arguments:
  //  pupil  Output parameter to hold the resulting pupil function.
  void GetPupilFunction(mats::PupilFunction* pupil) const;

 private:
  // Abstract method to get the mask of the aperture.
  //
  // Returns:
  //  Square array with size of params().array_size() that is the magnitude of
  //  the pupil function.
  virtual cv::Mat GetApertureTemplate() const = 0;

  // Abstract method to the get the optical path length difference across the
  // aperture. This is a virtual function, as different apertures may want to
  // model this differently, such as piston/tip/tilt only, Zernike polynomials,
  // or with actual data.
  //
  // Returns:
  //  Square array with size of params().array_size() and data type CV_64FC1
  //  that represents the optical path length difference in waves.
  virtual cv::Mat GetOpticalPathLengthDiff() const = 0;

 // Utility functions for subclasses
 protected:
  // Get an instantiation of random piston/tip/tilt (PTT) error.
  //
  // Arguments:
  //  ptt  Output: 3-element array to hold the piston/tip/tilt coefficients
  //       that were applied.
  //
  // Returns:
  //  See GetPistonTipTilt()
  cv::Mat GetRandomPistonTipTilt(double* ptt) const;

  // Given the true PTT error, construct an estimate given the simulations
  // wavefront error knowledge level.
  //
  // Arguments:
  //  ptt_truth      The true values of piston/tip/tilt
  //  ptt_estimates  Output: 3-element array for the piston/tip/tilt
  //                 coefficients in the wavefront error estimate.
  //
  // Returns:
  //  See GetPistonTipTilt()
  cv::Mat GetPistonTipTiltEstimate(double* ptt_truth,
                                   double* ptt_estimates) const;

 private:
  // Given piston/tip/tilt coefficients, compute the optical path length
  // differences.
  //
  // Arguments:
  //  ptt  3-element array [piston, tip, tilt] [waves]
  //       piston is the offset of the aperture
  //       tip is the difference between the center and edge of the aperture
  //       along the x-axis.
  //       tilt is the difference between the center and edge of the aperture
  //       along the y-axis.
  //
  // Returns:
  //  Square array with size of params().array_size() and data type CV_64FC1
  //  that represents the PTT optical path length difference in waves. This
  //  array will have to be scaled to fit the RMS error desired for the
  //  simulation.
  cv::Mat GetPistonTipTilt(double* ptt) const;

 private:
  const mats::SimulationConfig& params_;
  const mats::Simulation& sim_params_;
};

#endif  // APERTURE_H
