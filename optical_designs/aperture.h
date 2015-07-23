// Abstraction of the aperture function of an optical system.
// Author: Philip Salvaggio

#ifndef APERTURE_H
#define APERTURE_H

#include <opencv/cv.h>

#include "base/macros.h"
#include "base/simulation_config.pb.h"
#include "base/aperture_parameters.pb.h"
#include "registry/registry.h"

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

  const mats::ApertureParameters& aperture_params() const {
    return aperture_params_;
  }

  const std::vector<double>& aberrations() const { return aberrations_; }

  double fill_factor() const;
  double encircled_diameter() const;

  // Get the complex-valued pupil function of this aperture, which can be used
  // to calculate the MTF/PSF due to the diffraction of the aperture and the
  // aberrations modeled by the specific implementation of this class.
  //
  // Arguments:
  //  wavelength  The wavelength at which the pupil function is desired [m]
  //  pupil       Output parameter to hold the resulting pupil function
  void GetPupilFunction(double wavelength,
                        mats::PupilFunction* pupil) const;

  // Same as GetPupilFunction(), except calculated with the wavefront error
  // estimate.
  void GetPupilFunctionEstimate(double wavelength,
                                mats::PupilFunction* pupil) const;

  // Get the true wavefront error across the aperture.
  //
  // Parameters:
  //  size   Side length of the output array (-1 for params().array_size()).
  //
  // Returns:
  //  Square (CV_64FC1) array that represents the wavefront error in waves.
  cv::Mat GetWavefrontError(int size = -1) const;
  void GetWavefrontError(cv::Mat_<double>* output) const;

  // Get the estimate of the wavefront error across the aperture. The quality
  // of the estimate is based wfe_knowledge() attribute in the Simulation
  // parameters that were given.
  //
  // Parameters:
  //  size   Side length of the output array (-1 for params().array_size()).
  //
  // Returns:
  //  Square (CV_64FC1) array that represents the wvefront error estimate.
  cv::Mat GetWavefrontErrorEstimate(int size = -1) const;
  void GetWavefrontErrorEstimate(cv::Mat_<double>* output) const;

  // Get the mask of the aperture. This array will be represent the aperture
  // transmission at each point. Traditionally, it is a binary array, but it is
  // represented as a double array for generality. Unlike the pupil function,
  // the aperture fills the whole array, since the user should not be taking
  // the Fourier transform of this array.
  //
  // Parameters:
  //  size   Side length of the output array (-1 for params().array_size()).
  //
  // Retures:
  //   Square (CV_64FC1) array that represents the aperture mask.
  cv::Mat GetApertureMask(int size = -1) const;
  void GetApertureMask(cv::Mat_<double>* output) const;

 private:
  // Gets the encircled diameter of the aperture.
  virtual double GetEncircledDiameter() const;

  // Abstract method to get the mask of the aperture.
  //
  // Parameters:
  //  output  Output: The aperture template |p[x,y]|. The size of the array will
  //                  already be set and is guaranteed to be square.
  virtual void GetApertureTemplate(cv::Mat_<double>* output) const = 0;

  // Abstract method to the get the optical path length difference across the
  // aperture. This is a virtual function, as different apertures may want to
  // model this differently, such as piston/tip/tilt only, Zernike polynomials,
  // or with actual data.
  //
  // Returns:
  //  Square array with size of params().array_size() and data type CV_64FC1
  //  that represents the optical path length difference in waves.
  virtual void GetOpticalPathLengthDiff(cv::Mat_<double>* output) const = 0;

  // Abstract method to the get the estimate of the optical path length 
  // difference across the aperture. The quality of the estimate is determined
  // by the wfe_knowledge() method in the simulation parameters. This quality
  // setting may mean different things to different apertures, so this function
  // is left virtual.
  //
  // Returns:
  //  Square array with size of params().array_size() and data type CV_64FC1
  //  that represents the optical path length difference estimate in waves.
  virtual void GetOpticalPathLengthDiffEstimate(
      cv::Mat_<double>* output) const = 0;

  // Internal helper function for GetPupilFunction() and
  // GetPupilFunctionEstimate().
  void GetPupilFunctionHelper(
      double wavelength,
      mats::PupilFunction* pupil,
      std::function<void(cv::Mat_<double>*)> wfe_generator) const;

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

 private:  // Cache variables
  mutable cv::Mat_<double> mask_;
  mutable cv::Mat_<double> opd_;
  mutable cv::Mat_<double> opd_est_;
  mutable double encircled_diameter_;
  mutable double fill_factor_;
};


// Factory class that can be used to construct Aperture subclasses, based on
// the aperture_type() in the Simulation protobuf.
class ApertureFactory {
 public:
   static Aperture* Create(const mats::SimulationConfig& params, int sim_index);

 NO_CONSTRUCTION(ApertureFactory)
};

using ApertureFactoryImpl =
    registry::Registry<Aperture, const mats::SimulationConfig&, int>;
#define REGISTER_APERTURE(class_name, ap_config_enum) \
  REGISTER_SUBCLASS_W_IDENTIFIER(Aperture, class_name, ap_config_enum, \
                                 const mats::SimulationConfig&, int)

#endif  // APERTURE_H
