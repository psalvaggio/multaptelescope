// Abstraction of the aperture function of an optical system.
// Author: Philip Salvaggio

#ifndef APERTURE_H
#define APERTURE_H

#include <opencv2/core/core.hpp>

#include "base/macros.h"
#include "base/simulation_config.pb.h"
#include "base/aperture_parameters.pb.h"
#include "registry/registry.h"

namespace mats {

class Simulation;
class PupilFunction;

// Abstraction of an optical system's aperture function. The magnitude of the
// aperture function is a binary mask of the aperture, while the phase is the
// optical path length difference in radians.
class Aperture {
 public:
  // Constructor
  //
  // Arguments:
  //  params     The Simulation object containing the parameters for this
  //             run of the model.
  explicit Aperture(const Simulation& params);

  // Destructor
  virtual ~Aperture();

  // No copy or assignment
  Aperture(const Aperture& other) = delete;
  Aperture& operator=(const Aperture& other) = delete;

  // Accessors
  const Simulation& simulation_params() const { return sim_params_; }

  const ApertureParameters& aperture_params() const { return aperture_params_; }

  const std::vector<double>& on_axis_aberrations() const {
    return on_axis_aberrations_;
  }

  const std::vector<double>& off_axis_aberrations() const {
    return off_axis_aberrations_;
  }

  bool HasOffAxisAberration() const;

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
                        double image_height,
                        double angle,
                        PupilFunction* pupil) const;

  void GetPupilFunction(const std::vector<double>& wavelength,
                        double image_height,
                        double angle,
                        int size,
                        double reference_wavelength,
                        std::vector<PupilFunction>* pupil) const;

  // Get the true wavefront error across the aperture.
  //
  // Parameters:
  //  size           Side length of the output array
  //  image_height   Fractional height in the image plane (0-1) for off-axis
  //                 aberrations.
  //  angle          Angle in the image plane for off-axis aberrations. Radians
  //                 CCW of the +x axis.
  //
  // Returns:
  //  Square (CV_64FC1) array that represents the wavefront error in waves.
  cv::Mat GetWavefrontError(int size, double image_height, double angle) const;
  void GetWavefrontError(double image_height,
                         double angle,
                         cv::Mat_<double>* output) const;

  // Get the mask of the aperture. This array will be represent the aperture
  // transmission at each point. Traditionally, it is a binary array, but it is
  // represented as a double array for generality. Unlike the pupil function,
  // the aperture fills the whole array, since the user should not be taking
  // the Fourier transform of this array.
  //
  // Parameters:
  //  size   Side length of the output array
  //
  // Retures:
  //   Square (CV_64FC1) array that represents the aperture mask.
  cv::Mat GetApertureMask(int size) const;
  void GetApertureMask(cv::Mat_<double>* output) const;

 protected:
  bool IsOffAxis(ZernikeCoefficient::AberrationType type) const;
  bool HasAberration(ZernikeCoefficient::AberrationType type) const;
  double GetAberration(ZernikeCoefficient::AberrationType type) const;

  void ZernikeWavefrontError(double image_height,
                             double angle,
                             cv::Mat_<double>* output) const;

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
  virtual void GetOpticalPathLengthDiff(double image_height,
                                        double angle,
                                        cv::Mat_<double>* output) const;

 private:
  Simulation sim_params_;
  ApertureParameters aperture_params_;
  std::vector<double> on_axis_aberrations_;
  std::vector<double> off_axis_aberrations_;

 private:  // Cache variables
  mutable cv::Mat_<double> mask_;
  mutable cv::Mat_<double> on_axis_opd_;
  mutable double encircled_diameter_;
  mutable double fill_factor_;
};


// Factory class that can be used to construct Aperture subclasses, based on
// the aperture_type() in the Simulation protobuf.
class ApertureFactory {
 public:
   ApertureFactory() = delete;

   static Aperture* Create(const Simulation& params);
};

using ApertureFactoryImpl =
    registry::Registry<Aperture, const Simulation&>;
}

#define REGISTER_APERTURE(class_name, ap_config_enum) \
  REGISTER_SUBCLASS_W_IDENTIFIER(Aperture, class_name, ap_config_enum, \
                                 const mats::Simulation&)

#endif  // APERTURE_H
