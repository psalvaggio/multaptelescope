// A parameterization of sparse apertures as a number of arms, which contain
// collinear circular apertures.
// Author: Philip Salvaggio

#ifndef N_ARM_ARRAY_H
#define N_ARM_ARRAY_H

#include "circular_array.h"

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>

namespace genetic {

class NArmArray {
 public:
  NArmArray() = default;
  NArmArray(const NArmArray& other) = default;
  NArmArray(NArmArray&& other) = default;

  NArmArray& operator=(const NArmArray& other) = default;
  NArmArray& operator=(NArmArray&& other) = default;

  const CircularSubaperture& operator[](int index) const;

  struct ArmSubaperture {
    ArmSubaperture(int arm, double offset, double r)
        : arm(arm), offset(offset), r(r) {}

    int arm;
    double offset, r;
  };

  // ArmSubaperture Index
  const ArmSubaperture& operator()(size_t index) const {
    return apertures_[index];
  }
  ArmSubaperture& operator()(size_t index) {
    cache_dirty_ = true;
    return apertures_[index];
  }

  // Returns the number of arms in the array
  size_t NumArms() const { return arm_angles_.size(); }

  // Add an arm to the array with the given angle [radians CCW of +x]
  void AddArm(double angle);

  // Set the number of arms, this will clear all previous arms/apertures
  void SetNumArms(size_t arms);

  // Get the angle of the requested arm
  double ArmAngle(int arm) const { return arm_angles_[arm]; }

  // Set the angle of an arm
  void SetArmAngle(int arm, double angle);

  // Add an aperture to the array
  //
  // Arguments:
  //  arm     Arm to which to add the aperture
  //  offset  The offset from the center of the array [m]
  //  r       The radius of the aperture [m]
  void AddAperture(int arm, double offset, double r);

  // Get the number of apertures in the array
  size_t size() const { return apertures_.size(); }

  // Clear the arms and apertures of the array
  void clear();

 private:
  void CreateCircularArray() const;
 
 private:
  std::vector<double> arm_angles_;
  std::vector<ArmSubaperture> apertures_;

  mutable CircularArray circ_array_cache_;
  mutable bool cache_dirty_;
};

std::ostream& operator<<(std::ostream& os, const NArmArray& array);
std::istream& operator>>(std::istream& is, NArmArray& array);

}  // namespace genetic

#endif  // N_ARM_ARRAY_H
