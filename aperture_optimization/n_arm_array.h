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
    ArmSubaperture(double offset, double r) : offset(offset), r(r) {}

    double offset, r;
  };

  const ArmSubaperture& operator()(size_t index) const;

  const ArmSubaperture& operator()(size_t arm, size_t ap) const {
    return apertures_[arm][ap];
  }

  ArmSubaperture& operator()(size_t arm, size_t ap) {
    cache_dirty_ = false;
    return apertures_[arm][ap];
  }

  size_t NumArms() const { return apertures_.size(); }
  void SetNumArms(size_t arms);

  double ArmAngle(int arm) const { return arm_angles_[arm]; }
  void SetArmAngle(int arm, double angle);

  void AddAperture(int arm, double offset, double r);
  void RemoveAperture(int arm, int ap);

  size_t size() const { return size_; }
  size_t AperturesOnArm(int arm) const { return apertures_[arm].size(); }

  void SwapApertures(int arm1, int ap1, int arm2, int ap2) {
    std::swap(apertures_[arm1][ap1], apertures_[arm2][ap2]);
  }

  void clear();

 private:
  void CreateCircularArray() const;
 
 private:
  std::vector<double> arm_angles_;
  std::vector<std::vector<ArmSubaperture>> apertures_;
  size_t size_;

  mutable CircularArray circ_array_cache_;
  mutable bool cache_dirty_;
};

std::ostream& operator<<(std::ostream& os, const NArmArray& array);
std::istream& operator>>(std::istream& is, NArmArray& array);

}  // namespace genetic

#endif  // N_ARM_ARRAY_H
