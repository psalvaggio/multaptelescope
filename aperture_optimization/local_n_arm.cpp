// File Description
// Author: Philip Salvaggio

#include "local_n_arm.h"

namespace genetic {

void LocalNArm::Mutate(PopulationMember<model_t>& member) {
  model_t& array(member.model());

  const double kEncircledRadius = encircled_diameter_ / 2.0;

  // Randomly offset the arms
  for (size_t i = 1; i < array.NumArms(); i++) {
    double p = (double)rand() / RAND_MAX;
    if (p < mutate_probability_) {
      array.SetArmAngle(i, array.ArmAngle(i) + distribution_(generator_) *
                        M_PI / 8);
    }
  }

  // Randomly modify the aperture offsets on each arm
  for (size_t i = 0; i < array.size(); i++) {
    double p = (double)rand() / RAND_MAX;
    if (p < mutate_probability_) {
      auto& ap = array(i);
      ap.offset += distribution_(generator_) * subap_translate_stddev_;
      ap.offset = std::min(ap.offset, 0.999 * (kEncircledRadius - ap.r));

      if ((double)rand() / RAND_MAX < 0.01) {
        ap.arm = rand() % array.NumArms();
        ap.offset = ((double)rand() / RAND_MAX) * (kEncircledRadius - 2 * ap.r)
                    + ap.r;
      }
    }
  }
}

}  // namespace genetic
