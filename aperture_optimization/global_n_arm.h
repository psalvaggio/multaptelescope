// A global search strategy for sparse aperture optimization that use the
// NArmArray constrained parameterization.
// Author: Philip Salvaggio

#ifndef GLOBAL_N_ARM_H
#define GLOBAL_N_ARM_H

#include "aperture_optimization/n_arm_array.h"
#include "base/simulation_config.pb.h"
#include "genetic/genetic_algorithm.h"

#include <array>
#include <chrono>
#include <random>
#include <unordered_set>
#include <vector>

namespace genetic {

class GlobalNArm : public GeneticSearchStrategy<NArmArray> {
 public:
  using model_t = NArmArray;

  GlobalNArm(int num_arms,
             double encircled_diameter,
             const CircularSubapertureBudget& subap_radii,
             double crossover_probability,
             double mutate_probability)
      : num_arms_(num_arms),
        crossover_probability_(crossover_probability),
        mutate_probability_(mutate_probability),
        encircled_diameter_(encircled_diameter),
        subap_radii_(subap_radii),
        should_continue_(true) {}
  
  model_t Introduce(const GeneticFitnessFunction<model_t>& fitness_function)
      override;

  model_t Crossover(const PopulationMember<model_t>& member1,
                    const PopulationMember<model_t>& member2) override;

  void Mutate(PopulationMember<model_t>& member) override;

  bool ShouldContinue(
      const std::vector<PopulationMember<model_t>>&, size_t) override {
    return should_continue_;
  }

  void Stop() override { should_continue_ = false; }

 private:
  int num_arms_;
  double crossover_probability_;
  double mutate_probability_;
  double encircled_diameter_;
  CircularSubapertureBudget subap_radii_;
  bool should_continue_;
};

}  // namespace genetic

#endif  // GLOBAL_R_ARM_H
