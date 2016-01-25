// File Description
// Author: Philip Salvaggio

#ifndef LOCAL_N_ARM_H
#define LOCAL_N_ARM_H

#include "aperture_optimization/n_arm_array.h"
#include "base/simulation_config.pb.h"
#include "genetic/genetic_algorithm.h"

#include <array>
#include <chrono>
#include <random>
#include <unordered_set>
#include <vector>

namespace genetic {

class LocalNArm : public GeneticSearchStrategy<NArmArray> {
 public:
  using model_t = NArmArray;

  LocalNArm(const model_t& best_guess,
            double mutate_probability,
            double subap_translate_stddev,
            double encircled_diameter)
      : best_guess_(best_guess),
        mutate_probability_(mutate_probability),
        subap_translate_stddev_(subap_translate_stddev),
        encircled_diameter_(encircled_diameter),
        should_continue_(true),
        generator_(std::chrono::system_clock::now().time_since_epoch().count()),
        distribution_() {}
  
  model_t Introduce(const GeneticFitnessFunction<model_t>&) override {
    return best_guess_;
  }

  model_t Crossover(const PopulationMember<model_t>& member1,
                    const PopulationMember<model_t>& member2) override {
    return (double)rand() / RAND_MAX < 0.5 ? member1.model() : member2.model();
  }

  void Mutate(PopulationMember<model_t>& member) override;

  bool ShouldContinue(
      const std::vector<PopulationMember<model_t>>&, size_t) override {
    return should_continue_;
  }

  void Stop() override { should_continue_ = false; }

 private:
  model_t best_guess_;
  double mutate_probability_;
  double subap_translate_stddev_;
  double encircled_diameter_;
  bool should_continue_;

  std::default_random_engine generator_;
  std::normal_distribution<double> distribution_;
};

}  // namespace genetic

#endif  // LOCAL_R_ARM_H
