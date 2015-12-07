// A global search strategy for sparse aperture optimizations that use the
// CircularArray parameterization.
// Author: Philip Salvaggio

#ifndef GLOBAL_CIRCULAR_ARRAY_H
#define GLOBAL_CIRCULAR_ARRAY_H

#include "aperture_optimization/circular_array.h"
#include "base/simulation_config.pb.h"
#include "genetic/genetic_algorithm.h"

#include <array>
#include <chrono>
#include <random>
#include <unordered_set>
#include <vector>

namespace genetic {

class GlobalCircularArray : public GeneticSearchStrategy<CircularArray> {
 public:
  using model_t = CircularArray;
  using SubapertureRadii = std::vector<std::pair<double, int>>;

  GlobalCircularArray(int num_subapertures,
                      double encircled_diameter,
                      const SubapertureRadii& subap_radii,
                      double crossover_probability,
                      double mutate_probability)
      : num_subapertures_(num_subapertures),
        crossover_probability_(crossover_probability),
        mutate_probability_(mutate_probability),
        encircled_diameter_(encircled_diameter),
        subap_radii_(subap_radii),
        should_continue_(true) {}
  
  model_t Introduce(GeneticFitnessFunction<model_t>& fitness_function) override;

  model_t Crossover(const PopulationMember<model_t>& member1,
                    const PopulationMember<model_t>& member2) override;

  void Mutate(PopulationMember<model_t>& member) override;

  bool ShouldContinue(
      const std::vector<PopulationMember<model_t>>&, size_t) override {
    return should_continue_;
  }

  void Stop() override { should_continue_ = false; }

 private:
  int num_subapertures_;
  double crossover_probability_;
  double mutate_probability_;
  double encircled_diameter_;
  SubapertureRadii subap_radii_;
  bool should_continue_;
};

}  // namespace genetic

#endif  // GLOBAL_CIRCULAR_ARRAY_H
