// A local search strategy for sparse aperture optimizations that use the
// CircularArray parameterization.
// Author: Philip Salvaggio

#ifndef LOCAL_CIRCULAR_ARRAY_H
#define LOCAL_CIRCULAR_ARRAY_H

#include "aperture_optimization/circular_array.h"
#include "base/simulation_config.pb.h"
#include "genetic/genetic_algorithm.h"
#include <vector>
#include <chrono>
#include <random>

namespace genetic {

class LocalCircularArray : public GeneticSearchStrategy<CircularArray> {
 public:
  using model_t = CircularArray;

  LocalCircularArray(const model_t& best_guess,
                     double mutate_probability,
                     double subap_translate_stddev,
                     double encircled_diameter)
      : best_guess_(best_guess),
        mutate_probability_(mutate_probability),
        subap_translate_stddev_(encircled_diameter * subap_translate_stddev),
        encircled_diameter_(encircled_diameter),
        should_continue_(true),
        generator_(std::chrono::system_clock::now().time_since_epoch().count()),
        distribution_() {}
  
  model_t Introduce(GeneticFitnessFunction<model_t>&) override {
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

#endif  // LOCAL_CIRCULAR_ARRAY_H
