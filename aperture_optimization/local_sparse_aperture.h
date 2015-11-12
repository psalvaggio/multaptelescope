// File Description
// Author: Philip Salvaggio

#ifndef LOCAL_SPARSE_APERTURE_H
#define LOCAL_SPARSE_APERTURE_H

#include "base/simulation_config.pb.h"
#include "genetic/genetic_algorithm.h"
#include <vector>
#include <chrono>
#include <random>

namespace genetic {

template<typename Model>
class LocalSparseAperture : public GeneticSearchStrategy<Model> {
 public:
  using model_t = typename GeneticSearchStrategy<Model>::model_t;

  LocalSparseAperture(const model_t& best_guess,
                      double mutate_probability,
                      double encircled_diameter)
      : best_guess_(best_guess),
        mutate_probability_(mutate_probability),
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
  double encircled_diameter_;
  bool should_continue_;

  std::default_random_engine generator_;
  std::normal_distribution<double> distribution_;
};


template<typename Model>
void LocalSparseAperture<Model>::Mutate(PopulationMember<model_t>& member) {
  model_t& locations(member.model());

  double sigma = 0.1 * encircled_diameter_;
  double max_center_radius = encircled_diameter_ / 2.0;
  double max_center_radius2 = pow(max_center_radius, 2);

  for (size_t i = 0; i < locations.size(); i += 2) {
    double p = (double)rand() / RAND_MAX;
    if (p < mutate_probability_) {
      locations[i] += distribution_(generator_) * sigma;
      locations[i+1] += distribution_(generator_) * sigma;
      double r2 = pow(locations[i], 2) + pow(locations[i+1], 2);
      if (r2 > max_center_radius2) {
        double r = sqrt(r2);
        locations[i] *= max_center_radius / r;
        locations[i+1] *= max_center_radius / r;
      }
      if ((double)rand() / RAND_MAX < 0.01) {
        double new_x = 2 * (double(rand()) / RAND_MAX - 0.5);
        double new_y = 2 * (double(rand()) / RAND_MAX - 0.5);
        double r2 = new_x * new_x + new_y * new_y;
        if (r2 > 1) {
          double r = sqrt(r2);
          new_x /= r;
          new_y /= r;
        }
        locations[i] = max_center_radius * new_x;
        locations[i+1] = max_center_radius * new_y;
      }
    }
  }
}

}  // namespace genetic

#endif  // LOCAL_SPARSE_APERTURE_H
