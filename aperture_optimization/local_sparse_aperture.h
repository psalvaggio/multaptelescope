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

  void Stop() { should_continue_ = false; }

  void ZeroMean(model_t* locations);

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

  double sigma = 0.025 * encircled_diameter_;

  for (size_t i = 0; i < locations.size(); i += 2) {
    double p = (double)rand() / RAND_MAX;
    if (p < mutate_probability_) {
      locations[i] += distribution_(generator_) * sigma;
      locations[i+1] += distribution_(generator_) * sigma;
    }
  }

  //ZeroMean(&locations);
}

template<typename Model>
void LocalSparseAperture<Model>::ZeroMean(model_t* locations) {
  if (!locations) return;

  double mean_x = 0;
  double mean_y = 0;
  for (size_t i = 0; i < locations->size(); i += 2) {
    mean_x += (*locations)[i];
    mean_y += (*locations)[i+1];
  }
  mean_x /= locations->size() / 2;
  mean_y /= locations->size() / 2;

  for (size_t i = 0; i < locations->size(); i += 2) {
    (*locations)[i] -= mean_x;
    (*locations)[i+1] -= mean_y;
  }
}

}  // namespace genetic

#endif  // LOCAL_SPARSE_APERTURE_H
