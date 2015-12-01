// File Description
// Author: Philip Salvaggio

#ifndef LOCAL_SPARSE_APERTURE_H
#define LOCAL_SPARSE_APERTURE_H

#include "aperture_optimization/circular_array.h"
#include "base/simulation_config.pb.h"
#include "genetic/genetic_algorithm.h"
#include <vector>
#include <chrono>
#include <random>

namespace genetic {

class LocalSparseAperture : public GeneticSearchStrategy<CircularArray> {
 public:
  using model_t = CircularArray;

  LocalSparseAperture(const model_t& best_guess,
                      double mutate_probability,
                      double subap_translate_stdddev,
                      double encircled_diameter)
      : best_guess_(best_guess),
        mutate_probability_(mutate_probability),
        subap_translate_stddev_(encircled_diameter * subap_translate_stdddev),
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


/*
template<typename Model>
void LocalSparseAperture<Model>::Mutate(PopulationMember<model_t>& member) {
  model_t& locations(member.model());

  const double kEncircledRadius = encircled_diameter_ / 2.0;

  for (size_t i = 0; i < locations.size(); i++) {
    double p = (double)rand() / RAND_MAX;
    if (p < mutate_probability_) {
      locations[i].x += distribution_(generator_) * subap_translate_stddev_;
      locations[i].y += distribution_(generator_) * subap_translate_stddev_;
      double r = sqrt(pow(locations[i].x, 2) + pow(locations[i].y, 2));
      if (r > kEncircledRadius - locations[i].r) {
        double scale = 0.999 * (kEncircledRadius - locations[i].r) / r;
        locations[i].x *= scale;
        locations[i].y *= scale;
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
        locations[i].x = (kEncircledRadius - locations[i].r) * new_x;
        locations[i].y = (kEncircledRadius - locations[i].r) * new_y;
      }
    }
  }
}
*/

}  // namespace genetic

#endif  // LOCAL_SPARSE_APERTURE_H
