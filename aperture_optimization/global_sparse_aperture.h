// File Description
// Author: Philip Salvaggio

#ifndef GLOBAL_SPARSE_APERTURE_H
#define GLOBAL_SPARSE_APERTURE_H

#include "base/simulation_config.pb.h"
#include "genetic/genetic_algorithm.h"
#include <vector>
#include <chrono>
#include <random>

namespace genetic {

template<typename Model>
class GlobalSparseAperture : public GeneticSearchStrategy<Model> {
 public:
  using model_t = typename GeneticSearchStrategy<Model>::model_t;

  GlobalSparseAperture(int num_subapertures,
                       double encircled_diameter,
                       double subaperture_diameter,
                       double crossover_probability,
                       double mutate_probability)
      : num_subapertures_(num_subapertures),
        crossover_probability_(crossover_probability),
        mutate_probability_(mutate_probability),
        encircled_diameter_(encircled_diameter),
        subaperture_diameter_(subaperture_diameter),
        max_center_radius2_(
            pow(0.5 * (encircled_diameter - subaperture_diameter), 2)),
        should_continue_(true),
        generator_(std::chrono::system_clock::now().time_since_epoch().count()),
        distribution_() {}
  
  model_t Introduce(GeneticFitnessFunction<model_t>& fitness_function) override;

  model_t Crossover(const PopulationMember<model_t>& member1,
                    const PopulationMember<model_t>& member2) override;

  void Mutate(PopulationMember<model_t>& member) override;

  bool ShouldContinue(
      const std::vector<PopulationMember<model_t>>&, size_t) override {
    return should_continue_;
  }

  void Stop() { should_continue_ = false; }

  void ZeroMean(model_t* locations);

 private:
  int num_subapertures_;
  double crossover_probability_;
  double mutate_probability_;
  double encircled_diameter_;
  double subaperture_diameter_;
  double max_center_radius2_;
  bool should_continue_;

  std::default_random_engine generator_;
  std::normal_distribution<double> distribution_;
};

template<typename Model>
typename GlobalSparseAperture<Model>::model_t
GlobalSparseAperture<Model>::Introduce(
    GeneticFitnessFunction<model_t>& fitness_function) {
  model_t tmp_model;
  PopulationMember<model_t> member(std::move(tmp_model));
  model_t& locations(member.model());
  locations.resize(2*num_subapertures_, 0);

  bool keep_going = true;

  while (keep_going) {
    for (int i = 0; i < num_subapertures_; i++) {
      double r = (double)rand() / RAND_MAX * sqrt(max_center_radius2_);
      double theta = (double)rand() / RAND_MAX * 2 * M_PI;
      locations[2*i] = r * cos(theta);
      locations[2*i + 1] = r * sin(theta);
    }
    keep_going = !fitness_function(member);
  }

  model_t new_model = std::move(member.model());
  //ZeroMean(&new_model);
  return new_model;
}

template<typename Model>
typename GlobalSparseAperture<Model>::model_t
GlobalSparseAperture<Model>::Crossover(
    const PopulationMember<model_t>& member1,
    const PopulationMember<model_t>& member2) {
  const model_t& input1_locs(member1.model());
  const model_t& input2_locs(member2.model());
  model_t output_locs;
  output_locs.resize(2*num_subapertures_, 0);

  for (size_t i = 0; i < input1_locs.size(); i += 2) {
    double p = (double)rand() / RAND_MAX;
    if (p < crossover_probability_) {
      output_locs[i] = input2_locs[i];
      output_locs[i+1] = input2_locs[i+1];
    } else {
      output_locs[i] = input1_locs[i];
      output_locs[i+1] = input1_locs[i+1];
    }
  }

  return output_locs;
}

template<typename Model>
void GlobalSparseAperture<Model>::Mutate(PopulationMember<model_t>& member) {
  model_t& locations(member.model());

  for (size_t i = 0; i < locations.size(); i += 2) {
    double p = (double)rand() / RAND_MAX;
    if (p < mutate_probability_) {
      double new_r = sqrt(max_center_radius2_) * (double)rand() / RAND_MAX;
      double new_theta = 2 * M_PI * (double)rand() / RAND_MAX;
      double new_x = new_r * cos(new_theta);
      double new_y = new_r * sin(new_theta);

      if (new_r * new_r < max_center_radius2_) {
        locations[i] = new_x;
        locations[i+1] = new_y;
      }
    }
  }

  //ZeroMean(&locations);
}

template<typename Model>
void GlobalSparseAperture<Model>::ZeroMean(model_t* locations) {
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

#endif  // GLOBAL_SPARSE_APERTURE_H
