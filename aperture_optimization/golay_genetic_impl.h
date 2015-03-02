// File Description
// Author: Philip Salvaggio

#ifndef GOLAY_GENETIC_IMPL_H
#define GOLAY_GENETIC_IMPL_H

#include "aperture_optimization/local_sparse_aperture.h"
#include "base/simulation_config.pb.h"
#include "genetic/genetic_algorithm.h"
#include <vector>

namespace genetic {

class GolayFitnessFunction
    : public GeneticFitnessFunction<std::vector<double>> {
 public:
  GolayFitnessFunction(int num_subapertures,
                       double encircled_diameter,
                       double subaperture_diameter,
                       double reference_wavelength);

  bool operator()(PopulationMember<model_t>& member) override;

  void Visualize(const model_t& locations) override;

 private:
  void GetAutocorrelationPeaks(const model_t& locations, model_t* peaks);

 private:
  mats::SimulationConfig conf_;
  double max_center_radius2_;
  double subaperture_diameter2_;
};

/*
class GolayGeneticImpl : public GeneticAlgorithmImpl<std::vector<double>> {
 public:
  GolayGeneticImpl(int num_subapertures,
                   double max_center_radius,
                   double subaperture_diameter,
                   double mutate_probability,
                   double crossover_probability);

  bool Evaluate(PopulationMember<model_t>& member) override;

  model_t Introduce() override;

  model_t Crossover(const PopulationMember<model_t>& member1,
                    const PopulationMember<model_t>& member2) override;

  void Mutate(PopulationMember<model_t>& member) override;

  bool ShouldContinue(
      const std::vector<PopulationMember<model_t>>&, size_t) override {
    return should_continue_;
  }

  void Stop() { should_continue_ = false; }

  void Visualize(const model_t& model) override;

  void ZeroMean(model_t* locations);

  void GetAutocorrelationPeaks(const model_t& locations, model_t* peaks);

 private:
  int num_subapertures_;
  double max_center_radius2_;
  double subaperture_diameter2_;
  bool should_continue_;
  double mutate_probability_;
  double crossover_probability_;

  mats::SimulationConfig conf_;
};

class GolayLocalImpl : public GeneticAlgorithmImpl<std::vector<double>> {
  public:
   GolayLocalImpl(const model_t& best_guess,
                  double encircled_diameter,
                  double subaperture_diameter,
                  double mutate_probability);

  bool Evaluate(PopulationMember<model_t>& member) override;

  model_t Introduce() override;

  model_t Crossover(const PopulationMember<model_t>& member1,
                    const PopulationMember<model_t>& member2) override;

  void Mutate(PopulationMember<model_t>& member) override;

  bool ShouldContinue(
      const std::vector<PopulationMember<model_t>>&, size_t) override {
    return should_continue_;
  }

  void Stop() { should_continue_ = false; }

  void Visualize(const model_t& model) override;

  void GetAutocorrelationPeaks(const model_t& locations, model_t* peaks);

 private:
  int num_subapertures_;
  double max_center_radius2_;
  double subaperture_diameter2_;
  bool should_continue_;
  LocalSparseAperture<model_t> local_searcher_;

  mats::SimulationConfig conf_;
};
*/

}  // namespace genetic

#endif  // GOLAY_GENETIC_IMPL_H
