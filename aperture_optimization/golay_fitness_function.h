// A genetic algorithm fitness function for generating Golay apertures.
// Author: Philip Salvaggio

#ifndef GOLAY_GENETIC_IMPL_H
#define GOLAY_GENETIC_IMPL_H

#include "base/simulation_config.pb.h"
#include "genetic/genetic_algorithm.h"
#include <vector>

namespace genetic {

class GolayFitnessFunction
    : public GeneticFitnessFunction<std::vector<double>> {
 public:
  GolayFitnessFunction(int num_subapertures,
                       double encircled_diameter,
                       double subaperture_diameter);

  bool operator()(PopulationMember<model_t>& member) override;

  void Visualize(const model_t& locations) override;

 private:
  void GetAutocorrelationPeaks(const model_t& locations, model_t* peaks);

 private:
  double max_center_radius2_;
  double subaperture_diameter2_;
  double encircled_diameter_;
  std::vector<double> peaks_;
};

}  // namespace genetic

#endif  // GOLAY_GENETIC_IMPL_H
