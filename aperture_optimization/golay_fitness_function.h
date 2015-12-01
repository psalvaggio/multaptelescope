// A genetic algorithm fitness function for generating Golay apertures.
// Author: Philip Salvaggio

#ifndef GOLAY_GENETIC_IMPL_H
#define GOLAY_GENETIC_IMPL_H

#include "aperture_optimization/circular_array.h"
#include "base/simulation_config.pb.h"
#include "genetic/genetic_algorithm.h"
#include <vector>

namespace genetic {

class GolayFitnessFunction : public GeneticFitnessFunction<CircularArray> {
 public:
  GolayFitnessFunction(int num_subapertures,
                       double encircled_diameter,
                       const CircularSubapertureBudget& subap_radii);

  bool operator()(PopulationMember<model_t>& member) override;

  void Visualize(const model_t& locations) override;

 private:
  void GetAutocorrelationPeaks(const model_t& locations,
                               std::vector<CircularAutocorrelationPeak>* peaks);

 private:
  CircularSubapertureBudget subap_radii_;
  double encircled_diameter_;
  std::vector<CircularAutocorrelationPeak> peaks_;
  double max_subap_radius_;
  double total_r2_;
};

}  // namespace genetic

#endif  // GOLAY_GENETIC_IMPL_H
