// A genetic algorithm fitness function for maximizing the spatial quality
// factor of a sparse aperture system.
// Author: Philip Salvaggio

#ifndef ACUTANCE_GENETIC_IMPL_H
#define ACUTANCE_GENETIC_IMPL_H

#include "aperture_optimization/circular_array.h"
#include "base/simulation_config.pb.h"
#include "genetic/genetic_algorithm.h"

#include <opencv2/core/core.hpp>
#include <vector>

namespace genetic {

template<typename T>
class AcutanceFitnessFunction : public GeneticFitnessFunction<T> {
 public:
  using model_t = typename GeneticFitnessFunction<T>::model_t;

  AcutanceFitnessFunction(int num_subapertures,
                          double encircled_diameter,
                          double peak_frequency,
                          const CircularSubapertureBudget& subap_radii);

  bool operator()(PopulationMember<model_t>& member) override;

  void Visualize(const model_t& locations) override;

 private:
  void GetAutocorrelationPeaks(const model_t& locations,
                               std::vector<CircularAutocorrelationPeak>* peaks);

 private:
  static const int kSimulationSize;

 private:
  CircularSubapertureBudget subap_radii_;
  double encircled_diameter_;
  std::vector<CircularAutocorrelationPeak> peaks_;
  double max_subap_radius_;
  double total_r2_;
  cv::Mat_<double> csf_;
};

}  // namespace genetic

#include "acutance_fitness_function.hpp"

#endif  // ACUTANCE_GENETIC_IMPL_H
