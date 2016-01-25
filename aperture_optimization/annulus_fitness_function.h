// A genetic algorithm fitness function for generating apertures that
// approximate the MTF properties of an annulus.
// Author: Philip Salvaggio

#ifndef ANNULUS_GENETIC_IMPL_H
#define ANNULUS_GENETIC_IMPL_H

#include "aperture_optimization/circular_array.h"
#include "base/simulation_config.pb.h"
#include "genetic/genetic_algorithm.h"

#include <opencv2/core/core.hpp>

#include <vector>

namespace genetic {

template<typename T>
class AnnulusFitnessFunction : public GeneticFitnessFunction<T> {
 public:
  using model_t = T;

  AnnulusFitnessFunction(int num_subapertures,
                         double encircled_diameter,
                         const CircularSubapertureBudget& subap_radii);

  bool operator()(PopulationMember<model_t>& member) const override;

  void Visualize(const model_t& locations) const override;

 private:
  void GetAutocorrelationPeaks(
      const model_t& locations,
      std::vector<CircularAutocorrelationPeak>* peaks) const;

 private:
  static const int kMtfSize;

 private:
  CircularSubapertureBudget subap_radii_;
  double encircled_diameter_;
  mutable std::vector<CircularAutocorrelationPeak> peaks_;
  double max_subap_radius_;
  double total_r2_;
  std::vector<cv::Mat_<uint8_t>> radial_masks_;
  std::vector<int> mask_radii_;
};

}  // namespace genetic

#include "annulus_fitness_function.hpp"

#endif  // ANNULUS_GENETIC_IMPL_H
