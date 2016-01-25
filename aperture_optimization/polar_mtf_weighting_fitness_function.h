// A fitness function that applies a weighting function to the MTF as a function
// of radial/angular frequency
// Author: Philip Salvaggio


#ifndef POLAR_MTF_WEIGHTING_FITNESS_FUNCTION_H
#define POLAR_MTF_WEIGHTING_FITNESS_FUNCTION_H

#include "aperture_optimization/circular_array.h"
#include "genetic/genetic_algorithm.h"

#include <functional>
#include <opencv2/core/core.hpp>
#include <vector>

namespace genetic {

template<typename T>
class PolarMtfWeightingFitnessFunction : public GeneticFitnessFunction<T> {
 public:
  using model_t = typename GeneticFitnessFunction<T>::model_t;

  PolarMtfWeightingFitnessFunction(
      int num_subapertures,
      double encircled_diameter,
      double maximum_mtf_value,
      const CircularSubapertureBudget& subap_radii,
      std::function<double(double, double)> weighting);

  bool operator()(PopulationMember<model_t>& member) const override;

  void Visualize(const model_t& locations) const override;

 private:
  void GetAutocorrelationPeaks(
      const model_t& locations,
      std::vector<CircularAutocorrelationPeak>* peaks) const;

 private:
  static const int kSimulationSize;

 private:
  CircularSubapertureBudget subap_radii_;
  double encircled_diameter_;
  mutable std::vector<CircularAutocorrelationPeak> peaks_;
  double max_subap_radius_;
  double total_r2_;
  cv::Mat_<double> weighting_;
  double max_mtf_val_;
};

}

#include "polar_mtf_weighting_fitness_function.hpp"

#endif
