// Implementation file for genetic_algorithm.h
// Author: Philip Salvaggio

#ifndef GENETIC_ALGORITHM_HPP
#define GENETIC_ALGORITHM_HPP

#include <algorithm>
#include <iostream>
#include "genetic_algorithm.h"

namespace genetic {

template <typename Impl>
void GeneticAlgorithm(Impl& impl,
                      size_t population_size,
                      size_t breeds_per_generation,
                      typename Impl::model_t& best_model) {
  using Model = typename Impl::model_t;

  // A lambda for sorting the population based on fitness via std::sort.
  auto fitness_sort =
      [](const PopulationMember<Model>& a,
         const PopulationMember<Model>& b) -> bool {
          return a.fitness() > b.fitness();
      };

  // Randomly selects an index based off of a given probability distribution.
  // The CDF does not need to be normalized, just monotonically increasing.
  auto select_index =
      [](const std::vector<double>& cdf) -> int {
        double p = (double)rand() / RAND_MAX * cdf.back();
        int i = 0;
        while (p > cdf[i]) i++;
        return i;
      };

  // Given the population, construct a cumulative distribution function based
  // on the fitness of the members. Despite the name, the CDF is not
  // normalized.
  auto compute_fitness_cdf =
      [&fitness_sort](const std::vector<PopulationMember<Model>>& population,
                      std::vector<double>& fitness_cdf) -> void {
        fitness_cdf.resize(population.size());
        auto min_member = std::max_element(
            std::begin(population), std::end(population), fitness_sort);
        double min_fitness = min_member->fitness();

        int index = 0;
        for (const PopulationMember<Model>& tmp : population) {
          fitness_cdf[index] = (index == 0) ?  tmp.fitness() - min_fitness
              : fitness_cdf[index-1] + tmp.fitness() - min_fitness;
          index++;
        }
      };

  // Initialize the population.
  std::vector<PopulationMember<Model>> population;
  population.reserve(population_size + breeds_per_generation);
  for (size_t i = 0; i < population_size; i++) {
    PopulationMember<Model> member(std::move(impl.Introduce()), 0);
    impl.Evaluate(member);
    population.push_back(std::move(member));
  }
  std::sort(std::begin(population), std::end(population), fitness_sort);

  std::vector<double> cumulative_fitness(population_size, 0);
  std::vector<double> new_cumulative_fitness(
      population_size + breeds_per_generation, 0);
  std::vector<size_t> selection_indices(population_size, 0);

  size_t generation_num = 0;

  // Main loop of the genetic algorithm.
  do {
    generation_num++;

    compute_fitness_cdf(population, cumulative_fitness);

    // Construct the new population members from the existing ones.
    for (size_t i = 0; i < breeds_per_generation; i++) {
      int index1 = select_index(cumulative_fitness);
      int index2 = index1;
      while (index1 == index2) {
        index2 = select_index(cumulative_fitness);
      }

      do {
        PopulationMember<Model> member(
            std::move(impl.Crossover(population[index1], population[index2])));
        impl.Mutate(member);

        // Only add if the model is valid.
        if (impl.Evaluate(member)) {
          population.push_back(std::move(member));
          break;
        }
      } while (true);
    }

    // If we extra members hanging around from the previous generation, select
    // the best population member and fill out the population using the fitness
    // to weight selection.
    if (population.size() > population_size) {
      compute_fitness_cdf(population, new_cumulative_fitness);

      selection_indices[0] = 0;
      for (size_t i = 1; i < population_size; i++) {
        bool unique = false;
        size_t index = 0;

        while (!unique) {
          index = select_index(new_cumulative_fitness);
          unique = true;
          for (int j = i - 1; j >= 0; j--) {
            if (selection_indices[j] == index) {
              unique = false;
              break;
            }
          }
        }
        selection_indices[i] = index;
      }

      // Sort the selection indices in ascending order, so we can select them in
      // the population in one pass.
      std::sort(std::begin(selection_indices), std::end(selection_indices));

      // Construct the new population based on the selections.
      for (size_t i = 0; i < population_size; i++) {
        if (i == selection_indices[i]) continue;
        std::swap(population[i], population[selection_indices[i]]);
      }
      population.erase(std::begin(population) + population_size,
                       std::end(population));
    }

    std::sort(std::begin(population), std::end(population), fitness_sort);

    std::cout << "\rGeneration " << generation_num
              << ", Fitness = " << population[0].fitness();
    std::cout.flush();
    impl.Visualize(population[0].model());
  } while (impl.ShouldContinue(population, generation_num));

  best_model = Model(population[0].model());
}

template <typename Model>
PopulationMember<Model>::PopulationMember(Model&& model, double fitness)
    : model_(std::move(model)), fitness_(fitness) {}

template <typename Model>
PopulationMember<Model>::PopulationMember(Model&& model)
    : PopulationMember(std::move(model), 0) {}

template <typename Model>
PopulationMember<Model>::PopulationMember(PopulationMember<Model>&& other)
    : PopulationMember(std::move(other.model_), other.fitness_) {}

template <typename Model>
PopulationMember<Model>& PopulationMember<Model>::operator=(
    PopulationMember<Model>&& other) {
  model_ = std::move(other.model_);
  fitness_ = other.fitness_;
  return *this;
}

}  // namespace genetic

#endif  // GENETIC_ALGORITHM_HPP
