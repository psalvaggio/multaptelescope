// Implementation file for genetic_algorithm.h
// Author: Philip Salvaggio

#ifndef GENETIC_ALGORITHM_HPP
#define GENETIC_ALGORITHM_HPP

#include <algorithm>
#include <iostream>
#include "genetic_algorithm.h"
#include "io/logging.h"

namespace genetic {


template <typename Model>
GeneticAlgorithm<Model>::GeneticAlgorithm() 
    : generation_lock_(), population_(), running_(false) {}


template<typename Model>
GeneticAlgorithm<Model>::GeneticAlgorithm(GeneticAlgorithm<Model>&& other)
    : generation_lock_(),
      population_(std::move(other.population_)),
      running_(other.running_) {}

template <typename Model>
void GeneticAlgorithm<Model>::Run(
    GeneticFitnessFunction<Model>& fitness_function,
    GeneticSearchStrategy<Model>& searcher,
    size_t population_size,
    size_t breeds_per_generation) {

  // A lambda for sorting the population based on fitness via std::sort.
  auto fitness_sort =
      [](const member_t& a, const member_t& b) {
          return a.fitness() > b.fitness();
      };

  // Randomly selects an index based off of a given probability distribution.
  // The CDF does not need to be normalized, just monotonically increasing.
  auto select_index =
      [](const std::vector<double>& cdf) {
        double p = (double)rand() / RAND_MAX * cdf.back();
        int i = 0;
        while (p > cdf[i]) i++;
        return i;
      };

  // Given the population, construct a cumulative distribution function based
  // on the fitness of the members. Despite the name, the CDF is not
  // normalized.
  auto compute_fitness_cdf =
      [&fitness_sort](const population_t& population,
                      std::vector<double>& fitness_cdf) {
        fitness_cdf.resize(population.size());
        auto minmax_members = std::minmax_element(
            std::begin(population), std::end(population), fitness_sort);
        double min_fitness = minmax_members.second->fitness();
        double max_fitness = minmax_members.first->fitness();
        double range = max_fitness - min_fitness;
        double offset = (range == 0)
            ?  1 - min_fitness : -min_fitness;

        int index = 0;
        for (const auto& tmp : population) {
          fitness_cdf[index] = (index == 0) ?  tmp.fitness() + offset
              : fitness_cdf[index-1] + tmp.fitness() + offset;
          index++;
        }
      };

  // Initialize the population.
  {
    std::lock_guard<std::mutex> lock(generation_lock_);
    population_.clear();
    population_.reserve(population_size + breeds_per_generation);
    for (size_t i = 0; i < population_size; i++) {
      member_t member(std::move(searcher.Introduce(fitness_function)), 0);
      fitness_function(member);
      population_.push_back(std::move(member));
    }
    std::sort(std::begin(population_), std::end(population_), fitness_sort);
  }

  std::vector<double> cumulative_fitness(population_size, 0);
  std::vector<double> new_cumulative_fitness(
      population_size + breeds_per_generation, 0);
  std::vector<size_t> selection_indices(population_size, 0);

  generation_num_ = 0;

  // Main loop of the genetic algorithm.
  do {
    std::lock_guard<std::mutex> lock(generation_lock_);
    running_ = true;
    generation_num_++;

    compute_fitness_cdf(population_, cumulative_fitness);

    // Construct the new population members from the existing ones.
    for (size_t i = 0; i < breeds_per_generation; i++) {
      int index1 = select_index(cumulative_fitness);
      int index2 = index1;
      while (index1 == index2) {
        index2 = select_index(cumulative_fitness);
      }

      do {
        member_t member(std::move(searcher.Crossover(population_[index1],
                                                     population_[index2])));
        searcher.Mutate(member);

        // Only add if the model is valid.
        if (fitness_function(member)) {
          population_.push_back(std::move(member));
          break;
        }
      } while (true);
    }

    // If we extra members hanging around from the previous generation, select
    // the best population member and fill out the population using the fitness
    // to weight selection.
    if (population_.size() > population_size) {
      compute_fitness_cdf(population_, new_cumulative_fitness);

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
        std::swap(population_[i], population_[selection_indices[i]]);
      }
      population_.erase(std::begin(population_) + population_size,
                        std::end(population_));
    }

    std::sort(std::begin(population_), std::end(population_), fitness_sort);
  } while (searcher.ShouldContinue(population_, generation_num_));

  running_ = false;
}


template <typename Model>
const Model& GeneticAlgorithm<Model>::best_model(double* fitness) {
  if (fitness != nullptr) {
    *fitness = population_.at(0).fitness();
  }
  return population_.at(0).model();
}


template <typename Model>
PopulationMember<Model>::PopulationMember(Model&& model, double fitness)
    : model_(std::move(model)), fitness_(fitness) {}


template <typename Model>
PopulationMember<Model>::PopulationMember(Model&& model)
    : PopulationMember(std::move(model), 0) {}

}  // namespace genetic

#endif  // GENETIC_ALGORITHM_HPP
