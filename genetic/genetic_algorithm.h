// A generic interface for running genetic algorithms. To customize for your
// specific needs, you will need to inherit from the GeneticAlgorithmImpl base
// class. To be able to customize the data types used for models, the model
// class is a template parameter of GeneticAlgorithmImpl. The typedef model_t
// will be available for use by subclasses. If the desired model type is a
// pointer, than you MUST use a smart pointer, or the memory will be leaked.
// For efficiency's sake, the model type needs to be move-constructable and
// move-assignable.
//
// Author: Philip Salvaggio

#ifndef GENETIC_ALGORITHM_H
#define GENETIC_ALGORITHM_H

#include <cstddef>
#include <vector>

namespace genetic {

// Run the genetic algorithm using the user-defined implementation.
//
// Parameters:
//   impl                  The user-defined subclass of GeneticAlgorithmImpl
//   population_size       The size of the population
//   breeds_per_generation The number of new members to create every iteration
//   best_model            Output: The best model found.
template <typename Impl>
void GeneticAlgorithm(Impl& impl,
                      size_t population_size,
                      size_t breeds_per_generation,
                      typename Impl::model_t& best_model);

// Utility class for wrapping a model and its fitness.
template <typename Model>
class PopulationMember {
 public:
  PopulationMember(Model&& model, double fitness);
  explicit PopulationMember(Model&& model);
  PopulationMember(PopulationMember<Model>&& other);

  PopulationMember& operator=(PopulationMember<Model>&& other);

  Model& model() { return model_; }
  const Model& model() const { return model_; }

  double fitness() const { return fitness_; }
  void set_fitness(double fitness) { fitness_ = fitness; }

 private:
  Model model_;
  double fitness_;
};

template <typename Model>
class GeneticAlgorithmImpl {
 public:
  // Typedef available for user-defined classes to use.
  using model_t = Model;

  // Evaluate should evaluate the user-defined fitness function. The result
  // should be stored using set_fitness.
  //
  // Parameters:
  //   member  The model to evalute.
  //
  // Returns:
  //   A boolean indicating whether the model is valid. If false, the fitness
  //   stored using set_fitness is ignored and the model is dropped from the
  //   population.
  virtual bool Evaluate(PopulationMember<model_t>& member) = 0;

  // Introduce a new model into the population. It is assumed that the model is
  // valid, i.e. Evalute() would return true.
  virtual model_t Introduce() = 0;

  // Perform the genetic crossover opertation. Given two parent models from the
  // population, constructs a new model. The new model is not necessary valid.
  // Any non-determinism should be introduced by the subclass, as this function
  // is called every generation to create the new population members.
  //
  // Parameters:
  //   member1  The first model
  //   member2  The second model
  //
  // Returns:
  //   The model that resulted from the crossover operation.
  virtual model_t Crossover(const PopulationMember<model_t>& member1,
                            const PopulationMember<model_t>& member2) = 0;

  // Performs the mutation operation on the given model. This is called on
  // every new population member produced by Crossover, so any non-determinism
  // must be introduced by the subclass. The mutation may result in the model
  // becoming invalid.
  //
  // Parameters:
  //   member  Input/Output: The model to be modified in-place.
  virtual void Mutate(PopulationMember<model_t>& member) = 0;

  // A convergence test.
  //
  // Parameters:
  //   population      The collection of models currently in the population.
  //                   Their fitnesses will have been computed with Evaluate.
  //   generation_num  The generation number we are on.
  //
  // Returns:
  //   Whether the iteration should continue.
  virtual bool ShouldContinue(
      const std::vector<PopulationMember<model_t>>& population,
      size_t generation_num) = 0;
};

}  // namespace genetic

#include "genetic_algorithm.hpp"

#endif  // GENETIC_ALGORITHM_H
