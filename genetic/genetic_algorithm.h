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
#include <mutex>
#include <vector>

namespace genetic {

template <typename Model> class GeneticFitnessFunction;
template <typename Model> class GeneticSearchStrategy;
template <typename Model> class PopulationMember;

template <typename Model>
class GeneticAlgorithm {
 public:
   using member_t = PopulationMember<Model>;
   using population_t = std::vector<member_t>;

   GeneticAlgorithm();

   GeneticAlgorithm(GeneticAlgorithm<Model>&& other);

   // Run the genetic algorithm using the user-defined implementation.
   //
   // Parameters:
   //   fitness_function      The user-defined GolayFitnessFunction.
   //   searcher              The user-defined GolaySearchStragety.
   //   population_size       The size of the population
   //   breeds_per_generation The number of new members to create every
   //                         iteration
   //   best_model            Output: The best model found.
   void Run(const GeneticFitnessFunction<Model>& fitness_function,
            GeneticSearchStrategy<Model>& searcher,
            size_t population_size,
            size_t breeds_per_generation);

   std::mutex& generation_lock() { return generation_lock_; }

   // NOTE: If the algorithm is currently running, then acquire
   // generation_lock() before calling these method. The returned value will
   // only be valid while the lock is held.
   
   // Get the best model and optionally its fitness.
   const Model& best_model(double* fitness = NULL);

   // Get the population of the genetic algorithm.
   const population_t& population() const { return population_; }

   // Whether the algorithm is currently running
   bool running() const { return running_; }

   // The current generation number
   int generation_num() const { return generation_num_; }

 private:
  std::mutex generation_lock_;
  population_t population_;
  bool running_;
  int generation_num_;
};


// Concept for a fitness function.
template <typename Model>
class GeneticFitnessFunction {
 public:
  // Typedef available for user-defined classes to use.
  using model_t = Model;

  virtual ~GeneticFitnessFunction() {}

  // Required: Evaluate the fitness function.
  //
  // Arguments:
  //  member  The population member. The result should be stored in this
  //          object using set_fitness().
  virtual bool operator()(PopulationMember<model_t>& member) const = 0;

  // Optional: Visualize the best model. This is called at the end of each
  // iteration. The defualt implementation does nothing.
  //
  // Parameters:
  //   model   The best model from the current generation.
  virtual void Visualize(const model_t&) const {}
};


// Concept for a search strategy.
template <typename Model>
class GeneticSearchStrategy {
 public:
  // Typedef available for user-defined classes to use.
  using model_t = Model;

  // Introduce a new model into the population. It is assumed that the model is
  // valid, i.e. the fitness function would return true.
  virtual model_t Introduce(
      const GeneticFitnessFunction<model_t>& fitness_function) = 0;

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

  // Manually stops the iteration.
  virtual void Stop() = 0;
};


// Utility class for wrapping a model and its fitness.
template <typename Model>
class PopulationMember {
 public:
  PopulationMember(Model&& model, double fitness);
  explicit PopulationMember(Model&& model);
  PopulationMember(PopulationMember<Model>&& other) = default;

  PopulationMember& operator=(PopulationMember<Model>&& other) = default;

  Model& model() { return model_; }
  const Model& model() const { return model_; }

  double fitness() const { return fitness_; }
  void set_fitness(double fitness) { fitness_ = fitness; }

 private:
  Model model_;
  double fitness_;
};

}  // namespace genetic

#include "genetic_algorithm.hpp"

#endif  // GENETIC_ALGORITHM_H
