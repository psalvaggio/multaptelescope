// A main function to run aperture optimizations.
// Author: Philip Salvaggio

#ifndef OPTIMIZATION_MAIN_H_
#define OPTIMIZATION_MAIN_H_

#include "genetic/genetic_algorithm.h"

#include <ncurses.h>

#include <thread>

namespace genetic {

template<typename model_t>
GeneticAlgorithm<model_t> OptimizationMain(
    const GeneticFitnessFunction<model_t>& fitness_function,
    GeneticSearchStrategy<model_t>& search_strategy,
    int population_size,
    int breeds_per_generation) {
  GeneticAlgorithm<model_t> genetic;
  std::thread genetic_thread(
      [&] () {
        genetic.Run(fitness_function,
                    search_strategy,
                    population_size,
                    breeds_per_generation);
      });

  while (!genetic.running()) usleep(1000);

  initscr();
  cbreak();
  timeout(100);
  noecho();
  keypad(stdscr, TRUE);
  raw();
  nonl();

  mvprintw(LINES - 1, 0, "'q' to exit");

  int keycode = 0;
  double prev_best = 0;
  clock_t prev_time = clock();
  int prev_gen = 0;
  while ((keycode = getch()) != 'q') {
    usleep(1e5);
    std::lock_guard<std::mutex> lock(genetic.generation_lock());

    const auto& population = genetic.population();

    int gen_num = genetic.generation_num();
    if (gen_num - prev_gen > 1000) {
      clock_t time = clock();
      double elapsed_secs = double(time - prev_time) / CLOCKS_PER_SEC;
      double time_per_gen = elapsed_secs / (gen_num - prev_gen);
      mvprintw(LINES - 2, 0, "Time/generation = %f [ms]", time_per_gen * 1000);
      prev_gen = gen_num;
      prev_time = time;
    }

    mvprintw(0, 0, "Generation %d", gen_num);
    int member_idx = 0;
    for (const auto& member : population) {
      mvprintw(member_idx+1, 0, "Member %d: %f", member_idx, member.fitness());
      member_idx++;
    }
    if (population[0].fitness() != prev_best) {
      fitness_function.Visualize(population[0].model());
      prev_best = population[0].fitness();
    }
  }

  endwin();

  search_strategy.Stop();
  genetic_thread.join();
  return genetic;
}
                      

} 

#endif
