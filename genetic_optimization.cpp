// File Description
// Author: Philip Salvaggio

#include "mats.h"

#include "aperture_optimization/golay_fitness_function.h"
#include "aperture_optimization/global_sparse_aperture.h"
#include "aperture_optimization/local_sparse_aperture.h"

#include <opencv2/opencv.hpp>

#include <csignal>
#include <cmath>
#include <fstream>
#include <thread>
#include <vector>

#include <gflags/gflags.h>
#include <ncurses.h>

DEFINE_int32(subapertures, 6, "Number of subapertures");
DEFINE_double(encircled_diameter, 1, "Encircled Diameter [m]");
DEFINE_double(fill_factor, 0.18, "Fill Factor");


static const double kReferenceWavelength = 550e-9;

static const int kPopulationSize = 10;
static const int kBreedsPerGeneration = 12;

using namespace std;
using namespace genetic;
using model_t = vector<double>;

static unique_ptr<GeneticSearchStrategy<model_t>>
    search_strategy(nullptr);
static bool is_global;

static bool has_stopped = false;
void stop_iteration(int) {
  if (has_stopped) {
    std::cout << std::endl;
    exit(1);
  } else {
    if (search_strategy.get()) {
      if (is_global) {
        auto* searcher =
            dynamic_cast<GlobalSparseAperture<model_t>*>(search_strategy.get());
        searcher->Stop();
      } else {
        auto* searcher =
            dynamic_cast<LocalSparseAperture<model_t>*>(search_strategy.get());
        searcher->Stop();
      }
    }
    has_stopped = true;
  }
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  mats_io::Logging::Init();

  signal(SIGINT, stop_iteration);
  srand(time(NULL));

  // F = nd^2 / D^2 -> d = D * sqrt(F / n)
  double subap_diameter = FLAGS_encircled_diameter * sqrt(FLAGS_fill_factor /
      FLAGS_subapertures);


  vector<double> truth_pos{
    -0.269461, -0.145774,
    -0.136939, -0.375480,
     0.260628, -0.145774,
     0.393150,  0.083930,
    -0.004417,  0.313636,
    -0.269461,  0.313636};
  double max_rad = 0;
  for (size_t i = 0; i < truth_pos.size(); i += 2) {
    max_rad = max(sqrt(pow(truth_pos[i], 2) + pow(truth_pos[i+1], 2)), max_rad);
  }
  for (size_t i = 0; i < truth_pos.size(); i++) {
    truth_pos[i] *= 0.999 * (0.5 * (FLAGS_encircled_diameter - subap_diameter))
                    / max_rad;
  }

  GolayFitnessFunction fitness_function(FLAGS_subapertures,
                                        FLAGS_encircled_diameter,
                                        subap_diameter,
                                        kReferenceWavelength);

  PopulationMember<vector<double>> truth(move(truth_pos), 0);
  fitness_function(truth);
  cout << "Truth: " << truth.fitness() << endl;

  is_global = argc < 2;
  if (is_global) {
    search_strategy.reset(new GlobalSparseAperture<model_t>(
          FLAGS_subapertures,
          FLAGS_encircled_diameter,
          subap_diameter,
          0.25,
          0.85));
  } else {
    ifstream ifs(argv[1]);
    if (ifs.is_open()) {
      string line;
      model_t best_guess;
      while (getline(ifs, line)) {
        best_guess.push_back(atof(line.c_str()));
      }

      search_strategy.reset(new LocalSparseAperture<model_t>(
            best_guess,
            0.75,
            FLAGS_encircled_diameter));
    } else {
      cerr << "File: " << argv[1] << " is not readable." << endl;
      exit(1);
    }
  }

  cv::namedWindow("Best MTF", cv::WINDOW_AUTOSIZE);
  cv::moveWindow("Best MTF", 600, 0);
  cv::namedWindow("Best Mask", cv::WINDOW_AUTOSIZE);
  cv::moveWindow("Best Mask", 600, 600);

  auto raw_search_strategy = search_strategy.get();

  GeneticAlgorithm<typename GolayFitnessFunction::model_t> genetic;
  std::thread genetic_thread(
      [&genetic, &fitness_function, raw_search_strategy] () {
        genetic.Run(fitness_function,
                    *raw_search_strategy,
                    kPopulationSize,
                    kBreedsPerGeneration);
      });

  cout << "Waiting for genetic algorithm to start..." << endl;
  while (!genetic.running()) usleep(1000);
  cout << "Genetic algorithm started." << endl;


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
  while (!has_stopped && (keycode = getch()) != 'q') {
    usleep(1e5);
    lock_guard<mutex> lock(genetic.generation_lock());

    const auto& population = genetic.population();

    mvprintw(0, 0, "Generation %d", genetic.generation_num());
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

  if (!has_stopped) {
    stop_iteration(0);
  }
  genetic_thread.join();


  const auto& best_locations = genetic.best_model();

  ofstream loc_ofs("locations.txt");
  for (size_t i = 0; i < best_locations.size(); i++) {
    loc_ofs << best_locations[i] << endl;
  }

  ofstream ofs("best_aperture_plot.txt");
  ofs << "set parametric" << endl
      << "unset key" << endl
      << "set angle degree" << endl
      << "set size square" << endl
      << "set trange [0:360]" << endl
      << "r = " << FLAGS_encircled_diameter * 0.5 << endl
      << "r2 = " << subap_diameter * 0.5 << endl;
  
  for (size_t i = 0; i < best_locations.size(); i += 2) {
    ofs << "x" << (i/2) << " = " << best_locations[i] << "; y" << (i/2)
        << " = " << best_locations[i+1] << endl;
  }

  ofs << "plot \"-\" u 1:2, r*cos(t), r*sin(t)";

  for (size_t i = 0; i < best_locations.size(); i += 2) {
    ofs << ", r2*cos(t) + x" << (i/2) << ", r2*sin(t) + y" << (i/2);
  }
  ofs << endl;

  for (size_t i = 0; i < best_locations.size(); i += 2) {
    ofs << best_locations[i] << "\t" << best_locations[i+1] << endl;
  }
  std::cout << std::endl;

  return 0;
}
