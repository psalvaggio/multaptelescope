// File Description
// Author: Philip Salvaggio

#include "mats.h"

#include "aperture_optimization/golay_fitness_function.h"
#include "aperture_optimization/global_sparse_aperture.h"
#include "aperture_optimization/local_sparse_aperture.h"
#include "aperture_optimization/optimization_main.h"

#include <opencv2/opencv.hpp>

#include <csignal>
#include <cmath>
#include <fstream>
#include <thread>
#include <vector>

#include <gflags/gflags.h>
#include <ncurses.h>

DEFINE_int32(subapertures, 6, "Number of subapertures");
DEFINE_int32(population_size, 10, "Population size");
DEFINE_int32(breeds_per_generation, 12, "Number of new apertures to make each "
                                        "generation");
DEFINE_double(fill_factor, 0.18, "Fill Factor");


static const double kEncircledDiameter = 10;

using namespace std;
using namespace genetic;
using model_t = vector<double>;

static unique_ptr<GeneticSearchStrategy<model_t>>
    search_strategy(nullptr);

static bool has_stopped = false;
void stop_iteration(int) {
  if (has_stopped) {
    std::cout << std::endl;
    exit(1);
  } else {
    if (search_strategy.get()) search_strategy->Stop();
    has_stopped = true;
  }
}

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  mats_io::Logging::Init();

  signal(SIGINT, stop_iteration);
  srand(time(NULL));

  // F = nd^2 / D^2 -> d = D * sqrt(F / n)
  double subap_diameter = kEncircledDiameter * sqrt(FLAGS_fill_factor /
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
    truth_pos[i] *= 0.999 * (0.5 * (kEncircledDiameter - subap_diameter))
                    / max_rad;
  }

  GolayFitnessFunction fitness_function(FLAGS_subapertures,
                                        kEncircledDiameter,
                                        subap_diameter);

  PopulationMember<vector<double>> truth(move(truth_pos), 0);
  fitness_function(truth);
  cout << "Truth: " << truth.fitness() << endl;

  if (argc < 2) {
    search_strategy.reset(new GlobalSparseAperture<model_t>(
          FLAGS_subapertures,
          kEncircledDiameter,
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
            kEncircledDiameter));
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

  auto genetic =
      genetic::OptimizationMain(fitness_function,
                                *raw_search_strategy,
                                FLAGS_population_size,
                                FLAGS_breeds_per_generation);

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
      << "r = " << kEncircledDiameter * 0.5 << endl
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
