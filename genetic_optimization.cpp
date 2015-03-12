// File Description
// Author: Philip Salvaggio

#include "aperture_optimization/golay_genetic_impl.h"
#include "aperture_optimization/global_sparse_aperture.h"
#include "aperture_optimization/local_sparse_aperture.h"
#include "genetic/genetic_algorithm.h"
#include "base/pupil_function.h"
#include "optical_designs/compound_aperture.h"

//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "base/opencv_utils.h"
#include "base/str_utils.h"

#include <csignal>
#include <cmath>
#include <fstream>
#include <vector>

static const int kNumPoints = 6;
static const double kEncircledDiameter = 3;
static const double kSubapertureDiameter = 0.52;
static const double kReferenceWavelength = 550e-9;

static const int kPopulationSize = 8;
//static const double kCrossoverProbability = 0;
//static const double kMutateProbability = 1;
static const int kBreedsPerGeneration = 4;

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
  signal(SIGINT, stop_iteration);
  srand(time(NULL));

  GolayFitnessFunction fitness_function(kNumPoints,
                                        kEncircledDiameter,
                                        kSubapertureDiameter,
                                        kReferenceWavelength);

  is_global = argc < 2;
  if (is_global) {
    search_strategy.reset(new GlobalSparseAperture<model_t>(
          kNumPoints,
          kEncircledDiameter,
          kSubapertureDiameter,
          0.25,
          0.75));
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
            0.5,
            kEncircledDiameter));
    } else {
      cerr << "File: " << argv[1] << " is not readable." << endl;
      exit(1);
    }
  }
  /*
  cv::namedWindow("MTF", cv::WINDOW_AUTOSIZE);
  cv::moveWindow("MTF", 0, 0);
  cv::namedWindow("Mask", cv::WINDOW_AUTOSIZE);
  cv::moveWindow("Mask", 0, 600);
  cv::namedWindow("Best MTF", cv::WINDOW_AUTOSIZE);
  cv::moveWindow("Best MTF", 600, 0);
  cv::namedWindow("Best Mask", cv::WINDOW_AUTOSIZE);
  cv::moveWindow("Best Mask", 600, 600);
  */


  typename GolayFitnessFunction::model_t best_locations;
  GeneticAlgorithm(fitness_function,
                   *search_strategy,
                   kPopulationSize,
                   kBreedsPerGeneration,
                   best_locations);

  ofstream ofs("locations.txt");
  ofs << "set parametric" << endl
      << "unset key" << endl
      << "set angle degree" << endl
      << "set size square" << endl
      << "set trange [0:360]" << endl
      << "r = " << kEncircledDiameter * 0.5 << endl
      << "r2 = " << kSubapertureDiameter * 0.5 << endl;
  
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
