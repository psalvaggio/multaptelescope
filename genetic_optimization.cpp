// File Description
// Author: Philip Salvaggio

#include "aperture_optimization/golay_genetic_impl.h"
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
static const double kSubapertureDiameter = 0.625;

static const int kPopulationSize = 8;
static const double kCrossoverProbability = 0;
static const double kMutateProbability = 1;
static const int kBreedsPerGeneration = 4;

using namespace std;
using namespace genetic;

static GolayGeneticImpl impl(
    kNumPoints,
    0.5 * (kEncircledDiameter - kSubapertureDiameter),
    kSubapertureDiameter,
    kMutateProbability,
    kCrossoverProbability);

static bool has_stopped = false;
void stop_iteration(int) {
  if (has_stopped) {
    std::cout << std::endl;
    exit(1);
  } else {
    impl.Stop();
    has_stopped = true;
  }
}

int main() {
  signal(SIGINT, stop_iteration);
  srand(time(NULL));
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


  typename GolayGeneticImpl::model_t best_locations;
  GeneticAlgorithm(impl,
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
      << "r2 = " << kSubapertureDiameter * 0.5 << endl
      << "plot \"-\" u 1:2, r*cos(t), r*sin(t)";

  for (size_t i = 0; i < best_locations.size(); i += 2) {
    ofs << ", r2*cos(t)";

    if (best_locations[i] >= 0) {
      ofs << " + " << best_locations[i];
    } else {
      ofs << best_locations[i];
    }
    ofs << ",r2*sin(t)";
    if (best_locations[i+1] >= 0) {
      ofs << " + " << best_locations[i+1];
    } else {
      ofs << best_locations[i+1];
    }
  }
  ofs << endl;
  for (size_t i = 0; i < best_locations.size(); i += 2) {
    ofs << best_locations[i] << "\t" << best_locations[i+1] << endl;
  }
  std::cout << std::endl;

  return 0;
}
