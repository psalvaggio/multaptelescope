// A program to optimize a sparse aperture with free-floating circular
// subapertures.
// Author: Philip Salvaggio

#include "mats.h"

#include "aperture_optimization/acutance_fitness_function.h"
#include "aperture_optimization/annulus_fitness_function.h"
#include "aperture_optimization/golay_fitness_function.h"
#include "aperture_optimization/global_circular_array.h"
#include "aperture_optimization/local_circular_array.h"
#include "aperture_optimization/optimization_main.h"
#include "aperture_optimization/polar_mtf_weighting_fitness_function.h"
#include "optical_designs/compound_aperture_parameters.pb.h"

#include <opencv2/opencv.hpp>

#include <csignal>
#include <cmath>
#include <fstream>
#include <thread>
#include <vector>

#include <gflags/gflags.h>
#include <google/protobuf/text_format.h>
#include <ncurses.h>

DEFINE_int32(subapertures, 6, "Number of subapertures");
DEFINE_int32(population_size, 10, "Population size");
DEFINE_int32(breeds_per_generation, 12, "Number of new apertures to make each "
                                        "generation");
DEFINE_double(fill_factor, 0.18, "Fill Factor");
DEFINE_double(local_stddev, 0.05, "Standard deviation for translating "
                                  "subapertures in local search [fraction of "
                                  "encircled diameter]");
DEFINE_string(base_config, "", "Path to a SimulationConfig file. If specified "
                               "the aperture_params will be replaced with the "
                               "result and outputted to best_aperture.txt");
DEFINE_string(fitness_function, "Annulus", "The name of the fitness function");


static const double kEncircledDiameter = 1;

using namespace std;
using namespace mats;
using namespace genetic;
using model_t = CircularArray;

static unique_ptr<GeneticFitnessFunction<model_t>>
    fitness_function(nullptr);
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

  // 3*r^2 + 3*(a*r)^2 = FR^2
  // r^2 = FR^2/(3 * (a^2 + 1))
  const double kSubapRRatio = 1.0;
  double little_diameter = kEncircledDiameter * sqrt(FLAGS_fill_factor / (
      0.5 * FLAGS_subapertures * (kSubapRRatio * kSubapRRatio + 1)));
  double big_diameter = little_diameter * kSubapRRatio;

  auto subap_budget = MakeCircularSubapertureBudget(
     0.5 * little_diameter, 6);
     //0.5 * little_diameter, 3, 0.5 * big_diameter, 3);

  string fitness_func_str = strtolower(FLAGS_fitness_function);
  if (fitness_func_str == "annulus") {
    fitness_function.reset(new AnnulusFitnessFunction<model_t>(
        FLAGS_subapertures,
        kEncircledDiameter,
        subap_budget));
  } else if (fitness_func_str == "golay") {
    fitness_function.reset(new GolayFitnessFunction<model_t>(
        FLAGS_subapertures,
        kEncircledDiameter,
        subap_budget));
  } else if (fitness_func_str == "acutance") {
    fitness_function.reset(new AcutanceFitnessFunction<model_t>(
        FLAGS_subapertures,
        kEncircledDiameter,
        0.03333,
        subap_budget));
  } else if (fitness_func_str == "dog") {
    const double kStdDevRatio = 0.5;
    const double kPeakFreq = 0.0333;
    double stddev = kPeakFreq / (2 * kStdDevRatio) *
                    sqrt((pow(kStdDevRatio, 2) - 1) / log(kStdDevRatio));
    fitness_function.reset(new PolarMtfWeightingFitnessFunction<model_t>(
        FLAGS_subapertures,
        kEncircledDiameter,
        0.1,
        subap_budget,
        [kStdDevRatio, stddev](double rho, double) {
          return exp(-rho * rho / (2 * pow(stddev, 2))) -
                 exp(-rho * rho / (2 * pow(kStdDevRatio * stddev, 2)));
        }));
  } else {
    cerr << "Unrecognized fitness function: " << FLAGS_fitness_function << endl
         << "Acceptable options are:" << endl
         << "  Annulus" << endl
         << "  Golay" << endl
         << "  Acutance" << endl;
    return 1;
  }

  if (argc < 2) {
    search_strategy.reset(new GlobalCircularArray(
          FLAGS_subapertures,
          kEncircledDiameter,
          subap_budget,
          0.25,
          0.85));
  } else {
    ifstream ifs(argv[1]);
    if (ifs.is_open()) {
      model_t best_guess;
      ifs >> best_guess;
      search_strategy.reset(new LocalCircularArray(
            best_guess,
            0.75,
            FLAGS_local_stddev,
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

  auto genetic =
      genetic::OptimizationMain(*fitness_function,
                                *search_strategy,
                                FLAGS_population_size,
                                FLAGS_breeds_per_generation);

  const auto& best_locations = genetic.best_model();

  ofstream loc_ofs("locations.txt");
  loc_ofs << best_locations;

  if (!FLAGS_base_config.empty()) {
    string config_file = ResolvePath(FLAGS_base_config);
    SimulationConfig sim_config;
    DetectorParameters detector_params;
    if (!MatsInit(config_file, &sim_config, &detector_params)) return 1;

    Simulation ref_sim = sim_config.simulation(0);
    sim_config.clear_simulation();
    auto* sim = sim_config.add_simulation();
    sim->CopyFrom(ref_sim);

    Telescope telescope(sim_config, 0, detector_params);
    vector<double> wavelength{sim_config.reference_wavelength()}, sw{1};
    double q = telescope.EffectiveQ(wavelength, sw);
    double f = kEncircledDiameter * detector_params.pixel_pitch() * q /
               sim_config.reference_wavelength();

    sim->set_name("Optimized Aperture Layout");
    sim->set_focal_length(f);
    sim->clear_aperture_params();
    auto* ap_params = sim->mutable_aperture_params();

    ap_params->set_type(ApertureParameters::COMPOUND);
    ap_params->set_encircled_diameter(kEncircledDiameter);
    ap_params->set_fill_factor(FLAGS_fill_factor);
    
    auto* array_ext = ap_params->MutableExtension(compound_aperture_params);
    array_ext->set_combine_operation(CompoundApertureParameters::OR);
    for (size_t i = 0; i < best_locations.size(); i++) {
      auto* subap = array_ext->add_aperture();
      subap->set_type(ApertureParameters::CIRCULAR);
      subap->set_encircled_diameter(2 * best_locations[i].r);
      subap->set_offset_x(best_locations[i].x);
      subap->set_offset_y(best_locations[i].y);
    }
    
    string output;
    google::protobuf::TextFormat::Printer printer;
    printer.PrintToString(sim_config, &output);

    ofstream ofs("best_aperture.txt");
    ofs << output;
  }

  ofstream ofs("best_aperture_plot.txt");
  ofs << "set parametric" << endl
      << "set xlabel \"X Position [m]\"" << endl
      << "set ylabel \"Y Position [m]\"" << endl
      << "unset key" << endl
      << "set angle degree" << endl
      << "set size square" << endl
      << "set trange [0:360]" << endl
      << "r = " << kEncircledDiameter * 0.5 << endl;
  
  for (size_t i = 0; i < best_locations.size(); i++) {
    ofs << "x" << i << " = " << best_locations[i].x << "; y" << i << " = "
        << best_locations[i].y << "; r" << i << " = "
        << best_locations[i].r << endl;
  }

  ofs << "plot \"-\" u 1:2, r*cos(t), r*sin(t)";

  for (size_t i = 0; i < best_locations.size(); i++) {
    ofs << ", r" << i << "*cos(t) + x" << i << ", r" << i << "*sin(t) + y" << i;
  }
  ofs << endl;

  for (size_t i = 0; i < best_locations.size(); i++) {
    ofs << best_locations[i].x << "\t" << best_locations[i].y << endl;
  }
  cout << endl;

  return 0;
}
